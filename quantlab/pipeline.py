"""Orchestration: OHLCV + ExperimentConfig -> features, labels, walk-forward, OOS signals, backtest, metrics.

This is the only module that knows the order of operations. The Streamlit UI
calls into here and renders what comes back; the tests call into here with
synthetic data. Nothing in this file draws anything.
"""

from __future__ import annotations

import copy
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from quantlab.backtest.engine import BacktestResult, run_backtest
from quantlab.backtest.metrics import (
    DSRResult,
    RandomBenchmarkResult,
    deflated_sharpe,
    performance_table,
    random_strategy_benchmark,
)
from quantlab.config import ExperimentConfig, ModelConfig
from quantlab.data.calendar import bars_per_year
from quantlab.features.registry import build_feature_matrix
from quantlab.labeling import build_labels, class_balance
from quantlab.models import build_model
from quantlab.models.base import ModelAdapter
from quantlab.models.meta import SIDE_COL, rule_side
from quantlab.seeding import seed_everything
from quantlab.validation.leakage import check_alignment, check_feature_matrix, check_folds, check_labels_not_in_features
from quantlab.validation.walkforward import Fold, WalkForwardSplitter

ProgressFn = Callable[[dict[str, Any]], None]


# --------------------------------------------------------------------------- #
# Dataset assembly
# --------------------------------------------------------------------------- #
@dataclass
class Dataset:
    prices: pd.DataFrame  # trimmed OHLCV, aligned with X
    X: pd.DataFrame
    labels: pd.DataFrame  # label / ret / t1 / weight (+ extras)
    t1_pos: np.ndarray  # positional resolution index per row, -1 if none
    n_trimmed: int  # warm-up rows dropped from the top
    feature_columns: list[str]
    interval: str

    @property
    def y(self) -> pd.Series:
        return self.labels["label"]

    @property
    def weights(self) -> pd.Series:
        return self.labels["weight"].fillna(0.0)

    @property
    def n(self) -> int:
        return len(self.X)


def prepare_dataset(df: pd.DataFrame, cfg: ExperimentConfig) -> Dataset:
    """Features + labels on one index, warm-up trimmed from the top, leakage checks run."""
    cfg.features.validate()
    cfg.label.validate()
    X = build_feature_matrix(df, cfg.features.features, shift=1)
    check_feature_matrix(X, df.index, min_shift=1)
    labels = build_labels(df, cfg.label)
    if cfg.model.meta.enabled and cfg.model.meta.primary != "model":
        # The rule sees prices through the same one-bar shift as every feature.
        side = rule_side(df, cfg.model.meta.primary, cfg.model.meta.primary_params).shift(1)
        X[SIDE_COL] = side
    check_labels_not_in_features(X.drop(columns=[SIDE_COL], errors="ignore"), labels, df["close"])

    # Trim the warm-up: first row from which every feature is non-NaN. A global,
    # positional trim keeps X, labels and prices aligned by construction.
    feat_cols = [c for c in X.columns if c != SIDE_COL]
    complete = X[feat_cols].notna().all(axis=1).to_numpy()
    if not complete.any():
        raise ValueError("No row has all features defined. Some lookback is longer than the data.")
    first = int(np.argmax(complete))
    X, labels, prices = X.iloc[first:], labels.iloc[first:], df.iloc[first:]
    check_alignment(X, labels["label"], {"weight": labels["weight"], "t1": labels["t1"]})

    t1_pos = np.full(len(X), -1, dtype=int)
    has_t1 = labels["t1"].notna().to_numpy()
    t1_pos[has_t1] = X.index.get_indexer(labels.loc[has_t1, "t1"])
    return Dataset(prices=prices, X=X, labels=labels, t1_pos=t1_pos, n_trimmed=first, feature_columns=feat_cols, interval=str(df.attrs.get("interval", cfg.data.interval)))


# --------------------------------------------------------------------------- #
# Walk-forward
# --------------------------------------------------------------------------- #
@dataclass
class FoldResult:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_train: int
    n_test: int
    train_balance: dict[int, float]
    test_balance: dict[int, float]
    model: ModelAdapter
    params_used: dict[str, Any]
    oos: pd.DataFrame  # per test bar: y_true, p_*, y_pred, fold
    insample: pd.DataFrame  # per train bar: y_true, p_*, y_pred (greyed in UI)
    fit_seconds: float
    tune_seconds: float = 0.0
    context_start: pd.Timestamp | None = None
    train_accuracy: float = np.nan
    test_accuracy: float = np.nan
    test_logloss: float = np.nan


@dataclass
class WalkForwardResult:
    folds: list[FoldResult]
    oos: pd.DataFrame  # concatenation of every fold's OOS block
    classes: list[int]
    is_meta: bool
    total_seconds: float
    fold_table: pd.DataFrame = field(default_factory=pd.DataFrame)

    def signals(self) -> pd.DataFrame:
        """Columns the backtest engine understands."""
        cols = ["p_long", "p_short"] + (["side", "p_trade"] if self.is_meta else [])
        return self.oos[cols]


def _proba_to_frame(proba: pd.DataFrame, y_true: pd.Series, classes: list[int], meta: pd.DataFrame | None) -> pd.DataFrame:
    out = pd.DataFrame(index=proba.index)
    out["y_true"] = y_true.reindex(proba.index)
    for c in classes:
        out[f"p_{c}"] = proba[c] if c in proba.columns else 0.0
    out["p_long"] = proba[1] if 1 in proba.columns else 0.0
    out["p_short"] = proba[-1] if -1 in proba.columns else (1.0 - out["p_long"])
    out["p_neutral"] = proba[0] if 0 in proba.columns else 0.0
    scored = proba.notna().all(axis=1)
    out["y_pred"] = np.nan
    out.loc[scored, "y_pred"] = proba.loc[scored].idxmax(axis=1).astype(float)
    if meta is not None:
        out["side"] = meta["side"]
        out["p_trade"] = meta["p_trade"]
    return out


CONTEXT_BARS = 500  # pre-test history handed to sequence models; they take the last seq_len-1 rows


def _tune(model_cfg: ModelConfig, X: pd.DataFrame, y: pd.Series, w: pd.Series, seed: int, n_trials: int, gap: int, progress: ProgressFn | None) -> tuple[dict[str, Any], float]:
    """Optuna TPE search on the chronological tail of the train fold. Honest, slow, optional.

    Objective: log-loss on the last 20% of the fold, with `gap` bars of embargo
    between the tuning-train and tuning-validation slices so overlapping labels
    do not leak inside the tuner either. The user's own params are trial 0, so
    the search can never do worse than not searching.
    """
    import optuna

    t0 = time.time()
    if model_cfg.ensemble.enabled:
        if progress:
            progress({"stage": "tune_skip", "reason": "ensembles are not re-tuned per fold; tune the members individually first"})
        return dict(model_cfg.params), 0.0
    probe = build_model(model_cfg, seed)
    n = len(X)
    split = int(n * 0.8)
    tr, va = slice(0, split), slice(split + gap, n)
    if n - split - gap < 30:
        return dict(model_cfg.params), 0.0
    yv = y.iloc[va]
    ctx = X.iloc[max(0, split + gap - CONTEXT_BARS) : split + gap]

    def objective(trial: optuna.Trial) -> float:
        draw = probe.suggest_params(trial)
        if not draw:
            raise optuna.TrialPruned()
        cfg_k = copy.deepcopy(model_cfg)
        cfg_k.params = {**model_cfg.params, **draw}
        try:
            m = build_model(cfg_k, seed)
            m.fit(X.iloc[tr], y.iloc[tr], w.iloc[tr], None)
            p = m.predict_proba(X.iloc[va], ctx)
        except ValueError as e:
            raise optuna.TrialPruned() from e
        ok = p.notna().all(axis=1) & yv.notna()
        if ok.sum() < 10:
            raise optuna.TrialPruned()
        cls = [int(c) for c in m.classes_]
        ll = float(log_loss(yv[ok].astype(int), p.loc[ok, cls].to_numpy(), labels=cls))
        if progress:
            progress({"stage": "tune", "candidate": trial.number + 1, "n_candidates": n_trials, "val_logloss": ll})
        return ll

    optuna.logging.set_verbosity(optuna.logging.ERROR)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=min(5, max(1, n_trials // 3))))
    # Seed the study with the user's own params as trial 0. Only keys the search
    # space knows about can be enqueued; the rest ride along via model_cfg.params.
    space_keys = set(_space_keys(probe))
    baseline = {k: v for k, v in probe.params.items() if k in space_keys}
    if baseline:
        study.enqueue_trial(baseline)
    study.optimize(objective, n_trials=n_trials, catch=())
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not done:
        return dict(model_cfg.params), time.time() - t0
    best = {**model_cfg.params, **study.best_params}
    return best, time.time() - t0


def _space_keys(adapter: ModelAdapter) -> list[str]:
    """Names the adapter's search space draws, discovered with a fixed-choice dry run."""
    import optuna

    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    trial = study.ask()
    try:
        return list(adapter.suggest_params(trial).keys())
    finally:
        study.tell(trial, 0.0)


def run_walkforward(ds: Dataset, cfg: ExperimentConfig, progress: ProgressFn | None = None) -> WalkForwardResult:
    cfg.validate()
    seed_everything(cfg.seed)
    t_all = time.time()
    splitter = WalkForwardSplitter(cfg.walkforward, cfg.label.max_horizon)
    folds: list[Fold] = list(splitter.split(ds.n, ds.t1_pos))
    check_folds(folds, ds.n, cfg.walkforward.embargo, cfg.walkforward.mode)
    X, y, w = ds.X, ds.y, ds.weights
    is_meta = cfg.model.meta.enabled
    results: list[FoldResult] = []
    oos_blocks: list[pd.DataFrame] = []
    classes: set[int] = set()
    for f in folds:
        t0 = time.time()
        if progress:
            progress({"stage": "fold_start", "fold": f.fold_id, "n_folds": len(folds), "train": (X.index[f.train_idx[0]], X.index[f.train_idx[-1]]), "test": (X.index[f.test_idx[0]], X.index[f.test_idx[-1]])})
        # Purged train rows: keep the slice contiguous (sequence models) but blank the label.
        lo, hi = int(f.train_idx.min()), int(f.train_idx.max()) + 1
        X_tr = X.iloc[lo:hi]
        y_tr = y.iloc[lo:hi].copy()
        purged = np.setdiff1d(np.arange(lo, hi), f.train_idx)
        if len(purged):
            y_tr.iloc[purged - lo] = np.nan
        w_tr = w.iloc[lo:hi]
        X_te, y_te = X.iloc[f.test_idx], y.iloc[f.test_idx]

        model_cfg = cfg.model
        tune_s = 0.0
        if cfg.walkforward.retune_per_fold:
            best, tune_s = _tune(cfg.model, X_tr, y_tr, w_tr, cfg.seed + f.fold_id, cfg.walkforward.tune_iterations, cfg.walkforward.embargo, progress)
            model_cfg = copy.deepcopy(cfg.model)
            model_cfg.params = best
        model = build_model(model_cfg, cfg.seed)
        model.fit(X_tr, y_tr, w_tr, progress)
        fold_classes = [int(c) for c in model.classes_]
        classes.update(fold_classes)

        # History for sequence models: bars strictly before the test window (train + embargo).
        te0 = int(f.test_idx[0])
        ctx = X.iloc[max(0, te0 - CONTEXT_BARS) : te0]
        proba_te = model.predict_proba(X_te, ctx)
        proba_tr = model.predict_proba(X_tr)
        meta_te = model.predict_meta(X_te, ctx) if is_meta else None  # type: ignore[attr-defined]
        meta_tr = model.predict_meta(X_tr) if is_meta else None  # type: ignore[attr-defined]
        oos = _proba_to_frame(proba_te, y_te, fold_classes, meta_te)
        ins = _proba_to_frame(proba_tr, y_tr, fold_classes, meta_tr)
        oos["fold"] = f.fold_id
        ins["fold"] = f.fold_id
        ok_te = oos["y_pred"].notna() & oos["y_true"].notna()
        ok_tr = ins["y_pred"].notna() & ins["y_true"].notna()
        ll = np.nan
        if ok_te.sum() > 0:
            ll = float(log_loss(oos.loc[ok_te, "y_true"].astype(int), oos.loc[ok_te, [f"p_{c}" for c in fold_classes]].to_numpy(), labels=fold_classes))
        fr = FoldResult(
            fold_id=f.fold_id,
            train_start=X.index[lo],
            train_end=X.index[hi - 1],
            test_start=X_te.index[0],
            test_end=X_te.index[-1],
            n_train=int(y_tr.notna().sum()),
            n_test=len(X_te),
            train_balance=class_balance(y_tr)["share"].to_dict(),
            test_balance=class_balance(y_te)["share"].to_dict(),
            model=model,
            params_used=model.get_params(),
            oos=oos,
            insample=ins,
            fit_seconds=time.time() - t0 - tune_s,
            tune_seconds=tune_s,
            context_start=ctx.index[0] if len(ctx) else None,
            train_accuracy=float((ins.loc[ok_tr, "y_pred"] == ins.loc[ok_tr, "y_true"]).mean()) if ok_tr.any() else np.nan,
            test_accuracy=float((oos.loc[ok_te, "y_pred"] == oos.loc[ok_te, "y_true"]).mean()) if ok_te.any() else np.nan,
            test_logloss=ll,
        )
        results.append(fr)
        oos_blocks.append(oos)
        if progress:
            progress({"stage": "fold_done", "fold": f.fold_id, "n_folds": len(folds), "test_accuracy": fr.test_accuracy, "seconds": fr.fit_seconds + tune_s})
    oos_all = pd.concat(oos_blocks).sort_index()
    if oos_all.index.has_duplicates:
        raise RuntimeError("OOS blocks overlap. The splitter produced overlapping test windows; this is a bug.")
    table = pd.DataFrame(
        [
            {
                "fold": r.fold_id, "train_start": r.train_start, "train_end": r.train_end, "n_train": r.n_train,
                "test_start": r.test_start, "test_end": r.test_end, "n_test": r.n_test,
                "train_acc": r.train_accuracy, "test_acc": r.test_accuracy, "test_logloss": r.test_logloss,
                "fit_s": r.fit_seconds, "tune_s": r.tune_seconds,
            }
            for r in results
        ]
    )
    return WalkForwardResult(folds=results, oos=oos_all, classes=sorted(classes), is_meta=is_meta, total_seconds=time.time() - t_all, fold_table=table)


# --------------------------------------------------------------------------- #
# Backtest + evaluation
# --------------------------------------------------------------------------- #
@dataclass
class Evaluation:
    backtest: BacktestResult
    metrics: dict[str, float]
    dsr: DSRResult
    random_bench: RandomBenchmarkResult
    fold_metrics: pd.DataFrame
    n_trials: int


def backtest_oos(ds: Dataset, wf: WalkForwardResult, cfg: ExperimentConfig) -> BacktestResult:
    return run_backtest(ds.prices, wf.signals(), cfg.backtest, bars_per_year=bars_per_year(ds.interval), atr_window=cfg.label.atr_window)


def evaluate(
    ds: Dataset,
    wf: WalkForwardResult,
    cfg: ExperimentConfig,
    n_trials: int,
    trial_sharpes_per_bar: np.ndarray | None,
    n_random: int = 1000,
) -> Evaluation:
    bpy = bars_per_year(ds.interval)
    bt = backtest_oos(ds, wf, cfg)
    metrics = performance_table(bt.returns, bt.benchmark_returns, bt.positions, bt.trades, bpy)
    dsr = deflated_sharpe(bt.returns, n_trials, trial_sharpes_per_bar, bpy)
    c = cfg.backtest.costs
    per_side = (c.commission_bps + c.spread_bps / 2 + c.slippage_bps) / 1e4
    rb = random_strategy_benchmark(bt.positions, bt.benchmark_returns, per_side, bpy, n=n_random, seed=cfg.seed)
    rows = []
    for f in wf.folds:
        r = bt.returns.loc[f.test_start : f.test_end]
        b = bt.benchmark_returns.loc[f.test_start : f.test_end]
        rows.append(
            {
                "fold": f.fold_id,
                "test_start": f.test_start,
                "test_end": f.test_end,
                "strategy_return": float((1 + r).prod() - 1),
                "bench_return": float((1 + b).prod() - 1),
                "sharpe": float(r.mean() / r.std(ddof=1) * np.sqrt(bpy)) if r.std(ddof=1) > 0 else np.nan,
                "test_acc": f.test_accuracy,
            }
        )
    return Evaluation(bt, metrics, dsr, rb, pd.DataFrame(rows), n_trials)


# --------------------------------------------------------------------------- #
# Experiment record
# --------------------------------------------------------------------------- #
def experiment_record(cfg: ExperimentConfig, ds: Dataset, wf: WalkForwardResult, ev: Evaluation, session_id: str, run_id: str | None = None) -> dict[str, Any]:
    """Flat dict for the append-only log. Metrics prefixed so they sort together."""
    run_id = run_id or uuid.uuid4().hex[:12]
    rec: dict[str, Any] = {
        "run_id": run_id,
        "session_id": session_id,
        "timestamp": pd.Timestamp.utcnow().tz_localize(None),
        "ticker": cfg.data.ticker,
        "interval": cfg.data.interval,
        "start": cfg.data.start,
        "end": cfg.data.end,
        "oos_start": wf.oos.index[0],
        "oos_end": wf.oos.index[-1],
        "config_hash": cfg.config_hash(),
        "config_json": cfg.to_json(indent=None),
        "features": ",".join(ds.feature_columns),
        "n_features": len(ds.feature_columns),
        "label_scheme": cfg.label.scheme,
        "label_horizon": cfg.label.max_horizon,
        "model_family": _family_label(cfg.model),
        "hyperparameters": json.dumps(cfg.model.params, sort_keys=True, default=str),
        "wf_mode": cfg.walkforward.mode,
        "wf_train": cfg.walkforward.train_window,
        "wf_test": cfg.walkforward.test_window,
        "wf_step": cfg.walkforward.step,
        "wf_embargo": cfg.walkforward.embargo,
        "wf_retune": cfg.walkforward.retune_per_fold,
        "n_folds": len(wf.folds),
        "cost_commission_bps": cfg.backtest.costs.commission_bps,
        "cost_spread_bps": cfg.backtest.costs.spread_bps,
        "cost_slippage_bps": cfg.backtest.costs.slippage_bps,
        "cost_slippage_atr": cfg.backtest.costs.slippage_atr_frac,
        "sizing": cfg.backtest.sizing.scheme,
        "direction": cfg.backtest.direction,
        "execution": cfg.backtest.execution,
        "long_threshold": cfg.backtest.long_threshold,
        "short_threshold": cfg.backtest.short_threshold,
        "seed": cfg.seed,
        "n_trials_at_run": ev.n_trials,
        "sharpe_per_bar": ev.dsr.sharpe_per_bar,
        "ret_skew": ev.dsr.skew,
        "ret_kurtosis": ev.dsr.kurtosis,  # non-excess, as PSR wants it
        "dsr": ev.dsr.dsr,
        "psr": ev.dsr.psr,
        "random_percentile": ev.random_bench.percentile,
        "oos_accuracy": float(np.nanmean([f.test_accuracy for f in wf.folds])),
        "oos_logloss": float(np.nanmean([f.test_logloss for f in wf.folds])),
        "wf_seconds": wf.total_seconds,
    }
    for k, v in ev.metrics.items():
        rec[f"m_{k}"] = v
    return rec


def _family_label(m: ModelConfig) -> str:
    if m.ensemble.enabled:
        core = f"{m.ensemble.method}({'+'.join(m.ensemble.members)})"
    else:
        core = m.family
    if m.meta.enabled:
        return f"meta[{m.meta.primary}->{core}]"
    return core

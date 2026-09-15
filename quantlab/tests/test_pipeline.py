"""End to end on synthetic data: features -> labels -> walk-forward -> backtest -> log."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantlab.backtest.metrics import deflated_sharpe, pbo_cscv, random_strategy_benchmark
from quantlab.config import EnsembleConfig, ExperimentConfig, FeatureConfig, FeatureSpec, LabelConfig, MetaConfig, ModelConfig, WalkForwardConfig
from quantlab.experiments.compare import leaderboard, pbo_across_runs
from quantlab.experiments.store import ExperimentStore
from quantlab.pipeline import evaluate, experiment_record, prepare_dataset, run_walkforward

FEATS = FeatureConfig([FeatureSpec("rsi", {"window": 14}), FeatureSpec("macd", {}), FeatureSpec("natr", {"window": 14}), FeatureSpec("momentum", {"window": 10})])
WF = WalkForwardConfig(train_window=400, test_window=100, step=100, embargo=5)


def _cfg(family: str = "logreg", **model_kwargs) -> ExperimentConfig:
    from quantlab.config import DataConfig

    return ExperimentConfig(data=DataConfig(ticker="SYNTH"), features=FEATS, label=LabelConfig(horizon=5), model=ModelConfig(family=family, **model_kwargs), walkforward=WF)


def test_oos_series_is_contiguous_and_after_train(ohlcv: pd.DataFrame) -> None:
    cfg = _cfg()
    ds = prepare_dataset(ohlcv, cfg)
    wf = run_walkforward(ds, cfg)
    assert not wf.oos.index.has_duplicates
    assert wf.oos.index.is_monotonic_increasing
    for f in wf.folds:
        assert f.train_end < f.test_start
        gap = ds.X.index.get_loc(f.test_start) - ds.X.index.get_loc(f.train_end) - 1
        assert gap == 5
    assert set(wf.oos["y_true"].dropna().unique()) <= {-1, 1}
    assert wf.oos[["p_long", "p_short"]].dropna().sum(axis=1).round(6).eq(1.0).all()


@pytest.mark.parametrize("family,params", [("random_forest", {"n_estimators": 30}), ("gru", {"epochs": 2, "seq_len": 10})])
def test_other_families_run(ohlcv: pd.DataFrame, family: str, params: dict) -> None:
    cfg = _cfg(family, params=params)
    ds = prepare_dataset(ohlcv, cfg)
    wf = run_walkforward(ds, cfg)
    assert len(wf.folds) >= 3
    if family == "gru":
        # Pre-test context feeds the first windows, so every test bar is scored.
        assert wf.oos["p_long"].notna().all()
        assert wf.folds[0].context_start is not None and wf.folds[0].context_start < wf.folds[0].test_start


def test_meta_and_ensemble_paths(ohlcv: pd.DataFrame) -> None:
    cfg = _cfg("logreg", meta=MetaConfig(enabled=True, primary="ma_crossover", primary_params={"fast": 10, "slow": 30}))
    ds = prepare_dataset(ohlcv, cfg)
    wf = run_walkforward(ds, cfg)
    assert wf.is_meta and {"side", "p_trade"} <= set(wf.oos.columns)
    ev = evaluate(ds, wf, cfg, 1, None, n_random=20)
    assert ev.metrics["n_trades"] >= 0
    cfg2 = _cfg("logreg", ensemble=EnsembleConfig(enabled=True, method="soft", members=["logreg", "extra_trees"], member_params={"extra_trees": {"n_estimators": 20}}))
    ds2 = prepare_dataset(ohlcv, cfg2)
    wf2 = run_walkforward(ds2, cfg2)
    assert wf2.oos["p_long"].notna().all()


def test_store_and_leaderboard(ohlcv: pd.DataFrame, tmp_path: Path) -> None:
    store = ExperimentStore(tmp_path)
    for fam in ("logreg", "extra_trees"):
        cfg = _cfg(fam, params={} if fam == "logreg" else {"n_estimators": 20})
        ds = prepare_dataset(ohlcv, cfg)
        wf = run_walkforward(ds, cfg)
        ev = evaluate(ds, wf, cfg, store.n_trials("SYNTH") + 1, store.trial_sharpes("SYNTH"), n_random=20)
        rec = experiment_record(cfg, ds, wf, ev, "s")
        store.log_run(rec, ev.backtest.returns)
        with pytest.raises(ValueError, match="append-only"):
            store.log_run(rec, ev.backtest.returns)
    runs = store.load_runs()
    assert len(runs) == 2 and store.n_trials("SYNTH") == 2
    assert runs["ret_skew"].notna().all() and (runs["ret_kurtosis"] > 0).all()
    lb = leaderboard(store)
    assert "dsr_now" in lb and lb["n_trials_now"].iloc[0] == 2
    # The recomputed DSR must use the stored moments, not a normal approximation.
    from quantlab.backtest.metrics import expected_max_sharpe, probabilistic_sharpe

    row = lb.iloc[0]
    sr0 = expected_max_sharpe(2, float(runs["sharpe_per_bar"].var(ddof=1)))
    want = probabilistic_sharpe(row["sharpe_per_bar"] if "sharpe_per_bar" in row else runs.set_index("run_id").loc[row["run_id"], "sharpe_per_bar"], sr0, int(row["m_n_bars"]), float(row["ret_skew"]), float(row["ret_kurtosis"]))
    assert row["dsr_now"] == pytest.approx(want)
    pbo = pbo_across_runs(store, lb["run_id"].tolist(), n_partitions=4)
    assert 0 <= pbo.pbo <= 1
    assert store.load_returns(lb["run_id"].iloc[0]) is not None


def test_context_must_precede_x(ohlcv: pd.DataFrame) -> None:
    from quantlab.models import build_model

    cfg = _cfg("gru", params={"epochs": 1, "seq_len": 5})
    ds = prepare_dataset(ohlcv, cfg)
    m = build_model(cfg.model, 0).fit(ds.X.iloc[:400], ds.y.iloc[:400])
    ok = m.predict_proba(ds.X.iloc[400:450], ds.X.iloc[380:400])
    assert ok.notna().all().all()
    with pytest.raises(ValueError, match="future"):
        m.predict_proba(ds.X.iloc[400:450], ds.X.iloc[410:420])
    # Without context the first seq_len-1 rows are honestly unscored, not silently filled.
    assert m.predict_proba(ds.X.iloc[400:450])[1].iloc[:4].isna().all()


def test_optuna_retuning_changes_params_per_fold(ohlcv: pd.DataFrame) -> None:
    cfg = _cfg("logreg")
    cfg.walkforward = WalkForwardConfig(train_window=400, test_window=200, step=200, embargo=5, retune_per_fold=True, tune_iterations=3)
    ds = prepare_dataset(ohlcv, cfg)
    events = []
    wf = run_walkforward(ds, cfg, progress=lambda e: events.append(e) if e.get("stage") == "tune" else None)
    assert (wf.fold_table["tune_s"] > 0).all()
    assert len(events) == 3 * len(wf.folds)
    for f in wf.folds:
        assert set(f.params_used) >= {"C", "l1_ratio"}
    # Determinism: same seed, same tuned params.
    wf2 = run_walkforward(ds, cfg)
    assert [f.params_used for f in wf.folds] == [f.params_used for f in wf2.folds]


@pytest.mark.parametrize("family,params,scale", [("logreg", {}, "log-odds"), ("extra_trees", {"n_estimators": 20}, "probability"), ("gru", {"epochs": 1, "seq_len": 8}, "logit")])
def test_shap_covers_every_family(ohlcv: pd.DataFrame, family: str, params: dict, scale: str) -> None:
    from quantlab.diagnostics import shap_last_fold

    cfg = _cfg(family, params=params)
    ds = prepare_dataset(ohlcv, cfg)
    wf = run_walkforward(ds, cfg)
    blocks = shap_last_fold(ds, wf, max_rows=60)
    assert len(blocks) == 1
    b = blocks[0]
    assert scale in b.scale
    assert list(b.values.columns) == ds.feature_columns
    assert b.values.index.equals(b.X.index) and len(b.values) > 0
    assert np.isfinite(b.values.to_numpy()).all()


def test_dsr_penalises_trials() -> None:
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.0008, 0.01, 1000))
    one = deflated_sharpe(r, 1, None, 252)
    many = deflated_sharpe(r, 200, rng.normal(0, 0.05, 200), 252)
    assert many.dsr < one.dsr
    assert many.expected_max_sharpe > 0


def test_pbo_flags_pure_noise_and_random_bench_matches_turnover() -> None:
    rng = np.random.default_rng(2)
    M = pd.DataFrame(rng.normal(0, 0.01, (400, 10)))
    res = pbo_cscv(M, n_partitions=8)
    assert 0.15 <= res.pbo <= 0.85  # noise: the IS-best is a coin flip OOS (70 combos, so it is noisy itself)
    idx = pd.bdate_range("2020-01-01", periods=400)
    pos = pd.Series(np.repeat(rng.choice([-1.0, 0.0, 1.0], 40), 10), index=idx)
    r = pd.Series(rng.normal(0, 0.01, 400), index=idx)
    rb = random_strategy_benchmark(pos, r, 0.0005, 252, n=50)
    assert rb.n == 50 and 0 <= rb.percentile <= 1

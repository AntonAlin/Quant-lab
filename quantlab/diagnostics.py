"""Post-hoc diagnostics on the walk-forward result: importance, calibration, regimes.

Permutation importance is computed *per fold on that fold's OOS block* with that
fold's own model, then averaged. Importance computed in-sample tells you what
the model memorised, which is a different question.

SHAP covers every family: trees via TreeExplainer, logistic regression via
LinearExplainer, the torch nets via GradientExplainer with the sequence axis
summed out. Ensembles and meta-labeling are explained component by component.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, log_loss

from quantlab.models.classical import SklearnAdapter
from quantlab.models.meta import SIDE_COL
from quantlab.pipeline import CONTEXT_BARS, Dataset, WalkForwardResult


def _context_for(ds: Dataset, f: Any) -> pd.DataFrame:
    te0 = ds.X.index.get_loc(f.test_start)
    return ds.X.iloc[max(0, te0 - CONTEXT_BARS) : te0]


def permutation_importance_oos(ds: Dataset, wf: WalkForwardResult, n_repeats: int = 5, seed: int = 0, max_folds: int | None = None) -> pd.DataFrame:
    """Mean OOS log-loss increase when each feature is shuffled, per fold, averaged.

    The shuffle covers the test block *and* the pre-test context handed to
    sequence models, so a feature cannot sneak back in through the history window.
    """
    rng = np.random.default_rng(seed)
    folds = wf.folds if max_folds is None else wf.folds[-max_folds:]
    rows: dict[str, list[float]] = {c: [] for c in ds.feature_columns}
    for f in folds:
        X_te = ds.X.loc[f.test_start : f.test_end]
        y_te = ds.y.loc[f.test_start : f.test_end]
        ctx = _context_for(ds, f)
        base = _fold_logloss(f.model, X_te, y_te, ctx)
        if not np.isfinite(base):
            continue
        n_ctx = len(ctx)
        for col in ds.feature_columns:
            deltas = []
            for _ in range(n_repeats):
                full = pd.concat([ctx, X_te])
                full[col] = rng.permutation(full[col].to_numpy())
                ll = _fold_logloss(f.model, full.iloc[n_ctx:], y_te, full.iloc[:n_ctx])
                if np.isfinite(ll):
                    deltas.append(ll - base)
            if deltas:
                rows[col].append(float(np.mean(deltas)))
    out = pd.DataFrame({"feature": list(rows), "importance": [np.mean(v) if v else np.nan for v in rows.values()], "std": [np.std(v) if v else np.nan for v in rows.values()]})
    return out.sort_values("importance", ascending=False).reset_index(drop=True)


def _fold_logloss(model: Any, X: pd.DataFrame, y: pd.Series, ctx: pd.DataFrame | None = None) -> float:
    proba = model.predict_proba(X, ctx)
    ok = proba.notna().all(axis=1) & y.notna()
    if ok.sum() < 10:
        return np.nan
    cls = [int(c) for c in model.classes_]
    yv = y[ok].astype(int)
    if yv.nunique() < 2:
        return np.nan
    return float(log_loss(yv, proba.loc[ok, cls].to_numpy(), labels=cls))


@dataclass
class ShapBlock:
    component: str  # which part of the model this explains
    scale: str  # what the SHAP values are in: probability / log-odds / logit
    values: pd.DataFrame  # rows x features
    X: pd.DataFrame  # raw feature values on the same rows, for colouring


def shap_last_fold(ds: Dataset, wf: WalkForwardResult, max_rows: int = 400) -> list[ShapBlock]:
    """SHAP on the last fold's OOS block for every explainable component of the fitted model.

    Trees -> TreeExplainer, logistic regression -> LinearExplainer, torch nets ->
    GradientExplainer (summed over the sequence axis). Ensembles are explained
    member by member because their members live on different output scales and
    averaging log-odds with probabilities would be a lie with a colour bar.
    """
    f = wf.folds[-1]
    X_te = ds.X.loc[f.test_start : f.test_end].drop(columns=[SIDE_COL], errors="ignore").tail(max_rows)
    ctx = _context_for(ds, f).drop(columns=[SIDE_COL], errors="ignore")
    X_bg = ds.X.loc[f.train_start : f.train_end].drop(columns=[SIDE_COL], errors="ignore")
    blocks: list[ShapBlock] = []
    _explain(f.model, "model", X_te, ctx, X_bg, blocks)
    return blocks


def _explain(adapter: Any, name: str, X_te: pd.DataFrame, ctx: pd.DataFrame, X_bg: pd.DataFrame, out: list[ShapBlock]) -> None:
    from quantlab.models.deep import TorchSequenceAdapter
    from quantlab.models.ensemble import StackingAdapter, VotingAdapter
    from quantlab.models.meta import MetaLabelingAdapter

    if isinstance(adapter, MetaLabelingAdapter):
        if adapter.primary is not None:
            _explain(adapter.primary, f"{name}.primary", X_te, ctx, X_bg, out)
        _explain(adapter.secondary, f"{name}.secondary", X_te, ctx, X_bg, out)
        return
    if isinstance(adapter, (VotingAdapter, StackingAdapter)):
        for m in adapter.members:
            _explain(m, f"{name}.{m.name}", X_te, ctx, X_bg, out)
        return
    if isinstance(adapter, TorchSequenceAdapter):
        vals = adapter.explain(X_te, ctx, X_bg)
        if vals is not None:
            out.append(ShapBlock(f"{name} ({adapter.name})", "logit of P(+1), summed over the sequence", vals, X_te.loc[vals.index]))
        return
    if isinstance(adapter, SklearnAdapter):
        blk = _explain_sklearn(adapter, name, X_te, X_bg)
        if blk is not None:
            out.append(blk)


def _explain_sklearn(adapter: SklearnAdapter, name: str, X_te: pd.DataFrame, X_bg: pd.DataFrame) -> ShapBlock | None:
    import shap

    est = adapter.sklearn_estimator()
    Xt = adapter.transform_features(X_te)
    cols = adapter.feature_names_
    pos = int(np.searchsorted(adapter.classes_, 1)) if 1 in adapter.classes_ else len(adapter.classes_) - 1
    if hasattr(est, "feature_importances_"):
        vals = shap.TreeExplainer(est).shap_values(Xt)
        kind = type(est).__name__
        scale = "probability of +1" if kind.endswith(("ForestClassifier", "TreesClassifier")) else "log-odds of +1"
    elif hasattr(est, "coef_"):
        bg = adapter.transform_features(X_bg.tail(1000))
        vals = shap.LinearExplainer(est, bg).shap_values(Xt)
        scale = "log-odds of +1"
    else:
        return None
    arr = _pick_class(vals, pos)
    return ShapBlock(f"{name} ({adapter.name})", scale, pd.DataFrame(arr, index=X_te.index, columns=cols), X_te)


def _pick_class(vals: Any, pos: int) -> np.ndarray:
    """shap returns a list (per class), a 3-D array (n, f, classes) or a 2-D array. Normalise."""
    if isinstance(vals, list):
        return np.asarray(vals[pos] if len(vals) > pos else vals[-1])
    arr = np.asarray(vals)
    if arr.ndim == 3:
        return arr[:, :, min(pos, arr.shape[2] - 1)]
    return arr


def confusion(wf: WalkForwardResult) -> pd.DataFrame:
    o = wf.oos.dropna(subset=["y_true", "y_pred"])
    labels = wf.classes
    cm = confusion_matrix(o["y_true"].astype(int), o["y_pred"].astype(int), labels=labels)
    return pd.DataFrame(cm, index=[f"true {c}" for c in labels], columns=[f"pred {c}" for c in labels])


def calibration(wf: WalkForwardResult, n_bins: int = 10) -> pd.DataFrame:
    """Reliability curve for P(long) (or P(trade) under meta-labeling)."""
    o = wf.oos.dropna(subset=["y_true"])
    if wf.is_meta:
        o = o[o["side"] != 0].dropna(subset=["p_trade"])
        y = (o["y_true"] * o["side"] > 0).astype(int)
        p = o["p_trade"]
    else:
        o = o.dropna(subset=["p_long"])
        y = (o["y_true"] == 1).astype(int)
        p = o["p_long"]
    if len(o) < n_bins * 5 or y.nunique() < 2:
        return pd.DataFrame(columns=["mean_predicted", "fraction_positive"])
    frac, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
    return pd.DataFrame({"mean_predicted": mean_pred, "fraction_positive": frac})


def probability_by_class(wf: WalkForwardResult) -> pd.DataFrame:
    o = wf.oos.dropna(subset=["y_true", "p_long"])
    return o[["y_true", "p_long"]].rename(columns={"y_true": "true_class", "p_long": "p_long"})


def per_regime_performance(ds: Dataset, returns: pd.Series, benchmark: pd.Series, bars_per_year: float, vol_window: int = 20) -> pd.DataFrame:
    """Strategy vs benchmark inside vol terciles and trend/range states over the OOS period."""
    px = ds.prices["close"]
    r = np.log(px).diff()
    vol = r.rolling(vol_window).std(ddof=1).reindex(returns.index)
    terc = pd.qcut(vol.rank(method="first"), 3, labels=["low_vol", "mid_vol", "high_vol"])
    trend = (px.rolling(50).mean() > px.rolling(200).mean()).reindex(returns.index).map({True: "ma50>ma200", False: "ma50<ma200"})
    dd = (px / px.cummax() - 1.0).reindex(returns.index)
    dd_state = pd.cut(dd, [-1.0, -0.2, -0.1, 0.0001], labels=["dd>20%", "dd 10-20%", "dd<10%"])
    rows = []
    for name, groups in (("vol", terc), ("trend", trend), ("drawdown", dd_state)):
        for g, idx in returns.groupby(groups, observed=True).groups.items():
            rs, rb = returns.loc[idx], benchmark.reindex(idx)
            rows.append(
                {
                    "regime_type": name,
                    "regime": str(g),
                    "n_bars": len(idx),
                    "strategy_ann_return": float(rs.mean() * bars_per_year),
                    "strategy_sharpe": float(rs.mean() / rs.std(ddof=1) * np.sqrt(bars_per_year)) if rs.std(ddof=1) > 0 else np.nan,
                    "bench_ann_return": float(rb.mean() * bars_per_year),
                    "bench_sharpe": float(rb.mean() / rb.std(ddof=1) * np.sqrt(bars_per_year)) if rb.std(ddof=1) > 0 else np.nan,
                }
            )
    return pd.DataFrame(rows)

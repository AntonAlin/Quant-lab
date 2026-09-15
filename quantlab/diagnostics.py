"""Post-hoc diagnostics on the walk-forward result: importance, calibration, regimes.

Permutation importance is computed *per fold on that fold's OOS block* with that
fold's own model, then averaged. Importance computed in-sample tells you what
the model memorised, which is a different question.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, log_loss

from quantlab.models.classical import SklearnAdapter
from quantlab.models.meta import SIDE_COL
from quantlab.pipeline import Dataset, WalkForwardResult


def permutation_importance_oos(ds: Dataset, wf: WalkForwardResult, n_repeats: int = 5, seed: int = 0, max_folds: int | None = None) -> pd.DataFrame:
    """Mean OOS log-loss increase when each feature is shuffled, per fold, averaged."""
    rng = np.random.default_rng(seed)
    folds = wf.folds if max_folds is None else wf.folds[-max_folds:]
    rows: dict[str, list[float]] = {c: [] for c in ds.feature_columns}
    for f in folds:
        X_te = ds.X.loc[f.test_start : f.test_end]
        y_te = ds.y.loc[f.test_start : f.test_end]
        base = _fold_logloss(f.model, X_te, y_te)
        if not np.isfinite(base):
            continue
        for col in ds.feature_columns:
            deltas = []
            for _ in range(n_repeats):
                Xp = X_te.copy()
                Xp[col] = rng.permutation(Xp[col].to_numpy())
                ll = _fold_logloss(f.model, Xp, y_te)
                if np.isfinite(ll):
                    deltas.append(ll - base)
            if deltas:
                rows[col].append(float(np.mean(deltas)))
    out = pd.DataFrame({"feature": list(rows), "importance": [np.mean(v) if v else np.nan for v in rows.values()], "std": [np.std(v) if v else np.nan for v in rows.values()]})
    return out.sort_values("importance", ascending=False).reset_index(drop=True)


def _fold_logloss(model: Any, X: pd.DataFrame, y: pd.Series) -> float:
    proba = model.predict_proba(X)
    ok = proba.notna().all(axis=1) & y.notna()
    if ok.sum() < 10:
        return np.nan
    cls = [int(c) for c in model.classes_]
    yv = y[ok].astype(int)
    if yv.nunique() < 2:
        return np.nan
    return float(log_loss(yv, proba.loc[ok, cls].to_numpy(), labels=cls))


def shap_values_last_fold(ds: Dataset, wf: WalkForwardResult, max_rows: int = 500) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """SHAP on the last fold's OOS block for tree models. Returns (shap_df, X_used) or None."""
    f = wf.folds[-1]
    model = f.model
    inner = getattr(model, "secondary", model)  # meta-labeling: explain the secondary
    if not isinstance(inner, SklearnAdapter):
        return None
    est = inner.sklearn_estimator()
    if not hasattr(est, "feature_importances_"):
        return None  # not a tree model; SHAP's Linear explainer is a footnote here
    import shap

    X_te = ds.X.loc[f.test_start : f.test_end].drop(columns=[SIDE_COL], errors="ignore").tail(max_rows)
    Xt = inner.transform_features(X_te)
    explainer = shap.TreeExplainer(est)
    vals = explainer.shap_values(Xt)
    if isinstance(vals, list):
        vals = vals[-1]  # positive class for binary; last class otherwise
    elif vals.ndim == 3:
        vals = vals[:, :, -1]
    return pd.DataFrame(vals, index=X_te.index, columns=inner.feature_names_), pd.DataFrame(Xt, index=X_te.index, columns=inner.feature_names_)


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

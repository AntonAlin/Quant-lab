"""Label construction entry point used by the pipeline and the UI."""

from __future__ import annotations

import pandas as pd

from quantlab.config import LabelConfig
from quantlab.labeling.fixed_horizon import fixed_horizon_labels
from quantlab.labeling.trend_scanning import trend_scanning_labels
from quantlab.labeling.triple_barrier import triple_barrier_labels
from quantlab.labeling.weights import combined_weights


def build_labels(df: pd.DataFrame, cfg: LabelConfig, side: pd.Series | None = None) -> pd.DataFrame:
    """OHLCV + LabelConfig -> DataFrame with label / ret / t1 / weight (+ scheme extras)."""
    cfg.validate()
    if cfg.scheme == "fixed_horizon":
        lab = fixed_horizon_labels(df["close"], cfg.horizon, cfg.deadband, cfg.three_class)
    elif cfg.scheme == "triple_barrier":
        lab = triple_barrier_labels(df, cfg.horizon, cfg.pt_mult, cfg.sl_mult, cfg.atr_window, side=side)
    elif cfg.scheme == "trend_scanning":
        lab = trend_scanning_labels(df["close"], cfg.min_window, cfg.max_window, cfg.window_step)
    else:  # pragma: no cover - validate() already refuses
        raise ValueError(cfg.scheme)
    lab["weight"] = combined_weights(df.index, lab["t1"], df["close"], cfg.weighting).reindex(lab.index)
    lab.attrs["scheme"] = cfg.scheme
    return lab


def class_balance(labels: pd.Series) -> pd.DataFrame:
    """Counts and shares per class, plus a flag the UI turns into an ugly warning."""
    lab = labels.dropna().astype(int)
    vc = lab.value_counts().sort_index()
    out = pd.DataFrame({"count": vc, "share": vc / vc.sum()})
    out.attrs["dominant_share"] = float(out["share"].max()) if len(out) else 0.0
    out.attrs["imbalanced"] = bool(out.attrs["dominant_share"] > 0.65)
    return out

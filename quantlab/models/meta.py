"""Meta-labeling: a primary signal decides *direction*, a secondary model decides *whether to bet*.

The secondary model is trained only on bars where the primary fired, with the
target "was the primary right?". Its probability becomes the bet size / filter.
The primary can be a rule (MA crossover, momentum sign, RSI extremes) or any
ModelAdapter.

Rule-based sides need prices, which the model layer does not see. The pipeline
computes them (shifted, like every feature) and injects them as a column named
SIDE_COL into X. This adapter pops that column before anything is fitted.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from quantlab.features.technical import ema, rsi, sma
from quantlab.models.base import ModelAdapter, ProgressFn

SIDE_COL = "__primary_side__"

PRIMARY_RULES: dict[str, dict[str, Any]] = {
    "ma_crossover": {"fast": 20, "slow": 50},
    "momentum": {"window": 20},
    "rsi_extremes": {"window": 14, "lower": 30.0, "upper": 70.0},
}


def rule_side(df: pd.DataFrame, rule: str, params: dict[str, Any]) -> pd.Series:
    """Unshifted primary side in {-1, 0, 1}. The pipeline shifts it by one bar."""
    c = df["close"]
    if rule == "ma_crossover":
        f, s = int(params.get("fast", 20)), int(params.get("slow", 50))
        if f >= s:
            raise ValueError("ma_crossover primary: fast must be < slow.")
        return np.sign(ema(c, f) - ema(c, s)).fillna(0.0)
    if rule == "momentum":
        w = int(params.get("window", 20))
        return np.sign(c.pct_change(w)).fillna(0.0)
    if rule == "rsi_extremes":
        w = int(params.get("window", 14))
        lo, hi = float(params.get("lower", 30)), float(params.get("upper", 70))
        r = rsi(c, w)
        side = pd.Series(0.0, index=df.index)
        side[r < lo] = 1.0  # oversold -> long
        side[r > hi] = -1.0
        return side
    raise ValueError(f"Unknown primary rule {rule!r}. Known: {sorted(PRIMARY_RULES)} or 'model'.")


class MetaLabelingAdapter(ModelAdapter):
    name = "meta"

    def __init__(self, secondary: ModelAdapter, primary: ModelAdapter | None = None, seed: int = 42) -> None:
        super().__init__(params={}, seed=seed)
        self.secondary = secondary
        self.primary = primary  # None -> expect SIDE_COL in X
        self.is_sequence_model = secondary.is_sequence_model or (primary is not None and primary.is_sequence_model)

    def get_params(self) -> dict[str, Any]:
        return {
            "family": "meta",
            "primary": "rule" if self.primary is None else self.primary.get_params(),
            "secondary": self.secondary.get_params(),
        }

    def param_distributions(self, rng: np.random.Generator) -> dict[str, Any]:
        return {}

    def _sides(self, X: pd.DataFrame, fit: bool, y: pd.Series | None = None, w: pd.Series | None = None, progress: ProgressFn | None = None) -> tuple[pd.DataFrame, pd.Series]:
        if self.primary is None:
            if SIDE_COL not in X.columns:
                raise ValueError(f"Meta-labeling with a rule primary needs column {SIDE_COL!r} in X. The pipeline injects it.")
            side = X[SIDE_COL].fillna(0.0)
            return X.drop(columns=[SIDE_COL]), side
        Xf = X.drop(columns=[SIDE_COL], errors="ignore")
        if fit:
            assert y is not None
            self.primary.fit(Xf, y, w, progress)
        proba = self.primary.predict_proba(Xf)
        p_long = proba[1] if 1 in proba.columns else pd.Series(0.0, index=X.index)
        p_short = proba[-1] if -1 in proba.columns else 1.0 - p_long
        side = pd.Series(0.0, index=X.index)
        side[p_long > p_short] = 1.0
        side[p_short > p_long] = -1.0
        side[proba.isna().all(axis=1)] = 0.0
        return Xf, side

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: pd.Series | None = None, progress: ProgressFn | None = None) -> "MetaLabelingAdapter":
        self.feature_names_ = list(X.columns)
        Xf, side = self._sides(X, fit=True, y=y, w=sample_weight, progress=progress)
        # Secondary target: 1 if the primary's side agreed with the realised label.
        agree = (y * side > 0).astype(float)
        target = agree.where(y.notna() & (side != 0), np.nan)
        # Keep the full contiguous slice (sequence models), but blank targets
        # where the primary did not fire so those bars are never trained on.
        n_fired = int(target.notna().sum())
        if n_fired < 30:
            raise ValueError(f"Meta-labeling: primary fired on only {n_fired} training bars. Loosen the primary.")
        if target.dropna().nunique() < 2:
            raise ValueError("Meta-labeling: the primary was always right (or always wrong) in this fold. Nothing to learn.")
        self.secondary.fit(Xf, target, sample_weight, progress)
        self.classes_ = np.array([-1, 1])
        self.fitted_ = True
        return self

    def predict_meta(self, X: pd.DataFrame) -> pd.DataFrame:
        """side in {-1,0,1} and P(take the trade). This is what the backtest wants."""
        self._check_fitted()
        Xf, side = self._sides(X, fit=False)
        proba = self.secondary.predict_proba(Xf)
        p_trade = proba[1] if 1 in proba.columns else pd.Series(np.nan, index=X.index)
        return pd.DataFrame({"side": side, "p_trade": p_trade})

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Directional view for diagnostics: P(long) = p_trade when side=+1, etc."""
        m = self.predict_meta(X)
        out = pd.DataFrame(np.nan, index=X.index, columns=[-1, 1], dtype=float)
        p = m["p_trade"]
        out[1] = np.where(m["side"] > 0, p, np.where(m["side"] < 0, 1 - p, 0.5))
        out[-1] = 1.0 - out[1]
        out.loc[p.isna(), :] = np.nan
        return out

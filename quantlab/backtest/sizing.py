"""Position sizing: turn a directional signal into a position size (fraction of equity).

Everything here is a function of information available at the signal bar. The
vol estimate for bar t uses returns up to t; Kelly uses trailing realised win
statistics up to t. No sizing rule may touch the return it is about to earn.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.config import SizingConfig


def fixed_fractional(n: int, fraction: float) -> np.ndarray:
    return np.full(n, float(fraction))


def vol_target(returns: pd.Series, target_vol: float, window: int, bars_per_year: float, cap: float) -> np.ndarray:
    """Scale so that realised vol of the position hits target_vol, capped at `cap` leverage."""
    rv = returns.rolling(window, min_periods=max(5, window // 2)).std(ddof=1) * np.sqrt(bars_per_year)
    size = (target_vol / rv.replace(0, np.nan)).clip(upper=cap)
    # Before we have a vol estimate, trade at 1x rather than not at all; the
    # alternative silently skips the first month of every backtest.
    return size.fillna(1.0).to_numpy()


def kelly(signal_returns: pd.Series, window: int, max_fraction: float) -> np.ndarray:
    """Trailing Kelly fraction f = p/b_loss - q/b_win on the *strategy's own* recent returns.

    We compute it on what the signal would have earned had we followed it at 1x,
    over the trailing window. Kelly on a 60-bar sample is a noise generator, so
    the cap matters more than the formula.
    """
    r = signal_returns
    wins = r.where(r > 0)
    losses = -r.where(r < 0)
    p = (r > 0).rolling(window, min_periods=window).mean()
    avg_w = wins.rolling(window, min_periods=1).mean()
    avg_l = losses.rolling(window, min_periods=1).mean()
    q = 1 - p
    f = p / avg_l.replace(0, np.nan) - q / avg_w.replace(0, np.nan)
    f = f.clip(lower=0.0, upper=max_fraction)
    return f.fillna(0.0).to_numpy()


def confidence(prob: np.ndarray, floor: float, cap: float = 1.0) -> np.ndarray:
    """Linear ramp from 0 at p=floor to `cap` at p=1. Below the floor you should not be trading anyway."""
    p = np.nan_to_num(prob, nan=floor)
    return np.clip((p - floor) / max(1e-9, 1.0 - floor), 0.0, 1.0) * cap


def compute_sizes(
    cfg: SizingConfig,
    asset_returns: pd.Series,
    side: np.ndarray,
    prob: np.ndarray,
    bars_per_year: float,
) -> np.ndarray:
    """Dispatch on scheme. Returns |size| per bar; direction is applied by the engine."""
    cfg.validate()
    n = len(asset_returns)
    if cfg.scheme == "fixed_fractional":
        return fixed_fractional(n, cfg.fixed_fraction)
    if cfg.scheme == "vol_target":
        return vol_target(asset_returns, cfg.target_vol, cfg.vol_window, bars_per_year, cfg.leverage_cap)
    if cfg.scheme == "kelly":
        # Signal returns at t are what following side[t-1] earned over bar t. Shifted so
        # the Kelly estimate at t only sees closed bars.
        sig_r = pd.Series(np.r_[0.0, side[:-1]] * asset_returns.to_numpy(), index=asset_returns.index)
        return kelly(sig_r, cfg.kelly_window, cfg.kelly_max)
    if cfg.scheme == "confidence":
        return confidence(prob, cfg.confidence_floor, cap=cfg.leverage_cap)
    raise ValueError(f"Unknown sizing scheme {cfg.scheme!r}.")

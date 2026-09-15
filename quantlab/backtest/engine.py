"""Cost-aware backtest engine.

Conventions (read these twice):
- A *target* position is decided at the close of bar t from the signal at t.
- It is *executed* one bar later: `pos[t] = target[t-1]` for both execution modes.
- What differs is the return the executed position earns during bar t:
    next_open      : open[t+1] / open[t] - 1   (enter at t's open, mark at next open)
    close_to_close : close[t]  / close[t-1] - 1 (enter at t-1's close)
- Costs are charged on |pos[t] - pos[t-1]| at bar t, in return space, per side.
- Positions are fractions of current equity, so returns compound.

The core loop is Python over numpy arrays. Vectorising cooldown and max-holding
is possible and unreadable; 20k bars takes milliseconds either way.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quantlab.backtest.sizing import compute_sizes
from quantlab.config import BacktestConfig
from quantlab.features.technical import atr as _atr


@dataclass
class BacktestResult:
    returns: pd.Series  # strategy net return per bar, indexed by execution bar
    gross_returns: pd.Series
    costs: pd.Series
    positions: pd.Series  # executed position (signed size) per bar
    targets: pd.Series  # target decided at close of each bar
    equity: pd.Series
    benchmark_returns: pd.Series  # buy and hold, close to close, no costs
    benchmark_equity: pd.Series
    trades: pd.DataFrame
    exit_reasons: pd.Series
    config: BacktestConfig
    bars_per_year: float
    meta: dict = field(default_factory=dict)

    @property
    def exposure(self) -> float:
        return float((self.positions != 0).mean())

    @property
    def turnover_per_bar(self) -> float:
        return float(self.positions.diff().abs().fillna(self.positions.abs()).mean())


def signals_to_direction(signals: pd.DataFrame, cfg: BacktestConfig) -> np.ndarray:
    """Probabilities (+ optional meta side) -> desired direction in {-1, 0, +1} per bar.

    Expected columns:
        p_long, p_short         probability of the +1 / -1 class (binary: p_short = 1 - p_long)
        side, p_trade (optional) meta-labeling: primary side and P(take the trade)
    NaN probabilities mean "no prediction" and map to flat.
    """
    n = len(signals)
    d = np.zeros(n, dtype=float)
    if "side" in signals.columns and "p_trade" in signals.columns:
        side = signals["side"].fillna(0).to_numpy()
        p = signals["p_trade"].to_numpy(dtype=float)
        go_long = (side > 0) & (p > cfg.long_threshold)
        go_short = (side < 0) & (p > cfg.short_threshold)
    else:
        for c in ("p_long", "p_short"):
            if c not in signals.columns:
                raise ValueError(f"signals needs column {c!r} (or side/p_trade for meta-labeling).")
        pl = signals["p_long"].to_numpy(dtype=float)
        ps = signals["p_short"].to_numpy(dtype=float)
        go_long = pl > cfg.long_threshold
        go_short = ps > cfg.short_threshold
        both = go_long & go_short  # only possible when thresholds < 0.5; pick the stronger
        go_long[both] = pl[both] >= ps[both]
        go_short[both] = ~go_long[both]
    d[go_long] = 1.0
    d[go_short] = -1.0
    if cfg.direction == "long_only":
        d[d < 0] = 0.0
    elif cfg.direction == "short_only":
        d[d > 0] = 0.0
    return d


def _per_side_cost(cfg: BacktestConfig, atr_frac: np.ndarray) -> np.ndarray:
    c = cfg.costs
    fixed = (c.commission_bps + c.spread_bps / 2.0 + c.slippage_bps) / 1e4
    return fixed + c.slippage_atr_frac * np.nan_to_num(atr_frac, nan=0.0)


def run_backtest(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    cfg: BacktestConfig,
    bars_per_year: float = 252.0,
    atr_window: int = 14,
) -> BacktestResult:
    """Run the engine over the intersection of `prices` and `signals` indices."""
    cfg.validate()
    for c in ("open", "high", "low", "close"):
        if c not in prices.columns:
            raise ValueError(f"prices needs column {c!r}.")
    if not signals.index.isin(prices.index).all():
        raise ValueError("signals index contains timestamps not in prices. Alignment lost somewhere upstream.")
    px = prices.loc[signals.index.min() : signals.index.max()]
    sig = signals.reindex(px.index)  # gaps -> NaN -> flat
    idx = px.index
    n = len(idx)
    if n < 3:
        raise ValueError("Need at least 3 bars to backtest anything.")

    # Returns earned by a position held during bar t.
    close = px["close"].to_numpy(dtype=float)
    open_ = px["open"].to_numpy(dtype=float)
    if cfg.execution == "next_open":
        r_exec = np.r_[open_[1:] / open_[:-1] - 1.0, np.nan]
    else:
        r_exec = np.r_[np.nan, close[1:] / close[:-1] - 1.0]
    r_cc = pd.Series(np.r_[np.nan, close[1:] / close[:-1] - 1.0], index=idx)

    # ATR fraction for slippage, lagged so cost at t uses ATR known at t-1.
    atr_frac = (_atr(prices, atr_window) / prices["close"]).reindex(idx).shift(1).to_numpy()
    per_side = _per_side_cost(cfg, atr_frac)

    direction = signals_to_direction(sig, cfg)
    prob_for_sizing = _confidence_prob(sig, direction)
    size = compute_sizes(cfg.sizing, r_cc.fillna(0.0), direction, prob_for_sizing, bars_per_year)

    # State machine over target positions (decided at close t).
    target = np.zeros(n)
    reason = np.array([""] * n, dtype=object)
    pos_dir = 0.0
    hold = 0
    cool = 0
    for t in range(n):
        want = direction[t]
        if pos_dir != 0.0:
            hold += 1
            if cfg.max_holding > 0 and hold >= cfg.max_holding:
                pos_dir, hold, cool = 0.0, 0, cfg.cooldown
                reason[t] = "max_holding"
            elif want == 0.0:
                pos_dir, hold, cool = 0.0, 0, cfg.cooldown
                reason[t] = "signal"
            elif want != pos_dir:
                if cfg.cooldown > 0:
                    # Flip means an exit; honour the cooldown before re-entering.
                    pos_dir, hold, cool = 0.0, 0, cfg.cooldown
                    reason[t] = "flip"
                else:
                    pos_dir, hold = want, 0
                    reason[t] = "flip"
        else:
            if cool > 0:
                cool -= 1
            elif want != 0.0:
                pos_dir, hold = want, 0
        target[t] = pos_dir * size[t] if pos_dir != 0.0 else 0.0

    pos = np.r_[0.0, target[:-1]]  # executed one bar later
    trade_size = np.abs(np.diff(np.r_[0.0, pos]))
    cost = trade_size * per_side
    gross = pos * np.nan_to_num(r_exec, nan=0.0)
    net = gross - cost
    # The final bar in next_open mode has no next open to mark against. The
    # position is closed at the last available open with no P&L on that bar.
    if cfg.execution == "next_open":
        net[-1] = -cost[-1]
        gross[-1] = 0.0

    returns = pd.Series(net, index=idx, name="strategy")
    equity = (1.0 + returns).cumprod()
    bench = r_cc.fillna(0.0).rename("buy_and_hold")
    bench_eq = (1.0 + bench).cumprod()
    positions = pd.Series(pos, index=idx, name="position")
    trades = _extract_trades(idx, pos, returns.to_numpy(), reason)

    return BacktestResult(
        returns=returns,
        gross_returns=pd.Series(gross, index=idx),
        costs=pd.Series(cost, index=idx),
        positions=positions,
        targets=pd.Series(target, index=idx),
        equity=equity,
        benchmark_returns=bench,
        benchmark_equity=bench_eq,
        trades=trades,
        exit_reasons=pd.Series(reason, index=idx),
        config=cfg,
        bars_per_year=bars_per_year,
        meta={"n_bars": n, "direction_counts": pd.Series(direction).value_counts().to_dict()},
    )


def _confidence_prob(sig: pd.DataFrame, direction: np.ndarray) -> np.ndarray:
    """Probability backing the chosen direction, for confidence sizing."""
    if "p_trade" in sig.columns:
        return sig["p_trade"].to_numpy(dtype=float)
    pl = sig["p_long"].to_numpy(dtype=float)
    ps = sig["p_short"].to_numpy(dtype=float)
    return np.where(direction > 0, pl, np.where(direction < 0, ps, np.nan))


def _extract_trades(idx: pd.DatetimeIndex, pos: np.ndarray, net: np.ndarray, reason: np.ndarray) -> pd.DataFrame:
    """Group consecutive same-sign executed positions into trades."""
    rows = []
    n = len(pos)
    i = 0
    while i < n:
        if pos[i] == 0.0:
            i += 1
            continue
        side = np.sign(pos[i])
        j = i
        while j + 1 < n and np.sign(pos[j + 1]) == side:
            j += 1
        seg = net[i : j + 1]
        pnl = float(np.prod(1.0 + seg) - 1.0)
        # The exit decision was taken at the close of bar j (executed at j+1);
        # reason[j] holds why. End of data if the position was still open.
        why = reason[j] if j < n and reason[j] else ("end_of_data" if j == n - 1 else "signal")
        rows.append(
            {
                "entry": idx[i],
                "exit": idx[min(j + 1, n - 1)],
                "side": "long" if side > 0 else "short",
                "avg_size": float(np.mean(np.abs(pos[i : j + 1]))),
                "bars_held": int(j - i + 1),
                "pnl": pnl,
                "reason": why,
            }
        )
        i = j + 1
    cols = ["entry", "exit", "side", "avg_size", "bars_held", "pnl", "reason"]
    return pd.DataFrame(rows, columns=cols)

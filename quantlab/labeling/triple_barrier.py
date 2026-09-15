"""Triple-barrier labels: profit-take and stop-loss as ATR multiples, vertical barrier at h bars.

Why ATR multiples rather than fixed percentages: a 2% barrier is a coin-flip in a
30-vol regime and a week-long event in a 10-vol regime. Scaling by ATR keeps the
label meaning roughly the same across regimes.

The loop is explicit Python over numpy arrays. It is O(n*h) and fine for tens of
thousands of bars. If you need a million bars, numba it, but you do not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.features.technical import atr as _atr

BARRIER_CODES = {"pt": 1, "sl": -1, "vertical": 0}


def triple_barrier_labels(
    df: pd.DataFrame,
    horizon: int,
    pt_mult: float,
    sl_mult: float,
    atr_window: int = 14,
    side: pd.Series | None = None,
    use_high_low: bool = True,
) -> pd.DataFrame:
    """Return label / barrier / ret / t1 per entry bar.

    `side` (optional, -1/+1 per bar) flips the barriers for shorts; with a side
    given, label=1 means "the trade on that side hit profit-take", not "price
    went up". This is what meta-labeling consumes. Bars with side 0 get NaN.

    `use_high_low`: touch detection on intrabar high/low (realistic) versus on
    close only (conservative when you distrust Yahoo's intraday extremes).
    """
    for c in ("high", "low", "close"):
        if c not in df.columns:
            raise ValueError(f"triple_barrier_labels needs column {c!r}.")
    if horizon < 1 or pt_mult <= 0 or sl_mult <= 0:
        raise ValueError("horizon >= 1 and positive barrier multiples are required.")

    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float) if use_high_low else close
    low = df["low"].to_numpy(dtype=float) if use_high_low else close
    vol = _atr(df, atr_window).to_numpy(dtype=float)
    n = len(close)
    sides = np.ones(n) if side is None else side.reindex(df.index).to_numpy(dtype=float)

    label = np.full(n, np.nan)
    barrier = np.full(n, np.nan)
    ret = np.full(n, np.nan)
    t1_pos = np.full(n, -1, dtype=int)

    for i in range(n):
        s = sides[i]
        if np.isnan(vol[i]) or vol[i] <= 0 or s == 0 or np.isnan(s):
            continue
        end = i + horizon
        if end >= n:
            continue  # horizon runs off the end: no label, not a truncated one
        entry = close[i]
        upper = entry + (pt_mult if s > 0 else sl_mult) * vol[i]
        lower = entry - (sl_mult if s > 0 else pt_mult) * vol[i]
        hit = -1
        hit_kind = "vertical"
        for j in range(i + 1, end + 1):
            touched_up = high[j] >= upper
            touched_dn = low[j] <= lower
            if touched_up and touched_dn:
                # Both in the same bar. We cannot know the order; assume the
                # stop-loss came first because pessimism is cheaper than optimism.
                hit, hit_kind = j, ("sl" if s > 0 else "pt")
                break
            if touched_up:
                hit, hit_kind = j, ("pt" if s > 0 else "sl")
                break
            if touched_dn:
                hit, hit_kind = j, ("sl" if s > 0 else "pt")
                break
        if hit == -1:
            hit = end
            exit_price = close[end]
        else:
            # Fill at the barrier price, not the close of the touching bar.
            exit_price = upper if (high[hit] >= upper and hit_kind == ("pt" if s > 0 else "sl")) else lower
        r = s * (exit_price / entry - 1.0)
        ret[i] = r
        t1_pos[i] = hit
        barrier[i] = BARRIER_CODES[hit_kind]
        if hit_kind == "pt":
            label[i] = 1
        elif hit_kind == "sl":
            label[i] = -1
        else:
            label[i] = 1 if r > 0 else -1

    idx = df.index
    t1 = pd.Series([idx[p] if p >= 0 else pd.NaT for p in t1_pos], index=idx, dtype="datetime64[ns]")
    out = pd.DataFrame(
        {
            "label": label,
            "barrier": pd.Series(barrier, index=idx).map({1: "pt", -1: "sl", 0: "vertical"}),
            "ret": ret,
            "t1": t1,
        }
    )
    out.attrs["scheme"] = "triple_barrier"
    return out

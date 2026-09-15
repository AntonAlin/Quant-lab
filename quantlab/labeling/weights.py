"""Sample weights for overlapping labels.

Two ideas, both from Lopez de Prado (AFML ch. 4):
1. Uniqueness: a label spanning bars [t0, t1] shares those bars with every other
   label that overlaps. Average uniqueness = mean over its bars of 1/concurrency.
   Highly overlapping labels are near-duplicates and should not count as many.
2. Return attribution: weight by |sum over the label's bars of r_t / c_t| where
   c_t is concurrency. Big, uniquely-owned moves matter more than noise.

Both are normalised to mean 1 so the effective sample size is unchanged and the
regularisation strength of the models does not silently drift.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def concurrency(index: pd.DatetimeIndex, t1: pd.Series) -> pd.Series:
    """Number of live labels at each bar. t1 is the resolution time per label start."""
    t1 = t1.dropna()
    if t1.empty:
        return pd.Series(0.0, index=index)
    starts = index.get_indexer(t1.index)
    ends = index.get_indexer(t1.to_numpy())
    if (starts < 0).any() or (ends < 0).any():
        raise ValueError("t1 contains timestamps that are not in the bar index. Labels and bars are misaligned.")
    # Difference-array trick: +1 at start, -1 after end, cumsum.
    diff = np.zeros(len(index) + 1)
    np.add.at(diff, starts, 1.0)
    np.add.at(diff, ends + 1, -1.0)
    return pd.Series(np.cumsum(diff)[:-1], index=index)


def uniqueness_weights(index: pd.DatetimeIndex, t1: pd.Series) -> pd.Series:
    conc = concurrency(index, t1).to_numpy()
    out = pd.Series(np.nan, index=t1.index, dtype=float)
    pos = index.get_indexer(t1.index)
    for k, (i, end) in enumerate(zip(pos, t1.to_numpy())):
        if pd.isna(end):
            continue
        j = index.get_loc(end)
        c = conc[i : j + 1]
        c = c[c > 0]
        out.iloc[k] = float(np.mean(1.0 / c)) if len(c) else np.nan
    return _normalise(out)


def return_attribution_weights(index: pd.DatetimeIndex, t1: pd.Series, close: pd.Series) -> pd.Series:
    conc = concurrency(index, t1).to_numpy()
    r = np.log(close.reindex(index)).diff().to_numpy()
    out = pd.Series(np.nan, index=t1.index, dtype=float)
    pos = index.get_indexer(t1.index)
    for k, (i, end) in enumerate(zip(pos, t1.to_numpy())):
        if pd.isna(end):
            continue
        j = index.get_loc(end)
        seg_r = r[i + 1 : j + 1]
        seg_c = conc[i + 1 : j + 1]
        ok = (seg_c > 0) & ~np.isnan(seg_r)
        out.iloc[k] = float(np.abs((seg_r[ok] / seg_c[ok]).sum())) if ok.any() else np.nan
    return _normalise(out)


def combined_weights(index: pd.DatetimeIndex, t1: pd.Series, close: pd.Series, scheme: str) -> pd.Series:
    """Dispatcher for LabelConfig.weighting."""
    if scheme == "none":
        return pd.Series(1.0, index=t1.index)
    if scheme == "uniqueness":
        return uniqueness_weights(index, t1)
    if scheme == "return_attribution":
        return return_attribution_weights(index, t1, close)
    if scheme == "both":
        return _normalise(uniqueness_weights(index, t1) * return_attribution_weights(index, t1, close))
    raise ValueError(f"Unknown weighting scheme {scheme!r}.")


def _normalise(w: pd.Series) -> pd.Series:
    m = w.mean(skipna=True)
    if not np.isfinite(m) or m <= 0:
        return w
    return w / m

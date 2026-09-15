"""Fixed-horizon labels: sign of the forward return over h bars, with an optional deadband.

Output contract shared by all labelers (a DataFrame indexed like the input):
    label : int   -1 / 0 / 1 (0 only in three-class mode)
    ret   : float realised forward return used for the label
    t1    : Timestamp of the bar where the label is resolved (for purging/embargo/uniqueness)
Rows whose horizon runs past the end of the data are NaN, not clipped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def fixed_horizon_labels(
    close: pd.Series,
    horizon: int,
    deadband: float = 0.0,
    three_class: bool = False,
) -> pd.DataFrame:
    if horizon < 1:
        raise ValueError("horizon must be >= 1.")
    if deadband < 0:
        raise ValueError("deadband must be >= 0.")
    if not isinstance(close.index, pd.DatetimeIndex):
        raise TypeError("close must be indexed by a DatetimeIndex.")
    fwd = close.shift(-horizon) / close - 1.0
    label = pd.Series(np.nan, index=close.index, dtype=float)
    if three_class:
        label[fwd > deadband] = 1
        label[fwd < -deadband] = -1
        label[(fwd.abs() <= deadband) & fwd.notna()] = 0
    else:
        # Binary: the deadband is ignored on purpose; a two-class label with a
        # hole in the middle would just be a three-class label with missing rows.
        label[fwd > 0] = 1
        label[fwd <= 0] = -1
        label[fwd.isna()] = np.nan
    t1_pos = np.arange(len(close)) + horizon
    t1 = pd.Series(
        [close.index[i] if i < len(close) else pd.NaT for i in t1_pos], index=close.index, dtype="datetime64[ns]"
    )
    out = pd.DataFrame({"label": label, "ret": fwd, "t1": t1})
    out.attrs["scheme"] = "fixed_horizon"
    return out

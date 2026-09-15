"""Trend-scanning labels (Lopez de Prado, ML for Asset Managers, §5.4).

For each bar, fit OLS of price on time over forward windows of length L in
[min_window, max_window]; keep the window with the largest |t-stat| of the
slope; the label is the sign of that slope. The t-stat itself is a decent
"confidence" feature and is returned too.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _tstat_slope(y: np.ndarray) -> float:
    n = len(y)
    x = np.arange(n, dtype=float)
    xm, ym = x.mean(), y.mean()
    sxx = ((x - xm) ** 2).sum()
    if sxx == 0:
        return 0.0
    b = ((x - xm) * (y - ym)).sum() / sxx
    a = ym - b * xm
    resid = y - (a + b * x)
    dof = n - 2
    if dof <= 0:
        return 0.0
    s2 = (resid**2).sum() / dof
    se = np.sqrt(s2 / sxx) if s2 > 0 else 0.0
    if se == 0:
        return np.sign(b) * 1e6 if b != 0 else 0.0
    return float(b / se)


def trend_scanning_labels(
    close: pd.Series,
    min_window: int,
    max_window: int,
    step: int = 1,
    use_log: bool = True,
) -> pd.DataFrame:
    if min_window < 3 or max_window <= min_window or step < 1:
        raise ValueError("Need min_window >= 3, max_window > min_window, step >= 1.")
    y_all = np.log(close.to_numpy(dtype=float)) if use_log else close.to_numpy(dtype=float)
    n = len(y_all)
    windows = list(range(min_window, max_window + 1, step))
    label = np.full(n, np.nan)
    tval = np.full(n, np.nan)
    best_w = np.full(n, np.nan)
    t1_pos = np.full(n, -1, dtype=int)
    for i in range(n):
        if i + max_window > n:
            break  # not enough future for the longest window: no label
        best_t, best_len = 0.0, 0
        for L in windows:
            seg = y_all[i : i + L]
            if np.isnan(seg).any():
                continue
            t = _tstat_slope(seg)
            if abs(t) > abs(best_t):
                best_t, best_len = t, L
        if best_len == 0:
            continue
        label[i] = 1.0 if best_t > 0 else -1.0
        tval[i] = best_t
        best_w[i] = best_len
        t1_pos[i] = i + best_len - 1
    idx = close.index
    t1 = pd.Series([idx[p] if p >= 0 else pd.NaT for p in t1_pos], index=idx, dtype="datetime64[ns]")
    ret = pd.Series(np.nan, index=idx)
    ok = t1_pos >= 0
    ret[ok] = close.to_numpy()[t1_pos[ok]] / close.to_numpy()[ok] - 1.0
    out = pd.DataFrame({"label": label, "t_value": tval, "window": best_w, "ret": ret, "t1": t1})
    out.attrs["scheme"] = "trend_scanning"
    return out

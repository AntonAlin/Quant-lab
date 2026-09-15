"""Distributional and long-memory features: moments, Hurst, autocorrelation, fractional differentiation.

Fractional differentiation follows Lopez de Prado's fixed-width window method.
The point is to make a price series stationary while keeping as much memory as
possible; `d=1` is a plain return and throws the memory away.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.features.registry import ParamSpec, register

_W = lambda default, lo=5, hi=1000: ParamSpec("window", "int", default, lo, hi, 1)  # noqa: E731


@register("rolling_skew", "statistical", (_W(60),), "Rolling skewness of log returns.")
def f_skew(df: pd.DataFrame, window: int) -> pd.Series:
    return np.log(df["close"]).diff().rolling(window, min_periods=window).skew()


@register("rolling_kurt", "statistical", (_W(60),), "Rolling excess kurtosis of log returns.")
def f_kurt(df: pd.DataFrame, window: int) -> pd.Series:
    return np.log(df["close"]).diff().rolling(window, min_periods=window).kurt()


def hurst_rs(x: np.ndarray) -> float:
    """Rescaled-range Hurst on one window. ~0.5 random, >0.5 trending, <0.5 mean reverting.

    Uses a handful of sub-window sizes and a log-log fit. Noisy on short windows,
    which is why the default window is 100 and not 20.
    """
    n = len(x)
    if n < 20 or np.all(x == x[0]):
        return np.nan
    sizes = np.unique(np.floor(np.logspace(np.log10(10), np.log10(n // 2), 6)).astype(int))
    rs_vals = []
    for s in sizes:
        chunks = n // s
        if chunks < 1:
            continue
        vals = []
        for i in range(chunks):
            seg = x[i * s : (i + 1) * s]
            dev = np.cumsum(seg - seg.mean())
            r = dev.max() - dev.min()
            sd = seg.std(ddof=1)
            if sd > 0:
                vals.append(r / sd)
        if vals:
            rs_vals.append((s, np.mean(vals)))
    if len(rs_vals) < 3:
        return np.nan
    ls, lrs = np.log([a for a, _ in rs_vals]), np.log([b for _, b in rs_vals])
    return float(np.polyfit(ls, lrs, 1)[0])


@register("hurst", "statistical", (_W(100, 30),), "Rescaled-range Hurst exponent of log returns.", bounded=True)
def f_hurst(df: pd.DataFrame, window: int) -> pd.Series:
    r = np.log(df["close"]).diff()
    return r.rolling(window, min_periods=window).apply(hurst_rs, raw=True)


@register(
    "autocorr",
    "statistical",
    (_W(60), ParamSpec("lag", "int", 1, 1, 50, 1)),
    "Rolling autocorrelation of log returns at a given lag.",
    bounded=True,
)
def f_autocorr(df: pd.DataFrame, window: int, lag: int) -> pd.Series:
    if lag >= window:
        raise ValueError("autocorr: lag must be < window.")
    r = np.log(df["close"]).diff()
    return r.rolling(window, min_periods=window).apply(lambda x: _acf_lag(x, lag), raw=True)


def _acf_lag(x: np.ndarray, lag: int) -> float:
    a, b = x[:-lag], x[lag:]
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


# --------------------------------------------------------------------------- #
# Fractional differentiation
# --------------------------------------------------------------------------- #
def frac_diff_weights(d: float, threshold: float = 1e-4, max_size: int = 1000) -> np.ndarray:
    """Binomial weights of (1-B)^d, truncated once |w| < threshold. Returned oldest-first."""
    w = [1.0]
    k = 1
    while k < max_size:
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < threshold:
            break
        w.append(w_k)
        k += 1
    return np.array(w[::-1])


def frac_diff(series: pd.Series, d: float, threshold: float = 1e-4) -> pd.Series:
    """Fixed-width-window fractional differentiation. NaN until the window is full."""
    if not 0 <= d <= 2:
        raise ValueError(f"frac_diff: d={d} is outside [0, 2]. Nobody needs more than that.")
    w = frac_diff_weights(d, threshold)
    width = len(w)
    x = series.to_numpy(dtype=float)
    out = np.full(len(x), np.nan)
    if width > len(x):
        return pd.Series(out, index=series.index)
    # np.convolve with 'valid' does exactly the dot-product sweep we want.
    conv = np.convolve(x, w[::-1], mode="valid")
    out[width - 1 :] = conv
    # Any window that contained a NaN must be NaN, convolve would have propagated it anyway.
    return pd.Series(out, index=series.index)


@register(
    "frac_diff",
    "statistical",
    (ParamSpec("d", "float", 0.4, 0.0, 2.0, 0.05), ParamSpec("threshold", "float", 1e-4, 1e-6, 1e-2, 1e-5)),
    "Fractionally differentiated log close. Use the ADF helper on the Data tab to pick d.",
)
def f_frac_diff(df: pd.DataFrame, d: float, threshold: float) -> pd.Series:
    return frac_diff(np.log(df["close"]), d, threshold)


def adf_pvalue(x: pd.Series) -> float:
    from statsmodels.tsa.stattools import adfuller

    clean = x.dropna().to_numpy()
    if len(clean) < 30:
        return np.nan
    return float(adfuller(clean, maxlag=1, regression="c", autolag=None)[1])


def min_stationary_d(
    series: pd.Series,
    d_grid: np.ndarray | None = None,
    alpha: float = 0.05,
    threshold: float = 1e-4,
) -> pd.DataFrame:
    """Sweep d, report ADF p-value and correlation with the original series.

    The 'best' d is the smallest one with p < alpha. The correlation column tells
    you how much memory you kept; if it is 0.2 you might as well use returns.
    """
    if d_grid is None:
        d_grid = np.round(np.arange(0.0, 1.01, 0.1), 2)
    rows = []
    for d in d_grid:
        fd = frac_diff(series, float(d), threshold)
        p = adf_pvalue(fd)
        corr = float(pd.concat([series, fd], axis=1).dropna().corr().iloc[0, 1]) if fd.notna().sum() > 10 else np.nan
        rows.append({"d": float(d), "adf_pvalue": p, "corr_with_original": corr, "stationary": bool(p < alpha)})
    out = pd.DataFrame(rows)
    out.attrs["suggested_d"] = float(out.loc[out["stationary"], "d"].min()) if out["stationary"].any() else np.nan
    return out

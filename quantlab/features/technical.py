"""Classic indicators. Every lookback is a parameter, every calculation is backward-looking.

Implementation notes for future-me:
- Wilder smoothing (RSI, ATR, ADX) is an EMA with alpha = 1/n. We use pandas
  ewm(alpha=1/n, adjust=False) which matches the textbook recursion after warm-up.
- We never call .fillna(method=...) on prices. NaNs at the top are warm-up and
  the model layer deals with them explicitly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.features.registry import ParamSpec, register

_W = lambda default, lo=2, hi=500, name="window": ParamSpec(name, "int", default, lo, hi, 1)  # noqa: E731


def _need(df: pd.DataFrame, *cols: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Feature needs columns {missing}; frame has {list(df.columns)}.")


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def wma(s: pd.Series, n: int) -> pd.Series:
    w = np.arange(1, n + 1, dtype=float)
    return s.rolling(n, min_periods=n).apply(lambda x: float(np.dot(x, w) / w.sum()), raw=True)


def wilder(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    _need(df, "high", "low", "close")
    prev_close = df["close"].shift(1)
    return pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()], axis=1
    ).max(axis=1)


def atr(df: pd.DataFrame, n: int) -> pd.Series:
    return wilder(true_range(df), n)


# --------------------------------------------------------------------------- #
# Trend / momentum
# --------------------------------------------------------------------------- #
@register("sma_ratio", "trend", (_W(20),), "close / SMA(window) - 1. Raw SMA level is useless to a model; the ratio is not.")
def f_sma_ratio(df: pd.DataFrame, window: int) -> pd.Series:
    return df["close"] / sma(df["close"], window) - 1.0


@register("ema_ratio", "trend", (_W(20),), "close / EMA(window) - 1.")
def f_ema_ratio(df: pd.DataFrame, window: int) -> pd.Series:
    return df["close"] / ema(df["close"], window) - 1.0


@register("wma_ratio", "trend", (_W(20),), "close / WMA(window) - 1.")
def f_wma_ratio(df: pd.DataFrame, window: int) -> pd.Series:
    return df["close"] / wma(df["close"], window) - 1.0


@register(
    "ma_crossover",
    "trend",
    (ParamSpec("fast", "int", 20, 2, 300, 1), ParamSpec("slow", "int", 50, 3, 500, 1), ParamSpec("kind", "choice", "ema", choices=("sma", "ema"))),
    "(fast MA - slow MA) / slow MA. Positive = golden cross territory.",
)
def f_ma_crossover(df: pd.DataFrame, fast: int, slow: int, kind: str) -> pd.Series:
    if fast >= slow:
        raise ValueError(f"ma_crossover: fast ({fast}) must be < slow ({slow}).")
    f = ema if kind == "ema" else sma
    return (f(df["close"], fast) - f(df["close"], slow)) / f(df["close"], slow)


@register(
    "macd",
    "trend",
    (ParamSpec("fast", "int", 12, 2, 100, 1), ParamSpec("slow", "int", 26, 3, 300, 1), ParamSpec("signal", "int", 9, 2, 100, 1)),
    "MACD line, signal line and histogram, all divided by close so they are scale-free.",
)
def f_macd(df: pd.DataFrame, fast: int, slow: int, signal: int) -> pd.DataFrame:
    if fast >= slow:
        raise ValueError(f"macd: fast ({fast}) must be < slow ({slow}).")
    line = ema(df["close"], fast) - ema(df["close"], slow)
    sig = ema(line, signal)
    c = df["close"]
    return pd.DataFrame({"line": line / c, "signal": sig / c, "hist": (line - sig) / c})


@register("roc", "trend", (_W(10, 1),), "Rate of change: close / close[t-window] - 1.")
def f_roc(df: pd.DataFrame, window: int) -> pd.Series:
    return df["close"].pct_change(window)


@register("momentum", "trend", (_W(10, 1),), "Log return over the last k bars.")
def f_momentum(df: pd.DataFrame, window: int) -> pd.Series:
    return np.log(df["close"]).diff(window)


def _dm(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus = up.where((up > dn) & (up > 0), 0.0)
    minus = dn.where((dn > up) & (dn > 0), 0.0)
    return plus, minus


def adx(df: pd.DataFrame, n: int) -> pd.DataFrame:
    _need(df, "high", "low", "close")
    plus, minus = _dm(df)
    tr_n = wilder(true_range(df), n)
    pdi = 100 * wilder(plus, n) / tr_n
    mdi = 100 * wilder(minus, n) / tr_n
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return pd.DataFrame({"adx": wilder(dx, n), "plus_di": pdi, "minus_di": mdi})


@register("adx", "trend", (_W(14),), "Wilder ADX plus +DI/-DI. Trend *strength*, not direction.", bounded=True)
def f_adx(df: pd.DataFrame, window: int) -> pd.DataFrame:
    return adx(df, window)


@register("aroon", "trend", (_W(25),), "Aroon up/down and oscillator.", bounded=True)
def f_aroon(df: pd.DataFrame, window: int) -> pd.DataFrame:
    # argmax over a window of length window+1 -> bars since the high
    hi = df["high"].rolling(window + 1, min_periods=window + 1).apply(lambda x: window - int(np.argmax(x)), raw=True)
    lo = df["low"].rolling(window + 1, min_periods=window + 1).apply(lambda x: window - int(np.argmin(x)), raw=True)
    up = 100 * (window - hi) / window
    dn = 100 * (window - lo) / window
    return pd.DataFrame({"up": up, "down": dn, "osc": up - dn})


@register("price_ma_zscore", "trend", (_W(20),), "(close - SMA) / rolling std of close. How stretched is price vs its own mean.")
def f_price_ma_zscore(df: pd.DataFrame, window: int) -> pd.Series:
    c = df["close"]
    return (c - sma(c, window)) / c.rolling(window, min_periods=window).std(ddof=1)


@register("donchian_pos", "trend", (_W(20),), "Where close sits in the [rolling low, rolling high] channel, 0..1.", bounded=True)
def f_donchian(df: pd.DataFrame, window: int) -> pd.Series:
    hi = df["high"].rolling(window, min_periods=window).max()
    lo = df["low"].rolling(window, min_periods=window).min()
    return (df["close"] - lo) / (hi - lo).replace(0, np.nan)


# --------------------------------------------------------------------------- #
# Mean reversion
# --------------------------------------------------------------------------- #
def rsi(s: pd.Series, n: int) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0.0)
    dn = -d.clip(upper=0.0)
    rs = wilder(up, n) / wilder(dn, n).replace(0, np.nan)
    return 100 - 100 / (1 + rs)


@register("rsi", "mean_reversion", (_W(14),), "Wilder RSI, 0..100.", bounded=True)
def f_rsi(df: pd.DataFrame, window: int) -> pd.Series:
    return rsi(df["close"], window)


@register(
    "stochastic",
    "mean_reversion",
    (ParamSpec("k_window", "int", 14, 2, 200, 1), ParamSpec("d_window", "int", 3, 1, 50, 1)),
    "Stochastic %K and %D.",
    bounded=True,
)
def f_stoch(df: pd.DataFrame, k_window: int, d_window: int) -> pd.DataFrame:
    lo = df["low"].rolling(k_window, min_periods=k_window).min()
    hi = df["high"].rolling(k_window, min_periods=k_window).max()
    k = 100 * (df["close"] - lo) / (hi - lo).replace(0, np.nan)
    return pd.DataFrame({"k": k, "d": sma(k, d_window)})


@register(
    "bollinger",
    "mean_reversion",
    (_W(20), ParamSpec("n_std", "float", 2.0, 0.5, 5.0, 0.1)),
    "%B (position within bands) and bandwidth (band width / middle).",
)
def f_bollinger(df: pd.DataFrame, window: int, n_std: float) -> pd.DataFrame:
    c = df["close"]
    mid = sma(c, window)
    sd = c.rolling(window, min_periods=window).std(ddof=1)
    upper, lower = mid + n_std * sd, mid - n_std * sd
    return pd.DataFrame({"pct_b": (c - lower) / (upper - lower).replace(0, np.nan), "bandwidth": (upper - lower) / mid})


@register("williams_r", "mean_reversion", (_W(14),), "Williams %R, -100..0.", bounded=True)
def f_willr(df: pd.DataFrame, window: int) -> pd.Series:
    hi = df["high"].rolling(window, min_periods=window).max()
    lo = df["low"].rolling(window, min_periods=window).min()
    return -100 * (hi - df["close"]) / (hi - lo).replace(0, np.nan)


@register("cci", "mean_reversion", (_W(20),), "Commodity Channel Index on typical price.")
def f_cci(df: pd.DataFrame, window: int) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    ma = sma(tp, window)
    # mean absolute deviation, rolling. pandas dropped .mad() so we do it by hand.
    mad = tp.rolling(window, min_periods=window).apply(lambda x: float(np.mean(np.abs(x - x.mean()))), raw=True)
    return (tp - ma) / (0.015 * mad.replace(0, np.nan))


@register(
    "ols_residual_z",
    "mean_reversion",
    (_W(20), ParamSpec("ma_window", "int", 50, 3, 500, 1)),
    "Z-score of the residual from a rolling OLS of close on its own MA. Stretch, but with a slope.",
)
def f_ols_resid_z(df: pd.DataFrame, window: int, ma_window: int) -> pd.Series:
    y = df["close"]
    x = sma(y, ma_window)
    # Rolling OLS y = a + b x via rolling moments; the residual z-score uses the same window.
    mx, my = x.rolling(window).mean(), y.rolling(window).mean()
    cov = (x * y).rolling(window).mean() - mx * my
    var = (x * x).rolling(window).mean() - mx * mx
    b = cov / var.replace(0, np.nan)
    a = my - b * mx
    resid = y - (a + b * x)
    # The residual std over the window is computed from the *current* fit, which
    # would need a second pass. Rolling std of the residual series is close enough.
    return resid / resid.rolling(window, min_periods=window).std(ddof=1)


# --------------------------------------------------------------------------- #
# Volatility
# --------------------------------------------------------------------------- #
@register("realised_vol", "volatility", (_W(20),), "Rolling std of log returns (per bar, not annualised).")
def f_realised_vol(df: pd.DataFrame, window: int) -> pd.Series:
    return np.log(df["close"]).diff().rolling(window, min_periods=window).std(ddof=1)


@register("atr", "volatility", (_W(14),), "Average True Range in price units.")
def f_atr(df: pd.DataFrame, window: int) -> pd.Series:
    return atr(df, window)


@register("natr", "volatility", (_W(14),), "ATR / close. The one you actually want in a model.")
def f_natr(df: pd.DataFrame, window: int) -> pd.Series:
    return atr(df, window) / df["close"]


@register("parkinson", "volatility", (_W(20),), "Parkinson high-low volatility estimator.")
def f_parkinson(df: pd.DataFrame, window: int) -> pd.Series:
    hl = np.log(df["high"] / df["low"]) ** 2
    return np.sqrt(hl.rolling(window, min_periods=window).mean() / (4 * np.log(2)))


@register("garman_klass", "volatility", (_W(20),), "Garman-Klass OHLC volatility estimator.")
def f_gk(df: pd.DataFrame, window: int) -> pd.Series:
    hl = np.log(df["high"] / df["low"]) ** 2
    co = np.log(df["close"] / df["open"]) ** 2
    v = 0.5 * hl - (2 * np.log(2) - 1) * co
    return np.sqrt(v.rolling(window, min_periods=window).mean().clip(lower=0))


@register("yang_zhang", "volatility", (_W(20),), "Yang-Zhang estimator: handles overnight gaps, which Garman-Klass pretends do not exist.")
def f_yz(df: pd.DataFrame, window: int) -> pd.Series:
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    log_oc_prev = np.log(o / c.shift(1))  # overnight
    log_co = np.log(c / o)  # open-to-close
    rs = np.log(h / o) * np.log(h / c) + np.log(l / o) * np.log(l / c)  # Rogers-Satchell
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    v_on = log_oc_prev.rolling(window, min_periods=window).var(ddof=1)
    v_oc = log_co.rolling(window, min_periods=window).var(ddof=1)
    v_rs = rs.rolling(window, min_periods=window).mean()
    return np.sqrt((v_on + k * v_oc + (1 - k) * v_rs).clip(lower=0))


@register(
    "vol_ratio",
    "volatility",
    (ParamSpec("short", "int", 10, 2, 200, 1), ParamSpec("long", "int", 60, 3, 500, 1)),
    "Short realised vol / long realised vol. >1 means vol is expanding.",
)
def f_vol_ratio(df: pd.DataFrame, short: int, long: int) -> pd.Series:
    if short >= long:
        raise ValueError("vol_ratio: short must be < long.")
    r = np.log(df["close"]).diff()
    return r.rolling(short, min_periods=short).std(ddof=1) / r.rolling(long, min_periods=long).std(ddof=1)


@register(
    "vol_of_vol",
    "volatility",
    (ParamSpec("vol_window", "int", 20, 2, 200, 1), ParamSpec("window", "int", 60, 5, 500, 1)),
    "Rolling std of rolling vol, normalised by its mean.",
)
def f_vol_of_vol(df: pd.DataFrame, vol_window: int, window: int) -> pd.Series:
    v = np.log(df["close"]).diff().rolling(vol_window, min_periods=vol_window).std(ddof=1)
    return v.rolling(window, min_periods=window).std(ddof=1) / v.rolling(window, min_periods=window).mean()


# --------------------------------------------------------------------------- #
# Volume
# --------------------------------------------------------------------------- #
@register("obv_z", "volume", (_W(20),), "On-balance volume, z-scored over the window (raw OBV is a random walk with a unit).")
def f_obv(df: pd.DataFrame, window: int) -> pd.Series:
    _need(df, "volume")
    direction = np.sign(df["close"].diff()).fillna(0.0)
    obv = (direction * df["volume"]).cumsum()
    return (obv - obv.rolling(window, min_periods=window).mean()) / obv.rolling(window, min_periods=window).std(ddof=1)


@register("volume_z", "volume", (_W(20),), "Volume z-score vs its rolling window.")
def f_volume_z(df: pd.DataFrame, window: int) -> pd.Series:
    v = df["volume"]
    return (v - v.rolling(window, min_periods=window).mean()) / v.rolling(window, min_periods=window).std(ddof=1)


@register("vwap_distance", "volume", (_W(20),), "close / rolling VWAP - 1.")
def f_vwap(df: pd.DataFrame, window: int) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    pv = (tp * df["volume"]).rolling(window, min_periods=window).sum()
    vv = df["volume"].rolling(window, min_periods=window).sum()
    return df["close"] / (pv / vv.replace(0, np.nan)) - 1.0


@register("cmf", "volume", (_W(20),), "Chaikin Money Flow.", bounded=True)
def f_cmf(df: pd.DataFrame, window: int) -> pd.Series:
    hl = (df["high"] - df["low"]).replace(0, np.nan)
    mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / hl
    mfv = mfm * df["volume"]
    return mfv.rolling(window, min_periods=window).sum() / df["volume"].rolling(window, min_periods=window).sum().replace(0, np.nan)


@register("amihud", "volume", (_W(20),), "Amihud illiquidity: mean |return| / dollar volume, log-scaled so it fits on a chart.")
def f_amihud(df: pd.DataFrame, window: int) -> pd.Series:
    r = df["close"].pct_change().abs()
    dv = (df["close"] * df["volume"]).replace(0, np.nan)
    with np.errstate(divide="ignore"):
        return np.log((r / dv).rolling(window, min_periods=window).mean() * 1e9 + 1e-12)

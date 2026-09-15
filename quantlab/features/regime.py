"""Regime / state features and calendar dummies.

The HMM feature is the one that will bite you: `hmmlearn` fits by EM, which is
sensitive to initialisation and can flip state labels between fits. We fit on an
expanding window with a refit cadence and order states by variance so that
state 0 is always "calm". It is still slow-ish and still a bit of a black box,
which is why it is off by default.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.features.registry import ParamSpec, register
from quantlab.features.technical import adx

_W = lambda default, lo=5, hi=1000: ParamSpec("window", "int", default, lo, hi, 1)  # noqa: E731


@register(
    "vol_tercile",
    "regime",
    (ParamSpec("vol_window", "int", 20, 2, 200, 1), ParamSpec("window", "int", 250, 30, 2000, 1)),
    "Which tercile (0/1/2) today's realised vol sits in relative to its trailing window.",
    bounded=True,
)
def f_vol_tercile(df: pd.DataFrame, vol_window: int, window: int) -> pd.Series:
    v = np.log(df["close"]).diff().rolling(vol_window, min_periods=vol_window).std(ddof=1)
    pct = v.rolling(window, min_periods=window // 2).rank(pct=True)
    return np.floor(pct * 3).clip(upper=2)


@register(
    "trend_range",
    "regime",
    (ParamSpec("window", "int", 14, 2, 200, 1), ParamSpec("threshold", "float", 25.0, 5.0, 60.0, 1.0)),
    "1 if ADX > threshold (trending), else 0 (ranging).",
    bounded=True,
)
def f_trend_range(df: pd.DataFrame, window: int, threshold: float) -> pd.Series:
    a = adx(df, window)["adx"]
    return (a > threshold).astype(float).where(a.notna())


@register("drawdown", "regime", (), "Current drawdown from the running maximum close, <= 0.")
def f_drawdown(df: pd.DataFrame) -> pd.Series:
    c = df["close"]
    return c / c.cummax() - 1.0


@register("days_since_ath", "regime", (ParamSpec("log", "bool", True),), "Bars since the last all-time-high close (optionally log1p).")
def f_days_since_ath(df: pd.DataFrame, log: bool) -> pd.Series:
    c = df["close"].to_numpy()
    out = np.zeros(len(c))
    last = 0
    running = -np.inf
    for i, v in enumerate(c):
        if v >= running:
            running, last = v, i
        out[i] = i - last
    s = pd.Series(out, index=df.index)
    return np.log1p(s) if log else s


@register(
    "hmm_state",
    "regime",
    (
        ParamSpec("refit_every", "int", 60, 5, 500, 1),
        ParamSpec("min_train", "int", 250, 60, 2000, 1),
        ParamSpec("n_iter", "int", 50, 10, 500, 10),
    ),
    "P(high-vol state) from a 2-state Gaussian HMM on log returns, fitted on an expanding window with no lookahead.",
    bounded=True,
)
def f_hmm_state(df: pd.DataFrame, refit_every: int, min_train: int, n_iter: int) -> pd.Series:
    import logging

    from hmmlearn.hmm import GaussianHMM

    from quantlab import SEED

    # hmmlearn logs "not converging" for a delta of -0.003. Thanks, we know.
    logging.getLogger("hmmlearn").setLevel(logging.ERROR)

    r = np.log(df["close"]).diff().to_numpy()
    n = len(r)
    out = np.full(n, np.nan)
    model: GaussianHMM | None = None
    high_state = 1
    last_fit = -1
    for t in range(min_train, n):
        if model is None or t - last_fit >= refit_every:
            train = r[1:t].reshape(-1, 1)
            train = train[~np.isnan(train).any(axis=1)]
            if len(train) < min_train // 2:
                continue
            m = GaussianHMM(n_components=2, covariance_type="diag", n_iter=n_iter, random_state=SEED)
            try:
                m.fit(train)
            except ValueError:
                # EM occasionally collapses a state on flat data. Keep the previous model.
                continue
            model = m
            high_state = int(np.argmax(model.covars_.ravel()))
            last_fit = t
        # Posterior of the *current* bar using data up to and including t.
        window = r[max(1, t - 250) : t + 1].reshape(-1, 1)
        if np.isnan(window).any():
            continue
        post = model.predict_proba(window)
        out[t] = post[-1, high_state]
    return pd.Series(out, index=df.index)


# --------------------------------------------------------------------------- #
# Calendar
# --------------------------------------------------------------------------- #
@register("day_of_week", "calendar", (), "0=Monday .. 6=Sunday. Tree models eat this raw; linear models want one-hot, so use 'rank' or leave it.", bounded=True)
def f_dow(df: pd.DataFrame) -> pd.Series:
    return pd.Series(df.index.dayofweek.astype(float), index=df.index)


@register("month", "calendar", (), "Calendar month 1..12.", bounded=True)
def f_month(df: pd.DataFrame) -> pd.Series:
    return pd.Series(df.index.month.astype(float), index=df.index)


@register(
    "turn_of_month",
    "calendar",
    (ParamSpec("bars", "int", 3, 1, 10, 1),),
    "1 within the last `bars` or first `bars` trading bars of a month. Pension money arrives then, allegedly.",
    bounded=True,
)
def f_tom(df: pd.DataFrame, bars: int) -> pd.Series:
    idx = df.index
    ym = idx.to_period("M")
    pos_in_month = pd.Series(np.arange(len(idx)), index=idx).groupby(ym).cumcount()
    size = pd.Series(ym, index=idx).map(pd.Series(ym).value_counts())
    from_end = size.to_numpy() - 1 - pos_in_month.to_numpy()
    flag = (pos_in_month.to_numpy() < bars) | (from_end < bars)
    return pd.Series(flag.astype(float), index=idx)


@register("days_to_month_end", "calendar", (), "Trading bars remaining in the month, using the bars we actually have.", bounded=True)
def f_dtme(df: pd.DataFrame) -> pd.Series:
    idx = df.index
    ym = pd.Series(idx.to_period("M"), index=idx)
    pos = pd.Series(np.arange(len(idx)), index=idx).groupby(ym.to_numpy()).cumcount()
    size = ym.map(ym.value_counts())
    return (size - 1 - pos).astype(float)

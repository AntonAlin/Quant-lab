"""Yahoo Finance download with a parquet cache and an integrity report.

Why the paranoia: yfinance is free, which means the data is exactly as good as
free data. Duplicate timestamps, zero-volume phantom bars and 40% one-day moves
that never happened all show up eventually. The report exists so the user sees
them *before* they spend an afternoon fitting a model to a bad tick.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quantlab.config import DataConfig
from quantlab.data.calendar import GapReport, detect_gaps

log = logging.getLogger(__name__)

CACHE_DIR = Path.home() / ".quantlab" / "cache"
MIN_DAILY_BARS = 750
OHLCV = ["open", "high", "low", "close", "volume"]


@dataclass
class IntegrityReport:
    rows: int
    first: pd.Timestamp | None
    last: pd.Timestamp | None
    duplicate_index: int
    zero_volume_days: int
    nan_rows: int
    non_monotonic: bool
    negative_prices: int
    ohlc_violations: int  # high < low, close outside [low, high], etc.
    price_gaps: pd.DataFrame  # returns beyond N sigma
    suspected_bad_ticks: pd.DataFrame  # spike-and-revert pattern
    gaps: GapReport
    too_few_bars: bool
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "first": self.first,
            "last": self.last,
            "duplicate_index": self.duplicate_index,
            "zero_volume_days": self.zero_volume_days,
            "nan_rows": self.nan_rows,
            "non_monotonic": self.non_monotonic,
            "negative_prices": self.negative_prices,
            "ohlc_violations": self.ohlc_violations,
            "price_gaps": len(self.price_gaps),
            "suspected_bad_ticks": len(self.suspected_bad_ticks),
            **self.gaps.as_dict(),
            "too_few_bars": self.too_few_bars,
        }


def cache_path(cfg: DataConfig, cache_dir: Path = CACHE_DIR) -> Path:
    safe = cfg.ticker.replace("^", "IDX_").replace("/", "_").replace("=", "_")
    return cache_dir / f"{safe}_{cfg.interval}_{cfg.start}_{cfg.end}.parquet"


def download(cfg: DataConfig) -> pd.DataFrame:
    """Hit Yahoo. Returns lowercase OHLCV + 'close_raw' (unadjusted) + 'adj_factor'.

    We download twice: once adjusted, once raw. Two round-trips is cheap; the
    alternative (reconstructing raw from Adj Close) is wrong around splits.
    """
    import yfinance as yf  # imported lazily so the module is importable offline

    cfg.validate()
    kwargs = dict(
        start=cfg.start,
        end=cfg.end,
        interval=cfg.interval,
        progress=False,
        threads=False,
        multi_level_index=False,
    )
    adj = yf.download(cfg.ticker, auto_adjust=True, **kwargs)
    raw = yf.download(cfg.ticker, auto_adjust=False, **kwargs)
    if adj is None or adj.empty:
        raise RuntimeError(
            f"Yahoo returned nothing for {cfg.ticker!r} ({cfg.interval}, {cfg.start}..{cfg.end}). "
            "Check the symbol (indices need a '^'), the date range, or whether Yahoo is having a day."
        )
    df = _normalise_columns(adj)
    raw_n = _normalise_columns(raw) if raw is not None and not raw.empty else df
    df["close_raw"] = raw_n["close"].reindex(df.index)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["adj_factor"] = df["close"] / df["close_raw"]
    df.attrs["ticker"] = cfg.ticker
    df.attrs["interval"] = cfg.interval
    return df


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)
    out.columns = [str(c).lower().replace(" ", "_") for c in out.columns]
    missing = [c for c in OHLCV if c not in out.columns]
    if missing:
        raise RuntimeError(f"Downloaded frame is missing columns {missing}; got {list(out.columns)}.")
    out = out[OHLCV].astype(float)
    out.index = pd.DatetimeIndex(out.index)
    if out.index.tz is not None:
        out.index = out.index.tz_convert(None)
    out.index.name = "timestamp"
    return out


def load_ohlcv(cfg: DataConfig, cache_dir: Path = CACHE_DIR) -> pd.DataFrame:
    """Cache-first loader. `cfg.force_refresh=True` bypasses and overwrites the cache."""
    cfg.validate()
    path = cache_path(cfg, cache_dir)
    if path.exists() and not cfg.force_refresh:
        df = pd.read_parquet(path)
        df.index = pd.DatetimeIndex(df.index)
        df.attrs["ticker"] = cfg.ticker
        df.attrs["interval"] = cfg.interval
        df.attrs["from_cache"] = True
        return df
    df = download(cfg)
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    df.attrs["from_cache"] = False
    return df


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal, *visible* cleaning: sort, drop exact duplicate timestamps, drop all-NaN rows.

    We do not forward-fill prices. If Yahoo gave us a hole, the hole stays and the
    integrity report says so. Silent interpolation is how fake Sharpe gets made.
    """
    out = df.copy()
    out = out[~out.index.duplicated(keep="last")]
    out = out.sort_index()
    out = out.dropna(subset=["close"], how="all")
    return out


def integrity_report(
    df: pd.DataFrame,
    interval: str = "1d",
    sigma: float = 6.0,
    min_bars: int = MIN_DAILY_BARS,
) -> IntegrityReport:
    """Everything the user should read before trusting the series."""
    for c in OHLCV:
        if c not in df.columns:
            raise ValueError(f"integrity_report needs column {c!r}.")
    idx = df.index
    dup = int(idx.duplicated().sum())
    zero_vol = int((df["volume"] <= 0).sum())
    nan_rows = int(df[OHLCV].isna().any(axis=1).sum())
    non_mono = not idx.is_monotonic_increasing
    neg = int((df[["open", "high", "low", "close"]] <= 0).sum().sum())
    ohlc_viol = int(
        (
            (df["high"] < df["low"])
            | (df["close"] > df["high"])
            | (df["close"] < df["low"])
            | (df["open"] > df["high"])
            | (df["open"] < df["low"])
        ).sum()
    )

    logret = np.log(df["close"]).diff()
    z = (logret - logret.mean()) / logret.std(ddof=1)
    gaps = df.loc[z.abs() > sigma, ["close"]].assign(log_return=logret[z.abs() > sigma], z=z[z.abs() > sigma])

    # A bad tick looks like a big move that fully reverses the next bar. A real
    # crash does not politely undo itself in 24 hours (well, usually).
    nxt = logret.shift(-1)
    revert = (z.abs() > sigma / 2) & (np.sign(logret) == -np.sign(nxt)) & ((logret + nxt).abs() < 0.25 * logret.abs())
    bad = df.loc[revert, ["close"]].assign(log_return=logret[revert], next_return=nxt[revert])

    gap_report = detect_gaps(idx, interval)
    too_few = interval == "1d" and len(df) < min_bars

    warnings: list[str] = []
    if too_few:
        warnings.append(
            f"Only {len(df)} daily bars. Walk-forward validation of ML models on fewer than "
            f"{min_bars} bars is not meaningful: you will be fitting noise and validating on the "
            "rest of the noise. Extend the date range."
        )
    if dup:
        warnings.append(f"{dup} duplicate timestamps (dropped by clean_ohlcv, but ask why they exist).")
    if zero_vol:
        warnings.append(f"{zero_vol} zero-volume bars. For an index this is normal; for a stock it is a phantom bar.")
    if ohlc_viol:
        warnings.append(f"{ohlc_viol} bars where OHLC do not make geometric sense.")
    if len(gaps):
        warnings.append(f"{len(gaps)} returns beyond {sigma} sigma. Look at them before believing them.")
    if len(bad):
        warnings.append(f"{len(bad)} spike-and-revert patterns that smell like bad ticks.")
    if gap_report.missing_business_days:
        warnings.append(
            f"{len(gap_report.missing_business_days)} business days missing in long gaps "
            f"(longest gap around {gap_report.longest_gap_start:%Y-%m-%d})."
        )

    return IntegrityReport(
        rows=len(df),
        first=idx[0] if len(idx) else None,
        last=idx[-1] if len(idx) else None,
        duplicate_index=dup,
        zero_volume_days=zero_vol,
        nan_rows=nan_rows,
        non_monotonic=non_mono,
        negative_prices=neg,
        ohlc_violations=ohlc_viol,
        price_gaps=gaps,
        suspected_bad_ticks=bad,
        gaps=gap_report,
        too_few_bars=too_few,
        warnings=warnings,
    )


def synthetic_ohlcv(n: int = 1500, seed: int = 0, start: str = "2010-01-01", drift: float = 0.0002, vol: float = 0.012) -> pd.DataFrame:
    """Geometric random walk with plausible OHLC. For tests and for the 'no internet' case."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    r = rng.normal(drift, vol, n)
    close = 100 * np.exp(np.cumsum(r))
    open_ = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0, vol / 4, n))
    hi = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, vol / 2, n)))
    lo = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, vol / 2, n)))
    vol_ = rng.lognormal(12, 0.5, n)
    df = pd.DataFrame({"open": open_, "high": hi, "low": lo, "close": close, "volume": vol_}, index=idx)
    df.index.name = "timestamp"
    df["close_raw"] = df["close"]
    df["adj_factor"] = 1.0
    df.attrs["ticker"] = "SYNTH"
    df.attrs["interval"] = "1d"
    return df

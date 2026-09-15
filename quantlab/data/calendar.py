"""Trading-day bookkeeping: which days *should* exist, which ones do not, and gap detection.

We deliberately avoid exchange-calendar libraries. They are heavy, they disagree
with Yahoo about half-days, and for an integrity report "business days missing
from the series" is a perfectly good first approximation. The user gets a list,
not a verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class GapReport:
    missing_business_days: list[pd.Timestamp] = field(default_factory=list)
    longest_gap_bars: int = 0
    longest_gap_start: pd.Timestamp | None = None
    longest_gap_end: pd.Timestamp | None = None
    median_bar_spacing: pd.Timedelta | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "missing_business_days": len(self.missing_business_days),
            "longest_gap_bars": self.longest_gap_bars,
            "longest_gap_start": self.longest_gap_start,
            "longest_gap_end": self.longest_gap_end,
            "median_bar_spacing": self.median_bar_spacing,
        }


def expected_index(start: pd.Timestamp, end: pd.Timestamp, interval: str) -> pd.DatetimeIndex:
    """Naive expectation of bar timestamps. Business days for '1d', weeks for '1wk'.

    Hourly bars depend on the exchange session and Yahoo is inconsistent, so for
    '1h' we return an empty index and let the gap detector rely on spacing only.
    """
    if interval == "1d":
        return pd.bdate_range(start.normalize(), end.normalize())
    if interval == "1wk":
        return pd.date_range(start.normalize(), end.normalize(), freq="W-MON")
    if interval == "1h":
        return pd.DatetimeIndex([])
    raise ValueError(f"Unsupported interval {interval!r}. Use '1d', '1h' or '1wk'.")


def detect_gaps(index: pd.DatetimeIndex, interval: str, tolerance_bars: int = 5) -> GapReport:
    """Find missing business days and the longest stretch with no bars.

    `tolerance_bars`: how many expected bars can be missing in a row before we
    call it a gap worth reporting (holiday clusters are normal; a fortnight is not).
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("detect_gaps expects a DatetimeIndex.")
    if len(index) < 2:
        return GapReport()
    idx = index.sort_values()
    report = GapReport()

    spacing = pd.Series(idx[1:] - idx[:-1])
    report.median_bar_spacing = spacing.median()

    # Longest gap in units of the median spacing. Weekend gaps on daily data are
    # ~3 days / 1 day = 3 "bars", which is why the tolerance exists.
    if report.median_bar_spacing is not None and report.median_bar_spacing > pd.Timedelta(0):
        ratio = (spacing / report.median_bar_spacing).to_numpy()
        worst = int(np.argmax(ratio))
        report.longest_gap_bars = int(round(ratio[worst]))
        report.longest_gap_start = idx[worst]
        report.longest_gap_end = idx[worst + 1]

    expected = expected_index(idx[0], idx[-1], interval)
    if len(expected):
        have = pd.DatetimeIndex(idx.normalize().unique())
        missing = expected.difference(have)
        # Only keep missing days that sit inside a run longer than the tolerance,
        # otherwise the report is 90% Christmas.
        if len(missing):
            runs = _consecutive_runs(missing, expected)
            keep = [d for run in runs if len(run) > tolerance_bars for d in run]
            report.missing_business_days = list(keep)
    return report


def _consecutive_runs(missing: pd.DatetimeIndex, expected: pd.DatetimeIndex) -> list[list[pd.Timestamp]]:
    """Group missing expected dates into runs of consecutive expected slots."""
    pos = expected.get_indexer(missing)
    runs: list[list[pd.Timestamp]] = []
    current: list[pd.Timestamp] = []
    prev = None
    for p, d in zip(pos, missing):
        if prev is not None and p == prev + 1:
            current.append(d)
        else:
            if current:
                runs.append(current)
            current = [d]
        prev = p
    if current:
        runs.append(current)
    return runs


def bars_per_year(interval: str) -> float:
    """Annualisation constant. Hourly assumes ~6.5 trading hours; adjust if you trade crypto."""
    table = {"1d": 252.0, "1wk": 52.0, "1h": 252.0 * 6.5}
    if interval not in table:
        raise ValueError(f"No annualisation factor for interval {interval!r}.")
    return table[interval]

"""Walk-forward splitting with embargo and label purging.

Geometry (rolling mode, one fold):

    |<---- train_window ---->|<-embargo->|<-- test_window -->|
    ^train_start        train_end        test_start     test_end

Then everything slides by `step`. Expanding mode pins train_start at 0.

The embargo is a gap *between* train and test so that a label started at the
end of train (which resolves up to h bars later) does not overlap the first
test bars. Purging additionally drops any train sample whose t1 lands at or
after test_start, which matters when the label horizon is variable (triple
barrier, trend scanning) and could exceed the embargo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd

from quantlab.config import WalkForwardConfig


@dataclass(frozen=True)
class Fold:
    fold_id: int
    train_idx: np.ndarray  # positional indices into the aligned matrix
    test_idx: np.ndarray

    @property
    def train_range(self) -> tuple[int, int]:
        return int(self.train_idx.min()), int(self.train_idx.max())

    @property
    def test_range(self) -> tuple[int, int]:
        return int(self.test_idx.min()), int(self.test_idx.max())


class WalkForwardSplitter:
    """Positional splitter. Knows nothing about dates; the caller aligns everything first."""

    def __init__(self, cfg: WalkForwardConfig, label_horizon: int) -> None:
        cfg.validate(label_horizon=label_horizon)
        self.cfg = cfg
        self.label_horizon = label_horizon

    def n_folds(self, n: int) -> int:
        return sum(1 for _ in self.split(n))

    def split(self, n: int, t1_pos: np.ndarray | None = None) -> Iterator[Fold]:
        """Yield folds over `n` aligned rows.

        `t1_pos`: optional array of label-resolution positions per row (or -1).
        If given, train rows whose label resolves at or after test_start are purged.
        """
        c = self.cfg
        if n < c.train_window + c.embargo + 1:
            raise ValueError(
                f"Only {n} usable rows but train_window + embargo = {c.train_window + c.embargo}. "
                "Shorten the windows or load more data."
            )
        fold_id = 0
        train_end = c.train_window  # exclusive
        while True:
            test_start = train_end + c.embargo
            test_end = min(test_start + c.test_window, n)  # exclusive
            if test_start >= n:
                break
            train_start = 0 if c.mode == "expanding" else train_end - c.train_window
            train_idx = np.arange(train_start, train_end)
            if t1_pos is not None:
                # Purge: labels that have not resolved by test_start know about test bars.
                keep = (t1_pos[train_idx] < test_start) & (t1_pos[train_idx] >= 0)
                train_idx = train_idx[keep]
            test_idx = np.arange(test_start, test_end)
            if len(train_idx) >= 10 and len(test_idx) > 0:
                yield Fold(fold_id, train_idx, test_idx)
                fold_id += 1
            if test_end >= n:
                break
            train_end += c.step

    def describe(self, index: pd.DatetimeIndex, t1_pos: np.ndarray | None = None) -> pd.DataFrame:
        """Human-readable fold table for the UI."""
        rows = []
        for f in self.split(len(index), t1_pos):
            a, b = f.train_range
            c, d = f.test_range
            rows.append(
                {
                    "fold": f.fold_id,
                    "train_start": index[a],
                    "train_end": index[b],
                    "n_train": len(f.train_idx),
                    "test_start": index[c],
                    "test_end": index[d],
                    "n_test": len(f.test_idx),
                }
            )
        return pd.DataFrame(rows)

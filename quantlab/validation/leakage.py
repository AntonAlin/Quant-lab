"""Loud assertions against the classic ways of accidentally peeking at the future.

These run before every walk-forward and before every backtest. They are cheap.
If one of them fires, do not "fix" it by removing the check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.validation.walkforward import Fold


class LeakageError(AssertionError):
    """Raised when something downstream would be trained on the future."""


def check_folds(folds: list[Fold], n: int, embargo: int, mode: str) -> None:
    if not folds:
        raise LeakageError("No folds were generated. Check train/test window sizes against the data length.")
    prev_test_start = -1
    for f in folds:
        if len(f.train_idx) == 0 or len(f.test_idx) == 0:
            raise LeakageError(f"Fold {f.fold_id} has an empty train or test set.")
        if f.train_idx.max() >= f.test_idx.min():
            raise LeakageError(f"Fold {f.fold_id}: a train index ({f.train_idx.max()}) is not before the test start ({f.test_idx.min()}).")
        gap = f.test_idx.min() - f.train_idx.max() - 1
        if gap < embargo:
            raise LeakageError(f"Fold {f.fold_id}: embargo is {gap} bars, config says {embargo}.")
        if np.intersect1d(f.train_idx, f.test_idx).size:
            raise LeakageError(f"Fold {f.fold_id}: train and test overlap.")
        if f.test_idx.min() <= prev_test_start:
            raise LeakageError(f"Fold {f.fold_id}: test windows are not chronological.")
        if f.test_idx.max() >= n:
            raise LeakageError(f"Fold {f.fold_id}: test index runs past the data ({n}).")
        if mode == "expanding" and f.train_idx.min() != 0:
            raise LeakageError(f"Fold {f.fold_id}: expanding mode must anchor at 0, got {f.train_idx.min()}.")
        prev_test_start = f.test_idx.min()


def check_feature_matrix(X: pd.DataFrame, source_index: pd.DatetimeIndex, min_shift: int = 1) -> None:
    """The feature frame must be the source index, shifted by at least one bar."""
    shift = X.attrs.get("shift")
    if shift is None:
        raise LeakageError("Feature matrix has no 'shift' attribute. Build it with build_feature_matrix().")
    if shift < min_shift:
        raise LeakageError(f"Feature matrix shift is {shift}; must be >= {min_shift}.")
    if not X.index.equals(source_index):
        raise LeakageError("Feature matrix index differs from the price index. Alignment has been lost.")
    if not X.index.is_monotonic_increasing:
        raise LeakageError("Feature matrix index is not sorted.")
    if X.index.has_duplicates:
        raise LeakageError("Feature matrix index has duplicates.")
    expected = int(pd.util.hash_pandas_object(source_index).sum())
    if X.attrs.get("source_index_hash") not in (None, expected):
        raise LeakageError("Feature matrix was built on a different price index than the one you are using now.")


def check_alignment(X: pd.DataFrame, y: pd.Series, extra: dict[str, pd.Series] | None = None) -> None:
    """X, y (and weights/t1/etc) must share an identical index, no dropna surprises."""
    if not X.index.equals(y.index):
        raise LeakageError(
            f"X and y indices differ (X: {len(X)}, y: {len(y)}). Somebody called dropna() on one of them."
        )
    for name, s in (extra or {}).items():
        if not s.index.equals(X.index):
            raise LeakageError(f"'{name}' index differs from X. Alignment was lost.")


def check_labels_not_in_features(X: pd.DataFrame, labels: pd.DataFrame, close: pd.Series) -> None:
    """A feature perfectly correlated with the forward return is a leak, full stop."""
    if "ret" not in labels:
        return
    fwd = labels["ret"]
    for col in X.columns:
        both = pd.concat([X[col], fwd], axis=1).dropna()
        if len(both) < 50:
            continue
        c = both.corr().iloc[0, 1]
        if np.isfinite(c) and abs(c) > 0.98:
            raise LeakageError(f"Feature {col!r} has |corr|={c:.3f} with the forward return. That is the label wearing a hat.")


def future_spike_probe(feature_fn, df: pd.DataFrame, spike_at: int, magnitude: float = 5.0) -> bool:
    """Utility for tests: does a feature react *before* a future spike? Returns True if it leaks."""
    base = feature_fn(df)
    spiked = df.copy()
    spiked.iloc[spike_at:, spiked.columns.get_indexer(["open", "high", "low", "close"])] *= magnitude
    after = feature_fn(spiked)
    if isinstance(base, pd.Series):
        base, after = base.to_frame(), after.to_frame()
    before_idx = base.index[:spike_at]
    a, b = base.loc[before_idx].fillna(-999.0), after.loc[before_idx].fillna(-999.0)
    return not np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True)

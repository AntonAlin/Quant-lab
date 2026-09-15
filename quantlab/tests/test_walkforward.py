import numpy as np
import pytest

from quantlab.config import WalkForwardConfig
from quantlab.validation.leakage import LeakageError, check_folds
from quantlab.validation.walkforward import WalkForwardSplitter


@pytest.mark.parametrize("mode", ["rolling", "expanding"])
def test_geometry(mode: str) -> None:
    cfg = WalkForwardConfig(mode=mode, train_window=200, test_window=50, step=50, embargo=5)
    sp = WalkForwardSplitter(cfg, label_horizon=5)
    folds = list(sp.split(1000))
    assert len(folds) == 16  # test starts 205, 255, ..., 955; the last fold is 45 bars
    check_folds(folds, 1000, embargo=5, mode=mode)
    prev_end = -1
    for f in folds:
        assert f.train_idx.max() < f.test_idx.min()
        assert f.test_idx.min() - f.train_idx.max() - 1 == 5
        assert len(np.intersect1d(f.train_idx, f.test_idx)) == 0
        assert f.test_idx.min() > prev_end
        prev_end = f.test_idx.max()
        if mode == "rolling":
            assert len(f.train_idx) == 200
        else:
            assert f.train_idx.min() == 0
    # Test windows tile the OOS region with no gaps and no overlap.
    all_test = np.concatenate([f.test_idx for f in folds])
    assert len(all_test) == len(np.unique(all_test))
    assert all_test.min() == 205 and all_test.max() == 999


def test_purging_drops_unresolved_labels() -> None:
    cfg = WalkForwardConfig(train_window=100, test_window=20, step=20, embargo=2, i_know_what_i_am_doing=True)
    sp = WalkForwardSplitter(cfg, label_horizon=2)
    n = 300
    t1 = np.arange(n) + 10  # every label resolves 10 bars later, longer than the embargo
    f = next(sp.split(n, t1_pos=t1))
    assert f.train_idx.max() < f.test_idx.min() - 10 + 1
    assert all(t1[f.train_idx] < f.test_idx.min())


def test_embargo_below_horizon_is_refused() -> None:
    with pytest.raises(ValueError, match="embargo"):
        WalkForwardSplitter(WalkForwardConfig(embargo=2), label_horizon=5)
    WalkForwardSplitter(WalkForwardConfig(embargo=2, i_know_what_i_am_doing=True), label_horizon=5)


def test_check_folds_catches_overlap() -> None:
    from quantlab.validation.walkforward import Fold

    bad = [Fold(0, np.arange(0, 100), np.arange(95, 120))]
    with pytest.raises(LeakageError):
        check_folds(bad, 200, embargo=0, mode="rolling")
    bad2 = [Fold(0, np.arange(0, 100), np.arange(102, 120))]
    with pytest.raises(LeakageError, match="embargo"):
        check_folds(bad2, 200, embargo=5, mode="rolling")


def test_too_short_series_raises() -> None:
    sp = WalkForwardSplitter(WalkForwardConfig(train_window=500, embargo=5), label_horizon=5)
    with pytest.raises(ValueError, match="usable rows"):
        list(sp.split(300))

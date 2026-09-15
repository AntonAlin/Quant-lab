import numpy as np
import pandas as pd
import pytest

from quantlab.labeling.fixed_horizon import fixed_horizon_labels
from quantlab.labeling.trend_scanning import trend_scanning_labels
from quantlab.labeling.triple_barrier import triple_barrier_labels
from quantlab.labeling.weights import concurrency, uniqueness_weights


def _toy() -> pd.DataFrame:
    # high-low = 2 every bar. Wilder ATR (window 2, alpha 1/2) is 2 until the jump at
    # bar 5 widens the true range: ATR = [2,2,2,2,2,3,3.5,2.75,...]. Expectations below
    # use those numbers, not a flat 2.
    idx = pd.bdate_range("2020-01-01", periods=12)
    close = np.array([100, 100, 100, 100, 100, 103, 100, 100, 100, 94, 100, 100], dtype=float)
    return pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1.0}, index=idx)


def test_triple_barrier_hand_computed() -> None:
    df = _toy()
    # atr_window=2 so the ATR is defined from bar 2 onwards and equals 2.
    lab = triple_barrier_labels(df, horizon=4, pt_mult=1.0, sl_mult=1.5, atr_window=2)
    # Entry at bar 2 (close 100): pt at 102, sl at 97. Bar 5 high = 104 >= 102 -> pt at bar 5.
    assert lab.loc[df.index[2], "barrier"] == "pt"
    assert lab.loc[df.index[2], "label"] == 1
    assert lab.loc[df.index[2], "t1"] == df.index[5]
    assert lab.loc[df.index[2], "ret"] == pytest.approx(0.02)
    # Entry at bar 6 (close 100, ATR 3.5): sl at 94.75. Bar 9 low = 93 -> sl at bar 9.
    assert lab.loc[df.index[6], "barrier"] == "sl"
    assert lab.loc[df.index[6], "label"] == -1
    assert lab.loc[df.index[6], "ret"] == pytest.approx(-0.0525)
    assert lab.loc[df.index[6], "t1"] == df.index[9]
    # Entry at bar 3: window 4..7 hits bar 5's pt as well.
    assert lab.loc[df.index[3], "barrier"] == "pt"
    # Entry at bar 7 (ATR 2.75): sl at 95.875 -> bar 9 sl.
    assert lab.loc[df.index[7], "barrier"] == "sl"
    assert lab.loc[df.index[7], "ret"] == pytest.approx(-0.04125)
    # Entries whose horizon runs off the end get no label at all.
    assert lab["label"].iloc[-4:].isna().all()


def test_triple_barrier_short_side_flips() -> None:
    df = _toy()
    side = pd.Series(-1.0, index=df.index)
    lab = triple_barrier_labels(df, horizon=4, pt_mult=1.0, sl_mult=1.5, atr_window=2, side=side)
    # Short from bar 6 (ATR 3.5): profit-take at 96.5; bar 9 low 93 touches it.
    assert lab.loc[df.index[6], "barrier"] == "pt"
    assert lab.loc[df.index[6], "ret"] == pytest.approx(0.035)


def test_fixed_horizon_three_class() -> None:
    idx = pd.bdate_range("2020-01-01", periods=6)
    close = pd.Series([100, 101, 100.05, 99, 100, 100], index=idx, dtype=float)
    lab = fixed_horizon_labels(close, horizon=1, deadband=0.001, three_class=True)
    assert lab["label"].tolist()[:5] == [1, -1, -1, 1, 0]
    assert np.isnan(lab["label"].iloc[-1])
    assert lab["t1"].iloc[0] == idx[1]


def test_trend_scanning_labels_slope_sign() -> None:
    idx = pd.bdate_range("2020-01-01", periods=60)
    up = pd.Series(np.linspace(100, 130, 60) + np.sin(np.arange(60)) * 0.1, index=idx)
    lab = trend_scanning_labels(up, min_window=5, max_window=15)
    assert (lab["label"].dropna() == 1).all()
    down = pd.Series(np.linspace(130, 100, 60), index=idx)
    lab = trend_scanning_labels(down, min_window=5, max_window=15)
    assert (lab["label"].dropna() == -1).all()


def test_uniqueness_weights_hand_computed() -> None:
    idx = pd.bdate_range("2020-01-01", periods=6)
    # Two labels: bar0 spans 0..2, bar1 spans 1..3. Concurrency: [1,2,2,1,0,0].
    t1 = pd.Series([idx[2], idx[3]], index=idx[:2])
    c = concurrency(idx, t1)
    assert c.tolist() == [1, 2, 2, 1, 0, 0]
    w = uniqueness_weights(idx, t1)
    raw0 = np.mean([1 / 1, 1 / 2, 1 / 2])
    raw1 = np.mean([1 / 2, 1 / 2, 1 / 1])
    assert w.iloc[0] == pytest.approx(raw0 / np.mean([raw0, raw1]))
    assert w.mean() == pytest.approx(1.0)

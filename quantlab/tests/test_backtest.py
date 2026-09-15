import numpy as np
import pandas as pd
import pytest

from quantlab.backtest.engine import run_backtest
from quantlab.backtest.metrics import max_drawdown, performance_table, sharpe
from quantlab.config import BacktestConfig, CostConfig, SizingConfig


def _prices() -> pd.DataFrame:
    idx = pd.bdate_range("2021-01-01", periods=8)
    open_ = np.array([100, 102, 101, 105, 104, 108, 107, 110], dtype=float)
    close = open_ + 0.5
    return pd.DataFrame({"open": open_, "high": close + 1, "low": open_ - 1, "close": close, "volume": 1.0}, index=idx)


def test_hand_computed_pnl_with_costs() -> None:
    px = _prices()
    # Signals: long at bars 0,1; flat at 2; short at 3,4; flat after.
    p_long = pd.Series([0.9, 0.9, 0.5, 0.1, 0.1, 0.5, 0.5, 0.5], index=px.index)
    sig = pd.DataFrame({"p_long": p_long, "p_short": 1 - p_long})
    costs = CostConfig(commission_bps=10, spread_bps=20, slippage_bps=0)  # 20 bps per side
    cfg = BacktestConfig(long_threshold=0.6, short_threshold=0.6, execution="next_open", cooldown=0, max_holding=0, costs=costs, sizing=SizingConfig(scheme="fixed_fractional", fixed_fraction=1.0))
    res = run_backtest(px, sig, cfg)
    pos = res.positions.tolist()
    assert pos == [0, 1, 1, 0, -1, -1, 0, 0]
    o = px["open"].to_numpy()
    r = o[1:] / o[:-1] - 1  # open-to-open return earned by a position held during bar t
    c = 20 / 1e4
    expected = [
        0.0,  # bar 0: no position
        1 * r[1] - c,  # enter long at open 1, pay one side
        1 * r[2],  # hold
        0 * r[3] - c,  # exit at open 3
        -1 * r[4] - c,  # enter short at open 4
        -1 * r[5],  # hold
        -c,  # exit at open 6
        0.0,
    ]
    assert np.allclose(res.returns.to_numpy(), expected)
    assert len(res.trades) == 2
    assert res.trades["side"].tolist() == ["long", "short"]
    assert res.trades["bars_held"].tolist() == [2, 2]
    assert res.trades["pnl"].iloc[0] == pytest.approx((1 + expected[1]) * (1 + expected[2]) - 1)
    assert res.costs.sum() == pytest.approx(4 * c)


def test_close_to_close_and_long_only() -> None:
    px = _prices()
    p_long = pd.Series([0.9, 0.1, 0.9, 0.9, 0.1, 0.1, 0.5, 0.5], index=px.index)
    sig = pd.DataFrame({"p_long": p_long, "p_short": 1 - p_long})
    cfg = BacktestConfig(direction="long_only", execution="close_to_close", costs=CostConfig(0, 0, 0, allow_zero_cost=True))
    res = run_backtest(px, sig, cfg)
    assert res.positions.tolist() == [0, 1, 0, 1, 1, 0, 0, 0]
    cl = px["close"].to_numpy()
    assert res.returns.iloc[1] == pytest.approx(cl[1] / cl[0] - 1)
    assert (res.positions >= 0).all()


def test_max_holding_and_cooldown() -> None:
    px = _prices()
    p_long = pd.Series(0.9, index=px.index)
    sig = pd.DataFrame({"p_long": p_long, "p_short": 1 - p_long})
    cfg = BacktestConfig(max_holding=2, cooldown=1, costs=CostConfig(1, 0, 0))
    res = run_backtest(px, sig, cfg)
    # target: enter 0, hold 1, exit at 2 (max_holding), cooldown at 3, enter 4, hold 5, exit 6, cooldown 7
    assert res.targets.tolist() == [1, 1, 0, 0, 1, 1, 0, 0]
    assert (res.trades["reason"] == "max_holding").all()


def test_zero_cost_needs_explicit_consent() -> None:
    with pytest.raises(ValueError, match="fiction"):
        CostConfig(0, 0, 0).validate()


def test_sharpe_and_drawdown_hand_computed() -> None:
    r = pd.Series([0.01, -0.02, 0.03, 0.0, -0.01])
    expected = r.mean() / r.std(ddof=1) * np.sqrt(252)
    assert sharpe(r, 252) == pytest.approx(expected)
    eq = pd.Series([1.0, 1.1, 0.99, 1.05, 0.9, 1.2], index=pd.bdate_range("2021-01-01", periods=6))
    mdd, dur, peak, trough = max_drawdown(eq)
    assert mdd == pytest.approx(0.9 / 1.1 - 1)
    assert dur == 3  # bars 2,3,4 are below the 1.1 peak
    assert trough == eq.index[4] and peak == eq.index[1]


def test_performance_table_keys() -> None:
    idx = pd.bdate_range("2021-01-01", periods=300)
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0005, 0.01, 300), index=idx)
    b = pd.Series(rng.normal(0.0003, 0.01, 300), index=idx)
    pos = pd.Series(np.sign(rng.normal(size=300)), index=idx)
    trades = pd.DataFrame({"pnl": [0.01, -0.005, 0.02], "bars_held": [3, 2, 5]})
    t = performance_table(r, b, pos, trades, 252)
    for k in ("cagr", "sharpe", "sortino", "calmar", "max_drawdown", "hit_rate", "profit_factor", "turnover_annual", "avg_exposure", "tail_ratio", "worst_month", "alpha_annual", "beta"):
        assert k in t
    assert t["hit_rate"] == pytest.approx(2 / 3)
    assert t["profit_factor"] == pytest.approx(0.03 / 0.005)

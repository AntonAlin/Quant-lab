"""Tab 5: costs, sizing, thresholds, equity, drawdown, trades, rolling Sharpe, fold bars, and the honesty box."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from quantlab.backtest.metrics import rolling_sharpe
from quantlab.data.calendar import bars_per_year
from quantlab.pipeline import evaluate, experiment_record
from quantlab.ui import plots
from quantlab.ui.state import cfg, eval_key, grey_note, invalidate_from, red_box, store

METRIC_FMT = {
    "cagr": "{:.2%}", "ann_vol": "{:.2%}", "sharpe": "{:.2f}", "sortino": "{:.2f}", "calmar": "{:.2f}", "max_drawdown": "{:.2%}",
    "drawdown_duration_bars": "{:.0f}", "hit_rate": "{:.1%}", "avg_win": "{:.2%}", "avg_loss": "{:.2%}", "profit_factor": "{:.2f}",
    "expectancy": "{:.3%}", "turnover_annual": "{:.1f}", "avg_exposure": "{:.1%}", "n_trades": "{:.0f}", "tail_ratio": "{:.2f}",
    "worst_month": "{:.2%}", "alpha_annual": "{:.2%}", "beta": "{:.2f}", "total_return": "{:.1%}", "bench_total_return": "{:.1%}",
    "bench_sharpe": "{:.2f}", "bench_max_drawdown": "{:.2%}", "n_bars": "{:.0f}",
}


def render() -> None:
    wf = st.session_state.get("wf")
    ds = st.session_state.get("dataset")
    if wf is None or ds is None:
        st.info("Train a walk-forward model first (Model tab).")
        return
    c = cfg()
    B = c.backtest
    before = str(B.__dict__)

    st.subheader("Execution and thresholds")
    t1, t2, t3, t4 = st.columns(4)
    B.execution = t1.selectbox("Execution", ["next_open", "close_to_close"], index=["next_open", "close_to_close"].index(B.execution), help="Signal at close t -> fill at open t+1 (default) or at close t (optimistic).")
    B.direction = t2.selectbox("Direction", ["long_short", "long_only", "short_only"], index=["long_short", "long_only", "short_only"].index(B.direction))
    B.long_threshold = float(t3.slider("Long threshold P(long) >", 0.3, 0.95, float(B.long_threshold), 0.01))
    B.short_threshold = float(t4.slider("Short threshold P(short) >", 0.3, 0.95, float(B.short_threshold), 0.01))
    if wf.is_meta:
        st.caption("Meta-labeling: thresholds apply to P(take the trade) on the primary's side.")
    h1, h2 = st.columns(2)
    B.cooldown = int(h1.number_input("Cooldown after exit (bars)", 0, 200, B.cooldown))
    B.max_holding = int(h2.number_input("Max holding period (bars, 0 = none)", 0, 1000, B.max_holding))

    st.subheader("Costs (per side)")
    k1, k2, k3, k4 = st.columns(4)
    B.costs.commission_bps = float(k1.number_input("Commission (bps)", 0.0, 100.0, float(B.costs.commission_bps), 0.5))
    B.costs.spread_bps = float(k2.number_input("Spread (bps, full)", 0.0, 200.0, float(B.costs.spread_bps), 0.5))
    B.costs.slippage_bps = float(k3.number_input("Slippage (bps)", 0.0, 100.0, float(B.costs.slippage_bps), 0.5))
    B.costs.slippage_atr_frac = float(k4.number_input("Slippage (fraction of ATR)", 0.0, 1.0, float(B.costs.slippage_atr_frac), 0.01))
    total = B.costs.commission_bps + B.costs.spread_bps + B.costs.slippage_bps + B.costs.slippage_atr_frac
    if total == 0:
        B.costs.allow_zero_cost = st.checkbox("I understand that a zero-cost backtest is fiction and want it anyway", B.costs.allow_zero_cost)
    else:
        B.costs.allow_zero_cost = False

    st.subheader("Position sizing")
    S = B.sizing
    S.scheme = st.selectbox("Scheme", ["fixed_fractional", "vol_target", "kelly", "confidence"], index=["fixed_fractional", "vol_target", "kelly", "confidence"].index(S.scheme))
    s1, s2, s3 = st.columns(3)
    if S.scheme == "fixed_fractional":
        S.fixed_fraction = float(s1.number_input("Fraction of equity", 0.05, 10.0, float(S.fixed_fraction), 0.05))
    elif S.scheme == "vol_target":
        S.target_vol = float(s1.number_input("Target annual vol", 0.01, 1.0, float(S.target_vol), 0.01))
        S.vol_window = int(s2.number_input("Vol window (bars)", 5, 500, S.vol_window))
        S.leverage_cap = float(s3.number_input("Leverage cap", 0.1, 10.0, float(S.leverage_cap), 0.1))
    elif S.scheme == "kelly":
        S.kelly_max = float(s1.number_input("Max Kelly fraction", 0.05, 1.0, float(S.kelly_max), 0.05))
        S.kelly_window = int(s2.number_input("Kelly window (bars)", 10, 1000, S.kelly_window))
    else:
        S.confidence_floor = float(s1.number_input("Zero-size probability floor", 0.3, 0.9, float(S.confidence_floor), 0.01))
        S.leverage_cap = float(s2.number_input("Size at P=1", 0.1, 10.0, float(S.leverage_cap), 0.1))

    if str(B.__dict__) != before:
        invalidate_from("eval")
    try:
        B.validate()
    except ValueError as e:
        st.error(str(e))
        return

    st_store = store()
    ticker, interval = c.data.ticker, c.data.interval
    n_trials_all = st_store.n_trials(ticker, interval)
    n_trials_sess = st_store.n_trials(ticker, interval, st.session_state["session_id"])
    scope = st.radio("Trial count for the deflated Sharpe", [f"all logged runs on {ticker} ({n_trials_all})", f"this session only ({n_trials_sess})"], horizontal=True)
    sess_only = scope.startswith("this session")
    sid = st.session_state["session_id"] if sess_only else None
    is_new = (st_store.trials(ticker, interval, sid)["config_hash"] == c.config_hash()).sum() == 0 if n_trials_all else True
    n_trials = st_store.n_trials(ticker, interval, sid) + (1 if is_new else 0)

    if st.session_state.get("eval") is None or st.session_state.get("eval_key") != eval_key():
        with st.spinner("Backtesting OOS signals and running 1000 random strategies..."):
            ev = evaluate(ds, wf, c, n_trials, st_store.trial_sharpes(ticker, interval, sid), n_random=1000)
        st.session_state["eval"] = ev
        st.session_state["eval_key"] = eval_key()
        st.session_state["logged_run_id"] = None
    ev = st.session_state["eval"]
    bt = ev.backtest
    bpy = bars_per_year(ds.interval)

    # Logging is not optional. Every evaluated config gets written once.
    if st.session_state.get("logged_run_id") is None:
        rec = experiment_record(c, ds, wf, ev, st.session_state["session_id"])
        try:
            st.session_state["logged_run_id"] = st_store.log_run(rec, bt.returns)
        except ValueError as e:
            st.warning(f"Not logged: {e}")
    st.caption(f"Logged as run {st.session_state.get('logged_run_id')} (append-only, {st_store.runs_path}).")

    # ---- Honesty box first. The equity curve is not allowed to be the last word, so it is not the first either.
    st.subheader("Is this real?")
    dsr = ev.dsr
    a, b_, c_ = st.columns(3)
    a.metric("Sharpe (OOS, annualised)", f"{ev.metrics['sharpe']:.2f}", help="Raw. Net of costs. Computed on the concatenated out-of-sample series only.")
    b_.metric("Deflated Sharpe (DSR)", f"{dsr.dsr:.2f}" if np.isfinite(dsr.dsr) else "n/a", help=f"P(true SR > expected max SR of {dsr.n_trials} trials). Above 0.95 counts as significant.")
    c_.metric("Beats random strategies", f"{ev.random_bench.percentile:.0%}" if np.isfinite(ev.random_bench.percentile) else "n/a", help="Percentile among 1000 random strategies with identical turnover and exposure.")
    if not dsr.significant:
        red_box(dsr.verdict())
    else:
        st.success(dsr.verdict())
    if np.isfinite(ev.random_bench.percentile) and ev.random_bench.percentile < 0.95:
        red_box(ev.random_bench.verdict())
    st.plotly_chart(plots.random_benchmark_hist(ev.random_bench.random_sharpes, ev.random_bench.strategy_sharpe), use_container_width=True)

    st.subheader("Out-of-sample performance vs buy & hold")
    st.plotly_chart(plots.equity_curves(bt.equity, bt.benchmark_equity, folds=wf.fold_table), use_container_width=True)
    m = ev.metrics
    left = {k: m[k] for k in ("cagr", "ann_vol", "sharpe", "sortino", "calmar", "max_drawdown", "drawdown_duration_bars", "worst_month", "tail_ratio", "alpha_annual", "beta", "total_return")}
    right = {k: m[k] for k in ("hit_rate", "avg_win", "avg_loss", "profit_factor", "expectancy", "n_trades", "turnover_annual", "avg_exposure", "bench_total_return", "bench_sharpe", "bench_max_drawdown", "n_bars")}
    c1, c2 = st.columns(2)
    c1.dataframe(_fmt(left), use_container_width=True)
    c2.dataframe(_fmt(right), use_container_width=True)
    if m["turnover_annual"] > 50:
        st.warning(f"Annual turnover of {m['turnover_annual']:.0f}x. At {total:.1f} bps per side that is {m['turnover_annual'] * total / 1e4:.1%} of equity per year in costs. Check the gross vs net gap below.")
    gross_sharpe = float(bt.gross_returns.mean() / bt.gross_returns.std(ddof=1) * np.sqrt(bpy)) if bt.gross_returns.std(ddof=1) > 0 else np.nan
    st.caption(f"Gross Sharpe (before costs) {gross_sharpe:.2f} vs net {m['sharpe']:.2f}. Total costs paid: {bt.costs.sum():.1%} of equity over the OOS period.")

    with st.expander("In-sample fit (greyed: not evidence of anything)"):
        grey_note("These numbers are on training folds. They are shown so you can see the in-sample/out-of-sample gap, not because they mean something.")
        ins = pd.DataFrame({"fold": [f.fold_id for f in wf.folds], "train_accuracy": [f.train_accuracy for f in wf.folds], "test_accuracy": [f.test_accuracy for f in wf.folds]})
        st.dataframe(ins.style.format({"train_accuracy": "{:.3f}", "test_accuracy": "{:.3f}"}).set_properties(color="#8a8f98"), use_container_width=True, hide_index=True)

    c3, c4 = st.columns(2)
    with c3:
        win = st.slider("Rolling Sharpe window (bars)", 20, 500, 126)
        st.plotly_chart(plots.rolling_sharpe_plot(rolling_sharpe(bt.returns, win, bpy), win), use_container_width=True)
    with c4:
        st.plotly_chart(plots.fold_bars(ev.fold_metrics), use_container_width=True)
        pos_folds = int((ev.fold_metrics["strategy_return"] > 0).sum())
        st.caption(f"{pos_folds}/{len(ev.fold_metrics)} folds positive. A strategy that wins in one fold and loses in nine is a story about that one fold.")
    st.plotly_chart(plots.positions_plot(bt.positions, ds.prices["close"].reindex(bt.positions.index)), use_container_width=True)

    st.subheader("Trades")
    tr = bt.trades
    if len(tr):
        st.dataframe(tr.style.format({"avg_size": "{:.2f}", "pnl": "{:.2%}"}), use_container_width=True, hide_index=True)
        st.caption(f"Exit reasons: {tr['reason'].value_counts().to_dict()}. Median holding {tr['bars_held'].median():.0f} bars.")
    else:
        st.warning("No trades. Thresholds too strict, or the model never crosses them.")


def _fmt(d: dict[str, float]) -> pd.DataFrame:
    rows = {k: (METRIC_FMT.get(k, "{:.3f}").format(v) if isinstance(v, (int, float)) and np.isfinite(v) else str(v)) for k, v in d.items()}
    return pd.Series(rows, name="value").to_frame()

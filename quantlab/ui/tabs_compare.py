"""Tab 7: leaderboard over all logged runs, PBO across them, parallel coordinates."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from quantlab.config import ExperimentConfig
from quantlab.experiments.compare import leaderboard, parallel_coordinates_frame, pbo_across_runs
from quantlab.ui import plots
from quantlab.ui.state import invalidate_from, red_box, store


def render() -> None:
    s = store()
    runs = s.load_runs()
    if runs.empty:
        st.info(f"No runs logged yet. Every evaluated backtest is appended to {s.runs_path}.")
        return
    f1, f2, f3 = st.columns(3)
    tickers = sorted(runs["ticker"].unique())
    ticker = f1.selectbox("Ticker", ["(all)"] + tickers)
    models = sorted(runs["model_family"].unique())
    model_filter = f2.multiselect("Model family", models, default=models)
    min_trades = f3.number_input("Min trades", 0, 10000, 10)
    lb = leaderboard(s, None if ticker == "(all)" else ticker)
    lb = lb[lb["model_family"].isin(model_filter) & (lb["m_n_trades"] >= min_trades)] if "m_n_trades" in lb else lb
    if lb.empty:
        st.warning("Nothing matches the filters.")
        return
    st.caption(f"{len(lb)} runs, {lb['n_trials_now'].iloc[0]} distinct configs counted as trials. dsr_now is recomputed against today's trial count, so old runs lose their early significance as you keep trying things. That is the point.")
    st.dataframe(
        lb.style.format({c: "{:.3f}" for c in lb.columns if c.startswith("m_") or c in ("dsr_now", "oos_accuracy", "oos_logloss", "random_percentile")}, na_rep="-"),
        use_container_width=True,
        hide_index=True,
    )
    sig = lb[lb["dsr_now"] >= 0.95]
    if sig.empty:
        red_box("No logged run survives the deflated Sharpe test at the current trial count. Keep that in mind before choosing the top row.")
    else:
        st.success(f"{len(sig)} run(s) with DSR >= 0.95 at the current trial count.")

    st.subheader("Probability of backtest overfitting across logged runs")
    st.caption("CSCV over the OOS return series of every run in the table. Answers: if you pick the best config by in-sample Sharpe, how often does it underperform the median out of sample?")
    n_part = st.select_slider("Partitions", [4, 6, 8, 10, 12, 16], value=10)
    same_ticker = lb["ticker"].nunique() == 1
    if not same_ticker:
        st.warning("PBO across different tickers is meaningless; filter to one ticker.")
    elif len(lb) < 2:
        st.info("Need at least two runs on the same ticker.")
    else:
        pbo = pbo_across_runs(s, lb["run_id"].tolist(), n_partitions=int(n_part))
        if np.isfinite(pbo.pbo):
            if pbo.pbo > 0.5:
                red_box(pbo.verdict())
            else:
                st.info(pbo.verdict())
            c1, c2 = st.columns(2)
            c1.plotly_chart(plots.pbo_hist(pbo.logits), use_container_width=True)
            pairs = pd.DataFrame(pbo.is_oos_sharpe_pairs, columns=["is_sharpe", "oos_sharpe"])
            c2.plotly_chart(plots.scatter(pairs["is_sharpe"], pairs["oos_sharpe"], "IS-best config: in-sample vs out-of-sample Sharpe (per bar)", "IS Sharpe", "OOS Sharpe"), use_container_width=True)
            st.caption(f"{pbo.n_combinations} train/test combinations over {pbo.n_configs} configs on {len(s.returns_matrix(lb['run_id'].tolist()).dropna())} shared OOS bars.")
        else:
            st.info(pbo.verdict())

    st.subheader("Parameters vs OOS Sharpe")
    st.plotly_chart(plots.parallel_coords(parallel_coordinates_frame(lb)), use_container_width=True)

    st.subheader("Reload a run's config")
    rid = st.selectbox("Run", lb["run_id"].tolist())
    if st.button("Load this config into the session"):
        row = runs[runs["run_id"] == rid].iloc[0]
        st.session_state["cfg"] = ExperimentConfig.from_json(row["config_json"])
        invalidate_from("dataset")
        st.success("Config loaded. Load the data from the sidebar if the ticker or range differ, then retrain.")
    with st.expander("Raw config JSON"):
        st.code(runs[runs["run_id"] == rid].iloc[0]["config_json"], language="json")

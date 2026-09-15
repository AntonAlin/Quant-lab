"""Tab 6: importance (permutation on OOS, SHAP for trees), confusion, calibration, probability by class, regimes."""

from __future__ import annotations

import numpy as np
import streamlit as st

from quantlab.data.calendar import bars_per_year
from quantlab.diagnostics import calibration, confusion, per_regime_performance, permutation_importance_oos, probability_by_class, shap_last_fold
from quantlab.ui import plots
from quantlab.ui.state import cfg


def render() -> None:
    wf = st.session_state.get("wf")
    ds = st.session_state.get("dataset")
    ev = st.session_state.get("eval")
    if wf is None or ds is None:
        st.info("Train a walk-forward model first.")
        return
    c = cfg()
    st.subheader("Feature importance")
    c1, c2 = st.columns(2)
    with c1:
        n_rep = st.number_input("Permutation repeats", 1, 20, 3)
        max_folds = st.number_input("Folds to use (last N)", 1, max(1, len(wf.folds)), min(5, len(wf.folds)))
        if st.button("Compute permutation importance (OOS)"):
            with st.spinner("Shuffling columns and re-scoring every fold..."):
                st.session_state["perm_imp"] = permutation_importance_oos(ds, wf, n_repeats=int(n_rep), seed=c.seed, max_folds=int(max_folds))
        pi = st.session_state.get("perm_imp")
        if pi is not None:
            st.plotly_chart(plots.bar(pi.set_index("feature")["importance"].iloc[::-1], "Permutation importance: OOS log-loss increase when shuffled", plots.GREEN), use_container_width=True)
            neg = pi[pi["importance"] < 0]["feature"].tolist()
            if neg:
                st.caption(f"Shuffling these *improved* OOS log-loss, i.e. they are noise the model latched on to: {', '.join(neg[:8])}")
    with c2:
        if st.button("Compute SHAP (last fold, OOS)"):
            try:
                with st.spinner("Explaining the last fold..."):
                    st.session_state["shap_blocks"] = shap_last_fold(ds, wf)
            except Exception as e:  # noqa: BLE001 - shap is fragile across versions; degrade, do not crash the tab
                st.session_state["shap_blocks"] = []
                st.error(f"SHAP failed: {e}")
        blocks = st.session_state.get("shap_blocks")
        if blocks:
            names = [b.component for b in blocks]
            pick = st.selectbox("Component", names) if len(names) > 1 else names[0]
            blk = blocks[names.index(pick)]
            st.plotly_chart(plots.shap_beeswarm(blk.values, blk.X, scale=blk.scale), use_container_width=True)
            st.caption(f"SHAP values are on the {blk.scale} scale. Sequence models: each value is the sum over the lookback window.")
        elif blocks is not None:
            st.caption("Nothing explainable in this model.")

    st.subheader("Classification quality (OOS)")
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(plots.confusion_heatmap(confusion(wf)), use_container_width=True)
        o = wf.oos.dropna(subset=["y_true", "y_pred"])
        maj = o["y_true"].value_counts(normalize=True).max()
        acc = (o["y_true"] == o["y_pred"]).mean()
        st.caption(f"OOS accuracy {acc:.3f} vs majority-class baseline {maj:.3f}. If those are close, the model is a coin with a preferred side.")
    with c4:
        st.plotly_chart(plots.calibration_plot(calibration(wf)), use_container_width=True)
        st.caption("Below the diagonal: overconfident. Thresholds and confidence sizing assume the probabilities mean something.")
    st.plotly_chart(plots.prob_by_class(probability_by_class(wf)), use_container_width=True)
    o = wf.oos.dropna(subset=["p_long"])
    st.caption(f"P(long) spread: std {o['p_long'].std():.3f}, 5th-95th percentile {o['p_long'].quantile(0.05):.2f}-{o['p_long'].quantile(0.95):.2f}. If the whole distribution sits inside 0.45-0.55 your thresholds decide everything.")

    st.subheader("Per-regime performance")
    if ev is None:
        st.info("Open the Backtest tab once to evaluate, then come back.")
        return
    bt = ev.backtest
    reg = per_regime_performance(ds, bt.returns, bt.benchmark_returns, bars_per_year(ds.interval))
    st.plotly_chart(plots.regime_bars(reg), use_container_width=True)
    st.dataframe(reg.style.format({"strategy_ann_return": "{:.1%}", "strategy_sharpe": "{:.2f}", "bench_ann_return": "{:.1%}", "bench_sharpe": "{:.2f}"}), use_container_width=True, hide_index=True)
    vol_rows = reg[reg["regime_type"] == "vol"]
    if len(vol_rows) and np.isfinite(vol_rows["strategy_sharpe"]).all():
        best = vol_rows.loc[vol_rows["strategy_sharpe"].idxmax(), "regime"]
        st.caption(f"Best in {best}. If all the P&L lives in one regime, you have a regime bet, not a model.")

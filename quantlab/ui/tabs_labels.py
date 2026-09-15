"""Tab 3: labeling scheme, parameters, class balance, label drift, barrier breakdown."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from quantlab.labeling import build_labels, class_balance
from quantlab.ui import plots
from quantlab.ui.state import cfg, df, invalidate_from, ugly_warning


@st.cache_data(show_spinner="Building labels...")
def _labels(_df: pd.DataFrame, label_json: str, data_key: str) -> pd.DataFrame:
    from quantlab.config import ExperimentConfig

    return build_labels(_df, ExperimentConfig.from_json(label_json).label)


def render() -> None:
    d = df()
    if d is None:
        st.info("Load data first.")
        return
    c = cfg()
    L = c.label
    before = L.__dict__.copy()
    col1, col2 = st.columns([1, 2])
    with col1:
        L.scheme = st.selectbox("Scheme", ["fixed_horizon", "triple_barrier", "trend_scanning"], index=["fixed_horizon", "triple_barrier", "trend_scanning"].index(L.scheme))
        if L.scheme in ("fixed_horizon", "triple_barrier"):
            L.horizon = int(st.number_input("Horizon h (bars)", 1, 250, L.horizon, help="Also the default embargo."))
        if L.scheme == "fixed_horizon":
            L.three_class = st.checkbox("Three classes (with deadband)", L.three_class)
            if L.three_class:
                L.deadband = float(st.number_input("Deadband tau (abs. return)", 0.0, 0.2, float(L.deadband), 0.001, format="%.4f"))
        if L.scheme == "triple_barrier":
            L.pt_mult = float(st.number_input("Profit-take (ATR multiples)", 0.1, 20.0, float(L.pt_mult), 0.1))
            L.sl_mult = float(st.number_input("Stop-loss (ATR multiples)", 0.1, 20.0, float(L.sl_mult), 0.1))
            L.atr_window = int(st.number_input("ATR window", 2, 200, L.atr_window))
        if L.scheme == "trend_scanning":
            L.min_window = int(st.number_input("Min window", 3, 200, L.min_window))
            L.max_window = int(st.number_input("Max window", 4, 500, L.max_window))
            L.window_step = int(st.number_input("Window step", 1, 50, L.window_step))
        L.weighting = st.selectbox("Sample weights", ["none", "uniqueness", "return_attribution", "both"], index=["none", "uniqueness", "return_attribution", "both"].index(L.weighting), help="Overlapping labels are near-duplicates; uniqueness weights down-weight them. Return attribution up-weights big, uniquely-owned moves.")
    if L.__dict__ != before:
        invalidate_from("dataset")
        if c.walkforward.embargo < L.max_horizon:
            c.walkforward.embargo = L.max_horizon
    try:
        L.validate()
    except ValueError as e:
        st.error(str(e))
        return
    from quantlab.ui.state import data_key

    lab = _labels(d, c.to_json(indent=None), data_key())
    with col2:
        bal = class_balance(lab["label"])
        st.markdown("**Class balance**")
        st.dataframe(bal.style.format({"share": "{:.1%}"}), use_container_width=True)
        if bal.attrs.get("imbalanced"):
            ugly_warning(f"One class is {bal.attrs['dominant_share']:.0%} of the sample. Accuracy is a garbage metric here: a model that always predicts the majority class gets {bal.attrs['dominant_share']:.0%}. Look at log-loss, calibration and the backtest instead.")
        n_lab = int(lab["label"].notna().sum())
        st.caption(f"{n_lab} labelled bars; {len(lab) - n_lab} unlabelled (warm-up or horizon past the end of data). Mean |ret| per label: {lab['ret'].abs().mean():.3%}.")
        if "barrier" in lab:
            bt = lab["barrier"].value_counts(dropna=True)
            st.markdown("**Barrier touched**")
            st.dataframe((bt / bt.sum()).rename("share").to_frame().style.format("{:.1%}"), use_container_width=True)
            if bt.get("vertical", 0) / max(1, bt.sum()) > 0.6:
                st.warning("Most labels hit the vertical barrier. The ATR multiples are too wide for this horizon; the labels are mostly 'sign of return at h', i.e. fixed horizon with extra steps.")
        if "t_value" in lab:
            st.plotly_chart(plots.returns_histogram(lab["t_value"].rename("t-value")), use_container_width=True)
    st.plotly_chart(plots.labels_over_time(lab, d["close"]), use_container_width=True)
    if L.weighting != "none":
        st.plotly_chart(plots.returns_histogram(lab["weight"].rename("weight")), use_container_width=True)
        st.caption("Sample weight distribution (normalised to mean 1).")

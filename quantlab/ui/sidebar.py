"""Sidebar: data selection, load, health summary, config save/load, seed."""

from __future__ import annotations

import json
from datetime import date

import streamlit as st

from quantlab.config import DataConfig, ExperimentConfig
from quantlab.data.loader import clean_ohlcv, integrity_report, load_ohlcv
from quantlab.ui.state import cfg, invalidate_from, ugly_warning


@st.cache_data(show_spinner=False, ttl=6 * 3600)
def _cached_load(ticker: str, start: str, end: str, interval: str, force: bool):
    d = load_ohlcv(DataConfig(ticker=ticker, start=start, end=end, interval=interval, force_refresh=force))
    d = clean_ohlcv(d)
    return d


def render_sidebar() -> None:
    c = cfg()
    st.sidebar.title("QuantLab")
    st.sidebar.caption("Build a strategy. Then try to break it.")

    with st.sidebar.expander("Data", expanded=True):
        ticker = st.text_input("Ticker (Yahoo)", c.data.ticker, help="Indices need a caret, e.g. ^OMXS30, ^GSPC.")
        col1, col2 = st.columns(2)
        start = col1.date_input("Start", date.fromisoformat(c.data.start))
        end = col2.date_input("End", date.fromisoformat(c.data.end))
        interval = st.selectbox("Interval", ["1d", "1h", "1wk"], index=["1d", "1h", "1wk"].index(c.data.interval))
        force = st.checkbox("Force refresh (ignore cache)", value=False)
        if interval == "1h":
            st.caption("Yahoo only serves ~730 days of hourly bars. Expect a short series.")
        if st.button("Load data", type="primary", use_container_width=True):
            new_data = DataConfig(ticker=ticker.strip(), start=str(start), end=str(end), interval=interval, force_refresh=force)
            try:
                new_data.validate()
                with st.spinner(f"Downloading {new_data.ticker}..."):
                    d = _cached_load(new_data.ticker, new_data.start, new_data.end, new_data.interval, force)
                c.data = new_data
                st.session_state["df"] = d
                st.session_state["report"] = integrity_report(d, new_data.interval)
                invalidate_from("dataset")
                st.success(f"Loaded {len(d)} bars ({'cache' if d.attrs.get('from_cache') else 'download'}).")
            except Exception as e:  # noqa: BLE001 - surfaced to the user verbatim on purpose
                st.error(f"Load failed: {e}")

    rep = st.session_state.get("report")
    if rep is not None:
        with st.sidebar.expander("Data health", expanded=True):
            s = rep.summary()
            st.metric("Bars", s["rows"])
            st.caption(f"{s['first']:%Y-%m-%d} to {s['last']:%Y-%m-%d}")
            if rep.too_few_bars:
                ugly_warning(rep.warnings[0])
            issues = [w for w in rep.warnings if not w.startswith("Only ")]
            if issues:
                for w in issues:
                    st.warning(w)
            else:
                st.success("No integrity issues flagged.")

    with st.sidebar.expander("Config", expanded=False):
        st.download_button("Save config (JSON)", data=c.to_json(), file_name=f"quantlab_{c.data.ticker}_{c.config_hash()}.json", mime="application/json", use_container_width=True)
        up = st.file_uploader("Load config (JSON)", type=["json"])
        if up is not None and st.button("Apply loaded config", use_container_width=True):
            try:
                loaded = ExperimentConfig.from_json(up.read().decode())
                st.session_state["cfg"] = loaded
                invalidate_from("dataset")
                st.success("Config applied. Reload data if the ticker or range changed.")
                st.rerun()
            except (ValueError, TypeError, json.JSONDecodeError) as e:
                st.error(f"Config rejected: {e}")
        st.code(f"config hash: {c.config_hash()}", language=None)

    with st.sidebar.expander("Reproducibility", expanded=False):
        seed = st.number_input("Seed", min_value=0, max_value=2**31 - 1, value=int(c.seed), step=1)
        if seed != c.seed:
            c.seed = int(seed)
            invalidate_from("wf")
        st.caption(f"session id: {st.session_state['session_id']}")

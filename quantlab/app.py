"""Streamlit entrypoint. Layout only: every number on screen is computed elsewhere."""

from __future__ import annotations

import sys
from pathlib import Path

# `streamlit run quantlab/app.py` puts quantlab/ on sys.path, not the repo root.
# Make `import quantlab` work without demanding a pip install first.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402

from quantlab.ui import tabs_backtest, tabs_compare, tabs_data, tabs_diagnostics, tabs_features, tabs_labels, tabs_model  # noqa: E402
from quantlab.ui.sidebar import render_sidebar  # noqa: E402
from quantlab.ui.state import init_state  # noqa: E402


def main() -> None:
    st.set_page_config(page_title="QuantLab", page_icon="🧪", layout="wide", initial_sidebar_state="expanded")
    init_state()
    render_sidebar()
    tabs = st.tabs(["Data", "Features", "Labels", "Model", "Backtest", "Diagnostics", "Compare"])
    with tabs[0]:
        tabs_data.render()
    with tabs[1]:
        tabs_features.render()
    with tabs[2]:
        tabs_labels.render()
    with tabs[3]:
        tabs_model.render()
    with tabs[4]:
        tabs_backtest.render()
    with tabs[5]:
        tabs_diagnostics.render()
    with tabs[6]:
        tabs_compare.render()


if __name__ == "__main__":
    main()
else:
    # Streamlit imports the script as __main__ normally, but some launchers do not.
    main()

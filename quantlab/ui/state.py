"""Session-state plumbing shared by the tabs. Keeps the 'where is the config' question answered once."""

from __future__ import annotations

import uuid
from typing import Any

import pandas as pd
import streamlit as st

from quantlab.config import ExperimentConfig
from quantlab.experiments.store import ExperimentStore


def init_state() -> None:
    st.session_state.setdefault("cfg", ExperimentConfig())
    st.session_state.setdefault("session_id", uuid.uuid4().hex[:10])
    st.session_state.setdefault("df", None)
    st.session_state.setdefault("report", None)
    st.session_state.setdefault("dataset", None)
    st.session_state.setdefault("dataset_key", None)
    st.session_state.setdefault("wf", None)
    st.session_state.setdefault("wf_key", None)
    st.session_state.setdefault("eval", None)
    st.session_state.setdefault("eval_key", None)
    st.session_state.setdefault("logged_run_id", None)
    st.session_state.setdefault("store", ExperimentStore())


def cfg() -> ExperimentConfig:
    return st.session_state["cfg"]


def store() -> ExperimentStore:
    return st.session_state["store"]


def df() -> pd.DataFrame | None:
    return st.session_state["df"]


def data_key() -> str:
    d = df()
    if d is None:
        return "nodata"
    return f"{d.attrs.get('ticker')}|{d.attrs.get('interval')}|{len(d)}|{d.index[0]}|{d.index[-1]}"


def dataset_key() -> str:
    c = cfg()
    return f"{data_key()}|{hash_part(c.features)}|{hash_part(c.label)}|{hash_part(c.model.meta)}"


def wf_key() -> str:
    c = cfg()
    return f"{dataset_key()}|{hash_part(c.model)}|{hash_part(c.walkforward)}|{c.seed}"


def eval_key() -> str:
    return f"{wf_key()}|{hash_part(cfg().backtest)}"


def hash_part(obj: Any) -> str:
    import dataclasses
    import hashlib
    import json

    return hashlib.sha1(json.dumps(dataclasses.asdict(obj), sort_keys=True, default=str).encode()).hexdigest()[:10]


def invalidate_from(stage: str) -> None:
    """Downstream results are stale once an upstream config changes."""
    order = ["dataset", "wf", "eval"]
    if stage not in order:
        raise ValueError(f"Unknown stage {stage!r}.")
    for s in order[order.index(stage):]:
        st.session_state[s] = None
        st.session_state[f"{s}_key"] = None
    st.session_state["logged_run_id"] = None


def ugly_warning(msg: str) -> None:
    """The spec asked for ugly. Ugly delivered."""
    st.markdown(
        f"""<div style="background:#7f1d1d;border:3px dashed #fca5a5;padding:14px;border-radius:6px;
        font-weight:700;color:#fee2e2;font-size:1.05em">&#9888;&#65039; {msg}</div>""",
        unsafe_allow_html=True,
    )


def red_box(msg: str) -> None:
    st.markdown(
        f"""<div style="background:#3b0d0d;border-left:6px solid #ef4444;padding:12px 14px;border-radius:4px;color:#fecaca">{msg}</div>""",
        unsafe_allow_html=True,
    )


def grey_note(msg: str) -> None:
    st.markdown(f'<div style="color:#8a8f98;font-size:0.9em;font-style:italic">{msg}</div>', unsafe_allow_html=True)

"""Tab 4: model family, hyperparameters, walk-forward geometry, the train button, live progress."""

from __future__ import annotations

import time
from typing import Any

import pandas as pd
import streamlit as st

from quantlab.features.registry import ParamSpec
from quantlab.models import FAMILIES, family_schema
from quantlab.models.meta import PRIMARY_RULES
from quantlab.pipeline import prepare_dataset, run_walkforward
from quantlab.ui.state import cfg, df, invalidate_from, wf_key
from quantlab.ui.tabs_features import _widget
from quantlab.validation.walkforward import WalkForwardSplitter

CLASSICAL = ["logreg", "random_forest", "extra_trees", "xgboost", "lightgbm"]
DEEP = ["lstm", "gru", "transformer"]


def _params_editor(family: str, current: dict[str, Any], key: str) -> dict[str, Any]:
    schema = family_schema(family)
    out: dict[str, Any] = {}
    cols = st.columns(3)
    for i, p in enumerate(schema):
        with cols[i % 3]:
            out[p.name] = _widget(p, f"{key}_{family}_{p.name}", current.get(p.name, p.default))
    return out


def render() -> None:
    d = df()
    if d is None:
        st.info("Load data first.")
        return
    c = cfg()
    M, W = c.model, c.walkforward
    before_m, before_w = str(M.__dict__), str(W.__dict__)

    st.subheader("Model")
    mode = st.radio("Structure", ["single", "ensemble"], index=1 if M.ensemble.enabled else 0, horizontal=True)
    M.ensemble.enabled = mode == "ensemble"
    if not M.ensemble.enabled:
        M.family = st.selectbox("Family", CLASSICAL + DEEP, index=(CLASSICAL + DEEP).index(M.family))
        if M.family in DEEP:
            st.caption("Sequence models window inside each fold. The first seq_len-1 bars of every test fold get no prediction and sit flat.")
        M.params = _params_editor(M.family, M.params, "mp")
    else:
        E = M.ensemble
        E.method = st.selectbox("Method", ["soft", "hard", "stacking"], index=["soft", "hard", "stacking"].index(E.method))
        E.members = st.multiselect("Members", CLASSICAL + DEEP, default=E.members)
        if E.method == "stacking":
            E.stacking_folds = int(st.number_input("Inner OOF folds for the meta-learner", 2, 10, E.stacking_folds))
        for name in E.members:
            with st.expander(f"{name} hyperparameters"):
                E.member_params[name] = _params_editor(name, E.member_params.get(name, {}), f"ens_{name}")
    c1, c2 = st.columns(2)
    M.scaler = c1.selectbox("Scaler (fitted inside each fold)", ["standard", "robust", "none"], index=["standard", "robust", "none"].index(M.scaler))
    M.imputer = c2.selectbox("Imputer (fitted inside each fold)", ["median", "mean", "none"], index=["median", "mean", "none"].index(M.imputer))

    with st.expander("Meta-labeling", expanded=M.meta.enabled):
        mt = M.meta
        mt.enabled = st.checkbox("Enable meta-labeling", mt.enabled, help="Primary decides direction; the model above becomes the secondary that decides whether to take the trade.")
        if mt.enabled:
            choices = list(PRIMARY_RULES) + ["model"]
            mt.primary = st.selectbox("Primary signal", choices, index=choices.index(mt.primary))
            if mt.primary == "model":
                mt.primary_model = st.selectbox("Primary model family", CLASSICAL + DEEP, index=(CLASSICAL + DEEP).index(mt.primary_model))
                mt.primary_model_params = _params_editor(mt.primary_model, mt.primary_model_params, "pm")
            else:
                defaults = PRIMARY_RULES[mt.primary]
                cols = st.columns(len(defaults))
                newp = {}
                for i, (k, v) in enumerate(defaults.items()):
                    cur = mt.primary_params.get(k, v)
                    newp[k] = cols[i].number_input(k, value=float(cur) if isinstance(v, float) else int(cur), key=f"prim_{mt.primary}_{k}")
                mt.primary_params = newp

    st.subheader("Walk-forward")
    w1, w2, w3, w4 = st.columns(4)
    W.mode = w1.selectbox("Mode", ["rolling", "expanding"], index=["rolling", "expanding"].index(W.mode))
    W.train_window = int(w2.number_input("Train window (bars)", 50, 20000, W.train_window, 25))
    W.test_window = int(w3.number_input("Test window (bars)", 1, 5000, W.test_window, 5))
    W.step = int(w4.number_input("Step (bars)", 1, 5000, W.step, 5))
    e1, e2 = st.columns([1, 2])
    W.embargo = int(e1.number_input("Embargo (bars)", 0, 1000, W.embargo, help=f"Defaults to the label horizon ({c.label.max_horizon}). Anything smaller leaks overlapping labels into the test window."))
    if W.embargo < c.label.max_horizon:
        with e2:
            st.error(f"Embargo {W.embargo} < label horizon {c.label.max_horizon}. This leaks.")
            W.i_know_what_i_am_doing = st.checkbox("I know what I am doing (run anyway)", W.i_know_what_i_am_doing)
    else:
        W.i_know_what_i_am_doing = False
    W.retune_per_fold = st.checkbox("Re-tune hyperparameters inside every fold (random search on the fold's own tail)", W.retune_per_fold)
    if W.retune_per_fold:
        W.tune_iterations = int(st.number_input("Random-search candidates per fold", 2, 100, W.tune_iterations))
        st.caption(f"Cost: roughly {W.tune_iterations}x the training time of a plain run. Honest, but slow.")
    else:
        st.caption("Fixed hyperparameters across folds: fast, and the ones above were presumably chosen while looking at this data, which is a mild form of leakage. Re-tuning per fold is the honest option.")

    if str(W.__dict__) != before_w or str(M.__dict__) != before_m:
        invalidate_from("wf")

    try:
        c.validate()
    except ValueError as e:
        st.error(str(e))
        return

    # Geometry preview before spending compute.
    try:
        ds_preview_n = len(d) - _warmup_estimate(c)
        splitter = WalkForwardSplitter(W, c.label.max_horizon)
        n_folds = splitter.n_folds(max(ds_preview_n, 1))
        st.info(f"~{n_folds} folds over ~{ds_preview_n} usable bars ({W.mode}). OOS coverage: {min(1.0, n_folds * W.test_window / max(1, ds_preview_n)):.0%} of usable bars.")
    except ValueError as e:
        st.error(str(e))
        return

    fresh = st.session_state.get("wf") is not None and st.session_state.get("wf_key") == wf_key()
    if fresh:
        st.success(f"Walk-forward result is current for this config ({len(st.session_state['wf'].folds)} folds, {st.session_state['wf'].total_seconds:.0f}s).")
    if st.button("Train walk-forward", type="primary", disabled=fresh):
        _train(c, d)

    wf = st.session_state.get("wf")
    if wf is not None:
        st.subheader("Folds")
        st.dataframe(wf.fold_table.style.format({"train_acc": "{:.3f}", "test_acc": "{:.3f}", "test_logloss": "{:.3f}", "fit_s": "{:.1f}", "tune_s": "{:.1f}"}), use_container_width=True, hide_index=True)
        st.caption("train_acc is in-sample and only there to show the gap. test_acc is what matters, and even that is not money.")
        with st.expander("Hyperparameters used per fold"):
            st.dataframe(pd.DataFrame([{"fold": f.fold_id, **_flatten(f.params_used)} for f in wf.folds]), use_container_width=True, hide_index=True)


def _warmup_estimate(c) -> int:
    longest = 0
    for s in c.features.features:
        for v in s.params.values():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                longest = max(longest, int(v))
        if s.transform in ("zscore", "rank"):
            longest += s.transform_window
    return longest + 5


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flatten(v, f"{prefix}{k}."))
        elif isinstance(v, list):
            out[f"{prefix}{k}"] = str(v)
        else:
            out[f"{prefix}{k}"] = v
    return out


def _train(c, d: pd.DataFrame) -> None:
    status = st.status("Preparing dataset...", expanded=True)
    bar = st.progress(0.0)
    log = st.empty()
    epoch_box = st.empty()
    t0 = time.time()
    lines: list[str] = []

    def progress(ev: dict[str, Any]) -> None:
        stage = ev.get("stage")
        if stage == "fold_start":
            bar.progress(ev["fold"] / ev["n_folds"], text=f"Fold {ev['fold'] + 1}/{ev['n_folds']}: train {ev['train'][0]:%Y-%m-%d}..{ev['train'][1]:%Y-%m-%d}, test {ev['test'][0]:%Y-%m-%d}..{ev['test'][1]:%Y-%m-%d}")
        elif stage == "fold_done":
            lines.append(f"fold {ev['fold']}: test acc {ev['test_accuracy']:.3f} in {ev['seconds']:.1f}s")
            log.code("\n".join(lines[-12:]), language=None)
        elif stage == "epoch":
            epoch_box.caption(f"{ev['model']} epoch {ev['epoch']}: train loss {ev['train_loss']:.4f}, val loss {ev['val_loss']:.4f} (best {ev['best_val']:.4f})")
        elif stage == "tune":
            epoch_box.caption(f"tuning candidate {ev['candidate']}/{ev['n_candidates']}: val log-loss {ev['val_logloss']:.4f}")

    try:
        with status:
            ds = prepare_dataset(d, c)
            st.session_state["dataset"] = ds
            from quantlab.ui.state import dataset_key

            st.session_state["dataset_key"] = dataset_key()
            status.update(label=f"Training ({ds.n} usable bars, {ds.n_trimmed} warm-up rows trimmed)...")
            wf = run_walkforward(ds, c, progress)
        invalidate_from("eval")
        st.session_state["wf"] = wf
        st.session_state["wf_key"] = wf_key()
        status.update(label=f"Done: {len(wf.folds)} folds in {time.time() - t0:.0f}s", state="complete")
        bar.progress(1.0, text="Done")
        st.rerun()
    except Exception as e:  # noqa: BLE001
        status.update(label="Failed", state="error")
        st.exception(e)

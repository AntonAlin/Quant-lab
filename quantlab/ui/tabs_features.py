"""Tab 2: feature selection with auto-generated parameter widgets, correlation, VIF, mutual information."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from quantlab.config import FeatureSpec
from quantlab.features.registry import REGISTRY, FeatureDef, ParamSpec, _ensure_loaded, build_feature_matrix, groups
from quantlab.labeling import build_labels
from quantlab.ui import plots
from quantlab.ui.state import cfg, df, invalidate_from

TRANSFORMS = ["raw", "zscore", "rank", "diff", "log"]


def _widget(p: ParamSpec, key: str, current: Any) -> Any:
    if p.type == "int":
        return st.number_input(p.name, min_value=int(p.min) if p.min is not None else None, max_value=int(p.max) if p.max is not None else None, value=int(current), step=int(p.step or 1), key=key, help=p.help or None)
    if p.type == "float":
        step = float(p.step) if p.step else 0.01
        return st.number_input(p.name, min_value=float(p.min) if p.min is not None else None, max_value=float(p.max) if p.max is not None else None, value=float(current), step=step, format="%.5g", key=key, help=p.help or None)
    if p.type == "bool":
        return st.checkbox(p.name, value=bool(current), key=key, help=p.help or None)
    if p.type == "choice":
        return st.selectbox(p.name, list(p.choices), index=list(p.choices).index(current), key=key, help=p.help or None)
    raise ValueError(p.type)


def _spec_editor(fd: FeatureDef, spec: FeatureSpec | None, key: str) -> FeatureSpec | None:
    """Renders the checkbox + parameter widgets for one registry entry; returns the spec or None."""
    on = st.checkbox(f"**{fd.name}**", value=spec is not None, key=f"{key}_on", help=fd.description)
    if not on:
        return None
    current = spec.params if spec else {}
    params: dict[str, Any] = {}
    cols = st.columns(max(1, min(4, len(fd.params) + 1)))
    for i, p in enumerate(fd.params):
        with cols[i % len(cols)]:
            params[p.name] = _widget(p, f"{key}_{p.name}", current.get(p.name, p.default))
    with cols[len(fd.params) % len(cols)]:
        tr = st.selectbox("transform", TRANSFORMS, index=TRANSFORMS.index(spec.transform if spec else "raw"), key=f"{key}_tr", help="Applied after the indicator, before the one-bar shift. Rolling z-score/rank use the window below.")
        win = 60
        if tr in ("zscore", "rank"):
            win = st.number_input("window", 5, 1000, int(spec.transform_window if spec else 60), key=f"{key}_win")
    if fd.bounded and tr == "zscore":
        st.caption("This feature is already bounded; z-scoring it mostly adds noise. Your call.")
    return FeatureSpec(name=fd.name, params=params, transform=tr, transform_window=int(win))


@st.cache_data(show_spinner="Computing features...")
def _features(_df: pd.DataFrame, specs_json: str, data_key: str) -> pd.DataFrame:
    from quantlab.config import ExperimentConfig

    specs = ExperimentConfig.from_json(specs_json).features.features
    return build_feature_matrix(_df, specs)


def _vif(X: pd.DataFrame) -> pd.Series:
    Z = X.dropna()
    Z = (Z - Z.mean()) / Z.std(ddof=1)
    Z = Z.loc[:, Z.std() > 0]
    if Z.shape[1] < 2 or len(Z) < Z.shape[1] + 5:
        return pd.Series(dtype=float)
    out = {}
    A = Z.to_numpy()
    for j, c in enumerate(Z.columns):
        others = np.delete(A, j, axis=1)
        beta, *_ = np.linalg.lstsq(others, A[:, j], rcond=None)
        resid = A[:, j] - others @ beta
        r2 = 1 - resid.var() / A[:, j].var()
        out[c] = 1.0 / max(1e-9, 1 - r2)
    return pd.Series(out).sort_values(ascending=False)


def _mutual_info(X: pd.DataFrame, y: pd.Series, seed: int) -> pd.Series:
    from sklearn.feature_selection import mutual_info_classif

    both = pd.concat([X, y.rename("__y")], axis=1).dropna()
    if len(both) < 50:
        return pd.Series(dtype=float)
    mi = mutual_info_classif(both.drop(columns="__y"), both["__y"].astype(int), random_state=seed)
    return pd.Series(mi, index=X.columns).sort_values(ascending=False)


def render() -> None:
    d = df()
    if d is None:
        st.info("Load data first.")
        return
    _ensure_loaded()
    c = cfg()
    existing = {s.name: s for s in c.features.features}
    new_specs: list[FeatureSpec] = []
    st.caption("Every feature is computed from data up to and including bar t, then shifted one bar before it reaches the model. Multi-output indicators (MACD, Bollinger...) expand into several columns.")
    with st.expander("Quick presets"):
        col1, col2, col3 = st.columns(3)
        if col1.button("Trend + vol starter", use_container_width=True):
            _apply_preset(["sma_ratio", "ema_ratio", "ma_crossover", "macd", "roc", "adx", "natr", "realised_vol", "rsi", "bollinger"])
            st.rerun()
        if col2.button("Everything cheap", use_container_width=True):
            _apply_preset([n for n, fd in REGISTRY.items() if n not in ("hmm_state", "hurst")])
            st.rerun()
        if col3.button("Clear all", use_container_width=True):
            _apply_preset([])
            st.rerun()

    for group, defs in groups().items():
        with st.expander(f"{group.replace('_', ' ').title()} ({len(defs)})", expanded=any(fd.name in existing for fd in defs)):
            for fd in defs:
                spec = _spec_editor(fd, existing.get(fd.name), key=f"feat_{fd.name}")
                if spec is not None:
                    new_specs.append(spec)
                st.divider()

    if [s.__dict__ for s in new_specs] != [s.__dict__ for s in c.features.features]:
        c.features.features = new_specs
        invalidate_from("dataset")

    if not new_specs:
        st.warning("No features selected.")
        return
    try:
        c.features.validate()
    except ValueError as e:
        st.error(str(e))
        return

    from quantlab.ui.state import data_key

    X = _features(d, c.to_json(indent=None), data_key())
    st.subheader(f"Feature matrix: {X.shape[1]} columns, {int(X.notna().all(axis=1).sum())} complete rows of {len(X)}")
    warm = int(np.argmax(X.notna().all(axis=1).to_numpy())) if X.notna().all(axis=1).any() else len(X)
    st.caption(f"Warm-up: the first {warm} rows have at least one NaN feature and will be trimmed before walk-forward.")
    sel = st.multiselect("Plot features", list(X.columns), default=list(X.columns[:3]))
    if sel:
        st.plotly_chart(plots.feature_lines(X.tail(750), sel, d["close"].tail(750)), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        corr = X.corr()
        st.plotly_chart(plots.heatmap(corr, "Feature correlation"), use_container_width=True)
        high = [(a, b, corr.loc[a, b]) for i, a in enumerate(corr.columns) for b in corr.columns[i + 1 :] if abs(corr.loc[a, b]) > 0.9]
        if high:
            st.warning(f"{len(high)} feature pairs with |corr| > 0.9. Redundant inputs will not hurt trees much, but will make importances lie: " + ", ".join(f"{a}~{b}" for a, b, _ in high[:5]) + ("..." if len(high) > 5 else ""))
    with c2:
        vif = _vif(X)
        if len(vif):
            st.plotly_chart(plots.bar(vif, "Variance inflation factor", plots.AMBER), use_container_width=True)
            if (vif > 10).any():
                st.caption("VIF > 10 is the textbook 'these are the same feature' threshold.")
        try:
            lab = build_labels(d, c.label)
            mi = _mutual_info(X, lab["label"], c.seed)
            if len(mi):
                st.plotly_chart(plots.bar(mi, "Mutual information with the label", plots.GREEN), use_container_width=True)
                st.caption("Computed on the whole sample, so it is a sanity check, not a selection tool. Selecting features on this and then validating is a leak in slow motion.")
        except ValueError as e:
            st.caption(f"Mutual information skipped: {e}")


def _apply_preset(names: list[str]) -> None:
    c = cfg()
    c.features.features = [FeatureSpec(n, REGISTRY[n].defaults()) for n in names]
    for k in list(st.session_state.keys()):
        if k.startswith("feat_"):
            del st.session_state[k]
    invalidate_from("dataset")

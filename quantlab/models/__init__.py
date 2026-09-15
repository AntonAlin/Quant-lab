"""Model factory: ModelConfig -> ModelAdapter (plain, ensemble or meta-labeled)."""

from __future__ import annotations

from typing import Any

from quantlab.config import ModelConfig
from quantlab.features.registry import ParamSpec
from quantlab.models.base import ModelAdapter
from quantlab.models.classical import CLASSICAL
from quantlab.models.deep import DEEP
from quantlab.models.ensemble import StackingAdapter, VotingAdapter
from quantlab.models.meta import MetaLabelingAdapter

FAMILIES: dict[str, type[ModelAdapter]] = {**CLASSICAL, **DEEP}


def family_schema(family: str) -> tuple[ParamSpec, ...]:
    if family not in FAMILIES:
        raise KeyError(f"Unknown model family {family!r}. Known: {sorted(FAMILIES)}")
    return FAMILIES[family].param_schema


def make_adapter(family: str, params: dict[str, Any] | None, seed: int, scaler: str, imputer: str) -> ModelAdapter:
    if family not in FAMILIES:
        raise KeyError(f"Unknown model family {family!r}. Known: {sorted(FAMILIES)}")
    return FAMILIES[family](params=params or {}, seed=seed, scaler=scaler, imputer=imputer)


def build_model(cfg: ModelConfig, seed: int) -> ModelAdapter:
    """The one place that knows how ModelConfig maps onto adapters."""
    cfg.validate()
    if cfg.ensemble.enabled:
        members = [
            make_adapter(name, cfg.ensemble.member_params.get(name, {}), seed + i, cfg.scaler, cfg.imputer)
            for i, name in enumerate(cfg.ensemble.members)
        ]
        if cfg.ensemble.method == "stacking":
            core: ModelAdapter = StackingAdapter(members, n_inner_folds=cfg.ensemble.stacking_folds, seed=seed)
        else:
            core = VotingAdapter(members, method=cfg.ensemble.method, seed=seed)
    else:
        core = make_adapter(cfg.family, cfg.params, seed, cfg.scaler, cfg.imputer)
    if cfg.meta.enabled:
        primary = None
        if cfg.meta.primary == "model":
            primary = make_adapter(cfg.meta.primary_model, cfg.meta.primary_model_params, seed + 1000, cfg.scaler, cfg.imputer)
        return MetaLabelingAdapter(secondary=core, primary=primary, seed=seed)
    return core

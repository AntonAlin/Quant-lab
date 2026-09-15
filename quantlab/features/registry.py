"""Feature registry: name -> callable + parameter schema, so the UI can build itself.

Adding a feature is one decorated function. The schema drives Streamlit widgets,
config validation and the column naming, so nobody has to touch three files and
forget one of them.

The leakage rule lives here too: `build_feature_matrix` shifts every feature by
one bar. A feature at row t therefore describes information available at the
*close* of t-1, which is what you actually know when you decide what to do at t.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd

from quantlab.config import FeatureSpec

ParamType = Literal["int", "float", "bool", "choice"]
FeatureFn = Callable[..., pd.Series | pd.DataFrame]


@dataclass(frozen=True)
class ParamSpec:
    name: str
    type: ParamType
    default: Any
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: tuple[Any, ...] = ()
    help: str = ""

    def coerce(self, value: Any) -> Any:
        """Cast + range-check a user-supplied value, complaining specifically."""
        if self.type == "int":
            try:
                v = int(value)
            except (TypeError, ValueError) as e:
                raise ValueError(f"Parameter {self.name!r} expects an int, got {value!r}.") from e
        elif self.type == "float":
            try:
                v = float(value)
            except (TypeError, ValueError) as e:
                raise ValueError(f"Parameter {self.name!r} expects a float, got {value!r}.") from e
        elif self.type == "bool":
            v = bool(value)
        elif self.type == "choice":
            if value not in self.choices:
                raise ValueError(f"Parameter {self.name!r} must be one of {self.choices}, got {value!r}.")
            return value
        else:  # pragma: no cover
            raise ValueError(f"Unknown param type {self.type}")
        if self.min is not None and v < self.min:
            raise ValueError(f"Parameter {self.name!r}={v} is below its minimum {self.min}.")
        if self.max is not None and v > self.max:
            raise ValueError(f"Parameter {self.name!r}={v} is above its maximum {self.max}.")
        return v


@dataclass
class FeatureDef:
    name: str
    group: str
    fn: FeatureFn
    params: tuple[ParamSpec, ...] = ()
    description: str = ""
    # Some features are naturally bounded (RSI, %B) and z-scoring them is pointless.
    # Purely advisory; the UI shows it, nothing enforces it.
    bounded: bool = False

    def defaults(self) -> dict[str, Any]:
        return {p.name: p.default for p in self.params}

    def coerce_params(self, params: dict[str, Any]) -> dict[str, Any]:
        known = {p.name: p for p in self.params}
        unknown = set(params) - set(known)
        if unknown:
            raise ValueError(f"Feature {self.name!r} got unknown parameters {sorted(unknown)}.")
        out = self.defaults()
        for k, v in params.items():
            out[k] = known[k].coerce(v)
        return out


REGISTRY: dict[str, FeatureDef] = {}


def register(
    name: str,
    group: str,
    params: tuple[ParamSpec, ...] = (),
    description: str = "",
    bounded: bool = False,
) -> Callable[[FeatureFn], FeatureFn]:
    """Decorator. The wrapped function signature is fn(df: DataFrame, **params)."""

    def deco(fn: FeatureFn) -> FeatureFn:
        if name in REGISTRY:
            raise ValueError(f"Feature {name!r} registered twice. Pick another name.")
        REGISTRY[name] = FeatureDef(name=name, group=group, fn=fn, params=params, description=description, bounded=bounded)
        return fn

    return deco


def groups() -> dict[str, list[FeatureDef]]:
    out: dict[str, list[FeatureDef]] = {}
    for d in REGISTRY.values():
        out.setdefault(d.group, []).append(d)
    return out


def _ensure_loaded() -> None:
    # Feature modules register on import. Import them here so a caller that only
    # imported the registry still sees everything. Circular-import-safe because
    # the modules import only `register`/`ParamSpec` from here.
    from quantlab.features import regime, statistical, technical  # noqa: F401


def compute_feature(df: pd.DataFrame, name: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    """Compute one feature (possibly multi-column) on an OHLCV frame, unshifted."""
    _ensure_loaded()
    if name not in REGISTRY:
        raise KeyError(f"No feature named {name!r}. Known: {sorted(REGISTRY)}")
    fd = REGISTRY[name]
    p = fd.coerce_params(params or {})
    out = fd.fn(df, **p)
    if isinstance(out, pd.Series):
        out = out.to_frame(name)
    if not out.index.equals(df.index):
        raise RuntimeError(f"Feature {name!r} returned a misaligned index. That is a bug in the feature.")
    return out.astype(float)


def apply_transform(x: pd.DataFrame, transform: str, window: int) -> pd.DataFrame:
    """Post-transform, all rolling/backward-looking. Nothing here sees the future."""
    if transform == "raw":
        return x
    if transform == "zscore":
        mu = x.rolling(window, min_periods=max(5, window // 2)).mean()
        sd = x.rolling(window, min_periods=max(5, window // 2)).std(ddof=1)
        return (x - mu) / sd.replace(0.0, np.nan)
    if transform == "rank":
        # Percentile of the current value within its own trailing window.
        return x.rolling(window, min_periods=max(5, window // 2)).rank(pct=True)
    if transform == "diff":
        return x.diff()
    if transform == "log":
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.log(x.where(x > 0))
        return out
    raise ValueError(f"Unknown transform {transform!r}.")


def build_feature_matrix(
    df: pd.DataFrame,
    specs: list[FeatureSpec],
    shift: int = 1,
) -> pd.DataFrame:
    """OHLCV -> model matrix. Shifted by `shift` bars so nothing at t sees t itself.

    Returns the full-length frame (NaNs at the top from warm-up are kept, the
    walk-forward layer decides how to handle them so alignment is never lost).
    """
    if shift < 1:
        raise ValueError("shift must be >= 1. A shift of 0 means the model sees today's close today.")
    _ensure_loaded()
    if not specs:
        raise ValueError("build_feature_matrix called with no FeatureSpecs.")
    cols: list[pd.DataFrame] = []
    for spec in specs:
        spec.validate()
        raw = compute_feature(df, spec.name, spec.params)
        tr = apply_transform(raw, spec.transform, spec.transform_window)
        base = spec.column_name()
        if tr.shape[1] == 1:
            tr.columns = [base]
        else:
            tr.columns = [f"{base}__{c}" for c in tr.columns]
        cols.append(tr)
    X = pd.concat(cols, axis=1)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.shift(shift)
    X.attrs["shift"] = shift
    X.attrs["source_index_hash"] = int(pd.util.hash_pandas_object(df.index).sum())
    return X

"""Every knob the user can turn, as frozen-ish dataclasses with JSON round-tripping.

Why dataclasses and not a dict soup: a strategy config is the unit of
comparison in this app. If two runs differ, we want to know *exactly* which
field differs, and we want a stable hash so the experiment log can tell
"same config, re-run" apart from "new trial". Dicts do neither reliably.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Literal, get_args, get_origin

from quantlab import SEED

Interval = Literal["1d", "1h", "1wk"]
LabelScheme = Literal["fixed_horizon", "triple_barrier", "trend_scanning"]
WeightScheme = Literal["none", "uniqueness", "return_attribution", "both"]
WalkForwardMode = Literal["rolling", "expanding"]
Direction = Literal["long_only", "long_short", "short_only"]
Execution = Literal["next_open", "close_to_close"]
SizingScheme = Literal["fixed_fractional", "vol_target", "kelly", "confidence"]
TransformName = Literal["raw", "zscore", "rank", "diff", "log"]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@dataclass
class DataConfig:
    ticker: str = "^OMXS30"
    start: str = "2005-01-01"
    end: str = "2025-12-31"
    interval: Interval = "1d"
    force_refresh: bool = False

    def validate(self) -> None:
        if not self.ticker or not self.ticker.strip():
            raise ValueError("DataConfig.ticker is empty. Yahoo needs a symbol, e.g. '^OMXS30'.")
        if self.interval not in get_args(Interval):
            raise ValueError(f"interval must be one of {get_args(Interval)}, got {self.interval!r}")
        if self.start >= self.end:
            raise ValueError(f"start ({self.start}) must be before end ({self.end}).")


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
@dataclass
class FeatureSpec:
    """One feature instance: a registry name, its parameters, and a post-transform."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)
    transform: TransformName = "raw"
    transform_window: int = 60

    def column_name(self) -> str:
        """Stable, human-readable column id. Sorted params so order never matters."""
        p = "_".join(f"{k}{v}" for k, v in sorted(self.params.items()))
        base = f"{self.name}_{p}" if p else self.name
        if self.transform != "raw":
            base = f"{base}__{self.transform}{self.transform_window}"
        return base

    def validate(self) -> None:
        if self.transform not in get_args(TransformName):
            raise ValueError(f"Unknown transform {self.transform!r} for feature {self.name}.")
        if self.transform in ("zscore", "rank") and self.transform_window < 5:
            raise ValueError(
                f"transform_window={self.transform_window} for {self.name} is silly; use >= 5."
            )


@dataclass
class FeatureConfig:
    features: list[FeatureSpec] = field(default_factory=list)

    def validate(self) -> None:
        if not self.features:
            raise ValueError("No features selected. A model with zero inputs is a coin flip with extra steps.")
        names = [f.column_name() for f in self.features]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"Duplicate feature columns: {sorted(dupes)}. Change a parameter or drop one.")
        for f in self.features:
            f.validate()


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #
@dataclass
class LabelConfig:
    scheme: LabelScheme = "fixed_horizon"
    horizon: int = 5  # h, in bars. Also the default embargo.
    deadband: float = 0.0  # tau, absolute return; only used when three_class=True
    three_class: bool = False
    # triple barrier
    pt_mult: float = 2.0
    sl_mult: float = 1.0
    atr_window: int = 14
    # trend scanning
    min_window: int = 5
    max_window: int = 20
    window_step: int = 1
    # sample weights
    weighting: WeightScheme = "uniqueness"

    def validate(self) -> None:
        if self.scheme not in get_args(LabelScheme):
            raise ValueError(f"Unknown label scheme {self.scheme!r}.")
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1 bar.")
        if self.deadband < 0:
            raise ValueError("deadband must be >= 0.")
        if self.scheme == "triple_barrier":
            if self.pt_mult <= 0 or self.sl_mult <= 0:
                raise ValueError("Barrier multiples must be positive. A zero-width barrier is touched instantly.")
            if self.atr_window < 2:
                raise ValueError("atr_window must be >= 2.")
        if self.scheme == "trend_scanning":
            if self.min_window < 3:
                raise ValueError("trend scanning needs min_window >= 3 to fit a line with a t-stat.")
            if self.max_window <= self.min_window:
                raise ValueError("max_window must exceed min_window.")
            if self.window_step < 1:
                raise ValueError("window_step must be >= 1.")
        if self.weighting not in get_args(WeightScheme):
            raise ValueError(f"Unknown weighting {self.weighting!r}.")

    @property
    def max_horizon(self) -> int:
        """Longest forward look any label in this scheme can take. Drives the embargo."""
        if self.scheme == "trend_scanning":
            return self.max_window
        return self.horizon


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
@dataclass
class MetaConfig:
    enabled: bool = False
    # 'model' uses a separately configured primary adapter; anything else is a rule name
    primary: str = "ma_crossover"
    primary_params: dict[str, Any] = field(default_factory=lambda: {"fast": 20, "slow": 50})
    primary_model: str = "logreg"
    primary_model_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class EnsembleConfig:
    enabled: bool = False
    method: Literal["soft", "hard", "stacking"] = "soft"
    members: list[str] = field(default_factory=lambda: ["logreg", "random_forest"])
    member_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    stacking_folds: int = 4


@dataclass
class ModelConfig:
    family: str = "logreg"
    params: dict[str, Any] = field(default_factory=dict)
    scaler: Literal["standard", "robust", "none"] = "standard"
    imputer: Literal["median", "mean", "none"] = "median"
    meta: MetaConfig = field(default_factory=MetaConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)

    def validate(self) -> None:
        if not self.family:
            raise ValueError("ModelConfig.family is empty.")


# --------------------------------------------------------------------------- #
# Walk-forward
# --------------------------------------------------------------------------- #
@dataclass
class WalkForwardConfig:
    mode: WalkForwardMode = "rolling"
    train_window: int = 750
    test_window: int = 125
    step: int = 125
    embargo: int = 5
    i_know_what_i_am_doing: bool = False  # lets embargo < label horizon through
    retune_per_fold: bool = False
    tune_iterations: int = 10

    def validate(self, label_horizon: int | None = None) -> None:
        if self.mode not in get_args(WalkForwardMode):
            raise ValueError(f"Unknown walk-forward mode {self.mode!r}.")
        if self.train_window < 50:
            raise ValueError("train_window < 50 bars is not a training set, it is an anecdote.")
        if self.test_window < 1 or self.step < 1:
            raise ValueError("test_window and step must be >= 1.")
        if self.embargo < 0:
            raise ValueError("embargo cannot be negative. Time only goes one way.")
        if label_horizon is not None and self.embargo < label_horizon and not self.i_know_what_i_am_doing:
            raise ValueError(
                f"embargo={self.embargo} is smaller than the label horizon ({label_horizon}). "
                "Overlapping labels will leak across the train/test boundary. Raise the embargo "
                "or tick 'I know what I am doing'."
            )


# --------------------------------------------------------------------------- #
# Backtest
# --------------------------------------------------------------------------- #
@dataclass
class CostConfig:
    commission_bps: float = 2.0  # per side
    spread_bps: float = 5.0  # full spread; you pay half per side
    slippage_bps: float = 2.0  # per side
    slippage_atr_frac: float = 0.0  # alternative: fraction of ATR per side, added on top
    allow_zero_cost: bool = False

    def validate(self) -> None:
        for name in ("commission_bps", "spread_bps", "slippage_bps", "slippage_atr_frac"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative. Nobody pays you to trade.")
        total = self.commission_bps + self.spread_bps + self.slippage_bps + self.slippage_atr_frac
        if total == 0 and not self.allow_zero_cost:
            raise ValueError(
                "All transaction costs are zero. That is not a backtest, that is fan fiction. "
                "Tick 'allow zero cost' if you really want it."
            )


@dataclass
class SizingConfig:
    scheme: SizingScheme = "fixed_fractional"
    fixed_fraction: float = 1.0
    target_vol: float = 0.15  # annualised
    vol_window: int = 20
    leverage_cap: float = 2.0
    kelly_max: float = 0.5
    kelly_window: int = 60
    confidence_floor: float = 0.5  # p at which confidence sizing is zero

    def validate(self) -> None:
        if self.scheme not in get_args(SizingScheme):
            raise ValueError(f"Unknown sizing scheme {self.scheme!r}.")
        if not 0 < self.fixed_fraction <= 10:
            raise ValueError("fixed_fraction must be in (0, 10].")
        if self.target_vol <= 0 or self.leverage_cap <= 0:
            raise ValueError("target_vol and leverage_cap must be positive.")
        if not 0 < self.kelly_max <= 1:
            raise ValueError("kelly_max must be in (0, 1]. Full Kelly is already reckless.")
        if self.vol_window < 5 or self.kelly_window < 10:
            raise ValueError("vol_window >= 5 and kelly_window >= 10, please.")


@dataclass
class BacktestConfig:
    long_threshold: float = 0.55
    short_threshold: float = 0.55
    direction: Direction = "long_short"
    execution: Execution = "next_open"
    cooldown: int = 0
    max_holding: int = 0  # 0 = unlimited
    costs: CostConfig = field(default_factory=CostConfig)
    sizing: SizingConfig = field(default_factory=SizingConfig)

    def validate(self) -> None:
        for t in (self.long_threshold, self.short_threshold):
            if not 0 < t < 1:
                raise ValueError("Thresholds are probabilities and must sit strictly inside (0, 1).")
        if self.direction not in get_args(Direction):
            raise ValueError(f"Unknown direction {self.direction!r}.")
        if self.execution not in get_args(Execution):
            raise ValueError(f"Unknown execution {self.execution!r}.")
        if self.cooldown < 0 or self.max_holding < 0:
            raise ValueError("cooldown and max_holding must be >= 0.")
        self.costs.validate()
        self.sizing.validate()


# --------------------------------------------------------------------------- #
# The whole thing
# --------------------------------------------------------------------------- #
@dataclass
class ExperimentConfig:
    seed: int = SEED
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    label: LabelConfig = field(default_factory=LabelConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    walkforward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    def validate(self) -> None:
        self.data.validate()
        self.features.validate()
        self.label.validate()
        self.model.validate()
        self.walkforward.validate(label_horizon=self.label.max_horizon)
        self.backtest.validate()

    # -- serialisation ----------------------------------------------------- #
    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, default=_json_default)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExperimentConfig":
        return _from_dict(cls, d)

    @classmethod
    def from_json(cls, s: str) -> "ExperimentConfig":
        return cls.from_dict(json.loads(s))

    def config_hash(self, exclude_data_refresh: bool = True) -> str:
        """SHA-256 over the canonical JSON.

        force_refresh is excluded by default because re-downloading the same data
        does not make a new strategy, and we do not want to inflate trial counts.
        """
        d = self.to_dict()
        if exclude_data_refresh:
            d["data"].pop("force_refresh", None)
        payload = json.dumps(d, sort_keys=True, default=_json_default)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _json_default(o: Any) -> Any:
    # numpy scalars sneak in from Streamlit widgets all the time.
    if hasattr(o, "item"):
        return o.item()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serialisable")


def _from_dict(cls: type, d: Any) -> Any:
    """Recursive dataclass hydration that respects nested dataclasses and list[dataclass]."""
    if not is_dataclass(cls):
        return d
    if not isinstance(d, dict):
        raise TypeError(f"Expected a dict to build {cls.__name__}, got {type(d).__name__}.")
    known = {f.name: f for f in fields(cls)}
    unknown = set(d) - set(known)
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}. Old config file?")
    kwargs: dict[str, Any] = {}
    for name, f in known.items():
        if name not in d:
            continue  # default kicks in
        val = d[name]
        typ = f.type
        # Types are strings because of `from __future__ import annotations`; resolve by name.
        resolved = _resolve_type(typ)
        if is_dataclass(resolved):
            kwargs[name] = _from_dict(resolved, val)
        elif get_origin(resolved) is list and get_args(resolved) and is_dataclass(get_args(resolved)[0]):
            kwargs[name] = [_from_dict(get_args(resolved)[0], v) for v in val]
        else:
            kwargs[name] = val
    return cls(**kwargs)


def _resolve_type(typ: Any) -> Any:
    if isinstance(typ, str):
        return eval(typ, globals())  # noqa: S307 - types come from this module only
    return typ

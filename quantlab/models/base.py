"""ModelAdapter: one interface so the walk-forward loop never cares what it is training.

Contract:
- fit(X, y, sample_weight, progress): X is a contiguous DataFrame slice (may contain
  NaN features; the adapter imputes inside its own pipeline). y may contain NaN
  where the label is unusable (purged / warm-up). Adapters must skip those rows
  as targets and must never treat X contiguity as optional (sequence models
  need it).
- predict_proba(X) returns a DataFrame indexed like X with one column per class
  (the class labels themselves, ints). Rows the model cannot score are NaN.
- Everything that fits (imputer, scaler, model) is fitted in fit() and only there.
"""

from __future__ import annotations

import json
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

from quantlab.features.registry import ParamSpec

ProgressFn = Callable[[dict[str, Any]], None]


def make_preprocessor(scaler: str, imputer: str) -> Pipeline:
    """Imputer + scaler as a Pipeline so 'fit on train, transform test' is structural, not a habit."""
    steps: list[tuple[str, Any]] = []
    if imputer == "median":
        steps.append(("imputer", SimpleImputer(strategy="median")))
    elif imputer == "mean":
        steps.append(("imputer", SimpleImputer(strategy="mean")))
    elif imputer != "none":
        raise ValueError(f"Unknown imputer {imputer!r}.")
    if scaler == "standard":
        steps.append(("scaler", StandardScaler()))
    elif scaler == "robust":
        steps.append(("scaler", RobustScaler()))
    elif scaler != "none":
        raise ValueError(f"Unknown scaler {scaler!r}.")
    if not steps:
        steps.append(("identity", "passthrough"))
    return Pipeline(steps)


class ModelAdapter(ABC):
    name: str = "base"
    is_sequence_model: bool = False
    supports_sample_weight: bool = True
    param_schema: tuple[ParamSpec, ...] = ()

    def __init__(self, params: dict[str, Any] | None = None, seed: int = 42, scaler: str = "standard", imputer: str = "median") -> None:
        self.seed = seed
        self.scaler = scaler
        self.imputer = imputer
        self.params = self.coerce_params(params or {})
        self.classes_: np.ndarray = np.array([])
        self.feature_names_: list[str] = []
        self.fitted_: bool = False

    # -- params -------------------------------------------------------------- #
    @classmethod
    def coerce_params(cls, params: dict[str, Any]) -> dict[str, Any]:
        known = {p.name: p for p in cls.param_schema}
        unknown = set(params) - set(known)
        if unknown:
            raise ValueError(f"{cls.name}: unknown hyperparameters {sorted(unknown)}. Known: {sorted(known)}.")
        out = {p.name: p.default for p in cls.param_schema}
        for k, v in params.items():
            out[k] = known[k].coerce(v)
        return out

    def get_params(self) -> dict[str, Any]:
        return {"family": self.name, "seed": self.seed, "scaler": self.scaler, "imputer": self.imputer, **self.params}

    @abstractmethod
    def param_distributions(self, rng: np.random.Generator) -> dict[str, Any]:
        """One random hyperparameter draw, for per-fold re-tuning. Return {} to opt out."""

    # -- core ---------------------------------------------------------------- #
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: pd.Series | None = None, progress: ProgressFn | None = None) -> "ModelAdapter":
        ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        ...

    def predict(self, X: pd.DataFrame) -> pd.Series:
        proba = self.predict_proba(X)
        out = pd.Series(np.nan, index=X.index)
        ok = proba.notna().all(axis=1)
        out[ok] = proba.loc[ok].idxmax(axis=1).astype(float)
        return out

    # -- persistence --------------------------------------------------------- #
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        with open(path.with_suffix(".json"), "w") as fh:
            json.dump(self.get_params(), fh, indent=2, default=str)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ModelAdapter":
        with open(path, "rb") as fh:
            obj = pickle.load(fh)  # noqa: S301 - local files we wrote ourselves
        if not isinstance(obj, ModelAdapter):
            raise TypeError(f"{path} does not contain a ModelAdapter.")
        return obj

    # -- helpers ------------------------------------------------------------- #
    @staticmethod
    def _valid_rows(X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        if not X.index.equals(y.index):
            raise ValueError("X and y must share an index. Alignment was lost before fit().")
        return y.notna().to_numpy() & ~X.isna().all(axis=1).to_numpy()

    def _check_fitted(self) -> None:
        if not self.fitted_:
            raise RuntimeError(f"{self.name} has not been fitted. Call fit() first.")

    def _check_columns(self, X: pd.DataFrame) -> None:
        if list(X.columns) != self.feature_names_:
            raise ValueError(
                f"{self.name}: feature columns at predict time differ from fit time.\n"
                f"fit:     {self.feature_names_}\npredict: {list(X.columns)}"
            )

    def _proba_frame(self, index: pd.Index, proba: np.ndarray, valid: np.ndarray) -> pd.DataFrame:
        out = pd.DataFrame(np.nan, index=index, columns=[int(c) for c in self.classes_], dtype=float)
        out.loc[valid, :] = proba
        return out

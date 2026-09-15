"""sklearn-flavoured adapters: logistic regression, random forest, extra trees, XGBoost, LightGBM.

Everything goes through a Pipeline(imputer, scaler, estimator). Yes, scaling a
random forest is pointless. It is also harmless, and it means the code path is
identical for every family, which is worth more than the microseconds.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from quantlab.features.registry import ParamSpec
from quantlab.models.base import ModelAdapter, ProgressFn, make_preprocessor

P = ParamSpec


class SklearnAdapter(ModelAdapter):
    """Base for anything with a sklearn-style estimator supporting predict_proba."""

    def _build_estimator(self) -> Any:  # pragma: no cover - abstract-ish
        raise NotImplementedError

    def _encode(self, y: np.ndarray) -> np.ndarray:
        # XGBoost/LightGBM insist on 0..k-1 targets; sklearn does not care. Encode always.
        return np.searchsorted(self.classes_, y)

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: pd.Series | None = None, progress: ProgressFn | None = None) -> "SklearnAdapter":
        valid = self._valid_rows(X, y)
        if valid.sum() < 20:
            raise ValueError(f"{self.name}: only {int(valid.sum())} usable training rows. That is not a training set.")
        self.feature_names_ = list(X.columns)
        yv = y.to_numpy(dtype=float)[valid].astype(int)
        self.classes_ = np.unique(yv)
        if len(self.classes_) < 2:
            raise ValueError(f"{self.name}: training fold has a single class ({self.classes_}). Widen the window or change the labeler.")
        self.pipeline_: Pipeline = Pipeline([("pre", make_preprocessor(self.scaler, self.imputer)), ("est", self._build_estimator())])
        fit_kwargs: dict[str, Any] = {}
        if sample_weight is not None and self.supports_sample_weight:
            w = sample_weight.reindex(X.index).to_numpy(dtype=float)[valid]
            if np.isnan(w).any():
                raise ValueError(f"{self.name}: sample_weight contains NaN for usable rows.")
            fit_kwargs["est__sample_weight"] = w
        if progress:
            progress({"stage": "fit", "model": self.name, "n_rows": int(valid.sum()), "n_features": X.shape[1]})
        self.pipeline_.fit(X.loc[valid], self._encode(yv), **fit_kwargs)
        self.fitted_ = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        self._check_columns(X)
        valid = ~X.isna().all(axis=1).to_numpy()
        proba = self.pipeline_.predict_proba(X.loc[valid]) if valid.any() else np.zeros((0, len(self.classes_)))
        return self._proba_frame(X.index, proba, valid)

    def feature_importances(self) -> pd.Series | None:
        self._check_fitted()
        est = self.pipeline_.named_steps["est"]
        if hasattr(est, "feature_importances_"):
            return pd.Series(est.feature_importances_, index=self.feature_names_)
        if hasattr(est, "coef_"):
            return pd.Series(np.abs(est.coef_).mean(axis=0), index=self.feature_names_)
        return None

    def sklearn_estimator(self) -> Any:
        self._check_fitted()
        return self.pipeline_.named_steps["est"]

    def transform_features(self, X: pd.DataFrame) -> np.ndarray:
        """Preprocessed matrix, for SHAP on tree models."""
        self._check_fitted()
        return self.pipeline_.named_steps["pre"].transform(X)


class LogRegAdapter(SklearnAdapter):
    name = "logreg"
    param_schema = (
        P("C", "float", 1.0, 1e-4, 1e3, help="Inverse regularisation strength. Smaller = more shrinkage."),
        P("l1_ratio", "float", 0.5, 0.0, 1.0, 0.05, help="0 = ridge, 1 = lasso, in between = elastic net."),
        P("max_iter", "int", 2000, 100, 20000, 100),
        P("class_weight", "choice", "none", choices=("none", "balanced")),
    )

    def _build_estimator(self) -> LogisticRegression:
        p = self.params
        return LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            C=p["C"],
            l1_ratio=p["l1_ratio"],
            max_iter=p["max_iter"],
            class_weight=None if p["class_weight"] == "none" else "balanced",
            random_state=self.seed,
        )

    def param_distributions(self, rng: np.random.Generator) -> dict[str, Any]:
        return {"C": float(10 ** rng.uniform(-3, 2)), "l1_ratio": float(rng.choice([0.0, 0.25, 0.5, 0.75, 1.0]))}


class RandomForestAdapter(SklearnAdapter):
    name = "random_forest"
    param_schema = (
        P("n_estimators", "int", 300, 10, 3000, 10),
        P("max_depth", "int", 6, 1, 64, 1, help="0 would mean unlimited; we do not offer that because it overfits like mad."),
        P("min_samples_leaf", "int", 20, 1, 500, 1),
        P("max_features", "choice", "sqrt", choices=("sqrt", "log2", "0.5", "1.0")),
        P("class_weight", "choice", "none", choices=("none", "balanced", "balanced_subsample")),
    )

    def _build_estimator(self) -> RandomForestClassifier:
        p = self.params
        mf: Any = p["max_features"]
        if mf in ("0.5", "1.0"):
            mf = float(mf)
        return RandomForestClassifier(
            n_estimators=p["n_estimators"],
            max_depth=p["max_depth"],
            min_samples_leaf=p["min_samples_leaf"],
            max_features=mf,
            class_weight=None if p["class_weight"] == "none" else p["class_weight"],
            n_jobs=-1,
            random_state=self.seed,
        )

    def param_distributions(self, rng: np.random.Generator) -> dict[str, Any]:
        return {
            "max_depth": int(rng.integers(2, 12)),
            "min_samples_leaf": int(rng.choice([5, 10, 20, 50, 100])),
            "max_features": str(rng.choice(["sqrt", "log2", "0.5"])),
        }


class ExtraTreesAdapter(RandomForestAdapter):
    name = "extra_trees"

    def _build_estimator(self) -> ExtraTreesClassifier:
        p = self.params
        mf: Any = p["max_features"]
        if mf in ("0.5", "1.0"):
            mf = float(mf)
        return ExtraTreesClassifier(
            n_estimators=p["n_estimators"],
            max_depth=p["max_depth"],
            min_samples_leaf=p["min_samples_leaf"],
            max_features=mf,
            class_weight=None if p["class_weight"] == "none" else p["class_weight"],
            n_jobs=-1,
            random_state=self.seed,
        )


class XGBoostAdapter(SklearnAdapter):
    name = "xgboost"
    param_schema = (
        P("n_estimators", "int", 300, 10, 5000, 10),
        P("max_depth", "int", 4, 1, 16, 1),
        P("learning_rate", "float", 0.03, 1e-4, 1.0, 0.005),
        P("subsample", "float", 0.8, 0.1, 1.0, 0.05),
        P("colsample_bytree", "float", 0.8, 0.1, 1.0, 0.05),
        P("min_child_weight", "float", 5.0, 0.0, 100.0, 0.5),
        P("reg_alpha", "float", 0.0, 0.0, 100.0, 0.1),
        P("reg_lambda", "float", 1.0, 0.0, 100.0, 0.1),
    )

    def _build_estimator(self) -> Any:
        from xgboost import XGBClassifier

        p = self.params
        return XGBClassifier(
            n_estimators=p["n_estimators"],
            max_depth=p["max_depth"],
            learning_rate=p["learning_rate"],
            subsample=p["subsample"],
            colsample_bytree=p["colsample_bytree"],
            min_child_weight=p["min_child_weight"],
            reg_alpha=p["reg_alpha"],
            reg_lambda=p["reg_lambda"],
            random_state=self.seed,
            n_jobs=4,
            tree_method="hist",
            verbosity=0,
        )

    def param_distributions(self, rng: np.random.Generator) -> dict[str, Any]:
        return {
            "max_depth": int(rng.integers(2, 8)),
            "learning_rate": float(10 ** rng.uniform(-2.5, -0.7)),
            "subsample": float(rng.uniform(0.5, 1.0)),
            "colsample_bytree": float(rng.uniform(0.5, 1.0)),
            "min_child_weight": float(rng.choice([1, 5, 10, 20])),
            "reg_lambda": float(10 ** rng.uniform(-1, 1.5)),
        }


class LightGBMAdapter(SklearnAdapter):
    name = "lightgbm"
    param_schema = (
        P("n_estimators", "int", 300, 10, 5000, 10),
        P("num_leaves", "int", 15, 2, 512, 1),
        P("max_depth", "int", -1, -1, 64, 1, help="-1 = unlimited (LightGBM controls size through num_leaves)."),
        P("learning_rate", "float", 0.03, 1e-4, 1.0, 0.005),
        P("subsample", "float", 0.8, 0.1, 1.0, 0.05),
        P("colsample_bytree", "float", 0.8, 0.1, 1.0, 0.05),
        P("min_child_samples", "int", 20, 1, 1000, 1),
        P("reg_alpha", "float", 0.0, 0.0, 100.0, 0.1),
        P("reg_lambda", "float", 0.0, 0.0, 100.0, 0.1),
    )

    def _build_estimator(self) -> Any:
        from lightgbm import LGBMClassifier

        p = self.params
        return LGBMClassifier(
            n_estimators=p["n_estimators"],
            num_leaves=p["num_leaves"],
            max_depth=p["max_depth"],
            learning_rate=p["learning_rate"],
            subsample=p["subsample"],
            subsample_freq=1,
            colsample_bytree=p["colsample_bytree"],
            min_child_samples=p["min_child_samples"],
            reg_alpha=p["reg_alpha"],
            reg_lambda=p["reg_lambda"],
            random_state=self.seed,
            n_jobs=4,
            verbose=-1,
        )

    def param_distributions(self, rng: np.random.Generator) -> dict[str, Any]:
        return {
            "num_leaves": int(rng.choice([7, 15, 31, 63])),
            "learning_rate": float(10 ** rng.uniform(-2.5, -0.7)),
            "subsample": float(rng.uniform(0.5, 1.0)),
            "colsample_bytree": float(rng.uniform(0.5, 1.0)),
            "min_child_samples": int(rng.choice([10, 20, 50, 100])),
        }


CLASSICAL: dict[str, type[SklearnAdapter]] = {
    a.name: a for a in (LogRegAdapter, RandomForestAdapter, ExtraTreesAdapter, XGBoostAdapter, LightGBMAdapter)
}

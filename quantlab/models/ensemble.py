"""Voting and stacking over ModelAdapters.

Stacking trains the logistic meta-learner on *out-of-fold* member predictions
produced by a blocked, chronological inner split of the training fold. Training
the meta-learner on in-sample member predictions is the textbook way to build
an ensemble that is 100% confident and 50% right.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from quantlab.models.base import ModelAdapter, ProgressFn


class VotingAdapter(ModelAdapter):
    name = "voting"

    def __init__(self, members: list[ModelAdapter], method: str = "soft", seed: int = 42) -> None:
        super().__init__(params={}, seed=seed)
        if len(members) < 2:
            raise ValueError("An ensemble of one model is just a model with a longer name.")
        if method not in ("soft", "hard"):
            raise ValueError("method must be 'soft' or 'hard'.")
        self.members = members
        self.method = method
        self.is_sequence_model = any(m.is_sequence_model for m in members)

    def get_params(self) -> dict[str, Any]:
        return {"family": f"voting_{self.method}", "members": [m.get_params() for m in self.members]}

    def param_distributions(self, rng: np.random.Generator) -> dict[str, Any]:
        return {}

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: pd.Series | None = None, progress: ProgressFn | None = None) -> "VotingAdapter":
        self.feature_names_ = list(X.columns)
        for m in self.members:
            m.fit(X, y, sample_weight, progress)
        self.classes_ = np.array(sorted(set().union(*(set(m.classes_.tolist()) for m in self.members))))
        self.fitted_ = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        frames = [m.predict_proba(X).reindex(columns=[int(c) for c in self.classes_]).fillna(0.0) for m in self.members]
        if self.method == "soft":
            avg = sum(frames) / len(frames)
        else:
            votes = [pd.get_dummies(f.idxmax(axis=1)).reindex(columns=[int(c) for c in self.classes_], fill_value=0).astype(float) for f in frames]
            avg = sum(votes) / len(votes)
        # Rows nobody could score (sequence warm-up) should stay NaN, not 0.
        unscored = np.logical_and.reduce([m.predict_proba(X).isna().all(axis=1).to_numpy() for m in self.members])
        avg.loc[unscored, :] = np.nan
        row_sum = avg.sum(axis=1).replace(0, np.nan)
        return avg.div(row_sum, axis=0)


class StackingAdapter(ModelAdapter):
    name = "stacking"

    def __init__(self, members: list[ModelAdapter], n_inner_folds: int = 4, seed: int = 42) -> None:
        super().__init__(params={}, seed=seed)
        if len(members) < 2:
            raise ValueError("Stacking needs at least two members.")
        if n_inner_folds < 2:
            raise ValueError("n_inner_folds must be >= 2.")
        self.members = members
        self.n_inner_folds = n_inner_folds
        self.is_sequence_model = any(m.is_sequence_model for m in members)

    def get_params(self) -> dict[str, Any]:
        return {"family": "stacking", "n_inner_folds": self.n_inner_folds, "members": [m.get_params() for m in self.members]}

    def param_distributions(self, rng: np.random.Generator) -> dict[str, Any]:
        return {}

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: pd.Series | None = None, progress: ProgressFn | None = None) -> "StackingAdapter":
        import copy

        self.feature_names_ = list(X.columns)
        yv = y.to_numpy(dtype=float)
        self.classes_ = np.unique(yv[~np.isnan(yv)]).astype(int)
        cols = [int(c) for c in self.classes_]
        n = len(X)
        # Blocked chronological inner CV: fit on everything before block k, predict block k.
        # Block 0 has no history, so OOF rows start at block 1. Not symmetric, but not leaky.
        bounds = np.linspace(0, n, self.n_inner_folds + 1).astype(int)
        oof = [pd.DataFrame(np.nan, index=X.index, columns=cols, dtype=float) for _ in self.members]
        for k in range(1, self.n_inner_folds):
            tr = slice(0, bounds[k])
            te = slice(bounds[k], bounds[k + 1])
            for j, m in enumerate(self.members):
                mm = copy.deepcopy(m)
                try:
                    mm.fit(X.iloc[tr], y.iloc[tr], None if sample_weight is None else sample_weight.iloc[tr], None)
                except ValueError:
                    continue  # too few rows / single class in an early block: skip, the OOF row stays NaN
                oof[j].iloc[te] = mm.predict_proba(X.iloc[te]).reindex(columns=cols).to_numpy()
        Z = pd.concat(oof, axis=1)
        ok = Z.notna().all(axis=1).to_numpy() & ~np.isnan(yv)
        if ok.sum() < 30:
            raise ValueError("Stacking: fewer than 30 out-of-fold rows for the meta-learner. Increase the train window.")
        self.meta_ = LogisticRegression(C=1.0, max_iter=2000, random_state=self.seed)
        w = None if sample_weight is None else sample_weight.to_numpy(dtype=float)[ok]
        self.meta_.fit(Z.to_numpy()[ok], np.searchsorted(self.classes_, yv[ok].astype(int)), sample_weight=w)
        if progress:
            progress({"stage": "stacking_meta", "n_oof_rows": int(ok.sum())})
        for m in self.members:
            m.fit(X, y, sample_weight, progress)
        self.fitted_ = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        cols = [int(c) for c in self.classes_]
        Z = pd.concat([m.predict_proba(X).reindex(columns=cols) for m in self.members], axis=1)
        ok = Z.notna().all(axis=1).to_numpy()
        out = pd.DataFrame(np.nan, index=X.index, columns=cols, dtype=float)
        if ok.any():
            out.loc[ok, :] = self.meta_.predict_proba(Z.to_numpy()[ok])
        return out

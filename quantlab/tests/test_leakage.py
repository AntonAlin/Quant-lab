"""§4: no feature may respond before a future spike. Every registered feature, every transform."""

import numpy as np
import pandas as pd
import pytest

from quantlab.config import FeatureSpec
from quantlab.features.registry import REGISTRY, _ensure_loaded, build_feature_matrix, compute_feature
from quantlab.validation.leakage import LeakageError, check_feature_matrix, future_spike_probe

_ensure_loaded()
SLOW = {"hmm_state"}  # HMM refits are slow; covered separately with a smaller series


def _spike(df: pd.DataFrame, at: int, mult: float = 8.0) -> pd.DataFrame:
    out = df.copy()
    cols = ["open", "high", "low", "close"]
    out.loc[out.index[at:], cols] = out.loc[out.index[at:], cols] * mult
    out.loc[out.index[at:], "volume"] = out.loc[out.index[at:], "volume"] * mult
    return out


@pytest.mark.parametrize("name", sorted(n for n in REGISTRY if n not in SLOW))
def test_feature_does_not_see_future(name: str, ohlcv: pd.DataFrame) -> None:
    at = 900
    base = compute_feature(ohlcv, name, REGISTRY[name].defaults())
    spiked = compute_feature(_spike(ohlcv, at), name, REGISTRY[name].defaults())
    a = base.iloc[:at].to_numpy()
    b = spiked.iloc[:at].to_numpy()
    assert np.allclose(a, b, equal_nan=True), f"{name} changed before the spike"


def test_hmm_feature_does_not_see_future() -> None:
    from quantlab.data.loader import synthetic_ohlcv

    df = synthetic_ohlcv(400, seed=2)
    at = 350
    p = {"refit_every": 30, "min_train": 100, "n_iter": 20}
    base = compute_feature(df, "hmm_state", p)
    spiked = compute_feature(_spike(df, at), "hmm_state", p)
    assert np.allclose(base.iloc[:at].to_numpy(), spiked.iloc[:at].to_numpy(), equal_nan=True)


@pytest.mark.parametrize("transform", ["raw", "zscore", "rank", "diff", "log"])
def test_matrix_is_shifted_and_causal(transform: str, ohlcv: pd.DataFrame) -> None:
    specs = [FeatureSpec("rsi", {"window": 14}, transform, 30), FeatureSpec("macd", {}, transform, 30)]
    at = 800
    X = build_feature_matrix(ohlcv, specs)
    Xs = build_feature_matrix(_spike(ohlcv, at), specs)
    # Shift of one: the row *at* the spike must still be untouched.
    assert np.allclose(X.iloc[: at + 1].to_numpy(), Xs.iloc[: at + 1].to_numpy(), equal_nan=True)
    assert not np.allclose(X.iloc[at + 1 : at + 5].to_numpy(), Xs.iloc[at + 1 : at + 5].to_numpy(), equal_nan=True)
    check_feature_matrix(X, ohlcv.index)


def test_probe_detects_a_leaky_feature(ohlcv: pd.DataFrame) -> None:
    leaky = lambda df: df["close"].shift(-1) / df["close"] - 1  # noqa: E731
    assert future_spike_probe(leaky, ohlcv, spike_at=500)
    honest = lambda df: df["close"].pct_change(5)  # noqa: E731
    assert not future_spike_probe(honest, ohlcv, spike_at=500)


def test_unshifted_matrix_is_rejected(ohlcv: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        build_feature_matrix(ohlcv, [FeatureSpec("rsi", {"window": 14})], shift=0)
    X = build_feature_matrix(ohlcv, [FeatureSpec("rsi", {"window": 14})])
    X.attrs["shift"] = 0
    with pytest.raises(LeakageError):
        check_feature_matrix(X, ohlcv.index)

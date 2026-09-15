import numpy as np
import pandas as pd
import pytest

from quantlab.data.loader import synthetic_ohlcv


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    return synthetic_ohlcv(n=1200, seed=11)


@pytest.fixture
def small_ohlcv() -> pd.DataFrame:
    return synthetic_ohlcv(n=300, seed=3)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)

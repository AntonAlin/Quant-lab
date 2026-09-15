import pytest

from quantlab.config import BacktestConfig, CostConfig, EnsembleConfig, ExperimentConfig, FeatureConfig, FeatureSpec, LabelConfig, MetaConfig, ModelConfig, WalkForwardConfig


def _cfg() -> ExperimentConfig:
    return ExperimentConfig(
        seed=7,
        features=FeatureConfig([FeatureSpec("rsi", {"window": 21}, "zscore", 90), FeatureSpec("macd", {"fast": 8, "slow": 21, "signal": 5})]),
        label=LabelConfig(scheme="triple_barrier", horizon=10, pt_mult=1.5, sl_mult=1.0, weighting="both"),
        model=ModelConfig(family="xgboost", params={"max_depth": 3}, meta=MetaConfig(enabled=True, primary="momentum", primary_params={"window": 30}), ensemble=EnsembleConfig(enabled=False)),
        walkforward=WalkForwardConfig(mode="expanding", train_window=600, test_window=60, step=60, embargo=10, retune_per_fold=True),
        backtest=BacktestConfig(long_threshold=0.6, costs=CostConfig(3, 4, 1)),
    )


def test_round_trip_identical() -> None:
    c = _cfg()
    j = c.to_json()
    c2 = ExperimentConfig.from_json(j)
    assert c == c2
    assert c.config_hash() == c2.config_hash()
    assert isinstance(c2.features.features[0], FeatureSpec)
    assert isinstance(c2.model.meta, MetaConfig)


def test_hash_changes_with_params_but_not_refresh() -> None:
    a, b = _cfg(), _cfg()
    b.data.force_refresh = True
    assert a.config_hash() == b.config_hash()
    b.model.params["max_depth"] = 4
    assert a.config_hash() != b.config_hash()


def test_unknown_key_is_rejected() -> None:
    d = _cfg().to_dict()
    d["walkforward"]["banana"] = 1
    with pytest.raises(ValueError, match="Unknown keys"):
        ExperimentConfig.from_dict(d)


def test_validation_catches_nonsense() -> None:
    c = _cfg()
    c.walkforward.embargo = 3
    with pytest.raises(ValueError, match="embargo"):
        c.validate()
    c.walkforward.i_know_what_i_am_doing = True
    c.validate()
    c.features.features = []
    with pytest.raises(ValueError, match="No features"):
        c.validate()

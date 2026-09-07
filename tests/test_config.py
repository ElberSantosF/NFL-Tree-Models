"""The config is the contract of the workbench: if it accepts nonsense, runs lie."""

from __future__ import annotations

import pytest

from nfl_trees.config import ConfigError, ExperimentConfig, list_configs


def test_loads_a_valid_config(base_config_dict):
    cfg = ExperimentConfig.from_dict(base_config_dict)
    assert cfg.name == "test"
    assert cfg.data.target == "home_win"
    assert cfg.features.columns == ["HomeTeam", "AwayTeam", "GameSlot"]


def test_default_metrics_per_task(base_config_dict):
    del base_config_dict["evaluation"]
    cfg = ExperimentConfig.from_dict(base_config_dict)
    assert cfg.evaluation.primary_metric == "roc_auc"

    base_config_dict["task"] = "regression"
    cfg = ExperimentConfig.from_dict(base_config_dict)
    assert cfg.evaluation.primary_metric == "rmse"


def test_invalid_task(base_config_dict):
    base_config_dict["task"] = "ranking"
    with pytest.raises(ConfigError, match="task"):
        ExperimentConfig.from_dict(base_config_dict)


def test_unknown_field_is_rejected(base_config_dict):
    base_config_dict["split"]["test_fraction"] = 0.3
    with pytest.raises(ConfigError, match="unknown fields"):
        ExperimentConfig.from_dict(base_config_dict)


def test_no_features_is_rejected(base_config_dict):
    base_config_dict["features"] = {}
    with pytest.raises(ConfigError, match="no features"):
        ExperimentConfig.from_dict(base_config_dict)


def test_no_model_is_rejected(base_config_dict):
    base_config_dict["model"] = {}
    with pytest.raises(ConfigError, match="model.type"):
        ExperimentConfig.from_dict(base_config_dict)


def test_primary_metric_must_be_computed(base_config_dict):
    base_config_dict["evaluation"]["primary_metric"] = "pr_auc"
    with pytest.raises(ConfigError, match="primary_metric"):
        ExperimentConfig.from_dict(base_config_dict)


def test_invalid_source_is_rejected(base_config_dict):
    base_config_dict["data"]["source"] = "jogadas"
    with pytest.raises(ConfigError, match="invalid data.source"):
        ExperimentConfig.from_dict(base_config_dict)


def test_invalid_split_strategy_is_rejected(base_config_dict):
    """Caught when the config loads, not halfway through the run."""
    base_config_dict["split"]["strategy"] = "sazonal"
    with pytest.raises(ConfigError, match="invalid split.strategy"):
        ExperimentConfig.from_dict(base_config_dict)


def test_invalid_missing_strategy_is_rejected(base_config_dict):
    base_config_dict["features"]["missing_strategy"] = "medain"
    with pytest.raises(ConfigError, match="invalid features.missing_strategy"):
        ExperimentConfig.from_dict(base_config_dict)


def test_season_in_both_train_and_test(base_config_dict):
    base_config_dict["split"]["test_seasons"] = [2023, 2024]
    with pytest.raises(ConfigError, match="both train and test"):
        ExperimentConfig.from_dict(base_config_dict)


def test_repository_configs_are_valid():
    """Holds for whatever configs exist: `configs/` may still be empty."""
    for name in list_configs():
        ExperimentConfig.load(name)  # raises ConfigError if anything is wrong

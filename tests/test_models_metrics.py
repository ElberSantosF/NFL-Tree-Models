"""Model registry, metrics and the end-to-end run on synthetic data."""

from __future__ import annotations

import numpy as np
import pytest

from nfl_trees import metrics
from nfl_trees.config import ExperimentConfig
from nfl_trees.models import MODELS, build_model, get_spec, list_models, register


def test_catalog_starts_empty():
    """Models are added one at a time, as the study moves forward."""
    assert MODELS == {}, f"models registered outside the tests: {sorted(MODELS)}"


def test_empty_catalog_message():
    with pytest.raises(KeyError, match="model catalog is empty"):
        get_spec("anything")


def test_register_and_build(test_model):
    est = build_model(test_model, "classification", {}, seed=7)
    assert est.get_params()["random_state"] == 7
    assert est.get_params()["max_depth"] == 3  # catalog default


def test_yaml_params_win_over_defaults(test_model):
    est = build_model(test_model, "classification", {"max_depth": 9}, 42)
    assert est.get_params()["max_depth"] == 9


def test_regression_uses_the_regressor(test_model):
    est = build_model(test_model, "regression", {}, 42)
    assert "Regressor" in type(est).__name__


def test_spec_carries_a_summary_and_a_doc_path(test_model):
    spec = get_spec(test_model)
    assert spec.summary
    assert spec.doc == f"docs/models/{test_model}.md"
    assert spec in list_models()


def test_duplicate_registration_fails(test_model):
    with pytest.raises(ValueError, match="already registered"):
        register(test_model, "something else")(lambda task, params, seed: None)


def test_unknown_model(test_model):
    with pytest.raises(KeyError, match="unknown model"):
        build_model("magic_forest", "classification", {}, 42)


def test_missing_package_gives_a_useful_message():
    @register("ghost_model", "Depends on a package that does not exist.", requires="package_xyz")
    def _build(task, params, seed):  # pragma: no cover
        raise AssertionError("should never be called")

    try:
        with pytest.raises(ImportError, match="pip install package_xyz"):
            build_model("ghost_model", "classification", {}, 42)
    finally:
        MODELS.pop("ghost_model", None)


def test_classification_metrics():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.4, 0.6, 0.9])
    y_pred = (y_proba >= 0.5).astype(int)
    out = metrics.compute("classification", ["roc_auc", "accuracy", "f1"], y_true, y_pred, y_proba)
    assert out["roc_auc"] == 1.0
    assert out["accuracy"] == 1.0


def test_metric_that_does_not_apply_comes_back_as_nan():
    # roc_auc is undefined when a single class is present.
    y_true = np.array([1, 1, 1])
    y_proba = np.array([0.2, 0.7, 0.9])
    out = metrics.compute(
        "classification", ["roc_auc"], y_true, (y_proba > 0.5).astype(int), y_proba
    )
    assert np.isnan(out["roc_auc"])


def test_unknown_metric():
    with pytest.raises(KeyError, match="unknown metric"):
        metrics.compute("classification", ["secret_auc"], np.array([0, 1]), np.array([0, 1]))


def test_metric_direction():
    assert metrics.is_better("roc_auc", 0.8, 0.7)
    assert metrics.is_better("rmse", 3.0, 4.0)
    assert not metrics.is_better("rmse", 5.0, 4.0)


def test_end_to_end_run(monkeypatch, scores, base_config_dict):
    """Run the whole pipeline on synthetic data, without writing to results/."""
    from nfl_trees import experiment

    monkeypatch.setattr(experiment, "load_source", lambda *a, **k: scores)
    cfg = ExperimentConfig.from_dict(base_config_dict)
    result = experiment.run(cfg, save=False)

    assert set(result.metrics) == {"roc_auc", "accuracy"}
    assert result.split_info["test_seasons"] == [2024]
    assert {"y_true", "y_pred", "y_proba"} <= set(result.predictions.columns)
    assert result.importances is not None
    assert result.output_dir is None

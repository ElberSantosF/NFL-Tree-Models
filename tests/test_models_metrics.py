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


@pytest.mark.parametrize("name", ["roc_auc", "pr_auc", "log_loss", "brier"])
def test_probability_metrics_are_nan_without_probabilities(name):
    """Never computed on hard labels: roc_auc over 0/1 is balanced accuracy."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    out = metrics.compute("classification", [name], y_true, y_pred, None)
    assert np.isnan(out[name])


def test_a_model_without_predict_proba_does_not_report_an_auc(monkeypatch, scores,
                                                              base_config_dict):
    from sklearn.svm import LinearSVC

    from nfl_trees import experiment
    from nfl_trees.models import MODELS, register

    @register("no_proba", "Estimator without predict_proba, for the tests.")
    def _build(task, params, seed):
        return LinearSVC(random_state=seed, **params)

    try:
        monkeypatch.setattr(experiment, "load_source", lambda *a, **k: scores)
        base_config_dict["model"] = {"type": "no_proba", "params": {}}
        result = experiment.run(ExperimentConfig.from_dict(base_config_dict), save=False)
        assert np.isnan(result.metrics["roc_auc"])
        assert not np.isnan(result.metrics["accuracy"])
        assert "y_proba" not in result.predictions.columns
    finally:
        MODELS.pop("no_proba", None)


def test_cross_validation_scores_the_primary_metric(monkeypatch, scores, base_config_dict):
    """`cv_mean` reads on the scale of the metric it is named after."""
    from nfl_trees import experiment

    monkeypatch.setattr(experiment, "load_source", lambda *a, **k: scores)
    base_config_dict["evaluation"] = {"metrics": ["accuracy"], "primary_metric": "accuracy"}
    base_config_dict["split"]["cv_folds"] = 3
    result = experiment.run(ExperimentConfig.from_dict(base_config_dict), save=False)

    assert result.cv_scores["cv_metric"] == "accuracy"
    assert 0.0 <= result.cv_scores["cv_mean"] <= 1.0


def test_cross_validation_refuses_a_metric_it_cannot_score():
    from nfl_trees.experiment import CV_SCORERS

    assert set(CV_SCORERS) == set(metrics.CLASSIFICATION_METRICS) | set(
        metrics.REGRESSION_METRICS
    ), "every registered metric needs a cross-validation scorer"


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

"""Evaluation metrics.

A metric is a function `(y_true, y_pred, y_proba) -> float`. Registering one
here is all it takes to be able to name it in `evaluation.metrics` in the YAML.

Some metrics score a *probability*, not a decision. They are listed in
`NEEDS_PROBA` and come back as NaN when the estimator has no `predict_proba`:
computing `roc_auc` on hard 0/1 predictions is arithmetically possible but it
is no longer an AUC (it collapses to balanced accuracy), and a number under the
wrong name on the leaderboard is worse than a missing one.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np

log = logging.getLogger(__name__)

MetricFn = Callable[[np.ndarray, np.ndarray, np.ndarray | None], float]

CLASSIFICATION_METRICS: dict[str, MetricFn] = {}
REGRESSION_METRICS: dict[str, MetricFn] = {}

# Metrics where "higher is better" (used to sort the leaderboard).
HIGHER_IS_BETTER = {
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "r2",
}

# Metrics that score a probability: without `y_proba` they are not defined.
NEEDS_PROBA = frozenset({"roc_auc", "pr_auc", "log_loss", "brier"})


def _register(task: str, name: str) -> Callable[[MetricFn], MetricFn]:
    registry = CLASSIFICATION_METRICS if task == "classification" else REGRESSION_METRICS

    def decorate(fn: MetricFn) -> MetricFn:
        registry[name] = fn
        return fn

    return decorate


def registry_for(task: str) -> dict[str, MetricFn]:
    if task == "classification":
        return CLASSIFICATION_METRICS
    if task == "regression":
        return REGRESSION_METRICS
    raise ValueError(f"invalid task '{task}'")


def compute(
    task: str,
    names: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute the requested metrics. A metric that does not apply comes back as NaN."""
    registry = registry_for(task)
    out: dict[str, float] = {}
    for name in names:
        if name not in registry:
            raise KeyError(f"unknown metric '{name}' for {task}; use {sorted(registry)}")
        if name in NEEDS_PROBA and y_proba is None:
            log.warning(
                "metric '%s' needs probabilities and the model gave none: reported as NaN", name
            )
            out[name] = float("nan")
            continue
        try:
            out[name] = float(registry[name](y_true, y_pred, y_proba))
        except (ValueError, TypeError):
            # e.g. roc_auc with a single class present in the test set.
            out[name] = float("nan")
    return out


def is_better(metric: str, candidate: float, current: float) -> bool:
    """Compare two values of the same metric, respecting its direction."""
    if metric in HIGHER_IS_BETTER:
        return candidate > current
    return candidate < current


# --------------------------------------------------------------------------- #
# classification
# --------------------------------------------------------------------------- #
@_register("classification", "accuracy")
def _accuracy(y_true, y_pred, y_proba=None) -> float:
    from sklearn.metrics import accuracy_score

    return accuracy_score(y_true, y_pred)


@_register("classification", "balanced_accuracy")
def _balanced_accuracy(y_true, y_pred, y_proba=None) -> float:
    from sklearn.metrics import balanced_accuracy_score

    return balanced_accuracy_score(y_true, y_pred)


@_register("classification", "precision")
def _precision(y_true, y_pred, y_proba=None) -> float:
    from sklearn.metrics import precision_score

    return precision_score(y_true, y_pred, zero_division=0)


@_register("classification", "recall")
def _recall(y_true, y_pred, y_proba=None) -> float:
    from sklearn.metrics import recall_score

    return recall_score(y_true, y_pred, zero_division=0)


@_register("classification", "f1")
def _f1(y_true, y_pred, y_proba=None) -> float:
    from sklearn.metrics import f1_score

    return f1_score(y_true, y_pred, zero_division=0)


@_register("classification", "roc_auc")
def _roc_auc(y_true, y_pred, y_proba=None) -> float:
    from sklearn.metrics import roc_auc_score

    if y_proba is None:
        raise ValueError("roc_auc requires probabilities")
    return roc_auc_score(y_true, y_proba)


@_register("classification", "pr_auc")
def _pr_auc(y_true, y_pred, y_proba=None) -> float:
    from sklearn.metrics import average_precision_score

    if y_proba is None:
        raise ValueError("pr_auc requires probabilities")
    return average_precision_score(y_true, y_proba)


@_register("classification", "log_loss")
def _log_loss(y_true, y_pred, y_proba=None) -> float:
    from sklearn.metrics import log_loss

    if y_proba is None:
        raise ValueError("log_loss requires probabilities")
    return log_loss(y_true, y_proba, labels=[0, 1])


@_register("classification", "brier")
def _brier(y_true, y_pred, y_proba=None) -> float:
    from sklearn.metrics import brier_score_loss

    if y_proba is None:
        raise ValueError("brier requires probabilities")
    return brier_score_loss(y_true, y_proba)


# --------------------------------------------------------------------------- #
# regression
# --------------------------------------------------------------------------- #
@_register("regression", "rmse")
def _rmse(y_true, y_pred, y_proba=None) -> float:
    from sklearn.metrics import mean_squared_error

    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


@_register("regression", "mae")
def _mae(y_true, y_pred, y_proba=None) -> float:
    from sklearn.metrics import mean_absolute_error

    return mean_absolute_error(y_true, y_pred)


@_register("regression", "medae")
def _medae(y_true, y_pred, y_proba=None) -> float:
    from sklearn.metrics import median_absolute_error

    return median_absolute_error(y_true, y_pred)


@_register("regression", "r2")
def _r2(y_true, y_pred, y_proba=None) -> float:
    from sklearn.metrics import r2_score

    return r2_score(y_true, y_pred)

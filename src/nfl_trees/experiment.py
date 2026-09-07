"""Experiment runner: from the YAML to the results saved on disk.

A run always goes through the same steps, in the same order:

    config -> data -> features -> split -> training -> metrics -> artifacts

Each run writes its own folder under `results/<name>/` plus one row in
`results/leaderboard.csv`, so models can be compared later without
reprocessing anything.
"""

from __future__ import annotations

import json
import logging
import platform
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from . import metrics as metrics_mod
from .config import ExperimentConfig
from .data import load_source
from .features import build_dataset, make_preprocessor
from .models import build_model, get_spec
from .paths import LEADERBOARD, RESULTS_DIR, ensure_dir
from .split import describe as describe_split
from .split import split_indices

log = logging.getLogger(__name__)


@dataclass
class RunResult:
    """What a run produces, in memory."""

    config: ExperimentConfig
    pipeline: Pipeline
    metrics: dict[str, float]
    split_info: dict[str, Any]
    predictions: pd.DataFrame
    importances: pd.DataFrame | None
    cv_scores: dict[str, Any] = field(default_factory=dict)
    output_dir: Path | None = None
    duration_s: float = 0.0

    @property
    def primary(self) -> float:
        return self.metrics.get(self.config.evaluation.primary_metric, float("nan"))

    def summary(self) -> str:
        parts = [f"{k}={v:.4f}" for k, v in self.metrics.items()]
        return f"{self.config.name} [{self.config.model.type}] " + " ".join(parts)


def run(config: ExperimentConfig | str | Path, *, save: bool = True) -> RunResult:
    """Run an experiment end to end."""
    cfg = config if isinstance(config, ExperimentConfig) else ExperimentConfig.load(config)
    started = time.perf_counter()
    np.random.seed(cfg.seed)

    log.info("experiment '%s' | model=%s | target=%s", cfg.name, cfg.model.type, cfg.data.target)

    # 1. data
    df = load_source(
        cfg.data.source,
        cfg.data.seasons,
        include_preseason=cfg.data.include_preseason,
        include_postseason=cfg.data.include_postseason,
    )
    if cfg.data.sample_rows:
        df = df.sample(n=min(cfg.data.sample_rows, len(df)), random_state=cfg.seed)
        df = df.reset_index(drop=True)
    log.info("rows loaded: %d", len(df))

    # 2. features and target
    X, y, meta = build_dataset(df, cfg.features, cfg.data.target)
    log.info("feature matrix: %s", X.shape)

    # 3. split
    train_idx, test_idx = split_indices(meta, cfg.split, seed=cfg.seed)
    split_info = describe_split(meta, train_idx, test_idx)
    log.info("split: %s", split_info)

    # 4. training
    pipeline = Pipeline(
        [
            ("preprocess", make_preprocessor(cfg.features)),
            ("model", build_model(cfg.model.type, cfg.task, cfg.model.params, cfg.seed)),
        ]
    )
    y_arr = _as_array(y, cfg.task)
    pipeline.fit(X.iloc[train_idx], y_arr[train_idx])

    # 5. evaluation
    X_test = X.iloc[test_idx]
    y_test = y_arr[test_idx]
    y_pred = pipeline.predict(X_test)
    y_proba = _positive_proba(pipeline, X_test) if cfg.task == "classification" else None
    if cfg.task == "classification":
        if y_proba is not None:
            y_pred = (y_proba >= cfg.evaluation.threshold).astype(int)
        else:
            log.warning(
                "'%s' gives no probabilities: evaluation.threshold is ignored and %s "
                "cannot be computed",
                cfg.model.type,
                sorted(set(cfg.evaluation.metrics) & metrics_mod.NEEDS_PROBA) or "no metric",
            )

    scores = metrics_mod.compute(cfg.task, cfg.evaluation.metrics, y_test, y_pred, y_proba)
    cv_scores = (
        _cross_validate(pipeline, X.iloc[train_idx], y_arr[train_idx], cfg)
        if cfg.split.cv_folds
        else {}
    )

    predictions = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
    if y_proba is not None:
        predictions["y_proba"] = y_proba
    for col in meta.columns:
        predictions[col] = meta[col].iloc[test_idx].to_numpy()

    result = RunResult(
        config=cfg,
        pipeline=pipeline,
        metrics=scores,
        split_info=split_info,
        predictions=predictions,
        importances=extract_importances(pipeline, list(X.columns)),
        cv_scores=cv_scores,
        duration_s=time.perf_counter() - started,
    )
    log.info(result.summary())

    if save:
        result.output_dir = save_run(result)
        log.info("artifacts in %s", result.output_dir)
    return result


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _as_array(y: pd.Series, task: str) -> np.ndarray:
    return y.astype(int).to_numpy() if task == "classification" else y.astype(float).to_numpy()


def _positive_proba(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray | None:
    """Probability of the positive class, when the model offers one."""
    if not hasattr(pipeline, "predict_proba"):
        return None
    proba = pipeline.predict_proba(X)
    return proba[:, 1] if proba.ndim == 2 and proba.shape[1] == 2 else None


# `evaluation.primary_metric` -> the scikit-learn scorer that computes it. The
# `neg_` scorers come back negated, so the sign is flipped again on the way out
# and `cv_mean` always reads on the same scale as the metric it is named after.
CV_SCORERS = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "log_loss": "neg_log_loss",
    "brier": "neg_brier_score",
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "medae": "neg_median_absolute_error",
    "r2": "r2",
}


def _cross_validate(
    pipeline: Pipeline, X: pd.DataFrame, y: np.ndarray, cfg: ExperimentConfig
) -> dict[str, float]:
    """Cross-validation on the training set, as a stability reference only.

    Scores the same metric the config declares as primary. Without a scorer for
    it the fold scores would silently be the estimator's default `.score()` --
    a different metric under the same name -- so that case is refused.
    """
    from sklearn.model_selection import cross_val_score

    metric = cfg.evaluation.primary_metric
    if metric not in CV_SCORERS:
        raise KeyError(
            f"no cross-validation scorer for primary_metric '{metric}'; "
            f"available: {sorted(CV_SCORERS)}"
        )
    scoring = CV_SCORERS[metric]
    scores = cross_val_score(pipeline, X, y, cv=cfg.split.cv_folds, scoring=scoring, n_jobs=None)
    if scoring.startswith("neg_"):
        scores = -scores
    return {
        "cv_metric": metric,
        "cv_folds": float(cfg.split.cv_folds),
        "cv_mean": float(np.mean(scores)),
        "cv_std": float(np.std(scores)),
    }


def extract_importances(pipeline: Pipeline, columns: list[str]) -> pd.DataFrame | None:
    """Feature importances from the model, if it exposes `feature_importances_`."""
    model = pipeline.named_steps["model"]
    values = getattr(model, "feature_importances_", None)
    if values is None:
        return None
    try:
        names = [str(n) for n in pipeline.named_steps["preprocess"].get_feature_names_out()]
    except Exception:
        names = columns
    if len(names) != len(values):
        names = [f"f{i}" for i in range(len(values))]
    return (
        pd.DataFrame({"feature": names, "importance": values})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------- #
# persistence
# --------------------------------------------------------------------------- #
def save_run(result: RunResult) -> Path:
    """Write config, metrics, predictions, importances and the fitted model."""
    cfg = result.config
    out = ensure_dir(RESULTS_DIR / cfg.name)

    payload = {
        "name": cfg.name,
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "model": cfg.model.type,
        "model_summary": get_spec(cfg.model.type).summary,
        "task": cfg.task,
        "target": cfg.data.target,
        "metrics": result.metrics,
        "cv": result.cv_scores,
        "split": result.split_info,
        "duration_s": round(result.duration_s, 2),
        "python": platform.python_version(),
        "config": cfg.to_dict(),
    }
    (out / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    if cfg.evaluation.save_predictions:
        result.predictions.to_csv(out / "predictions.csv", index=False)
    if cfg.evaluation.save_importances and result.importances is not None:
        result.importances.to_csv(out / "importances.csv", index=False)

    try:
        import joblib

        joblib.dump(result.pipeline, out / "model.joblib")
    except ImportError:  # pragma: no cover
        log.warning("joblib unavailable: model not serialized")

    append_leaderboard(result)
    return out


def append_leaderboard(result: RunResult) -> Path:
    """Add (or refresh) this run's row in the leaderboard."""
    cfg = result.config
    row = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "experiment": cfg.name,
        "model": cfg.model.type,
        "task": cfg.task,
        "target": cfg.data.target,
        "primary_metric": cfg.evaluation.primary_metric,
        "primary_value": result.primary,
        "n_train": result.split_info.get("n_train"),
        "n_test": result.split_info.get("n_test"),
        "duration_s": round(result.duration_s, 2),
        **{f"metric_{k}": v for k, v in result.metrics.items()},
    }
    ensure_dir(RESULTS_DIR)
    frame = pd.DataFrame([row])
    if LEADERBOARD.exists():
        previous = pd.read_csv(LEADERBOARD)
        previous = previous[previous["experiment"] != cfg.name]
        frame = pd.concat([previous, frame], ignore_index=True)
    frame.to_csv(LEADERBOARD, index=False)
    return LEADERBOARD


def load_leaderboard() -> pd.DataFrame:
    """Read the leaderboard, sorted by each run's primary metric."""
    if not LEADERBOARD.exists():
        return pd.DataFrame()
    df = pd.read_csv(LEADERBOARD)
    if df.empty:
        return df
    ascending = ~df["primary_metric"].isin(metrics_mod.HIGHER_IS_BETTER)
    df = df.assign(_asc=ascending)
    # Sort within each target, respecting the direction of its metric.
    df = pd.concat(
        [
            group.sort_values("primary_value", ascending=bool(group["_asc"].iloc[0]))
            for _, group in df.groupby("target", sort=True)
        ]
    )
    return df.drop(columns="_asc").reset_index(drop=True)

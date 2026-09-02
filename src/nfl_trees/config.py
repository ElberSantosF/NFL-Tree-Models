"""Experiment configuration.

Every experiment is a self-contained YAML file in `configs/`. The point is that
the diff between two experiments reads clearly: what changed from one run to
the next is explicit in the file.

Format:

    name: decision_tree
    description: free text
    task: classification          # classification | regression
    seed: 42

    data:
      source: scores              # scores (one row per game) | plays
      target: home_win            # see nfl_trees.data.TARGETS
      seasons: [2015, 2016, ...]
      include_preseason: false

    features:
      numeric: [...]              # the features are yours: see features.builders
      categorical: [HomeTeam, AwayTeam]

    split:
      strategy: season            # season | random
      train_seasons: [...]
      test_seasons: [...]

    model:
      type: decision_tree         # see nfl_trees.models.MODELS
      params: {max_depth: 6}

    evaluation:
      metrics: [roc_auc, accuracy, f1]
      primary_metric: roc_auc
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .paths import CONFIG_DIR

TASKS = ("classification", "regression")

DEFAULT_METRICS = {
    "classification": ["roc_auc", "accuracy", "f1", "log_loss"],
    "regression": ["rmse", "mae", "r2"],
}
DEFAULT_PRIMARY = {"classification": "roc_auc", "regression": "rmse"}


class ConfigError(ValueError):
    """Invalid or incomplete configuration."""


@dataclass
class DataConfig:
    source: str = "scores"
    target: str = "home_win"
    seasons: list[int] = field(default_factory=list)
    include_preseason: bool = False
    include_postseason: bool = True
    sample_rows: int | None = None  # handy for fast iteration in a notebook


@dataclass
class FeatureConfig:
    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    builders: list[str] = field(default_factory=list)
    missing_strategy: str = "median"  # median | sentinel | keep

    @property
    def columns(self) -> list[str]:
        return [*self.numeric, *self.categorical]


@dataclass
class SplitConfig:
    strategy: str = "season"  # season | random
    train_seasons: list[int] = field(default_factory=list)
    test_seasons: list[int] = field(default_factory=list)
    test_size: float = 0.2
    cv_folds: int = 0  # 0 = no cross-validation


@dataclass
class ModelConfig:
    type: str = ""
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationConfig:
    metrics: list[str] = field(default_factory=list)
    primary_metric: str = ""
    threshold: float = 0.5
    save_predictions: bool = True
    save_importances: bool = True


@dataclass
class ExperimentConfig:
    name: str
    task: str = "classification"
    description: str = ""
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    source_path: Path | None = None

    # -- construction ----------------------------------------------------- #
    @classmethod
    def from_dict(cls, raw: dict[str, Any], source_path: Path | None = None) -> ExperimentConfig:
        if "name" not in raw:
            raise ConfigError("required field missing: 'name'")

        task = str(raw.get("task", "classification")).lower()
        if task not in TASKS:
            raise ConfigError(f"invalid task '{task}'; use one of {TASKS}")

        evaluation = _build(EvaluationConfig, raw.get("evaluation", {}), "evaluation")
        if not evaluation.metrics:
            evaluation.metrics = list(DEFAULT_METRICS[task])
        if not evaluation.primary_metric:
            evaluation.primary_metric = DEFAULT_PRIMARY[task]

        cfg = cls(
            name=str(raw["name"]),
            task=task,
            description=str(raw.get("description", "")),
            seed=int(raw.get("seed", 42)),
            data=_build(DataConfig, raw.get("data", {}), "data"),
            features=_build(FeatureConfig, raw.get("features", {}), "features"),
            split=_build(SplitConfig, raw.get("split", {}), "split"),
            model=_build(ModelConfig, raw.get("model", {}), "model"),
            evaluation=evaluation,
            source_path=source_path,
        )
        cfg.validate()
        return cfg

    @classmethod
    def load(cls, ref: str | Path) -> ExperimentConfig:
        """Load by file path or by file name inside `configs/`."""
        path = Path(ref)
        if not path.suffix:
            path = CONFIG_DIR / f"{ref}.yaml"
        if not path.exists():
            raise ConfigError(f"config not found: {path}")
        with path.open(encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        if not isinstance(raw, dict):
            raise ConfigError(f"the config must be a YAML mapping: {path}")
        return cls.from_dict(raw, source_path=path.resolve())

    # -- validation ------------------------------------------------------- #
    def validate(self) -> None:
        if not self.model.type:
            raise ConfigError(f"'{self.name}': model.type is required")
        if not self.features.columns and not self.features.builders:
            raise ConfigError(
                f"'{self.name}': no features declared "
                "(fill in features.numeric / features.categorical / features.builders)"
            )
        if self.evaluation.primary_metric not in self.evaluation.metrics:
            raise ConfigError(
                f"'{self.name}': primary_metric '{self.evaluation.primary_metric}' "
                "is not listed in evaluation.metrics"
            )
        if self.split.strategy == "season":
            overlap = set(self.split.train_seasons) & set(self.split.test_seasons)
            if overlap:
                raise ConfigError(
                    f"'{self.name}': seasons in both train and test: {sorted(overlap)}"
                )

    # -- serialization ---------------------------------------------------- #
    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        data = asdict(self)
        data["source_path"] = str(self.source_path) if self.source_path else None
        return data


def _build(cls: type, raw: Any, section: str):
    """Instantiate a config dataclass, rejecting unknown fields."""
    raw = dict(raw or {})
    known = set(cls.__dataclass_fields__)
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ConfigError(f"unknown fields in '{section}': {unknown}. Valid: {sorted(known)}")
    return cls(**raw)


def list_configs() -> list[str]:
    """Names of the configs available in `configs/`."""
    if not CONFIG_DIR.exists():
        return []
    return sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))

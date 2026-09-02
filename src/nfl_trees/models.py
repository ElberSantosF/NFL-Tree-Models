"""Tree model catalog.

**The catalog starts empty on purpose.** Each model is added when you get to
study it, not before -- registering seven models at once would turn the
workbench into a list of things you cannot explain yet.

To add a model, write a constructor and register it:

    @register("decision_tree", "Single decision tree (CART). Interpretable baseline.")
    def _decision_tree(task, params, seed):
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

        cls = DecisionTreeClassifier if task == "classification" else DecisionTreeRegressor
        return cls(random_state=seed, **params)

The constructor takes `(task, params, seed)` and returns an estimator with the
scikit-learn API (`fit` / `predict` / optionally `predict_proba`). After that:

1. create `configs/decision_tree.yaml` with `model.type: decision_tree`;
2. create `docs/models/decision_tree.md` (copy `docs/models/_template.md`);
3. run `python -m nfl_trees run decision_tree`.

Nothing else in the workbench needs to change.

Import the library **inside** the constructor, never at module level: that
keeps `list_models()` working even without xgboost/lightgbm installed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

MODELS: dict[str, ModelSpec] = {}


@dataclass
class ModelSpec:
    """Metadata for a model in the catalog."""

    name: str
    summary: str
    build: Callable[[str, dict[str, Any], int], Any]
    requires: str | None = None  # external package needed, if any
    defaults: dict[str, Any] = field(default_factory=dict)
    doc: str = ""  # path of the documentation under docs/models/

    @property
    def available(self) -> bool:
        """Can this model be instantiated in the current environment?"""
        if self.requires is None:
            return True
        from importlib.util import find_spec

        return find_spec(self.requires) is not None


def register(
    name: str,
    summary: str,
    *,
    requires: str | None = None,
    defaults: dict[str, Any] | None = None,
) -> Callable[[Callable], Callable]:
    """Register a model constructor in the catalog.

    `summary` is a one-liner about what the model does -- it shows up in
    `python -m nfl_trees models` and in the `metrics.json` of every run.
    `requires` is the name of the external package, when the model does not
    come from scikit-learn. `defaults` are hyperparameters applied before the
    ones coming from the YAML (the YAML always wins).
    """

    def decorate(fn: Callable[[str, dict[str, Any], int], Any]) -> Callable:
        if name in MODELS:
            raise ValueError(f"model '{name}' already registered")
        MODELS[name] = ModelSpec(
            name=name,
            summary=summary,
            build=fn,
            requires=requires,
            defaults=dict(defaults or {}),
            doc=f"docs/models/{name}.md",
        )
        return fn

    return decorate


def build_model(model_type: str, task: str, params: dict[str, Any], seed: int = 42) -> Any:
    """Instantiate the requested estimator, with the catalog defaults applied."""
    spec = get_spec(model_type)
    if not spec.available:
        raise ImportError(
            f"model '{model_type}' needs the '{spec.requires}' package "
            f"(pip install {spec.requires})"
        )
    merged = {**spec.defaults, **params}
    return spec.build(task, merged, seed)


def get_spec(model_type: str) -> ModelSpec:
    if not MODELS:
        raise KeyError(
            "the model catalog is empty: register a model with @register in "
            "nfl_trees/models.py before running an experiment"
        )
    if model_type not in MODELS:
        raise KeyError(f"unknown model '{model_type}'; available: {sorted(MODELS)}")
    return MODELS[model_type]


def list_models() -> list[ModelSpec]:
    """Models in the catalog, alphabetically."""
    return [MODELS[name] for name in sorted(MODELS)]

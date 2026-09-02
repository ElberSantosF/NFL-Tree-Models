"""nfl-trees: a workbench for experimenting with tree models on NFL data.

Typical use in a notebook:

    from nfl_trees import ExperimentConfig, run
    result = run(ExperimentConfig.load("decision_tree"))
    result.metrics
"""

from .config import ExperimentConfig
from .experiment import RunResult, load_leaderboard, run
from .models import MODELS, list_models

__version__ = "0.1.0"

__all__ = [
    "ExperimentConfig",
    "MODELS",
    "RunResult",
    "list_models",
    "load_leaderboard",
    "run",
]

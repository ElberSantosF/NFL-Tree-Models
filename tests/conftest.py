"""Shared fixtures.

The tests run without the raw CSVs: they use synthetic DataFrames with the same
schema as the real files. Tests that do depend on the real data should be marked
with `@pytest.mark.requires_data`.

The model catalog starts empty, so tests that need a model use the `test_model`
fixture, which registers one and unregisters it afterwards.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Allows running the tests without `pip install -e .`
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def scores() -> pd.DataFrame:
    """Synthetic games with the schema of data/raw/scores/2010-2026_scores.csv."""
    rng = np.random.default_rng(0)
    n = 300
    home = rng.integers(0, 45, size=n).astype(float)
    away = rng.integers(0, 45, size=n).astype(float)
    home[0], away[0] = 21.0, 21.0  # tie: target undefined
    home[1], away[1] = np.nan, np.nan  # scheduled game, no score
    status = np.array(["FINAL"] * n, dtype=object)
    status[1] = "TBD"
    return pd.DataFrame(
        {
            "Season": rng.choice([2022, 2023, 2024], size=n),
            "Week": rng.choice(["WEEK 1", "WEEK 5", "SUPER BOWL", "PRESEASON WEEK 1"], size=n),
            "GameStatus": status,
            "GameSlot": rng.choice(["Sunday", "Monday", "Thursday"], size=n),
            "GameDate": "September 1st",
            "AwayTeam": rng.choice(["DAL", "CIN", "KC"], size=n),
            "AwayScore": away,
            "HomeTeam": rng.choice(["SF", "BUF", "PHI"], size=n),
            "HomeScore": home,
        }
    )


@pytest.fixture
def plays() -> pd.DataFrame:
    """Synthetic plays with the schema of data/raw/plays/<season>_plays.csv."""
    rng = np.random.default_rng(1)
    n = 400
    return pd.DataFrame(
        {
            "Season": rng.choice([2022, 2023, 2024], size=n),
            "Week": rng.choice(["WEEK 1", "WEEK 5", "SUPER BOWL", "PRESEASON WEEK 1"], size=n),
            "GameSlot": rng.choice(["Sunday", "Monday", "Thursday"], size=n),
            "Date": "September 1st",
            "AwayTeam": rng.choice(["DAL", "CIN", "KC"], size=n),
            "HomeTeam": rng.choice(["SF", "BUF", "PHI"], size=n),
            "Quarter": rng.choice(["1st Quarter", "2nd Quarter"], size=n),
            "DriveNumber": rng.integers(1, 20, size=n),
            "TeamWithPossession": rng.choice(["San Francisco 49ers", "Buffalo Bills"], size=n),
            "IsScoringDrive": rng.integers(0, 2, size=n),
            "PlayNumberInDrive": rng.integers(1, 12, size=n),
            "IsScoringPlay": rng.integers(0, 2, size=n),
            "PlayOutcome": rng.choice(["9 Yard Pass", "-2 Yard Run", "Touchdown"], size=n),
            "PlayStart": "1st & 10 at DET 28",
            "PlayTimeFormation": "14:51 1st Shotgun",
            "PlayDescription": "some description",
        }
    )


@pytest.fixture
def test_model():
    """Register a shallow tree in the catalog and remove it after the test."""
    from nfl_trees.models import MODELS, register

    name = "test_tree"

    @register(name, "Shallow tree used only in the tests.", defaults={"max_depth": 3})
    def _build(task, params, seed):
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

        cls = DecisionTreeClassifier if task == "classification" else DecisionTreeRegressor
        return cls(random_state=seed, **params)

    yield name
    MODELS.pop(name, None)


@pytest.fixture
def base_config_dict(test_model) -> dict:
    """Minimal valid config, used as the base in the tests."""
    return {
        "name": "test",
        "task": "classification",
        "data": {"source": "scores", "target": "home_win", "seasons": [2022, 2023, 2024]},
        "features": {"categorical": ["HomeTeam", "AwayTeam", "GameSlot"]},
        "split": {"strategy": "season", "train_seasons": [2022, 2023], "test_seasons": [2024]},
        "model": {"type": test_model, "params": {}},
        "evaluation": {"metrics": ["roc_auc", "accuracy"], "primary_metric": "roc_auc"},
    }

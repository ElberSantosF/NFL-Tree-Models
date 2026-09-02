"""Project paths.

Import from here instead of building relative paths by hand: that way the code
behaves the same whether it runs from a notebook, the terminal or the tests.
The raw data layout is described in data/raw/README.md.
"""

from __future__ import annotations

from pathlib import Path

# .../src/nfl_trees/paths.py -> repository root
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Raw data
PLAYS_DIR = RAW_DIR / "plays"  # <season>_plays.csv
PLAYS_BY_WEEK_DIR = RAW_DIR / "plays_by_week"  # <season>/<season>_<week>_plays.csv
SCORES_DIR = RAW_DIR / "scores"  # 2010-2026_scores.csv and <season>_scores.csv
SCORES_FILE = SCORES_DIR / "2010-2026_scores.csv"

CONFIG_DIR = ROOT / "configs"
RESULTS_DIR = ROOT / "results"
LEADERBOARD = RESULTS_DIR / "leaderboard.csv"


def ensure_dir(path: Path) -> Path:
    """Create the directory (and parents) if missing and return the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path

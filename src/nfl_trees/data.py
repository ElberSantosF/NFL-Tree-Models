"""Raw data loading and target definitions.

Two sources, two grains:

- `scores` : one row per game (`data/raw/scores/2010-2026_scores.csv`)
- `plays`  : one row per play (`data/raw/plays/<season>_plays.csv`)

The goal of the project is predicting the winner of a game, so the working
grain is the game (`scores`). Plays are available because they are the raw
material for team-level aggregated features.

Targets live in `TARGETS`: functions `DataFrame -> Series`. To try a new
response variable, register a function here and point to it from the YAML
(`data.target`).
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from .paths import PLAYS_DIR, SCORES_FILE

# `Week` labels that are not regular season.
PRESEASON_WEEKS = frozenset(
    {
        "HALL OF FAME",
        "PRESEASON WEEK 1",
        "PRESEASON WEEK 2",
        "PRESEASON WEEK 3",
        "PRESEASON WEEK 4",
    }
)
POSTSEASON_WEEKS = frozenset(
    {
        "WILD CARD WEEKEND",
        "DIVISIONAL PLAYOFFS",
        "CONFERENCE CHAMPIONSHIPS",
        "SUPER BOWL",
    }
)
EXHIBITION_WEEKS = frozenset({"PRO BOWL"})

# `GameStatus` values in the scores file. Only `FINAL` is a game that was
# actually played: `BYE` and `Clinched Playoffs` are rows for a team without an
# opponent (rest week and playoff bye), and `TBD` is a game not played yet.
STATUS_PLAYED = "FINAL"
NON_GAME_STATUSES = frozenset({"BYE", "Clinched Playoffs"})
STATUS_SCHEDULED = "TBD"


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_scores(
    seasons: list[int] | None = None,
    *,
    include_preseason: bool = False,
    include_postseason: bool = True,
    statuses: tuple[str, ...] = (STATUS_PLAYED,),
) -> pd.DataFrame:
    """Load game scores (one row per game).

    By default returns only games that were played (`GameStatus == FINAL`). To
    build the set of games to predict, pass `statuses=("TBD",)`.
    """
    df = pd.read_csv(SCORES_FILE)
    df = df[df["Season"].astype(str).str.isdigit()].copy()
    df["Season"] = df["Season"].astype(int)
    if seasons:
        df = df[df["Season"].isin(seasons)]
    df = filter_status(df, statuses)
    df = filter_weeks(
        df, include_preseason=include_preseason, include_postseason=include_postseason
    )
    return df.reset_index(drop=True)


def load_plays(
    seasons: list[int] | None = None,
    *,
    include_preseason: bool = False,
    include_postseason: bool = True,
) -> pd.DataFrame:
    """Load plays for the requested seasons (one row per play)."""
    seasons = seasons or available_seasons()
    frames = []
    for season in seasons:
        path = PLAYS_DIR / f"{season}_plays.csv"
        if not path.exists():
            raise FileNotFoundError(f"plays file not found: {path}")
        frames.append(pd.read_csv(path, low_memory=False))
    df = pd.concat(frames, ignore_index=True)
    return filter_weeks(
        df, include_preseason=include_preseason, include_postseason=include_postseason
    )


def load_source(
    source: str,
    seasons: list[int] | None = None,
    *,
    include_preseason: bool = False,
    include_postseason: bool = True,
) -> pd.DataFrame:
    """Dispatch to the loader of the source picked in the config (`data.source`)."""
    loaders = {"scores": load_scores, "plays": load_plays}
    if source not in loaders:
        raise KeyError(f"unknown source '{source}'; use one of {sorted(loaders)}")
    return loaders[source](
        seasons, include_preseason=include_preseason, include_postseason=include_postseason
    )


def available_seasons() -> list[int]:
    """Seasons that have a plays file in data/raw/plays."""
    seasons = []
    for path in PLAYS_DIR.glob("*_plays.csv"):
        stem = path.stem.split("_")[0]
        if stem.isdigit():
            seasons.append(int(stem))
    return sorted(seasons)


def filter_status(df: pd.DataFrame, statuses: tuple[str, ...]) -> pd.DataFrame:
    """Keep only rows whose `GameStatus` is in `statuses`."""
    if "GameStatus" not in df.columns:
        return df
    return df[df["GameStatus"].astype(str).str.strip().isin(statuses)]


def filter_weeks(
    df: pd.DataFrame, *, include_preseason: bool, include_postseason: bool
) -> pd.DataFrame:
    """Drop preseason / postseason / Pro Bowl according to the flags."""
    week = df["Week"].astype(str).str.upper().str.strip()
    keep = ~week.isin(EXHIBITION_WEEKS)
    if not include_preseason:
        keep &= ~week.isin(PRESEASON_WEEKS)
    if not include_postseason:
        keep &= ~week.isin(POSTSEASON_WEEKS)
    return df[keep].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# targets
# --------------------------------------------------------------------------- #
TARGETS: dict[str, Callable[[pd.DataFrame], pd.Series]] = {}


def target(name: str) -> Callable[[Callable], Callable]:
    """Register a target function under `name`."""

    def decorate(fn: Callable[[pd.DataFrame], pd.Series]) -> Callable:
        if name in TARGETS:
            raise ValueError(f"target '{name}' already registered")
        TARGETS[name] = fn
        return fn

    return decorate


def build_target(name: str, df: pd.DataFrame) -> pd.Series:
    """Build the response variable registered under `name`."""
    if name not in TARGETS:
        raise KeyError(f"unknown target '{name}'; available: {sorted(TARGETS)}")
    return TARGETS[name](df).rename(name)


@target("home_win")
def _home_win(df: pd.DataFrame) -> pd.Series:
    """Classification: did the home team win?

    Games without a score (scheduled, not played) and ties become missing and
    are dropped from training: "won" is not defined in those cases. Ties are
    rare in the NFL but they happen -- if you ever want to predict the tie as
    well, that becomes a three-class target, registered separately.
    """
    home = pd.to_numeric(df["HomeScore"], errors="coerce")
    away = pd.to_numeric(df["AwayScore"], errors="coerce")
    diff = home - away
    return (diff > 0).astype("Int64").where(diff.notna() & (diff != 0))

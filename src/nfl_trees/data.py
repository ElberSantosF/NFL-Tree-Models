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

# Chronological position of each `Week` label inside its season. The postseason
# continues the regular season's count and the preseason runs negative, so
# sorting by this number sorts a season in the order it was played. There is a
# gap at 18 for 2010-2020, which had 17 regular-season weeks -- that is the
# point: `week >= 19` is the postseason in every season, whichever era it is.
# `PRO BOWL` is left out; it is not a competitive game and `filter_weeks` always
# drops it.
WEEK_ORDINAL: dict[str, int] = {
    "HALL OF FAME": -5,
    "PRESEASON WEEK 1": -4,
    "PRESEASON WEEK 2": -3,
    "PRESEASON WEEK 3": -2,
    "PRESEASON WEEK 4": -1,
    **{f"WEEK {n}": n for n in range(1, 19)},
    "WILD CARD WEEKEND": 19,
    "DIVISIONAL PLAYOFFS": 20,
    "CONFERENCE CHAMPIONSHIPS": 21,
    "SUPER BOWL": 22,
}

# Relocated franchises are spelled by the abbreviation they used at the time, in
# both sources. A team feature is about the franchise and not the city, so the
# two spellings are folded into one.
FRANCHISE_ALIASES = {"AZ": "ARI", "JAC": "JAX", "STL": "LAR", "SD": "LAC", "OAK": "LV"}

# `TeamWithPossession` in `plays` carries the full name; `HomeTeam`/`AwayTeam`
# carry the abbreviation. The plays files use each franchise's *current* name in
# every season, so this map needs no historical entries.
TEAM_NAME_TO_ABBR = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LAR", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}


def week_number(week: pd.Series) -> pd.Series:
    """`Week` label -> its position in the season (`WEEK_ORDINAL`), or missing."""
    return week.astype(str).str.upper().str.strip().map(WEEK_ORDINAL).astype("Int64")


def is_postseason(week: pd.Series) -> pd.Series:
    """Whether each `Week` label is a playoff round.

    Not "after week 17": that rule was right until 2020 and stopped being right
    in 2021, when the regular season grew to 18 weeks. The four postseason
    labels are the definition that holds in both eras.
    """
    return week.astype(str).str.upper().str.strip().isin(POSTSEASON_WEEKS)


def canonical_team(team: pd.Series) -> pd.Series:
    """One abbreviation per franchise, across the whole period."""
    return team.astype("string").str.strip().replace(FRANCHISE_ALIASES)


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


# `DriveNumber` restarts at every quarter, so on its own it is not a drive
# counter for the game: the quarter belongs in the key. A possession that runs
# across a quarter break is split in two by this, which the source has already
# done for all but 0.1% of drives.
DRIVE_KEY = ["Season", "Week", "AwayTeam", "HomeTeam", "Quarter", "DriveNumber"]

# `PlayOutcome` values that put points on the board for the offense. `Extra
# Point` is out (automatic, and already implied by the touchdown that preceded
# it) and so is `Safety`, which is two points for the *defense*.
SCORING_OUTCOMES = frozenset({"Touchdown", "Field Goal"})

# Enough of the plays file to build a drive. `PlayDescription` is most of the
# file's weight and is dropped again as soon as it has been reduced to a flag.
_DRIVE_PLAY_COLS = [*DRIVE_KEY, "TeamWithPossession", "PlayOutcome", "PlayDescription"]

# What `load_drives` returns.
DRIVE_COLUMNS = ["Season", "Week", "team", "opponent", "plays", "scored"]


def load_drives(
    seasons: list[int] | None = None,
    *,
    include_preseason: bool = False,
    include_postseason: bool = True,
) -> pd.DataFrame:
    """Load plays and reduce them to one row per drive.

    Columns: `Season`, `Week`, `team` (who ran the drive), `opponent`, `plays`
    and `scored` (the drive ended in a touchdown or field goal for `team`).

    Two things the raw file gets wrong for this purpose are repaired here, both
    described in docs/data.md: `DriveNumber` restarts every quarter, and a drive
    that ended in a defensive touchdown is credited to the *defense* on every
    one of its rows. The second one is handed back to the offense that ran the
    drive, as the drive without points that it actually was.

    One season is read at a time: the sixteen files together are ~580 MB in
    memory, while a drive table for all of them is a few megabytes.
    """
    seasons = seasons or available_seasons()
    frames = [
        _season_drives(
            season, include_preseason=include_preseason, include_postseason=include_postseason
        )
        for season in seasons
    ]
    # A season can come back empty -- 2026 is preseason only -- and concatenating
    # an empty frame would drag every column to `object`.
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=DRIVE_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def _season_drives(season: int, *, include_preseason: bool, include_postseason: bool):
    path = PLAYS_DIR / f"{season}_plays.csv"
    if not path.exists():
        raise FileNotFoundError(f"plays file not found: {path}")
    plays = pd.read_csv(path, usecols=_DRIVE_PLAY_COLS, low_memory=False)
    return build_drives(
        filter_weeks(
            plays, include_preseason=include_preseason, include_postseason=include_postseason
        )
    )


def build_drives(plays: pd.DataFrame) -> pd.DataFrame:
    """Fold a plays frame into one row per drive. See `load_drives`."""
    if plays.empty:
        return pd.DataFrame(columns=DRIVE_COLUMNS)

    # An interception or a fumble returned for a touchdown. `PlayOutcome` cannot
    # tell one of those from an offensive touchdown -- both read `Touchdown` --
    # so the play text is the only handle.
    description = plays["PlayDescription"].astype(str)
    returned_td = plays["PlayOutcome"].eq("Touchdown") & (
        description.str.contains("INTERCEPTED", case=False)
        | (
            description.str.contains("FUMBLES", case=False)
            & description.str.contains("RECOVERED by", case=False)
        )
    )
    del description

    drive_id = plays.groupby(DRIVE_KEY, sort=False).ngroup()
    by_drive = plays.groupby(drive_id, sort=False)
    scored = plays["PlayOutcome"].isin(SCORING_OUTCOMES).groupby(drive_id, sort=False).max()

    drives = pd.DataFrame(
        {
            "Season": by_drive["Season"].first(),
            "Week": by_drive["Week"].first(),
            "team": by_drive["TeamWithPossession"].first().map(TEAM_NAME_TO_ABBR).astype("string"),
            "home": canonical_team(by_drive["HomeTeam"].first()),
            "away": canonical_team(by_drive["AwayTeam"].first()),
            "plays": by_drive.size(),
            "scored": scored.astype(int),
        }
    )
    drives["opponent"] = drives["home"].where(drives["team"] != drives["home"], drives["away"])

    returned = returned_td.groupby(drive_id, sort=False).max().astype(bool)
    drives.loc[returned, ["team", "opponent"]] = drives.loc[
        returned, ["opponent", "team"]
    ].to_numpy()
    drives.loc[returned, "scored"] = 0

    return drives[DRIVE_COLUMNS].reset_index(drop=True)


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

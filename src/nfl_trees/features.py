"""Feature matrix assembly and preprocessing.

Tree models need neither scaling nor normalization, so the preprocessing here
is deliberately thin: imputation for numeric columns and ordinal encoding for
categorical ones (a tree can separate categories through successive splits, no
one-hot required).

New feature engineering goes into `FEATURE_BUILDERS`: a function
`DataFrame -> DataFrame` (the new columns only) referenced from
`features.builders` in the YAML. Three are registered, all of them game-grain,
so they expect `data.source: scores`:

- `calendar`    -- `month`, `week`, `day`, `playoff`
- `win_rates`   -- `pct_home_win`, `pct_away_win`
- `drive_rates` -- `home_pct_score_drive`, `home_pct_allowed_drive`,
                   `away_pct_score_drive`, `away_pct_allowed_drive`

A rule that holds for every builder in this project: team-level aggregation
must use **only games that happened earlier** than the one being predicted. A
full-season average includes the game itself and leaks the result. `win_rates`
and `drive_rates` share one window for that, described above `_prior_rate`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from .config import FeatureConfig
from .data import (
    WEEK_ORDINAL,
    build_target,
    canonical_team,
    is_postseason,
    load_drives,
    load_scores,
    week_number,
)

# Name -> function that takes the raw DataFrame and returns a DataFrame with
# the derived columns only. Filled in by the `@builder` calls at the bottom of
# this module; every entry is a deliberate decision, not a default.
FEATURE_BUILDERS: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {}


def builder(name: str) -> Callable[[Callable], Callable]:
    """Register a feature builder under `name`."""

    def decorate(fn: Callable[[pd.DataFrame], pd.DataFrame]) -> Callable:
        if name in FEATURE_BUILDERS:
            raise ValueError(f"builder '{name}' already registered")
        FEATURE_BUILDERS[name] = fn
        return fn

    return decorate


def apply_builders(df: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Apply the requested builders and return the DataFrame with the new columns."""
    out = df
    for name in names:
        if name not in FEATURE_BUILDERS:
            raise KeyError(f"unknown builder '{name}'; available: {sorted(FEATURE_BUILDERS)}")
        derived = FEATURE_BUILDERS[name](df)
        out = pd.concat([out, derived], axis=1)
    return out


# --------------------------------------------------------------------------- #
# the window that does not leak
# --------------------------------------------------------------------------- #
# `win_rates` and `drive_rates` both answer the same question -- what had this
# team done *before* kickoff? -- over the same window: the previous season in
# full, plus the current season up to the week before this game.
#
# That sits between the two obvious choices. A career average is stable but
# describes a roster that no longer exists; a season-to-date average describes
# the right roster but is empty in week 1 and rests on two games in week 3.
# Carrying the previous season means week 1 already has a number, and by
# December the current season dominates the average anyway.
#
# The **week is the time step**: two games in the same week are simultaneous as
# far as the window is concerned, so Thursday night never feeds into Sunday.
# `GameDate` carries no year, so ordering inside a week would mean rebuilding
# dates -- and a model that predicts a round before it is played would not have
# Thursday's result either.
#
# Season 2010 is the warm-up. It has no previous season, so its early weeks rest
# on very little and its week 1 is missing outright. Train from 2011 on.

_OFF_GRID_WEEK = -999  # week ordinal for a `Week` label that is not recognized


def _prior_rate(
    records: pd.DataFrame,
    keys: list[str],
    *,
    numerator: str,
    denominator: str,
    seasons: Iterable[int],
) -> pd.Series:
    """`(keys..., Season, week)` -> percentage over the window described above.

    `records` is one row per event, already carrying the two counts to divide
    (`[*keys, "Season", "week", numerator, denominator]`). The result is indexed
    over the **full grid** of keys x seasons x weeks rather than over the rows
    that happen to exist, so it can be read for a week the team did not play (a
    bye) and for a season that has not started yet. A window holding no games at
    all gives missing, never zero -- a rate of zero is one a team has to earn.
    """
    weeks = list(range(int(min(1, records["week"].min())), max(WEEK_ORDINAL.values()) + 1))
    grid = pd.MultiIndex.from_product(
        [*(sorted(records[key].dropna().unique()) for key in keys), sorted(set(seasons)), weeks],
        names=[*keys, "Season", "week"],
    )
    counts = (
        records.groupby([*keys, "Season", "week"])[[numerator, denominator]]
        .sum()
        .reindex(grid, fill_value=0.0)
    )

    # `from_product` varies the week fastest, so the frame is already in
    # chronological order inside each season -- which is what cumsum needs.
    by_season = counts.groupby(level=[*keys, "Season"], sort=False)
    earlier = by_season.cumsum() - counts  # this season, the weeks before this one
    season_total = by_season.transform("sum")  # this season in full

    previous = season_total.reset_index()
    previous["Season"] += 1  # season S-1 becomes the history season S starts with
    previous = previous.set_index([*keys, "Season", "week"]).reindex(grid).fillna(0.0)

    window = earlier + previous
    rate = 100.0 * window[numerator] / window[denominator]
    return rate.where(window[denominator] > 0)


def _game_clock(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """`(season, week)` arrays to look the grid up with, in the frame's row order."""
    season = df["Season"].astype(int).to_numpy()
    week = week_number(df["Week"]).fillna(_OFF_GRID_WEEK).astype(int).to_numpy()
    return season, week


def _read_at(rate: pd.Series, *levels: np.ndarray) -> np.ndarray:
    """Read `rate` once per game, keeping the caller's row order.

    Anything off the grid -- an unknown week label, a franchise with no history
    yet -- comes back missing rather than raising, and the imputer in the
    pipeline decides what to do with it.
    """
    return rate.reindex(pd.MultiIndex.from_arrays(list(levels))).to_numpy()


def win_rate_table(games: pd.DataFrame, *, seasons: Iterable[int] | None = None) -> pd.Series:
    """`(team, venue, Season, week)` -> win rate over the window, in percent.

    A tie counts half a win, the NFL convention notebook `01` uses throughout. A
    game with no score on file contributes to neither side of the fraction.

    `seasons` widens the grid past the seasons `games` covers, which is how a
    season that has not been played yet still gets last season's number.
    """
    home = pd.to_numeric(games["HomeScore"], errors="coerce")
    away = pd.to_numeric(games["AwayScore"], errors="coerce")
    week = week_number(games["Week"])
    played = home.notna() & away.notna() & week.notna()

    def side(team_col: str, own: pd.Series, other: pd.Series, venue: str) -> pd.DataFrame:
        frame = pd.DataFrame(
            {
                "team": canonical_team(games[team_col]),
                "venue": venue,
                "Season": games["Season"].astype(int),
                "week": week,
                "credit": (own > other).astype(float) + 0.5 * (own == other),
                "games": 1.0,
            }
        )
        return frame[played & frame["team"].notna()]

    records = pd.concat(
        [side("HomeTeam", home, away, "home"), side("AwayTeam", away, home, "away")],
        ignore_index=True,
    )
    records["week"] = records["week"].astype(int)  # plain int, to match the grid
    return _prior_rate(
        records,
        ["team", "venue"],
        numerator="credit",
        denominator="games",
        seasons=set(records["Season"]) | set(seasons or ()),
    )


def drive_rate_tables(
    drives: pd.DataFrame, *, seasons: Iterable[int] | None = None
) -> tuple[pd.Series, pd.Series]:
    """`(team, Season, week)` -> `(scoring rate, rate allowed)`, both in percent.

    The first is the share of the team's own drives that ended in a touchdown or
    field goal, the second the same measure on the drives it faced. Both pool
    home and away games: the venue split already has two features of its own,
    and halving the sample here would only make the numbers noisier.

    Input is `data.load_drives`, which has already handed the drives that ended
    in a defensive touchdown back to the offense that ran them.
    """
    week = week_number(drives["Week"])
    frame = pd.DataFrame(
        {
            "Season": drives["Season"].astype(int),
            "week": week,
            "team": drives["team"],
            "opponent": drives["opponent"],
            "scored": drives["scored"].astype(float),
            "drives": 1.0,
        }
    )
    frame = frame[week.notna() & frame["team"].notna() & frame["opponent"].notna()].copy()
    frame["week"] = frame["week"].astype(int)
    grid_seasons = set(frame["Season"]) | set(seasons or ())

    scored = _prior_rate(
        frame, ["team"], numerator="scored", denominator="drives", seasons=grid_seasons
    )
    # The same drives read from the other sideline: a drive this team faced.
    allowed = _prior_rate(
        frame.drop(columns="team").rename(columns={"opponent": "team"}),
        ["team"],
        numerator="scored",
        denominator="drives",
        seasons=grid_seasons,
    )
    return scored, allowed


# The folded tables are the same for every run, and `drive_rates` reads every
# play file to build its own. Cached per grid, and never mutated by the builders
# below -- `reindex` copies.
@lru_cache(maxsize=4)
def _cached_win_rates(seasons: tuple[int, ...]) -> pd.Series:
    return win_rate_table(load_scores(include_preseason=False), seasons=seasons)


@lru_cache(maxsize=4)
def _cached_drive_rates(seasons: tuple[int, ...]) -> tuple[pd.Series, pd.Series]:
    return drive_rate_tables(load_drives(include_preseason=False), seasons=seasons)


# --------------------------------------------------------------------------- #
# builders
# --------------------------------------------------------------------------- #
# `GameDate` carries the month by name and no year at all, which is all a
# calendar month needs: `September 14th` is month 9 in every season.
_MONTH_NUMBER = {
    name.upper(): number
    for number, name in enumerate(
        (
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ),
        start=1,
    )
}

# `GameSlot` mixes the day with the broadcast window: `Sunday Night Football` is
# a Sunday. `day` is the day of the week, so the primetime half is dropped here
# -- it is real information, and a good candidate for a feature of its own.
_WEEKDAYS = frozenset(
    {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}
)


def _month_number(date: pd.Series) -> pd.Series:
    """`August 8th` -> 8. Missing when the month cannot be read."""
    month = date.astype("string").str.strip().str.split().str[0].str.upper()
    return month.map(_MONTH_NUMBER).astype("Float64").astype("Int64")


def _day_of_week(slot: pd.Series) -> pd.Series:
    """`Sunday Night Football` -> `sunday`. `TBD` and anything else -> missing."""
    day = slot.astype("string").str.strip().str.lower().str.split().str[0]
    return day.where(day.isin(_WEEKDAYS)).astype("string")


@builder("calendar")
def _calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """`month`, `week`, `day`, `playoff` -- when the game was played.

    All four come off the game's own row, so there is no history here and no
    leakage to think about. Two of them are worth saying out loud:

    `week` is the ordinal, not the label: 1-18 through the regular season and
    19-22 across the playoff rounds, so a tree can split on "later than week X".
    The text label would be ordered alphabetically by the encoder, which files
    week 10 next to week 1 and drops the Super Bowl in the middle of December.

    `playoff` is the four postseason labels, *not* "after week 17". That rule
    was right until 2020 and stopped being right in 2021, when the regular
    season grew to 18 weeks: it would file 80 regular-season games under
    playoffs. See `data.is_postseason`.
    """
    return pd.DataFrame(
        {
            "month": _month_number(df["GameDate"]),
            "week": week_number(df["Week"]),
            "day": _day_of_week(df["GameSlot"]),
            "playoff": is_postseason(df["Week"]).astype("Int64"),
        },
        index=df.index,
    )


@builder("win_rates")
def _win_rate_features(df: pd.DataFrame) -> pd.DataFrame:
    """`pct_home_win` / `pct_away_win`: how each side does in the venue it is in.

    The home team's number counts its home games only and the away team's its
    away games only -- the split notebook `01` measured the home-field advantage
    on, and the one the `home_win` target has to beat.

    History is read from the whole scores file rather than from `df`, so a
    game's feature value is the same whichever seasons the config asked for.
    """
    season, week = _game_clock(df)
    rate = _cached_win_rates(tuple(sorted(set(season.tolist()))))
    return pd.DataFrame(
        {
            "pct_home_win": _read_at(
                rate,
                canonical_team(df["HomeTeam"]).to_numpy(),
                np.full(len(df), "home"),
                season,
                week,
            ),
            "pct_away_win": _read_at(
                rate,
                canonical_team(df["AwayTeam"]).to_numpy(),
                np.full(len(df), "away"),
                season,
                week,
            ),
        },
        index=df.index,
    )


@builder("drive_rates")
def _drive_rate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Four drive rates: what each side scores on, and what it gives up.

    `*_pct_score_drive` is the share of that team's own drives that ended in a
    touchdown or field goal; `*_pct_allowed_drive` is the same share on the
    drives it faced. Notebook `01` found the first correlates more strongly with
    winning than the second, which is the reason both are here instead of only
    one: the gap between them is something a model can use.

    Reads every play file, which takes a few seconds, so the folded tables are
    cached for the life of the process.
    """
    season, week = _game_clock(df)
    scored, allowed = _cached_drive_rates(tuple(sorted(set(season.tolist()))))
    home = canonical_team(df["HomeTeam"]).to_numpy()
    away = canonical_team(df["AwayTeam"]).to_numpy()
    return pd.DataFrame(
        {
            "home_pct_score_drive": _read_at(scored, home, season, week),
            "home_pct_allowed_drive": _read_at(allowed, home, season, week),
            "away_pct_score_drive": _read_at(scored, away, season, week),
            "away_pct_allowed_drive": _read_at(allowed, away, season, week),
        },
        index=df.index,
    )


def build_dataset(
    df: pd.DataFrame, features: FeatureConfig, target_name: str
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return `(X, y, meta)`.

    `meta` holds the columns that do not feed the model but are needed later
    (season for the split, game identification for inspection). Rows with a
    missing target are dropped.
    """
    df = apply_builders(df, features.builders)
    y = build_target(target_name, df)

    missing = [c for c in features.columns if c not in df.columns]
    if missing:
        raise KeyError(f"columns missing from the data: {missing}")

    meta_cols = [c for c in ("Season", "Week", "HomeTeam", "AwayTeam") if c in df.columns]

    keep = y.notna()
    X = df.loc[keep, features.columns].copy()
    meta = df.loc[keep, meta_cols].copy()
    y = y[keep]

    if features.categorical:
        X[features.categorical] = X[features.categorical].astype("string")
    if features.numeric:
        X[features.numeric] = X[features.numeric].apply(pd.to_numeric, errors="coerce")

    return X.reset_index(drop=True), y.reset_index(drop=True), meta.reset_index(drop=True)


def make_preprocessor(features: FeatureConfig) -> ColumnTransformer:
    """Lean preprocessor: impute numeric columns, encode categorical ones."""
    transformers: list[tuple[str, object, list[str]]] = []

    if features.numeric:
        if features.missing_strategy == "keep":
            numeric_step: object = "passthrough"
        elif features.missing_strategy == "sentinel":
            numeric_step = SimpleImputer(strategy="constant", fill_value=-999.0)
        else:
            numeric_step = SimpleImputer(strategy=features.missing_strategy)
        transformers.append(("numeric", numeric_step, features.numeric))

    if features.categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "encode",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                                encoded_missing_value=-2,
                            ),
                        ),
                    ]
                ),
                features.categorical,
            )
        )

    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def feature_names(preprocessor: ColumnTransformer, features: FeatureConfig) -> list[str]:
    """Column names on the preprocessor output."""
    try:
        return [str(n) for n in preprocessor.get_feature_names_out()]
    except Exception:  # preprocessor not fitted yet
        return list(features.columns)

"""Targets, row filters, feature builders, matrix assembly and splitting."""

from __future__ import annotations

import pandas as pd
import pytest

from nfl_trees.config import FeatureConfig, SplitConfig
from nfl_trees.data import (
    TARGETS,
    build_drives,
    build_target,
    canonical_team,
    filter_status,
    filter_weeks,
    is_postseason,
    load_drives,
    load_scores,
    week_number,
)
from nfl_trees.features import (
    apply_builders,
    build_dataset,
    drive_rate_tables,
    make_preprocessor,
    win_rate_table,
)
from nfl_trees.split import split_indices


def test_home_win_target_is_registered():
    assert "home_win" in TARGETS


def test_unknown_target(scores):
    with pytest.raises(KeyError, match="unknown target"):
        build_target("does_not_exist", scores)


def test_home_win_compares_the_scores(scores):
    y = build_target("home_win", scores)
    valid = y.notna()
    expected = scores.loc[valid, "HomeScore"] > scores.loc[valid, "AwayScore"]
    assert (y[valid].astype(bool) == expected).all()


def test_home_win_ignores_ties_and_games_without_a_score(scores):
    y = build_target("home_win", scores)
    assert y.isna().iloc[0]  # 21-21 tie
    assert y.isna().iloc[1]  # game with no score


def test_status_filter_keeps_only_games_that_were_played(scores):
    """`BYE` and `Clinched Playoffs` are rows for a team without an opponent."""
    import numpy as np
    import pandas as pd

    # Reuse three real rows and change only what characterizes a non-game, so
    # the DataFrame keeps the same dtypes.
    extra = scores.head(3).copy()
    extra["GameStatus"] = ["BYE", "Clinched Playoffs", "TBD"]
    extra[["HomeScore", "AwayScore"]] = np.nan
    full = pd.concat([scores, extra], ignore_index=True)

    played = filter_status(full, ("FINAL",))
    assert set(played["GameStatus"]) == {"FINAL"}
    assert len(played) == len(full) - 4  # 3 new rows + the fixture's TBD row

    scheduled = filter_status(full, ("TBD",))
    assert len(scheduled) == 2


def test_week_filter_drops_preseason(scores):
    filtered = filter_weeks(scores, include_preseason=False, include_postseason=True)
    assert not filtered["Week"].str.contains("PRESEASON").any()

    with_preseason = filter_weeks(scores, include_preseason=True, include_postseason=True)
    assert len(with_preseason) > len(filtered)


def test_week_filter_also_applies_to_plays(plays):
    filtered = filter_weeks(plays, include_preseason=False, include_postseason=False)
    assert not filtered["Week"].str.contains("PRESEASON|SUPER BOWL").any()


def test_build_dataset_drops_rows_with_missing_target(scores):
    features = FeatureConfig(categorical=["HomeTeam", "AwayTeam"])
    X, y, meta = build_dataset(scores, features, "home_win")
    assert len(X) == len(y) == len(meta)
    assert y.notna().all()
    assert len(X) < len(scores)  # the tie and the scoreless game are gone
    assert list(X.columns) == ["HomeTeam", "AwayTeam"]
    assert "Season" in meta.columns


def test_build_dataset_complains_about_a_missing_column(scores):
    features = FeatureConfig(numeric=["ColumnThatDoesNotExist"])
    with pytest.raises(KeyError, match="columns missing"):
        build_dataset(scores, features, "home_win")


def test_preprocessor_produces_a_numeric_matrix(scores):
    features = FeatureConfig(categorical=["HomeTeam", "AwayTeam"])
    X, _, _ = build_dataset(scores, features, "home_win")
    matrix = make_preprocessor(features).fit_transform(X)
    assert matrix.shape == (len(X), 2)


def test_season_split_does_not_mix_years(scores):
    features = FeatureConfig(categorical=["HomeTeam"])
    _, _, meta = build_dataset(scores, features, "home_win")
    split = SplitConfig(strategy="season", train_seasons=[2022, 2023], test_seasons=[2024])
    train_idx, test_idx = split_indices(meta, split)

    assert set(meta["Season"].iloc[train_idx]) == {2022, 2023}
    assert set(meta["Season"].iloc[test_idx]) == {2024}
    assert not set(train_idx) & set(test_idx)


def test_season_split_without_test_seasons(scores):
    features = FeatureConfig(categorical=["HomeTeam"])
    _, _, meta = build_dataset(scores, features, "home_win")
    with pytest.raises(ValueError, match="test_seasons"):
        split_indices(meta, SplitConfig(strategy="season"))


def test_unknown_builder_fails(scores):
    features = FeatureConfig(categorical=["HomeTeam"], builders=["recent_form"])
    with pytest.raises(KeyError, match="unknown builder"):
        build_dataset(scores, features, "home_win")


# --------------------------------------------------------------------------- #
# the season calendar
# --------------------------------------------------------------------------- #
def test_week_number_orders_a_season_the_way_it_was_played():
    labels = pd.Series(["PRESEASON WEEK 1", "WEEK 1", "WEEK 18", "WILD CARD WEEKEND", "SUPER BOWL"])
    assert list(week_number(labels)) == [-4, 1, 18, 19, 22]


def test_week_number_is_missing_for_a_label_it_does_not_know():
    assert week_number(pd.Series(["PRO BOWL", "week nine"])).isna().all()


def test_the_playoff_flag_leaves_week_18_alone():
    """Week 18 has been regular season since 2021: "after week 17" mislabels it."""
    labels = pd.Series(["WEEK 17", "WEEK 18", "WILD CARD WEEKEND", "SUPER BOWL"])
    assert list(is_postseason(labels)) == [False, False, True, True]


def test_canonical_team_folds_the_franchises_that_moved():
    moved = pd.Series(["STL", "SD", "OAK", "JAC", "AZ", "GB"])
    assert list(canonical_team(moved)) == ["LAR", "LAC", "LV", "JAX", "ARI", "GB"]


def test_calendar_builder_reads_the_game_row(scores):
    out = apply_builders(scores, ["calendar"])
    assert (out["month"] == 9).all()  # every fixture row is in September
    assert set(out["day"]) == {"sunday", "monday", "thursday"}
    assert (out.loc[out["Week"] == "WEEK 5", ["week", "playoff"]] == [5, 0]).all().all()
    assert (out.loc[out["Week"] == "SUPER BOWL", ["week", "playoff"]] == [22, 1]).all().all()


def test_day_drops_the_broadcast_window():
    """`Sunday Night Football` is a Sunday; `TBD` is not a day at all."""
    frame = pd.DataFrame(
        {
            "Season": 2024,
            "Week": "WEEK 1",
            "GameDate": "September 8th",
            "GameSlot": ["Sunday Night Football", "Sunday", "Monday Night Football", "TBD"],
        }
    )
    day = apply_builders(frame, ["calendar"])["day"]
    assert list(day[:3]) == ["sunday", "sunday", "monday"]
    assert pd.isna(day.iloc[3])


def test_month_comes_from_a_date_carrying_no_year():
    frame = pd.DataFrame(
        {
            "Season": 2024,
            "Week": ["WEEK 1", "WEEK 17", "SUPER BOWL"],
            "GameDate": ["September 8th", "December 29th", "February 9th"],
            "GameSlot": "Sunday",
        }
    )
    assert list(apply_builders(frame, ["calendar"])["month"]) == [9, 12, 2]


# --------------------------------------------------------------------------- #
# the window the team rates are computed over
# --------------------------------------------------------------------------- #
def _league(seasons=(2023, 2024)) -> pd.DataFrame:
    """Two seasons of a four-team league, one game a week, no ties.

    Every team hosts every other one, so each is home three times and away
    three times per season. The scores are rigged by position in the list:
    `AAA` loses every game and `DDD` wins every game, which makes the expected
    rate of any window a number the test can write down.
    """
    teams = ["AAA", "BBB", "CCC", "DDD"]
    rows = []
    for season in seasons:
        week = 0
        for home in teams:
            for away in teams:
                if home == away:
                    continue
                week += 1
                rows.append(
                    {
                        "Season": season,
                        "Week": f"WEEK {week}",
                        "GameStatus": "FINAL",
                        "GameSlot": "Sunday",
                        "GameDate": "October 1st",
                        "HomeTeam": home,
                        "AwayTeam": away,
                        "HomeScore": 20.0 + teams.index(home),
                        "AwayScore": 20.0 + teams.index(away),
                    }
                )
    return pd.DataFrame(rows)


def test_win_rate_window_is_last_season_plus_this_one_so_far():
    rate = win_rate_table(_league())

    # Week 1 of the first season has nothing behind it.
    assert pd.isna(rate.loc[("AAA", "home", 2023, 1)])

    # Week 1 of the second season already has one: the whole of the first.
    assert rate.loc[("AAA", "home", 2024, 1)] == pytest.approx(0.0)
    assert rate.loc[("DDD", "home", 2024, 1)] == pytest.approx(100.0)


def test_win_rate_never_sees_the_game_it_describes():
    """The whole point of the window, stated as a test.

    Flipping the result of a game cannot move the feature value of that same
    game -- and *must* move the following week's, or the test would pass on a
    table that simply ignores the score.
    """
    games = _league()
    before = win_rate_table(games)

    flipped = games.copy()
    row = flipped.index[flipped["Week"] == "WEEK 9"][0]
    flipped.loc[row, ["HomeScore", "AwayScore"]] = flipped.loc[
        row, ["AwayScore", "HomeScore"]
    ].to_numpy()
    after = win_rate_table(flipped)

    host = games.loc[row, "HomeTeam"]
    assert after.loc[(host, "home", 2023, 9)] == before.loc[(host, "home", 2023, 9)]
    assert after.loc[(host, "home", 2023, 10)] != before.loc[(host, "home", 2023, 10)]


def test_a_tie_counts_half_a_win():
    games = _league(seasons=(2023,))
    games.loc[games["Week"] == "WEEK 1", "AwayScore"] = games.loc[
        games["Week"] == "WEEK 1", "HomeScore"
    ]
    rate = win_rate_table(games)
    # AAA hosts weeks 1-3 and loses 2 and 3, so week 4 looks back on 0.5 of 3.
    assert rate.loc[("AAA", "home", 2023, 4)] == pytest.approx(100 * 0.5 / 3)


def test_a_window_with_no_games_behind_it_is_missing_not_zero():
    """Zero is a rate a team has to earn by losing, not one it starts with."""
    rate = win_rate_table(_league())
    assert pd.isna(rate.loc[("BBB", "home", 2023, 1)])


def _drive_log() -> pd.DataFrame:
    """Two seasons in which `AAA` scores on half its drives and `BBB` on none."""
    rows = []
    for season in (2023, 2024):
        for week in (1, 2):
            for team, opponent, scored in (
                ("AAA", "BBB", 1),
                ("AAA", "BBB", 0),
                ("BBB", "AAA", 0),
                ("BBB", "AAA", 0),
            ):
                rows.append(
                    {
                        "Season": season,
                        "Week": f"WEEK {week}",
                        "team": team,
                        "opponent": opponent,
                        "plays": 6,
                        "scored": scored,
                    }
                )
    return pd.DataFrame(rows)


def test_drive_rates_are_one_table_read_from_two_sidelines():
    """What a team gives up is what its opponents scored on."""
    scored, allowed = drive_rate_tables(_drive_log())
    assert scored.loc[("AAA", 2024, 1)] == pytest.approx(50.0)
    assert allowed.loc[("AAA", 2024, 1)] == pytest.approx(0.0)
    assert scored.loc[("BBB", 2024, 1)] == pytest.approx(0.0)
    assert allowed.loc[("BBB", 2024, 1)] == pytest.approx(50.0)


def test_drive_rates_can_be_read_for_a_week_the_team_did_not_play():
    """The grid is dense, so a bye -- or a season not started yet -- still answers."""
    scored, _ = drive_rate_tables(_drive_log(), seasons=[2025])
    assert scored.loc[("AAA", 2024, 7)] == pytest.approx(50.0)
    assert scored.loc[("AAA", 2025, 1)] == pytest.approx(50.0)


# --------------------------------------------------------------------------- #
# folding plays into drives
# --------------------------------------------------------------------------- #
def _play(**overrides) -> dict:
    play = {
        "Season": 2024,
        "Week": "WEEK 1",
        "AwayTeam": "BAL",
        "HomeTeam": "KC",
        "Quarter": "1st Quarter",
        "DriveNumber": 1,
        "TeamWithPossession": "Kansas City Chiefs",
        "PlayOutcome": "5 Yard Pass",
        "PlayDescription": "P.Mahomes pass short right",
    }
    return {**play, **overrides}


def test_build_drives_hands_a_pick_six_back_to_the_offense():
    """The trap in `plays`: the source credits the whole drive to the defense."""
    plays = pd.DataFrame(
        [
            _play(TeamWithPossession="Baltimore Ravens"),
            _play(TeamWithPossession="Baltimore Ravens"),
            _play(
                TeamWithPossession="Baltimore Ravens",
                PlayOutcome="Touchdown",
                PlayDescription="P.Mahomes pass INTERCEPTED by M.Humphrey, 40 yards, TOUCHDOWN",
            ),
        ]
    )
    drives = build_drives(plays)
    assert len(drives) == 1
    assert drives.loc[0, "team"] == "KC"  # the offense that actually ran it
    assert drives.loc[0, "opponent"] == "BAL"
    assert drives.loc[0, "scored"] == 0  # the possession ended in a turnover


def test_build_drives_does_not_touch_the_frame_it_was_given():
    plays = pd.DataFrame([_play()])
    columns = list(plays.columns)
    build_drives(plays)
    assert list(plays.columns) == columns


def test_a_drive_number_is_only_unique_inside_its_quarter():
    """`DriveNumber` restarts every quarter, so the quarter belongs in the key."""
    plays = pd.DataFrame(
        [
            _play(Quarter="1st Quarter", PlayOutcome="Touchdown"),
            _play(Quarter="2nd Quarter", PlayOutcome="Punt"),
        ]
    )
    drives = build_drives(plays)
    assert len(drives) == 2
    assert list(drives["scored"]) == [1, 0]


def test_only_a_touchdown_or_a_field_goal_scores_a_drive():
    """An extra point is automatic and a safety belongs to the other team."""
    for outcome in ("Extra Point", "Safety", "Field Goal No Good"):
        drives = build_drives(pd.DataFrame([_play(PlayOutcome=outcome)]))
        assert drives.loc[0, "scored"] == 0, outcome
    for outcome in ("Touchdown", "Field Goal"):
        drives = build_drives(pd.DataFrame([_play(PlayOutcome=outcome)]))
        assert drives.loc[0, "scored"] == 1, outcome


# --------------------------------------------------------------------------- #
# the builders against the real CSVs
# --------------------------------------------------------------------------- #
@pytest.mark.requires_data
def test_win_rate_builder_carries_last_season_into_week_1():
    games = load_scores([2024])
    out = apply_builders(games, ["win_rates"])
    assert out[["pct_home_win", "pct_away_win"]].notna().all().all()
    assert out["pct_home_win"].between(0, 100).all()


@pytest.mark.requires_data
def test_drive_rate_builder_puts_each_rate_on_the_right_side():
    """Guards the wiring: a swap between the four columns would survive everything above."""
    games = load_scores([2024])
    out = apply_builders(games, ["drive_rates"])
    drives = load_drives([2023])  # week 1 of 2024 looks back on the whole of 2023

    opener = out[out["Week"] == "WEEK 1"].iloc[0]
    for prefix, team_col in (("home", "HomeTeam"), ("away", "AwayTeam")):
        team = canonical_team(pd.Series([opener[team_col]])).iloc[0]
        own = drives[drives["team"] == team]
        faced = drives[drives["opponent"] == team]
        assert opener[f"{prefix}_pct_score_drive"] == pytest.approx(100 * own["scored"].mean())
        assert opener[f"{prefix}_pct_allowed_drive"] == pytest.approx(100 * faced["scored"].mean())

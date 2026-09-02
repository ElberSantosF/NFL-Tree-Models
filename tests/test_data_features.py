"""Targets, row filters, feature matrix assembly and splitting."""

from __future__ import annotations

import pytest

from nfl_trees.config import FeatureConfig, SplitConfig
from nfl_trees.data import TARGETS, build_target, filter_status, filter_weeks
from nfl_trees.features import build_dataset, make_preprocessor
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

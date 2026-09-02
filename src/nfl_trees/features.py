"""Feature matrix assembly and preprocessing.

Tree models need neither scaling nor normalization, so the preprocessing here
is deliberately thin: imputation for numeric columns and ordinal encoding for
categorical ones (a tree can separate categories through successive splits, no
one-hot required).

New feature engineering goes into `FEATURE_BUILDERS`: a function
`DataFrame -> DataFrame` (the new columns only) referenced from
`features.builders` in the YAML. The dict starts empty: every feature is a
deliberate decision.

A rule that holds for every builder in this project: team-level aggregation
must use **only games that happened earlier** than the one being predicted. A
full-season average includes the game itself and leaks the result.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from .config import FeatureConfig
from .data import build_target

# Name -> function that takes the raw DataFrame and returns a DataFrame with
# the derived columns only. Empty on purpose: each new feature is an experiment.
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

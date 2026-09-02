"""Train/test partitioning.

The temporal grain of the data is the season. Shuffling 2010 with 2025 trains
the model on information from the future relative to the test set, so the
default is a season holdout. `random` exists only as a teaching comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SplitConfig


def split_indices(
    meta: pd.DataFrame, split: SplitConfig, *, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Return `(train_idx, test_idx)` as positional index arrays."""
    if split.strategy == "season":
        return _season_split(meta, split)
    if split.strategy == "random":
        return _random_split(len(meta), split, seed)
    raise ValueError(f"invalid split strategy '{split.strategy}'; use 'season' or 'random'")


def _season_split(meta: pd.DataFrame, split: SplitConfig) -> tuple[np.ndarray, np.ndarray]:
    if "Season" not in meta.columns:
        raise KeyError("the season split requires a 'Season' column in the data")
    seasons = meta["Season"].astype(int)

    if not split.test_seasons:
        raise ValueError("split.test_seasons is empty: set the test seasons in the config")

    test_mask = seasons.isin(split.test_seasons)
    if split.train_seasons:
        train_mask = seasons.isin(split.train_seasons)
    else:
        # With no explicit list, train on everything that is not test.
        train_mask = ~test_mask

    train_idx = np.flatnonzero(train_mask.to_numpy())
    test_idx = np.flatnonzero(test_mask.to_numpy())
    for name, idx in (("train", train_idx), ("test", test_idx)):
        if idx.size == 0:
            raise ValueError(f"the season split produced an empty {name} set")
    return train_idx, test_idx


def _random_split(n: int, split: SplitConfig, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    cut = int(round(n * (1 - split.test_size)))
    return np.sort(idx[:cut]), np.sort(idx[cut:])


def describe(meta: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray) -> dict[str, object]:
    """Summary of the split, stored alongside the run metrics."""
    info: dict[str, object] = {"n_train": int(train_idx.size), "n_test": int(test_idx.size)}
    if "Season" in meta.columns:
        info["train_seasons"] = sorted(meta["Season"].iloc[train_idx].unique().tolist())
        info["test_seasons"] = sorted(meta["Season"].iloc[test_idx].unique().tolist())
    return info

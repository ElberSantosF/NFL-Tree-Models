# Architecture

## Goal

Predict the **winner of a game** from the 2010–2026 history, using tree
models — and, along the way, document what each model does.

Two things are deliberately left empty, to be filled in as the study moves
forward:

- **the model catalog starts empty** — each model is added when it is studied
  (`models.py`);
- **the feature list starts empty** — each feature is added when it is defined
  (`features.py`).

Everything around them is in place and tested.

## The core idea

An experiment is a **configuration file**, not a script. The package code is
fixed; what changes from one experiment to the next is the YAML in `configs/`.

This solves the classic problem of a study repository: ten similar notebooks,
each with a variation of the pipeline, and no way to tell which result came
from which combination. Here the answer is always the YAML plus the folder
under `results/`.

## The path of a run

```
configs/decision_tree.yaml
        │
        ▼
   config.py ......... validates the YAML and turns it into dataclasses
        │
        ▼
    data.py .......... reads the CSVs, filters out non-games, builds the target
        │
        ▼
 features.py ......... selects columns, applies builders, assembles the preprocessor
        │
        ▼
   split.py .......... separates train/test by season (no future leaking in)
        │
        ▼
  models.py .......... instantiates the requested estimator (MODELS)
        │
        ▼
 metrics.py .......... computes the metrics declared in the config
        │
        ▼
experiment.py ........ orchestrates it all and writes results/<name>/ + leaderboard.csv
```

`cli.py` is just the terminal entry point to that.

## Modules

| Module | Responsibility | What it does **not** do |
| --- | --- | --- |
| `paths.py` | Every project path in one place. | Read or write data. |
| `config.py` | Read the YAML, validate it, reject unknown fields. | Know what an NFL season is. |
| `data.py` | Read CSVs, filter weeks and statuses, define targets. | Feature engineering. |
| `features.py` | Assemble `X`, `y` and the preprocessing. | Train. |
| `split.py` | Decide which rows are train and which are test. | Evaluate. |
| `models.py` | Model catalog and its defaults. | Know anything about the data. |
| `metrics.py` | Compute metrics and know the direction of each. | Save anything. |
| `experiment.py` | Orchestrate the run and persist the artifacts. | Domain logic. |
| `cli.py` | Terminal interface. | Any logic of its own. |

The rule that keeps this clean: **every module has a registry, and the YAML
refers to its entries by name.** No module imports `experiment.py`.

## The four extension points

Virtually every new experiment falls into one of these four cases:

| I want to try... | Where I touch | How I use it |
| --- | --- | --- |
| a new model | `models.py` → `@register(...)` | `model.type: <name>` |
| a new feature | `features.py` → `@builder(...)` | `features.builders: [<name>]` |
| a new response variable | `data.py` → `@target(...)` | `data.target: <name>` |
| a new metric | `metrics.py` → `_register(...)` | `evaluation.metrics: [<name>]` |

If a new experiment requires touching `experiment.py`, an extension point is
probably missing — worth stopping to think before adding an `if`.

### Anatomy of a feature builder

A builder takes the raw DataFrame and returns **only the new columns**:

```python
@builder("recent_form")
def _recent_form(df: pd.DataFrame) -> pd.DataFrame:
    """Wins by the home team in the N games before this one."""
    ...
    return pd.DataFrame({"home_wins_last5": ..., "away_wins_last5": ...})
```

Then name it in `features.builders` and list the generated columns under
`features.numeric` / `features.categorical`.

The caveat worth repeating: team-level aggregation must use **only games that
happened earlier** than the one being predicted. A full-season average
includes the game itself and leaks the result.

## Design decisions (and why)

**The season split is the default.** Shuffling 2010 with 2025 trains the model
on information from the future relative to the test set. `strategy: random`
exists in the config, but it is there to *show* that difference, not to draw
conclusions from.

**`load_scores` filters `GameStatus`.** The scores file contains rows that are
not games (`BYE`, `Clinched Playoffs`) and games not played yet (`TBD`) — 11%
of the total. Without that filter they would enter training as games with no
score. See [data.md](data.md).

**Minimal preprocessing.** Trees do not care about scale and are indifferent to
monotonic transformations of a numeric variable. So: imputation for numeric
columns, ordinal encoding for categorical ones, and nothing else. One-hot on a
team column (32 categories) would only fragment the splits.

**No `.fit()` before the split.** The preprocessor lives inside the
scikit-learn `Pipeline`, so imputation and encoding are fitted on the training
set only. That is what prevents silent leakage.

**A run overwrites the previous one of the same name.** `results/<name>/` is
the current state of that experiment, and `leaderboard.csv` holds one row per
experiment (not per execution). To keep two variants, use two config names —
history lives in Git, not in timestamped folders.

**A model is an optional dependency.** `python -m nfl_trees models` shows an
`ok` column telling which libraries are installed. The catalog works without
xgboost/lightgbm in the environment.

## Directory layout

```
nfl-tree-models/
├── configs/              # one YAML per experiment (starts empty)
├── data/
│   ├── raw/              # the CSVs: scores/, plays/, plays_by_week/
│   └── processed/         # derived data, if you generate any
├── docs/
│   ├── architecture.md   # this file
│   ├── config-reference.md
│   ├── data.md
│   └── models/           # _template.md + one .md per model in the catalog
├── notebooks/            # exploration (not where experiments live)
├── results/              # run output + leaderboard.csv (outside Git)
├── src/nfl_trees/        # the package
└── tests/                # fast tests, no dependency on the CSVs
```

## What was deliberately left out

Things a "production" workbench would have that would be dead weight here:

- **MLflow / W&B** — `results/leaderboard.csv` is enough for the number of runs
  this project will see. If it ever passes a few dozen, worth switching.
- **Optuna** — tuning comes after an honest baseline exists. Until then, adjust
  `params` in the YAML and compare on the leaderboard.
- **SHAP** — `importances.csv` already comes out of every run; finer
  interpretation fits in notebook `04`.
- **A prediction command** — predicting the upcoming round (`GameStatus == TBD`)
  comes after the first trained model. `load_scores(statuses=("TBD",))` already
  hands over those games.
- **DVC / Git LFS** — the CSVs are small enough to live in Git.

# nfl-tree-models

A workbench for experimenting with **tree models** to predict the **winner of
NFL games**, using data from 2010 to 2026.

A study and portfolio repository. The goal is to have one place where tree
models can be compared under identical conditions, and where what each model
does gets documented as it is studied.

## The idea in one sentence

> An experiment is a YAML file, not a script.

The package code is fixed. What changes from one experiment to the next is the
config in `configs/`. That keeps runs comparable and results traceable: every
metric came from a versioned YAML, not from notebook cells that have since been
overwritten.

## Current state

The workbench is in place and tested. One thing is still **empty on purpose**,
to be filled in as the study moves forward:

| | State | Where it goes |
| --- | --- | --- |
| Model catalog | empty | `@register` in [models.py](src/nfl_trees/models.py) |
| Features | 3 builders, 10 columns | `@builder` in [features.py](src/nfl_trees/features.py) |
| Target | `home_win` ready | `@target` in [data.py](src/nfl_trees/data.py) |
| Data | 2010–2026 organized | [data/raw/](data/raw/README.md) |

## Installation

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate on Linux/macOS
pip install -e .                # the package and its core (numpy, pandas, scikit-learn)
pip install -e ".[boosting]"    # optional: xgboost and lightgbm
pip install -e ".[notebooks]"   # optional: matplotlib and jupyterlab (notebooks/)
pip install -e ".[dev]"         # optional: pytest and ruff
```

## Usage

```bash
python -m nfl_trees models              # model catalog and what is installed
python -m nfl_trees configs             # available experiments
python -m nfl_trees run decision_tree   # run one experiment
python -m nfl_trees run --all           # run every config
python -m nfl_trees leaderboard         # compare the runs
```

In a notebook or script:

```python
from nfl_trees import ExperimentConfig, run

result = run(ExperimentConfig.load("decision_tree"))
result.metrics          # {'roc_auc': ..., 'accuracy': ...}
result.importances      # DataFrame: feature -> importance
```

Every run writes `results/<name>/` with `metrics.json`, `predictions.csv`,
`importances.csv` and `model.joblib`, plus one row in
`results/leaderboard.csv`.

## First steps

1. **Register the first model** in `src/nfl_trees/models.py` — start with a
   decision tree, which is both the comparison floor and the only model you can
   draw in full. The module itself carries a ready example.
2. **Create its config** in `configs/decision_tree.yaml` — starting point in
   [configs/README.md](configs/README.md).
3. **Document it** by copying `docs/models/_template.md`.
4. `python -m nfl_trees run decision_tree`

The features are already there — the block to paste into the config is below.

## The features

Three builders, ten columns, all at game grain (`data.source: scores`):

| Column | Type | What it is |
| --- | --- | --- |
| `pct_home_win` | numeric | the home team's win rate **at home** |
| `pct_away_win` | numeric | the away team's win rate **on the road** |
| `home_pct_score_drive` | numeric | share of the home team's drives that ended in points |
| `home_pct_allowed_drive` | numeric | share of the drives it faced that ended in points |
| `away_pct_score_drive` | numeric | the same two, measured on the away team |
| `away_pct_allowed_drive` | numeric | " |
| `month` | numeric | calendar month, 1–12 |
| `week` | numeric | 1–18 through the regular season, 19–22 across the playoff rounds |
| `day` | categorical | `sunday`, `monday`, `thursday`, … |
| `playoff` | numeric | 1 in a postseason game, 0 otherwise |

The first six are history, and history is where leakage lives. All six are
computed over the same window: **the previous season in full, plus the current
season up to the week before this game**. A tie counts half a win (the NFL
convention), and a drive that ended in a defensive touchdown is handed back to
the offense that actually ran it — see [docs/data.md](docs/data.md).

Three things to know before running anything on them:

- **Season 2010 is a warm-up.** It has no previous season, so its week 1 comes
  out missing and its early weeks rest on very little. Train from 2011 on.
- **`playoff` is the four postseason labels, not "after week 17".** That rule
  was right until 2020 and stopped being right in 2021, when the regular season
  grew to 18 weeks — it would file 80 regular-season games under playoffs.
- **`week` is the ordinal, not the label.** A tree can split on "later than
  week X"; the text label would be sorted alphabetically by the encoder, which
  files week 10 next to week 1.

The config block that uses all of them:

```yaml
features:
  numeric:
    - pct_home_win
    - pct_away_win
    - home_pct_score_drive
    - home_pct_allowed_drive
    - away_pct_score_drive
    - away_pct_allowed_drive
    - month
    - week
    - playoff
  categorical: [day]
  builders: [calendar, win_rates, drive_rates]
```

## Layout

```
configs/          one YAML per experiment (this is where you work)
data/raw/         the CSVs: scores/, plays/, plays_by_week/
src/nfl_trees/    the package: config, data, features, split, models, metrics, runner
docs/             architecture, data dictionary, one .md per model
notebooks/        exploration (experiments do not live here)
results/          run output (outside Git)
tests/            fast tests, they run without the CSVs
```

## The four extension points

Almost every new experiment falls into one of these cases:

| I want to try... | Where I register it | How I use it in the YAML |
| --- | --- | --- |
| a new model | `models.py` → `@register` | `model.type: <name>` |
| a new feature | `features.py` → `@builder` | `features.builders: [<name>]` |
| a new response variable | `data.py` → `@target` | `data.target: <name>` |
| a new metric | `metrics.py` → `_register` | `evaluation.metrics: [<name>]` |

## Three things worth knowing before running

**The split is by season, not random.** Shuffling 2010 with 2025 trains the
model on information from the future relative to the test set.
`strategy: random` exists in the config to *show* that difference, not to
report results from.

**Not every row of the scores file is a game.** `GameStatus` has `BYE`,
`Clinched Playoffs` and `TBD` besides `FINAL` — 885 of the 6185 rows, 14% of
the file. The loader filters that out by default.

**Leakage is the easy mistake here.** A team-strength feature must be computed
from games **earlier** than the one being predicted. A full-season average
includes the game itself. The two history builders share one window for that,
and `test_win_rate_never_sees_the_game_it_describes` is the test that says so.

## Tests

```bash
pytest
```

The tests use synthetic DataFrames with the same schema as the CSVs, so they
run without the real data.

## Documentation

| Document | Content |
| --- | --- |
| [docs/architecture.md](docs/architecture.md) | how the pieces fit together and why they are that way |
| [docs/config-reference.md](docs/config-reference.md) | every YAML field, valid values, common errors |
| [docs/data.md](docs/data.md) | dictionary of both sources and their gotchas |
| [docs/models/](docs/models/README.md) | how to add a model + the documentation template |
| [data/raw/README.md](data/raw/README.md) | CSV layout |
| [configs/README.md](configs/README.md) | how to write a config |

# Config reference

Every experiment is a self-contained YAML file in [`configs/`](../configs/). An
unknown field is an **error**, not a warning — if you type `test_fraction`
instead of `test_size`, the run fails right away instead of silently using the
default.

## Full annotated config

```yaml
name: decision_tree              # required. Also the folder name under results/
description: >                   # free text: what this run is testing
  Baseline: the first tree, to get a floor for comparison.
task: classification             # classification | regression
seed: 42                         # fixes model randomness and the random split

data:
  source: scores                 # scores (one row per game) | plays (one per play)
  target: home_win               # a name registered in nfl_trees.data.TARGETS
  seasons: [2015, 2016, 2017]    # seasons to load. Empty = all available
  include_preseason: false       # preseason rosters differ; out by default
  include_postseason: true       # playoffs are in by default
  sample_rows: null              # sample N rows (dev only; null = everything)

features:
  numeric: []                    # columns treated as numbers
  categorical: [HomeTeam]        # columns treated as categories
  builders: []                   # names in nfl_trees.features.FEATURE_BUILDERS
  missing_strategy: median       # median | mean | most_frequent | sentinel | keep

split:
  strategy: season               # season (recommended) | random
  train_seasons: [2015, 2016]    # empty + season strategy = everything that is not test
  test_seasons: [2017]           # required with the season strategy
  test_size: 0.2                 # only used by the random strategy
  cv_folds: 0                    # >0 cross-validates the primary metric on train

model:
  type: decision_tree            # required; must be registered in nfl_trees.models
  params:                        # passed straight to the estimator
    max_depth: 6

evaluation:
  metrics: [roc_auc, accuracy]   # names registered in nfl_trees.metrics
  primary_metric: roc_auc        # must be in the list above; sorts the leaderboard
  threshold: 0.5                 # probability cutoff for the predicted class
  save_predictions: true         # writes results/<name>/predictions.csv
  save_importances: true         # writes results/<name>/importances.csv
```

## Valid values

**`data.source`** — `scores` (game grain, the one matching the project goal) or
`plays` (play grain). See [data.md](data.md).

**`data.target`** — registered in `nfl_trees/data.py`:

| Target | Grain | Task | What it is |
| --- | --- | --- | --- |
| `home_win` | game | classification | the home team won (ties and unplayed games drop out) |

To add another (point margin, favorite winning, three classes including the
tie), register a function in `data.py`:

```python
@target("home_margin")
def _home_margin(df):
    """Regression: home team points minus away team points."""
    return (df["HomeScore"] - df["AwayScore"]).astype("Float64")
```

**`model.type`** — required, and must be registered in the catalog.
`python -m nfl_trees models` lists what exists. See [models/](models/README.md).

**`features.builders`** — names registered with `@builder` in `features.py`.
Starts empty: features are the next step of the project.

**`evaluation.metrics`**

| Task | Metrics |
| --- | --- |
| classification | `accuracy`, `balanced_accuracy`, `precision`, `recall`, `f1`, `roc_auc`, `pr_auc`, `log_loss`, `brier` |
| regression | `rmse`, `mae`, `medae`, `r2` |

For predicting a winner, `roc_auc` measures ranking (does the model put the
games the home team won above the rest?) while `log_loss` / `brier` measure
whether the **probability** is calibrated. `accuracy` alone hides both.

**`features.missing_strategy`**

| Value | Effect |
| --- | --- |
| `median` | impute the training median (default) |
| `mean` / `most_frequent` | the other `SimpleImputer` strategies |
| `sentinel` | fill with `-999`, letting the model treat "missing" as its own value |
| `keep` | keep `NaN` — only works with models that handle missing natively (`HistGradientBoosting`, XGBoost, LightGBM) |

Any other value is rejected when the config loads, not halfway through the run.
The same holds for `data.source` and `split.strategy`.

## Caveats that keep results honest

**Target leakage.** To predict the winner, no feature may depend on what
happened in the game itself. `HomeScore` and `AwayScore` are obvious, but the
subtle case is an aggregated feature: "team's average points this season"
computed over the whole season includes the very game you are predicting. The
aggregation has to use **earlier games only**.

**A test season in the past of the training set.** The validator rejects a
season listed on both sides, but it does not reject training on 2024 and
testing on 2015. Keep the test set in the future of the training set.

**`sample_rows` in a final result.** It is a development shortcut. Remove it
before reporting a metric.

**`strategy: random` at play grain.** Plays from the same drive would land in
train and test at once. At game grain the problem is smaller, but the random
split still shuffles seasons — use `season`.

**A model with no `predict_proba`.** `roc_auc`, `pr_auc`, `log_loss` and
`brier` score a probability. When the estimator cannot give one they come back
as `NaN` with a warning, rather than being computed on the hard 0/1 predictions
— `roc_auc` over labels is arithmetically fine but it is balanced accuracy, not
an AUC, and it would land on the leaderboard under the wrong name.
`evaluation.threshold` is ignored in that case too.

**`cv_folds` scores the primary metric.** `cv_mean` and `cv_std` are always on
the scale of `primary_metric`, and `metrics.json` records which one under
`cv.cv_metric`. A primary metric with no cross-validation scorer is an error,
not a silent fallback to the estimator's default `.score()`.

## Common errors and what they mean

| Message | Cause |
| --- | --- |
| `unknown fields in 'split': [...]` | a typo in a YAML field |
| `model.type is required` | no model picked |
| `the model catalog is empty` | no model registered in `models.py` yet |
| `unknown model 'x'` | `model.type` matches no `@register` |
| `model 'x' needs the 'y' package` | `pip install y` |
| `no features declared` | `features` is empty |
| `primary_metric 'x' is not listed in evaluation.metrics` | the primary metric has to be computed |
| `seasons in both train and test` | overlapping season lists |
| `columns missing from the data: [...]` | a column name that does not exist in the chosen source |
| `unknown target 'x'` | the target is not registered in `data.py` |
| `unknown builder 'x'` | the builder is not registered in `features.py` |

# Experiment configs

Each `.yaml` file in this folder is **one experiment**. The folder starts
empty: the first config is born together with the first model you register in
`src/nfl_trees/models.py`.

The full field reference, valid values and common errors are in
[docs/config-reference.md](../docs/config-reference.md).

## Starting point

Copy this into `configs/<model_name>.yaml`, swap the `model` block and fill in
the features:

```yaml
name: decision_tree
description: >
  Baseline: the first tree, to get a floor for comparison.
task: classification
seed: 42

data:
  source: scores
  target: home_win
  # 2010 is left out on purpose: the rate features look back on the previous
  # season, and 2010 does not have one.
  seasons: [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
  include_preseason: false
  include_postseason: true

features:
  numeric:
    - pct_home_win           # home team, at home
    - pct_away_win           # away team, on the road
    - home_pct_score_drive   # drives that ended in points
    - home_pct_allowed_drive # ... and the ones it gave up
    - away_pct_score_drive
    - away_pct_allowed_drive
    - month                  # 1-12
    - week                   # 1-18 regular season, 19-22 playoffs
    - playoff                # 1 in the postseason
  categorical: [day]         # sunday, monday, thursday, ...
  builders: [calendar, win_rates, drive_rates]

split:
  strategy: season
  train_seasons: [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
  test_seasons: [2023, 2024]

model:
  type: decision_tree    # must be registered in nfl_trees.models
  params:
    max_depth: 6

evaluation:
  metrics: [roc_auc, accuracy, f1, log_loss]
  primary_metric: roc_auc
```

## Convention

One config per model, with the **same target, the same features and the same
split**. That way the only variable between two runs is the model, and
`results/leaderboard.csv` compares like with like.

When you want to vary features instead of the model, create a second config
whose name says so (`random_forest_with_recent_form.yaml`) rather than editing
the existing one — the leaderboard keeps one row per experiment *name*.

```bash
python -m nfl_trees configs        # list what is here
python -m nfl_trees run <name>      # run one
python -m nfl_trees run --all       # run all of them
```

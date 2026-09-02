# Notebooks

Notebooks are the **exploration space**, not where an experiment lives. The
rule that keeps the repository reproducible:

> If a result matters, it has to come out of a YAML in `configs/` — not out of
> notebook cells.

The expected flow:

1. explore in the notebook (look at the data, try a feature idea);
2. whatever survives becomes package code (`@builder`, `@target`, `@register`)
   or a config in `configs/`;
3. the notebook goes back to just reading and plotting the artifacts written
   under `results/`.

| Notebook | What it is for |
| --- | --- |
| `01_data_exploration.ipynb` | Understand the CSVs: columns, cardinality, missing values, distributions. |
| `02_features.ipynb` | Try feature ideas before promoting them to a `@builder`. |
| `03_model_comparison.ipynb` | Read `leaderboard.csv` and compare the models. |
| `04_model_interpretation.ipynb` | Open up the chosen model: importances, tree drawing, errors. |

The notebooks are empty on purpose — they are your working pad.

## Useful snippets

Loading the raw data:

```python
from nfl_trees.data import load_scores, load_plays

games = load_scores([2023, 2024])                  # only games that were played
upcoming = load_scores([2026], statuses=("TBD",))  # games to predict
plays = load_plays([2024])
```

Running an experiment without writing to `results/`:

```python
from nfl_trees import ExperimentConfig, run

result = run(ExperimentConfig.load("decision_tree"), save=False)
result.metrics
result.importances
```

Reading artifacts from runs already executed:

```python
import pandas as pd
from nfl_trees import load_leaderboard

load_leaderboard()
pd.read_csv("../results/decision_tree/importances.csv")
```

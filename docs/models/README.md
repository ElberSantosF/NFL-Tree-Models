# Models

One `.md` file per model, created when the model enters the catalog. They all
follow the same structure ([`_template.md`](_template.md)) — that is what makes
it possible to read two of them side by side and compare.

## Catalog

The catalog is **empty**. Models are added one at a time, as the study moves
forward.

```bash
python -m nfl_trees models   # lists the current catalog
```

| Model | Family | Doc |
| --- | --- | --- |
| _(none yet)_ | | |

<!-- add a row here for each registered model -->

## How to add a model

**1. Register the constructor** in [`src/nfl_trees/models.py`](../../src/nfl_trees/models.py):

```python
@register("decision_tree", "Single decision tree (CART). Interpretable baseline.")
def _decision_tree(task, params, seed):
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    cls = DecisionTreeClassifier if task == "classification" else DecisionTreeRegressor
    return cls(random_state=seed, **params)
```

The constructor takes `(task, params, seed)` and returns an estimator with the
scikit-learn API. Import the library **inside** the function, not at module
level.

Optional `@register` arguments:

| Argument | What it is for |
| --- | --- |
| `requires="xgboost"` | external package; the catalog shows `ok = no` when missing |
| `defaults={...}` | hyperparameters applied before the YAML (the YAML always wins) |

**2. Create the config** `configs/decision_tree.yaml` — see
[configs/README.md](../../configs/README.md).

**3. Write the doc** by copying [`_template.md`](_template.md) to
`docs/models/decision_tree.md`, and add its row to the table above.

**4. Run it:**

```bash
python -m nfl_trees run decision_tree
python -m nfl_trees leaderboard
```

Nothing else in the workbench needs to change.

## A suggested order

Not a rule, but each step below answers a specific question — and because
every config uses the same target, the same features and the same split, the
answer comes out clean:

| Model | The question it answers |
| --- | --- |
| decision tree | What is the floor? Does a shallow tree capture anything at all? |
| bagging (random forest) | How much does bagging gain over the single tree? |
| boosting | Does boosting beat bagging here, and at what cost? |

Start with a single tree: besides being the comparison floor, it is the only
model you can draw in full and read by hand.

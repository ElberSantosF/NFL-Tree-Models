# Decision tree

> One tree, grown greedily. The comparison floor, and the only model in this
> family you can draw in full and read by hand.

| | |
| --- | --- |
| Family | single tree |
| Library | `scikit-learn` (core dependency) |
| Classes | `DecisionTreeClassifier` / `DecisionTreeRegressor` |
| Algorithm | CART (Breiman et al., 1984) |
| Suggested name in the config | `decision_tree` |

## How it works

Training is recursive and greedy. At a node holding a set of rows:

1. For every feature, and every candidate threshold on it, split the rows in two.
2. Score each split by the **impurity decrease** it produces.
3. Keep the best split, then repeat on each child until a stopping rule fires.

The score is the impurity of the parent minus the weighted impurity of the
children:

```
gain = I(parent) - (n_left/n) * I(left) - (n_right/n) * I(right)
```

`I` is what changes between tasks:

| Criterion | Task | Formula | Note |
| --- | --- | --- | --- |
| Gini | classification | `1 - Σ p_k²` | default; cheaper, no logarithm |
| Entropy | classification | `-Σ p_k log₂ p_k` | peaks harder at p=0.5, slightly more balanced trees |
| Log loss | classification | same as entropy | sklearn alias |
| Squared error | regression | variance of `y` in the node | mean at the leaf |
| Absolute error | regression | mean `|y - median|` | median at the leaf, much slower |

Gini and entropy almost always pick the same splits. It is not a knob worth
tuning first.

CART splits are **binary** (`feature <= threshold`), always. Candidate
thresholds are the midpoints between consecutive observed values, so a feature
with `n` distinct values offers `n-1` candidates — which is where the importance
bias comes from.

A leaf predicts the class distribution of the training rows that reached it, so
`predict_proba` returns those frequencies. A leaf with 3 rows returns
probabilities of 0, 1/3, 2/3 or 1 — confident-looking numbers built on nothing.
That is why `min_samples_leaf` matters more than it looks.

### Stopping and pruning

Grown to completion, a tree puts one row per leaf and reaches 100% training
accuracy. Two ways to prevent that:

**Pre-pruning** — stop early with `max_depth`, `min_samples_split`,
`min_samples_leaf`, `min_impurity_decrease`. Fast, but greedy: it can stop
before a split that would only pay off two levels down.

**Post-pruning** — grow it out, then cut back with cost-complexity pruning
(`ccp_alpha`). It minimizes

```
R_α(T) = R(T) + α · |leaves(T)|
```

`R(T)` is the training error, `α` the price of a leaf. Larger `α`, smaller tree.
`cost_complexity_pruning_path()` gives the `α` values where the tree actually
changes, so cross-validating over that path beats guessing at `max_depth`.

## Hyperparameters that matter

| Parameter | Default | Controls | Regularizes when |
| --- | --- | --- | --- |
| `max_depth` | `None` (unlimited) | longest question chain | decreased |
| `min_samples_leaf` | 1 | rows required in a leaf | increased |
| `min_samples_split` | 2 | rows required to attempt a split | increased |
| `max_features` | `None` (all) | features considered per split | decreased |
| `ccp_alpha` | 0.0 | post-pruning strength | increased |
| `criterion` | `gini` | impurity measure | — |
| `class_weight` | `None` | per-class cost; `balanced` for skewed targets | — |

`max_depth` and `min_samples_leaf` do most of the work. Start there.

## Strengths

- **Readable end to end.** `sklearn.tree.plot_tree` / `export_text` gives you
  the whole model. Nothing else in this family offers that.
- **No preprocessing.** No scaling, no distribution assumptions, monotone
  transforms of a feature are invisible to it.
- **Interactions and non-linearity for free**, without being specified.
- **Fast**, on any dataset this size.

## Weaknesses

- **High variance.** Resample the data and the root split can change, taking
  every branch with it. This is the defining problem, and the reason forests and
  boosting exist.
- **Axis-aligned splits only.** A boundary like `pct_home_win > pct_away_win`
  becomes a staircase of dozens of splits. If you believe in such a boundary,
  build the difference as a feature — a tree will not find it on its own.
- **No extrapolation.** Predictions are flat outside the training range.
- **Biased importance** toward high-cardinality features.
- **Unstable probabilities** at small leaves, as above.

## On this dataset

~5,300 games, 10 features, a target near 50/50 with a real home-field tilt.
Expect leaf size to bind before depth does: with enough games per leaf a depth-8
tree scores like a depth-5 one, and it is small leaves, not deep ones, that fall
apart on the test seasons. NFL outcomes carry a lot of irreducible noise; a
model that fits the training seasons perfectly is fitting that noise.

Two useful things this model gives you that the ensembles will not:

- The **root split** tells you which single feature carries most of the signal.
- A depth-3 tree printed with `export_text` is a readable summary of the
  dataset, worth keeping in the model doc even after better models arrive.

Watch out for season 2010 — it has no previous season, so its history features
are missing or thin. Under `missing_strategy: sentinel` they arrive as `-999`,
far below every real value, and the tree will gladly spend its root split on
"is this 2010?". Under the default `median` they are quietly filled in and 2010
looks like an average team. Neither is informative: train from 2011 on. A
`DecisionTreeClassifier` on the scikit-learn version this repo requires (≥ 1.5)
can also route `NaN` down a learned default branch, so `missing_strategy: keep`
is worth trying against the other two — see
[config-reference](../../config-reference.md).

**Measured, not guessed.** [Notebook 03](../../../notebooks/03_model_comparison.ipynb)
tunes this model instead of asserting it: eighty TPE trials scored over twelve
rolling-origin folds (2013–2024), with 2025 walled off from the search and from
the feature selection. The tuned tree reaches **roc_auc 0.617** across the folds
and **0.601** on the held-out 2025 season. Its accuracy there, 0.549, barely
clears the 0.535 of always picking the home team — the ranking is worth more
than the 0.5 cut makes it look, which is the reason to read both numbers.

The same notebook settles which knob is doing the work. Both searches land
between 77 and 98 games per leaf and then choose depths as far apart as 5 and 8
for the same score: once a leaf has to speak for eighty games, the depth bound
has nothing left to do. And out-of-fold permutation importance puts `month`,
`week`, `day` and `playoff` at exactly zero in all twelve seasons — the tree
never splits on them, dropping them costs 0.0003 of roc_auc, and the model runs
on the six history rates alone, `away_pct_score_drive` alone worth more than the
next two together.

## Registering it here

```python
@register(
    "decision_tree",
    "Single decision tree (CART). Interpretable baseline.",
    defaults={"criterion": "log_loss", "max_depth": 8, "min_samples_leaf": 77},
)
def _decision_tree(task, params, seed):
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    cls = DecisionTreeClassifier if task == "classification" else DecisionTreeRegressor
    return cls(random_state=seed, **params)
```

```bash
python -m nfl_trees run decision_tree
```

## References

- Breiman, Friedman, Olshen, Stone — *Classification and Regression Trees* (1984)
- [scikit-learn: Decision Trees](https://scikit-learn.org/stable/modules/tree.html)
- [Cost-complexity pruning example](https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html)
- Hastie, Tibshirani, Friedman — *The Elements of Statistical Learning*, ch. 9.2

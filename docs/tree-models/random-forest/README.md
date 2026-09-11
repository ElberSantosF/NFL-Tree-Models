# Random forest

> Hundreds of trees grown on bootstrap samples, each split offered only a random
> subset of the columns, and the answer is their average. The standard fix for
> the single tree's variance.

| | |
| --- | --- |
| Family | bagging |
| Library | `scikit-learn` (core dependency) |
| Classes | `RandomForestClassifier` / `RandomForestRegressor` |
| Algorithm | Random forests (Breiman, 2001) |
| Suggested name in the config | `random_forest` |

## How it works

One decision tree is unstable: change a handful of games and the root split can
move, taking every branch with it. A forest does not make any single tree
better — it makes that instability cancel.

Three steps, `n_estimators` times over:

1. **Bootstrap.** Draw `n` rows with replacement from the training set (or
   `max_samples · n`). Each tree sees a different sample of the seasons.
2. **Grow a tree on it**, with one change to CART: at every node, only a random
   subset of `max_features` columns is *offered* as candidates for the split.
   The tree still picks the best split it is shown; it is the menu that is
   randomized, not the choice.
3. **Average.** For classification scikit-learn averages the trees'
   `predict_proba` — the per-leaf class frequencies — rather than taking a
   majority vote. A forest's probability is therefore much finer-grained than
   any of its trees': 300 leaves of `2/3` and `1/4` average to something in
   between.

Trees are grown **deep on purpose**. Deep trees have low bias and high variance,
and variance is exactly what step 3 removes. Pre-pruning a forest the way you
would prune a single tree is regularizing the thing the averaging already
handles.

### Why averaging helps, and where it stops

Take `B` trees, each with variance `σ²`, and let `ρ` be the average pairwise
correlation between their predictions. The variance of their mean is

```
Var(mean) = ρ·σ²  +  (1 - ρ)·σ² / B
```

Two terms, and they behave very differently:

- The **second** vanishes as `B` grows. This is why `n_estimators` cannot
  overfit and why it is not a hyperparameter to tune: past a few hundred trees
  that term is already negligible, and adding more only buys steadier decimals.
- The **first** does not move with `B` at all. `ρ·σ²` is the floor, and the only
  way below it is to make the trees disagree more — smaller `ρ`.

Everything Breiman added to plain bagging is an attack on `ρ`. Bootstrapping
decorrelates the trees through the rows; `max_features` decorrelates them
through the columns, and it is the stronger of the two — without it, a dominant
feature is the root split of all 300 trees and they stay near-copies of each
other. The catch is that both raise `σ²`: a tree that was not allowed to see the
best column at a node is a worse tree. `max_features` is where that trade-off is
set, which is why it is the knob that matters most here.

### The free test set

A bootstrap sample of size `n` misses `(1 - 1/n)^n → 1/e ≈ 37%` of the rows.
Those rows are, for that tree, unseen data. Average each row's prediction over
just the trees that did not draw it and you get an estimate that costs no extra
fit: `oob_score=True`, read from `oob_decision_function_`.

It is genuinely useful, and it has one assumption worth naming: it treats the
training rows as exchangeable. On a panel of NFL seasons they are not — the
out-of-bag rows for a 2011 game include trees grown on 2024, which is
information from its future. OOB is a good sanity check and a bad substitute for
the rolling-origin folds used in [notebook 03](../../../notebooks/03_model_comparison.ipynb).

## Hyperparameters that matter

| Parameter | Default | Controls | Regularizes when |
| --- | --- | --- | --- |
| `max_features` | `sqrt` | columns offered per split | decreased — the decorrelating knob |
| `min_samples_leaf` | 1 | rows required in a leaf | increased |
| `max_samples` | `None` (all `n`) | size of each bootstrap draw | decreased |
| `max_depth` | `None` (unlimited) | longest question chain | decreased — rarely the binding one |
| `bootstrap` | `True` | whether rows are sampled at all | — off leaves only feature randomness |
| `n_estimators` | 100 | trees averaged | **never** — a compute setting |
| `criterion` | `gini` | impurity measure | — |
| `class_weight` | `None` | per-class cost; `balanced` for skewed targets | — |
| `oob_score` | `False` | the free estimate above | — |
| `n_jobs` | `None` | cores used to fit and predict | — |

Start with `max_features`, then `min_samples_leaf`. Set `n_estimators` as high
as you are willing to wait for and leave it there.

## Strengths

- **Kills the single tree's variance**, which is the one thing that model does
  worst, and asks for almost no tuning to do it. A forest with defaults is a
  hard baseline to beat.
- **Cannot be overfitted by adding trees.** The number that costs compute is the
  number that is safe to raise.
- **Scores itself** out of bag, without a validation split.
- **Parallel by construction** — the trees are independent, so `n_jobs` is close
  to linear, which boosting cannot offer.
- **Keeps every convenience of the tree**: no scaling, mixed types, interactions
  found rather than specified.

## Weaknesses

- **Not readable.** The single tree's one real advantage is gone: 300 trees
  cannot be drawn, only summarized. Partial dependence and permutation
  importance replace reading the model.
- **The bias is unchanged.** Averaging axis-aligned step functions gives a
  smoother staircase, not a diagonal. If the boundary you need is
  `pct_home_win > pct_away_win`, a forest is no better placed than a tree to
  find it — build the difference as a feature.
- **Still no extrapolation.** Every prediction is an average of training leaves,
  so it is bounded by what training saw.
- **Impurity importance is biased twice over** — toward high-cardinality
  columns, as in any tree, and then spread thinly across correlated columns that
  each get offered in place of the other. Use permutation importance on held-out
  rows.
- **Probabilities are smooth, not calibrated.** Averaging pulls them toward the
  middle; a forest rarely says 0.95 even when it should.
- **Cost.** Hundreds of deep trees are hundreds of times the memory and the
  prediction latency of one, which is nothing at 4,000 games and a real
  constraint at millions.

## On this dataset

~5,300 games, 10 features, a target near 50/50. The single tree here ends up
shallow and heavily leaf-limited, which is another way of saying that most of
what it did was fight variance. A forest fights variance by construction, so it
can afford the deep trees the single tree could not — and that is where the gain
turns out to come from.

Season 2010 is still the warm-up season with no history behind it: train from
2011 on. The `missing_strategy` note from the
[decision tree](../decision-tree/README.md#on-this-dataset) applies unchanged —
the forest is no better at telling an imputed median from a real one.

**Measured, not guessed.** [Notebook 03](../../../notebooks/03_model_comparison.ipynb)
fits this model under the protocol the tree got: rolling-origin folds over twelve
seasons (2013–2024), 2025 walled off from both the search and the feature
selection, 300 trees. Forty TPE trials rather than the tree's eighty — and the
search says why forty was enough. The best trial and the tenth-best are **0.0025
roc_auc** apart, against 0.0498 down to the worst: the surface is flat almost
everywhere except at the edges, which is the practical shape of "bagging is the
family least sensitive to its knobs".

The tuned forest reaches **roc_auc 0.655** across the folds and **0.616** on the
held-out 2025 season, against **0.617** and **0.601** for the tuned tree. On the
holdout that is +0.015 of roc_auc and +0.028 of accuracy — 0.577 against the
tree's 0.549, over the 0.535 floor of always picking the home team — with the
log loss down from 0.693 to 0.666 and the Brier score from 0.248 to 0.237. The
probabilities got better, not only their ordering.

Three things the run settled that the theory could only point at:

- **It wants as much decorrelation as it can get.** The retuned forest ends at
  `max_features=0.198` over six columns, which is one randomly chosen column
  offered per split, and `max_samples=0.36` on top of that. With six features
  that all say roughly the same thing in slightly different words, weakening
  each tree costs little and `ρ` is where all the gain is — exactly what the
  variance identity above predicts, and the opposite of what the `sqrt` default
  would have done.
- **It does split on the columns the tree ignored, and still loses nothing when
  they go.** The step 1 forest spends **4%** of its impurity decrease on `month`,
  `week`, `day` and `playoff` — it has to, since `max_features` keeps offering
  them when nothing better is on the menu — where the tree never split on them
  at all. But dropping them costs the forest 0.0002 of roc_auc, the same nothing
  it cost the tree. The splits it spent there were noise, and averaging three
  hundred trees is what makes noise cost nothing.
- **Out of bag came out at 0.656, against 0.655 on the rolling folds.** The
  pooling problem is real but small here: the free estimate landed 0.0015 above
  the estimate that respects the calendar. Close enough to be useful as a sanity
  check, and still the wrong estimator to report.

One more thing worth seeing in the 2026 forecast, because it is the averaging
made visible. The tree's Super Bowl came out at exactly 0.500 — both finalists
had landed in the same leaf, and the notebook had to break the tie on expected
wins. The forest, asked the same question, answers 53.4%: three hundred leaves
averaged do not collide the way one leaf does. The two models also disagree on
who wins it, which is what 2026 is there to settle.

## Registering it here

The hyperparameters below are the ones the search landed on, and they assume the
six history rates as the feature set — `max_features=0.198` means one column per
split at six, and two at ten.

```python
@register(
    "random_forest",
    "Bagged trees with per-split feature sampling. The variance fix.",
    defaults={
        "n_estimators": 300,
        "criterion": "entropy",
        "max_depth": 19,
        "min_samples_leaf": 30,
        "min_samples_split": 125,
        "max_features": 0.198,
        "max_samples": 0.361,
        "n_jobs": -1,
    },
)
def _random_forest(task, params, seed):
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    cls = RandomForestClassifier if task == "classification" else RandomForestRegressor
    return cls(random_state=seed, **params)
```

```bash
python -m nfl_trees run random_forest
```

## References

- Breiman — *Random Forests* (2001), Machine Learning 45(1) —
  [the paper](https://link.springer.com/article/10.1023/A:1010933404324)
- Breiman — *Bagging Predictors* (1996), the half of the idea that came first
- [scikit-learn: Forests of randomized trees](https://scikit-learn.org/stable/modules/ensemble.html#forest)
- [scikit-learn: the pitfalls of impurity-based importance](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html)
- Hastie, Tibshirani, Friedman — *The Elements of Statistical Learning*, ch. 15
  (the variance identity above is 15.2)

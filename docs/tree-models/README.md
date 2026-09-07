# Tree models

Study notes on the model family this repository is built around. One folder per
model, all with the same sections, so two of them can be read side by side.

This folder is about **the models themselves** — how they split, what they
optimize, where they fail. It is separate from [`docs/models/`](../models/),
which documents the models actually registered in this workbench and the
results they produced here. Theory lives here, runs live there.

## What a tree model is

A decision tree asks a sequence of yes/no questions about a row and reads the
answer off the leaf it lands in:

```
                 pct_home_win <= 0.52 ?
                  /                  \
                yes                   no
                 │                     │
        week <= 3 ?              away_pct_score_drive <= 0.31 ?
         /       \                  /            \
   P(home)=0.44  P(home)=0.51   P(home)=0.71   P(home)=0.62
```

Each internal node is one threshold on one feature. Each leaf is a constant —
a probability for classification, a mean for regression. Prediction is walking
from the root to a leaf; training is choosing the questions.

Nobody solves that choice optimally: finding the smallest tree that fits a
dataset is NP-hard. Every model here is a **greedy** approximation. At each
node it tries the candidate splits, scores each by how much it reduces an
impurity measure, takes the best one, and never reconsiders it.

That single idea explains most of what follows:

- **No scaling, no distributions.** A split is an ordering question. Multiplying
  a feature by 1000 or taking its log changes nothing. There is no linearity or
  normality assumption to violate.
- **Interactions come free.** A split inside a branch is conditional on every
  split above it, so `week > 15 AND pct_home_win > 0.6` is expressible without
  anyone writing that term.
- **Mixed types are natural.** `week` (numeric), `day` (categorical) and
  `playoff` (binary) coexist without one-hot expansion — see
  [the preprocessor](../../src/nfl_trees/features.py) for what this repo does
  with them.
- **The output is a step function.** Trees do not extrapolate. Beyond the range
  seen in training the prediction is flat, whatever the trend was.
- **One tree is unstable.** Change a handful of rows and the root split may
  change, and everything below it with it. High variance, low bias.

That last point is what the rest of the family exists to fix.

## The two ways out of instability

Everything past the single tree is one of two answers to "one greedy tree
overfits":

**Bagging — average many trees fitted in parallel.** Each tree sees a bootstrap
sample and a random subset of features at each split, so their errors are
decorrelated and averaging cancels them out. Variance drops, bias stays roughly
where the single tree had it. Trees are grown deep on purpose: the averaging is
what does the regularizing. Random forest and extra trees live here.

**Boosting — fit trees in sequence, each on what the previous ones got wrong.**
Each new tree is small (a stump, or depth 3–6) and is fitted to the gradient of
the loss at the current prediction, then added with a shrinkage factor. Bias
drops step by step. Nothing decorrelates the errors, so boosting overfits if you
keep adding trees — the number of trees is a hyperparameter to tune, not to
maximize. AdaBoost, gradient boosting, XGBoost, LightGBM and CatBoost live here.

```
bagging      T1  T2  T3 ... Tn        (parallel, independent)   average
                                       ↓ kills variance

boosting     T1 → T2 → T3 ... → Tn    (sequential, each on the residual)  sum
                                       ↓ kills bias
```

## The catalog

One folder per model, **written when the model gets studied** — same rule as the
[model catalog](../models/README.md): documenting seven models at once would
produce seven summaries of things not yet understood.

| Model | Family | Library | Folder |
| --- | --- | --- | --- |
| [Decision tree](decision-tree/) | single tree | scikit-learn | written |
| Random forest | bagging | scikit-learn | to come |
| Extra trees | bagging | scikit-learn | to come |
| AdaBoost | boosting | scikit-learn | to come |
| Gradient boosting | boosting | scikit-learn | to come |
| XGBoost | boosting | `xgboost` | to come |
| LightGBM | boosting | `lightgbm` | to come |
| CatBoost | boosting | `catboost` | to come |

<!-- when a model is studied: create docs/tree-models/<name>/README.md and
     turn its row above into a link -->

The order above is not arbitrary — each step answers one question:

1. **Decision tree** — what is the floor, and what does a split actually do?
2. **Random forest** — how much does averaging buy over one tree?
3. **Extra trees** — how much of that gain came from bootstrapping versus from
   plain randomness?
4. **AdaBoost** — the original boosting idea, in its simplest form.
5. **Gradient boosting** — the same idea stated as gradient descent on a loss.
6. **XGBoost / LightGBM / CatBoost** — three engineering answers to the same
   algorithm, and what each one optimizes for.

## What each folder contains

Same sections everywhere — that is what lets two of them be read side by side:

- **How it works** — the mechanism, in enough detail to reimplement the idea.
- **Hyperparameters that matter** — what each one controls and which direction
  regularizes.
- **Strengths and weaknesses.**
- **On this dataset** — what to expect on ~5,300 NFL games with 10 features.
- **Registering it here** — the `@register` constructor to paste into
  [`models.py`](../../src/nfl_trees/models.py).
- **References.**

## Two things that apply to all of them

**Impurity-based importance is biased.** `feature_importances_` sums the
impurity decrease each feature produced. Features with many distinct values get
more chances to split well by luck, so continuous and high-cardinality columns
score higher than they deserve — `week` and `pct_home_win` over `playoff`, here.
Every run writes `importances.csv` from that attribute; treat it as a hint, and
confirm anything you plan to claim with
[permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html)
on the test set.

**No tree model fixes leakage.** These models are very good at finding a column
that encodes the answer. The history features in this repo are computed over a
window that ends the week before the game for exactly that reason — see
[docs/data.md](../data.md) and the split rules in [architecture](../architecture.md).

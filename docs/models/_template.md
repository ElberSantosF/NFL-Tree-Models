<!--
Model documentation template.

Copy it to `docs/models/<name>.md` when you register a new model in the
catalog, fill it in and delete this comment. The sections are the same across
all models on purpose: that is what lets you read two files side by side and
compare.
-->

> <one line: what this model does. The same sentence you passed to `@register`.>

| | |
| --- | --- |
| Name in the config | `<name>` |
| Family | single tree / bagging / boosting |
| Library | `scikit-learn` / `xgboost` / `lightgbm` / ... |
| Classes | `<Classifier>` / `<Regressor>` |
| Reference config | [`configs/<name>.yaml`](../../configs/<name>.yaml) |

```bash
python -m nfl_trees run <name>
```

## How it works

How does the model choose its splits? What does it do with the error of the
previous step? Where does the randomness come from, if any? Write it in your
own words — this is the section that shows whether you understood it.

## Hyperparameters in this config

| Parameter | Value used | What it controls | Effect of increasing it |
| --- | --- | --- | --- |
| | | | |

## When to use it

In which situation this model is the right choice, and why.

## Limitations and caveats

What it does poorly, where it misleads (e.g. feature importance biased by
cardinality), what needs care in preprocessing.

## Observed results

Metrics land in `results/<name>/metrics.json` and importances in
`results/<name>/importances.csv`.

| Date | Primary metric | Value | Note |
| --- | --- | --- | --- |
| | | | |

## Compared to the others

What this model showed that the previous one did not.

## References

Official documentation, the original paper, links that helped.

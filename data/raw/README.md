# Raw data

The CSV layout. The code only reads the consolidated files (`plays/` and
`scores/2010-2026_scores.csv`); the other folders are the same data sliced up,
useful for manual inspection.

```
data/raw/
├── scores/
│   ├── 2010-2026_scores.csv       <- used by the code (source `scores`)
│   └── <season>_scores.csv        <- the same, sliced per season
├── plays/
│   └── <season>_plays.csv         <- used by the code (source `plays`)
└── plays_by_week/
    └── <season>/<season>_<week>_plays.csv
```

Paths are centralized in [`src/nfl_trees/paths.py`](../../src/nfl_trees/paths.py) —
if you move anything, that is the only file to change.

The column dictionary, the `GameStatus` values and the gotchas of each source
are in [docs/data.md](../../docs/data.md).

## Size

449 CSVs, ~370 MB in total, versioned in Git by default so the repository is
reproducible on clone. Roughly half of that is redundant: `plays_by_week/` is
the same rows as `plays/`, only split per week — dropping it would cost ~185 MB
and no information. If you would rather keep GitHub light, uncomment the last
block of `.gitignore` and distribute the data through a release or Git LFS.

## Known gap

The three 2013 Wild Card games other than KC at IND have no rows in `plays/`
(and none in `plays_by_week/2013/`). `scores/` has all four. Anything joining
the two sources on the game will come up three games short — see
[docs/data.md](../../docs/data.md).

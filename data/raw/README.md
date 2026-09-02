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

449 files, ~190 MB in total, versioned in Git by default so the repository is
reproducible on clone. If you would rather keep GitHub light, uncomment the
last block of `.gitignore` and distribute the data through a release or Git LFS.

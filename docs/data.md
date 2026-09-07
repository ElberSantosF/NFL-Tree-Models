# Data

The file layout is in [data/raw/README.md](../data/raw/README.md). The code
reads the consolidated files: `data/raw/scores/2010-2026_scores.csv` and
`data/raw/plays/<season>_plays.csv`.

The goal of the project is to **predict the winner of a game**, so the working
grain is the game (`scores`). Plays (`plays`) come in as raw material for
team-level aggregated features.

## Source `scores` — one row per game

| Column | Type | Example | Note |
| --- | --- | --- | --- |
| `Season` | int | `2024` | used by the season split |
| `Week` | text | `WEEK 1`, `SUPER BOWL` | a label, not a number |
| `GameStatus` | text | `FINAL` | **decides what counts as a game** — see below |
| `GameSlot` | text | `Sunday` | day/slot of the game |
| `GameDate` | text | `August 8th` | **no year**; combine with `Season` |
| `AwayTeam` / `HomeTeam` | text | `DAL` / `CIN` | 2–3 letter abbreviation |
| `AwayScore` / `HomeScore` | float | `16.0` / `7.0` | missing when the game was not played |

### `GameStatus`: not every row is a game

This is the main gotcha of the scores file. There are 4 values:

| Value | Count | What it is |
| --- | --- | --- |
| `FINAL` | 5300 | a game that was played — **the only one with a result** |
| `BYE` | 542 | a team's rest week: `AwayTeam` filled in, `HomeTeam` empty |
| `TBD` | 291 | a scheduled game, not played yet |
| `Clinched Playoffs` | 52 | a team with a first-round playoff bye, no opponent |

`load_scores` returns **only `FINAL`** by default. Without that filter, the
other 885 rows — 14% of the file — would enter training as games with no score.

To build the set of games to predict (an ongoing season):

```python
from nfl_trees.data import load_scores

upcoming = load_scores([2026], statuses=("TBD",))
```

## Source `plays` — one row per play

| Column | Type | Example | Note |
| --- | --- | --- | --- |
| `Season` | int | `2025` | |
| `Week` | text | `WEEK 5` | |
| `GameSlot` | text | `Sunday` | |
| `Date` | text | `September 1st` | **no year** |
| `AwayTeam` / `HomeTeam` | text | `LAC` / `DET` | abbreviation |
| `Quarter` | text | `1st Quarter` | **text, not a number** |
| `DriveNumber` | int | `1` | drive sequence within the game |
| `TeamWithPossession` | text | `Detroit Lions` | full name, not the abbreviation |
| `IsScoringDrive` | 0/1 | `1` | the drive ended in points |
| `PlayNumberInDrive` | int | `3` | play sequence within the drive |
| `IsScoringPlay` | 0/1 | `0` | the play resulted in points |
| `PlayOutcome` | text | `9 Yard Pass`, `Fumble` | short outcome label |
| `PlayStart` | text | `1st & 10 at DET 28` | down, yards to go and field position |
| `PlayTimeFormation` | text | `14:51 1st Shotgun` | clock, quarter and formation, all in one |
| `PlayDescription` | text | `— T.Lance pass short right to...` | full play-by-play text |

Roughly 46 thousand plays per season once preseason is dropped (56 thousand
with it), 2010 through 2025. The 2026 file holds preseason only.

`plays_by_week/<season>/` is the same data split per week: identical rows and
columns, useful for opening one week by hand. Nothing in the code reads it.

## `Week` labels (both sources)

| Group | Values |
| --- | --- |
| preseason | `HALL OF FAME`, `PRESEASON WEEK 1..4` |
| regular season | `WEEK 1` .. `WEEK 18` |
| postseason | `WILD CARD WEEKEND`, `DIVISIONAL PLAYOFFS`, `CONFERENCE CHAMPIONSHIPS`, `SUPER BOWL` |
| exhibition | `PRO BOWL` |

`nfl_trees.data.filter_weeks` works off these groups. **`PRO BOWL` is always
dropped** (not a competitive game); preseason and postseason are controlled by
`include_preseason` / `include_postseason` in the config. Preseason is out by
default: rosters and intensity are not comparable to the regular season.

## Caveats

**Team names in two formats.** `TeamWithPossession` carries the full name
(`Detroit Lions`), while `HomeTeam`/`AwayTeam` carry the abbreviation (`DET`).
They are not directly comparable — to join plays with games, an
abbreviation ↔ name map is the first builder you will need. The plays files use
each franchise's *current* name in every season, so the map needs no historical
entries; the scores files do not, so `STL`/`SD`/`OAK`/`JAC`/`AZ` still appear
and have to be folded into `LAR`/`LAC`/`LV`/`JAX`/`ARI`.

**A drive that ends in a defensive touchdown is labelled with the defense.**
This is the sharpest trap in `plays`. On a pick-six or a fumble returned for a
score, `TeamWithPossession` names the team that reached the end zone — and not
only on the scoring play: **every row of that drive** carries it, including the
snaps the other team's offense actually ran. So the drive looks, from first play
to last, like a scoring drive by the defense. 1169 drives in 2010–2025 (1.2% of
all drives, 5.2% of touchdowns) are shaped this way.

Two consequences for any per-drive aggregate. The offense that ran the drive
loses it from its denominator; the defense gains a drive it never had, plus a
score. And checking "did the scorer own the drive?" does **not** catch it — the
owner is already the scorer, on the drive's very first play. The only way to
find these drives is the play text: `PlayOutcome == "Touchdown"` together with
`INTERCEPTED`, or `FUMBLES` and `RECOVERED by`, in `PlayDescription`.

`IsScoringDrive` follows the same convention, so it cannot be used to check a
derived flag against — the two agree because they share the mistake.

**Three games are missing from `plays`.** Of the four 2013 Wild Card games,
only KC at IND has plays. `scores` has all four, so joining the two sources on
the game yields 4360 games where `scores` alone yields 4363.

**Composite columns.** `PlayStart` and `PlayTimeFormation` pack several pieces
of information into one string (down, yards to go, field position, clock,
formation). Pulling that apart is `@builder` work in `features.py`.

**`Quarter` and `Week` are text.** Declaring them as `numeric` produces a column
of all `NaN` (`to_numeric` will not convert `"1st Quarter"`). Use
`categorical`.

**Dates without a year.** `Date` and `GameDate` carry only day and month.

**2026 is partial.** The file exists but covers the start of the season, and
most of its games are still `TBD`.

**Ties happen.** Rare in the NFL, but real. The `home_win` target returns
missing on a tie (and those rows leave the training set): "the home team won"
is not defined there. Predicting the tie would be a three-class target.

**Leakage is the easy mistake here.** To predict the winner, nothing that is
only known *after* the game can be a feature — score, game statistics, play
outcomes. A team-strength feature has to be computed from **earlier games
only**.

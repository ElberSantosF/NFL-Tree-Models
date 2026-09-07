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

## Derived grain `drives` — one row per possession

`plays` is not usable as-is for anything measured per possession: the two
caveats below (`DriveNumber` restarting every quarter, and defensive
touchdowns credited to the defense) both distort it. `data.load_drives` reads
the plays files and folds them into the grain that survives both:

| Column | Type | Note |
| --- | --- | --- |
| `Season` / `Week` | int / text | same labels as the other two sources |
| `team` | text | the offense that **ran** the drive, as an abbreviation |
| `opponent` | text | the defense that faced it |
| `plays` | int | rows in the drive (penalties and timeouts included) |
| `scored` | 0/1 | the drive ended in a touchdown or field goal **for `team`** |

```python
from nfl_trees.data import load_drives

drives = load_drives([2024])          # one season
drives = load_drives()                # 2010-2025, ~10 s and a few MB
```

98,907 drives over 2010–2025, 35.8% of them ending in points. One season is
read at a time — the sixteen plays files together are ~580 MB in memory, while
the drive table for all of them is a few megabytes.

Two numbers do **not** match the raw file, on purpose. `scored` counts only
`Touchdown` and `Field Goal`: an extra point is automatic and already implied
by the touchdown, and a safety is two points for the *defense*. And the 1,169
drives the source credits to a defense that returned a turnover are handed back
to the offense that ran them, as drives that ended without points — which is
what actually happened.

`load_drives` covers 4,360 of the 4,363 games in `scores`; the three missing
ones are the 2013 Wild Card games noted below.

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

`WEEK_ORDINAL` puts the same labels on a number line — preseason −5 to −1,
regular season 1 to 18, postseason 19 to 22 — and `week_number` applies it to a
column. That is the `week` feature, and it is what makes "later in the season"
a thing a tree can split on.

**The regular season is 17 weeks up to 2020 and 18 from 2021 on.** So "after
week 17 is the playoffs" is a rule that was true and then quietly stopped being
true: it files 80 real regular-season games (`WEEK 18`, 2021–2025) under
playoffs. `is_postseason` uses the four labels instead, which holds in both
eras — that is what the `playoff` feature is built on.

## Caveats

**Team names in two formats.** `TeamWithPossession` carries the full name
(`Detroit Lions`), while `HomeTeam`/`AwayTeam` carry the abbreviation (`DET`).
They are not directly comparable, so joining plays with games needs a map both
ways. Both live in `data.py`:

- `TEAM_NAME_TO_ABBR` — full name to abbreviation. The plays files use each
  franchise's *current* name in every season, so it needs no historical entries.
- `canonical_team` — one abbreviation per franchise. `HomeTeam`/`AwayTeam` name
  a relocated franchise by the abbreviation it used at the time, in **both**
  sources, so `STL`/`SD`/`OAK`/`JAC`/`AZ` still appear and are folded into
  `LAR`/`LAC`/`LV`/`JAX`/`ARI`. A team feature is about the franchise, not the
  city.

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

`load_drives` repairs this, so anything built on the drive grain is already
past it. Anything that groups the raw plays by possession is not.

**Three games are missing from `plays`.** Of the four 2013 Wild Card games,
only KC at IND has plays. `scores` has all four, so joining the two sources on
the game yields 4360 games where `scores` alone yields 4363.

**Composite columns.** `PlayStart` and `PlayTimeFormation` pack several pieces
of information into one string (down, yards to go, field position, clock,
formation). Pulling that apart is `@builder` work in `features.py`.

**`Quarter` and `Week` are text.** Declaring either as `numeric` produces a
column of all `NaN` (`to_numeric` will not convert `"1st Quarter"`). Use
`categorical` — or, for `Week`, the `week` feature, which is `week_number`
applied to the label and is numeric on purpose.

**Dates without a year.** `Date` and `GameDate` carry only day and month,
which is enough for a calendar month (`September 14th` is month 9 in every
season) and not enough to order two games inside the same week. The `month`
feature takes the first; nothing takes the second, and the rate features use
the week as their time step because of it.

**2026 is partial.** The file exists but covers the start of the season, and
most of its games are still `TBD`.

**Ties happen.** Rare in the NFL, but real. The `home_win` target returns
missing on a tie (and those rows leave the training set): "the home team won"
is not defined there. Predicting the tie would be a three-class target.

**Leakage is the easy mistake here.** To predict the winner, nothing that is
only known *after* the game can be a feature — score, game statistics, play
outcomes. A team-strength feature has to be computed from **earlier games
only**.

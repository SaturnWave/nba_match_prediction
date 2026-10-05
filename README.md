# NBA Match Prediction & Player Impact Scoring

An end-to-end NBA analytics pipeline that:

1. **Scrapes** raw NBA game data (play-by-play, box scores, player tracking) from the
   [`nba_api`](https://github.com/swar/nba_api).
2. **Computes a custom "impact score"** for every player from play-by-play events,
   with context-aware modifiers (clutch time, scoring runs, shot difficulty,
   foul trouble, lead changes), normalized per-100-possessions.
3. **Trains machine-learning models** (LightGBM) to predict game outcomes:
   home win, point differential and total score.

## Installation

```bash
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Requires Python 3.11+ (developed on 3.13).

After cloning, restore the caches git ignores (the local copies of the
database tables, the matchup cache and two smaller ones - 792 MB, kept
compressed in `local_backup/`):

```bash
python tools/local_backup.py restore
```

Everything else the pipeline reads is tracked directly. The one thing not in
the repository is `.env` (database credentials); `.env.example` lists its
keys, and nothing needs the database while the caches are in place.

## Project layout

```
nba_match_prediction/
├── impact_common.py             # shared stateless helpers (clutch time, shot
│                                #   distance bins, possessions, ...) imported
│                                #   across the impact-score scripts
│
├── data_retriaval/              # API scrapers -> writes into nba_data/
│   ├── nba_api_utils.py            # shared retrieval helpers (format_season, read_game_ids)
│   ├── retrieve_nba_data.py        # main scraper (PBP + tracking, all seasons)
│   ├── retrieve_box_scores.py      # traditional/advanced/defensive/matchup box scores
│   ├── play_by_play_data_retrieve.py
│   ├── retrieve_tracking_data.py
│   ├── player_track_retriaval.py
│   ├── playbyplay2.py
│   ├── game_data_retriaval.py      # NOTE: redundant subset of the root game_data_retriaval.py
│   └── test.py                     # small single-game smoke test
│
├── impact_score_calculation/    # the heuristic player-impact engine
│   ├── season_impact_engine.py     # per-play impact functions shared by both analyze scripts
│   ├── impact_score.py             # full single-game engine + visualizations
│   ├── analyze_season_impact.py    # scales impact scoring to a full season
│   └── analyze_season_impact_1.py  # optimized variant (multiprocessing + caching)
│
├── prediction_engines/
│   └── 2023_2024.py                # LightGBM training + prediction (NBAPredictor)
│
├── asasa.py                     # earlier PBP-only impact-score prototype
├── game_data_retriaval.py       # fetch season game-ID lists (superset of the data_retriaval/ copy)
│
├── game_ids/                    # GAME_ID,GAME_DATE,MATCHUP per season
│   └── game_id_<season>.csv
│
├── nba_data/                    # scraped raw data (one folder per season/game)
│   └── <season>/<game_id>/
│       ├── play_by_play/
│       ├── box_scores/
│       └── player_tracking/
│
├── impact_score_files/          # sample single-game CSVs used by impact_score.py / asasa.py
├── models/                      # trained .pkl models + feature-importance plots
└── requirements.txt
```

Seasons currently covered: `2019_2020` … `2024_2025`.

## Usage

All scripts resolve their paths relative to their own location, so run them from
anywhere inside the repository.

### 1. Retrieve data

```bash
python data_retriaval/retrieve_nba_data.py      # play-by-play + player tracking
python data_retriaval/retrieve_box_scores.py    # box scores
```

These read the per-season game lists from `game_ids/` and write into
`nba_data/<season>/<game_id>/`. The scrapers use adaptive throttling and retry
logic to stay within the NBA stats API rate limits.

### 2. Compute impact scores

Single game (uses the sample CSVs in `impact_score_files/`):

```bash
python impact_score_calculation/impact_score.py
```

Whole season (reads `nba_data/2023_2024/`):

```bash
python impact_score_calculation/analyze_season_impact.py
```

### 3. Train & predict

```bash
python prediction_engines/2023_2024.py
```

This loads the games, engineers time-aware features (rolling averages, win
streaks, season averages, head-to-head history — all shifted to avoid leakage),
trains three LightGBM models, and saves them with feature-importance plots to
`models/`. It then runs an example prediction (e.g. `GSW` vs `LAL`).

## The impact score (v5)

Every play a player is named on is priced by how it moves expected points
against a baseline of 1.00 points per possession, then weighted by how much
the game still hung on it. Implemented in `prediction_engines/value_engine.py`;
`prediction_engines/impact_v5.py` writes the result in the cache layout the
rest of the pipeline reads (`game_impact_cache_v5.pkl`).

| Event              | Value                      | Why                                              |
|--------------------|---------------------------:|--------------------------------------------------|
| Made 2 / made 3    | +1.0 / +2.0                | points scored minus the possession spent         |
| Missed shot        | −0.74                      | the possession is usually gone (26% come back)   |
| Free throw         | +1 made, − possession used | 1/n of a possession per attempt of an n-shot trip; none for an and-one, technical, flagrant or clear path |
| Assist             | 30% of the basket, on top  | +0.3 on a two, +0.6 on a three; the scorer keeps his full credit |
| Secondary assist   | +0.15                      | the pass to the passer, from the tracking table  |
| Free-throw assist  | +0.17                      | a pass that drew a shooting foul, from tracking  |
| Turnover / steal   | −1.15 / +1.15              | a possession plus a transition premium           |
| Offensive rebound  | +0.74                      | a possession recovered                           |
| Defensive rebound  | +0.26                      | a 26% chance denied — the expected outcome       |
| Block              | +0.6                       | the shooter already pays for the miss            |
| Foul               | −0.3 personal, −0.6 shooting | offensive fouls are paid by their turnover row  |

Leverage: 1.0 while the game is live, sliding to 0.3 at 25 points apart in the
fourth quarter, 1.25 in the last five minutes of a game within five points.
The leverage weight is applied in the player ratings only; the model's team
features use the same pricing with every weight at 1.0 (`--no-leverage`),
because the size of a win predicts the next game and weighted values were
measured to cost the forecast (Brier +0.003) while unweighted ones cost nothing.
No style bonuses — a step-back three and an open one both put three points on
the board. Rebound type is read from whose miss preceded it, not from the
description (which carries the player's running totals).

Passing, as measured against on-court plus-minus over three seasons: no assist
credit 0.394, a flat +0.35 per assist 0.442, 30% of the basket on top 0.445,
30% taken out of the scorer's credit 0.430 — so the passer is paid on top. The
passer is read from the description `(Name N AST)`; matching that name is the
hard part (no accents, no suffixes, a first-name prefix where teammates share a
surname, and the running total N to separate the rest), and with all of it
handled the engine credits 99.8–100% of the box score's assists. A credit per
pass made (the tracking table's PASS column) made the score worse and is not
paid; potential assists are not in the data.

Why v5 replaced the original engine: that engine fired only on made shots (a
miss cost nothing), typed every rebound as offensive because `"Off" in
description` matches `REBOUND (Off:0 Def:1)` on every row, scored neither
assists nor free throws, and discounted nothing once a game was decided. On
2017-18 regulars it correlated 0.14 with on-court plus-minus per 36 minutes;
v5 correlates 0.42 (0.39 and 0.43 on 2023-24 and 2025-26). The old functions
remain in `impact_engine.py` for reference.

## Shared modules

Common code has been pulled out of the individual scripts to avoid duplication:

- **`impact_common.py`** — stateless helpers (`is_clutch_time`, `get_score_margin`,
  `categorize_shot_distance`, `calculate_expected_points`, `calculate_team_possessions`, …)
  used by `asasa.py` and every script in `impact_score_calculation/`.
- **`impact_engine.py`** — the single set of six per-play `calculate_*_impact`
  functions, used by `asasa.py`, `impact_score.py` and both
  `analyze_season_impact*.py` scripts. The full-game DataFrame is passed
  explicitly, and optional `df_defensive` / `df_player_track` arguments add the
  tracking bonuses (no bonus when omitted), so every former variant is a
  parameterisation of this one engine.
- **`data_retriaval/nba_api_utils.py`** — `format_season` and `read_game_ids`,
  shared by the season scrapers.

Each consumer adds the relevant directory to `sys.path` at import time, so the
scripts still run directly (`python impact_score_calculation/impact_score.py`).

The merge was verified against the previous code: the season-analysis path is
**byte-identical** (720 sampled calls, 0 differences), and the tracking-bonus
path matches `impact_score.py` exactly — except `calculate_foul_impact`, which
in the old `asasa.py` / `impact_score.py` copies crashed on every foul
(`df_pbp.loc[:row.name-1]` iterates column labels, not rows). The unified engine
uses the corrected version, so those two scripts now run instead of aborting.

## Notes / known limitations

- `prediction_engines/2023_2024.py` currently uses a **simplified** stub of the
  impact calculator; the full logic lives in `impact_score_calculation/impact_score.py`
  and is not yet wired into the model features.
- `data_retriaval/game_data_retriaval.py` is a strict subset of the root
  `game_data_retriaval.py` and can likely be removed once its single-game
  entry point is no longer needed.
- Some scrapers and `impact_score.py` are pinned to a single sample game
  (`0022400058`).
```

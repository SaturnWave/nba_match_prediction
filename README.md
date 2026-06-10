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

Requires Python 3.9+.

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

## The impact score

Each play is assigned a base value by event type, then adjusted:

| Event       | Base value | Example modifiers                                  |
|-------------|-----------:|----------------------------------------------------|
| Made shot   | 2.0 / 3.0  | shot difficulty, lead change, late shot clock      |
| Steal       | 1.4        | leads to score, opponent frontcourt, steal type    |
| Block       | 1.2        | rim protection, stops a run, shot-clock violation  |
| Rebound     | 0.6 / 0.9  | offensive vs defensive, putback, after a block     |
| Foul        | −0.3…−1.5  | foul trouble, bonus situation, foul type           |
| Turnover    | −0.8       | turnover type, leads to opponent score             |

Plays in clutch time (last 5 min of Q4 / OT) are boosted ×1.5, and totals are
normalized to 100 possessions per team.

## Shared modules

Common code has been pulled out of the individual scripts to avoid duplication:

- **`impact_common.py`** — stateless helpers (`is_clutch_time`, `get_score_margin`,
  `categorize_shot_distance`, `calculate_expected_points`, `calculate_team_possessions`, …)
  used by `asasa.py` and every script in `impact_score_calculation/`.
- **`impact_score_calculation/season_impact_engine.py`** — the six per-play
  `calculate_*_impact` functions that were identical between
  `analyze_season_impact.py` and `analyze_season_impact_1.py`.
- **`data_retriaval/nba_api_utils.py`** — `format_season` and `read_game_ids`,
  shared by the season scrapers.

Each consumer adds the relevant directory to `sys.path` at import time, so the
scripts still run directly (`python impact_score_calculation/impact_score.py`).

## Notes / known limitations

- `prediction_engines/2023_2024.py` currently uses a **simplified** stub of the
  impact calculator; the full logic lives in `impact_score_calculation/impact_score.py`
  and is not yet wired into the model features.
- `asasa.py` and `impact_score.py` keep their own variants of the
  `calculate_*_impact` functions because they differ (PBP-only vs.
  tracking-integrated) — they were intentionally left separate rather than
  force-merged, which would change the scores.
- `data_retriaval/game_data_retriaval.py` is a strict subset of the root
  `game_data_retriaval.py` and can likely be removed once its single-game
  entry point is no longer needed.
- Some scrapers and `impact_score.py` are pinned to a single sample game
  (`0022400058`).
```

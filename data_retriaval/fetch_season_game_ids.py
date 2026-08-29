"""
Write game_ids/game_id_<season>.csv for seasons the repo does not have yet.

LeagueGameFinder returns one row per TEAM per game, so each game appears twice
(once from each side). Only the home row is kept — its MATCHUP reads
"HOME vs. AWAY", which is the convention every downstream parser in this repo
expects (see NBAPredictor._parse_matchup). Regular-season games only: their
game ids start with 002.

Usage:
  py data_retriaval/fetch_season_game_ids.py 2017-18 2018-19
"""
import os
import sys
import time

import pandas as pd
from nba_api.stats.endpoints import LeagueGameFinder

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAME_IDS_DIR = os.path.join(PROJECT_ROOT, "game_ids")
API_DELAY = 1.0
TIMEOUT = 60


def season_dir_name(season):
    """'2017-18' -> '2017_2018' (the directory/file convention on disk)."""
    start = int(season.split("-")[0])
    return f"{start}_{start + 1}"


def fetch(season):
    finder = LeagueGameFinder(season_nullable=season,
                              league_id_nullable="00",
                              season_type_nullable="Regular Season",
                              timeout=TIMEOUT)
    df = finder.get_data_frames()[0]
    df = df[df["GAME_ID"].astype(str).str.startswith("002")]
    home = df[df["MATCHUP"].str.contains(" vs. ", na=False)]
    out = (home[["GAME_ID", "GAME_DATE", "MATCHUP"]]
           .drop_duplicates(subset=["GAME_ID"])
           .sort_values("GAME_DATE")
           .reset_index(drop=True))
    out["GAME_ID"] = out["GAME_ID"].astype(str).str.zfill(10)
    return out


def main():
    seasons = sys.argv[1:] or ["2017-18", "2018-19"]
    os.makedirs(GAME_IDS_DIR, exist_ok=True)
    for season in seasons:
        name = season_dir_name(season)
        path = os.path.join(GAME_IDS_DIR, f"game_id_{name}.csv")
        if os.path.exists(path):
            print(f"{name}: zaten var, atlaniyor ({path})")
            continue
        games = fetch(season)
        games.to_csv(path, index=False)
        print(f"{name}: {len(games)} mac -> {path}")
        print(f"   {games.GAME_DATE.min()} .. {games.GAME_DATE.max()}")
        time.sleep(API_DELAY)


if __name__ == "__main__":
    main()

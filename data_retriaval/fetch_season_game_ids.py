"""
Write game_ids/game_id_<season>.csv for seasons the repo does not have yet.

LeagueGameFinder returns one row per TEAM per game, so each game appears twice
(once from each side). Only the home row is kept — its MATCHUP reads
"HOME vs. AWAY", which is the convention every downstream parser in this repo
expects (see NBAPredictor._parse_matchup). Regular-season games only: their
game ids start with 002.

A finished season's file is written once and then left alone. A season in
progress grows by the day, so --refresh fetches it again and rewrites it.

Usage:
  py data_retriaval/fetch_season_game_ids.py 2017-18 2018-19
  py data_retriaval/fetch_season_game_ids.py 2026-27 --refresh
"""
import argparse
import os
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seasons", nargs="*", default=["2017-18", "2018-19"])
    parser.add_argument("--refresh", action="store_true",
                        help="dosyasi olan sezonu da yeniden cek (suren sezon icin)")
    args = parser.parse_args()
    os.makedirs(GAME_IDS_DIR, exist_ok=True)
    for season in args.seasons:
        name = season_dir_name(season)
        path = os.path.join(GAME_IDS_DIR, f"game_id_{name}.csv")
        if os.path.exists(path) and not args.refresh:
            print(f"{name}: zaten var, atlaniyor ({path}; suren sezonsa --refresh)")
            continue
        games = fetch(season)
        games.to_csv(path, index=False)
        print(f"{name}: {len(games)} mac -> {path}")
        print(f"   {games.GAME_DATE.min()} .. {games.GAME_DATE.max()}")
        time.sleep(API_DELAY)


if __name__ == "__main__":
    main()

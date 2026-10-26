"""
Write game_ids/schedule_<season>.csv: the league's published schedule for a
season, every game of every type, with its status and - once played - its score.

game_ids/game_id_<season>.csv (fetch_season_game_ids.py) lists the games that
have been PLAYED; it is what the per-game retrieval scripts walk. This file is
the other half: the whole season as published, so games still to come are
known before they are played. Run it again during the season and it rewrites
the file with the current statuses and any newly fixed dates (the NBA Cup
knockout games, for one, get their teams only after the group stage).

season_type comes from the game id's first three digits: 001 preseason,
002 regular season, 003 All-Star, 004 playoffs, 005 play-in, 006 NBA Cup final.
A game whose teams are not fixed yet is kept, with home and away left empty.

Usage:
  py data_retriaval/fetch_schedule.py 2026-27
"""
import os
import sys

import pandas as pd
from nba_api.stats.endpoints import ScheduleLeagueV2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAME_IDS_DIR = os.path.join(PROJECT_ROOT, "game_ids")
TIMEOUT = 60

SEASON_TYPES = {"001": "preseason", "002": "regular_season", "003": "all_star",
                "004": "playoffs", "005": "play_in", "006": "nba_cup_final"}
STATUSES = {1: "scheduled", 2: "live", 3: "final"}


def season_dir_name(season):
    """'2026-27' -> '2026_2027' (the directory/file convention on disk)."""
    start = int(season.split("-")[0])
    return f"{start}_{start + 1}"


def tidy(raw):
    """The published rows, reduced to what describes a game."""
    game_id = raw["gameId"].astype(str).str.zfill(10)
    final = raw["gameStatus"] == 3
    schedule = pd.DataFrame({
        "game_id": game_id,
        "season_type": game_id.str[:3].map(SEASON_TYPES),
        "game_date": raw["gameDateEst"].astype(str).str[:10],
        "tipoff_utc": raw["gameDateTimeUTC"].astype(str),
        "status": raw["gameStatus"].map(STATUSES),
        "home_id": raw["homeTeam_teamId"],
        "home": raw["homeTeam_teamTricode"],
        "away_id": raw["awayTeam_teamId"],
        "away": raw["awayTeam_teamTricode"],
        "home_score": raw["homeTeam_score"].where(final).astype("Int64"),
        "away_score": raw["awayTeam_score"].where(final).astype("Int64"),
        "arena": raw["arenaName"],
        "city": raw["arenaCity"],
        "neutral_site": raw["isNeutral"],
        "week": raw["weekNumber"],
        "label": raw["gameLabel"],
        "sublabel": raw["gameSubLabel"],
        "national_tv": raw["nationalBroadcasters_0_broadcasterDisplay"],
        "postponed": raw["postponedStatus"],
    })
    if schedule["season_type"].isna().any():
        unknown = sorted(set(game_id[schedule["season_type"].isna()].str[:3]))
        raise ValueError(f"bilinmeyen mac turu on ekleri: {unknown}")
    undecided = schedule["home"].fillna("").astype(str).str.strip().eq("")
    schedule.loc[undecided, ["home", "away"]] = pd.NA
    return schedule.sort_values(["tipoff_utc", "game_id"]).reset_index(drop=True)


def main():
    seasons = sys.argv[1:]
    if not seasons:
        sys.exit("sezon verin, ornegin: py data_retriaval/fetch_schedule.py 2026-27")
    os.makedirs(GAME_IDS_DIR, exist_ok=True)
    for season in seasons:
        raw = ScheduleLeagueV2(season=season, league_id="00", timeout=TIMEOUT).get_data_frames()[0]
        schedule = tidy(raw)
        path = os.path.join(GAME_IDS_DIR, f"schedule_{season_dir_name(season)}.csv")
        schedule.to_csv(path, index=False)
        print(f"{season}: {len(schedule)} mac -> {path}")
        counts = schedule.groupby(["season_type", "status"]).size().unstack(fill_value=0)
        print(counts.to_string())
        played = schedule[(schedule["season_type"] == "regular_season") & (schedule["status"] == "final")]
        if len(played):
            print(f"   oynanan son normal sezon maci: {played['game_date'].max()}")
        undecided = int(schedule["home"].isna().sum())
        if undecided:
            print(f"   takimlari henuz belli olmayan: {undecided} mac")


if __name__ == "__main__":
    main()

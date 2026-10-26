"""
Report, season by season, which per-game files exist under nba_data/.

Every game listed in game_ids/game_id_<season>.csv should have ten files: the
play-by-play, seven box scores (traditional, advanced and defensive, each for
players and for teams, plus the matchups) and two tracking files. A file of 64
bytes or less counts as missing - the retrieval scripts use the same bound,
because an endpoint with nothing to return still writes a header line.

Run:  py tools/check_data.py              (every season)
      py tools/check_data.py 2026_2027    (one season)
"""
import glob
import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "nba_data")
GAME_IDS_DIR = os.path.join(PROJECT_ROOT, "game_ids")
MIN_BYTES = 65
SHOWN_IDS = 5

# label -> (subfolder, file name suffix after the game id)
FILES = {
    "pbp": ("play_by_play", "pbp.csv"),
    "traditional": ("box_scores", "box_score_traditional.csv"),
    "traditional_team": ("box_scores", "box_score_traditional_team.csv"),
    "advanced": ("box_scores", "box_score_advanced.csv"),
    "advanced_team": ("box_scores", "box_score_advanced_team.csv"),
    "defensive": ("box_scores", "box_score_defensive.csv"),
    "defensive_team": ("box_scores", "box_score_defensive_team.csv"),
    "matchups": ("box_scores", "box_score_matchups.csv"),
    "player_tracking": ("player_tracking", "player_tracking.csv"),
    "team_tracking": ("player_tracking", "team_tracking.csv"),
}


def sizes_in(folder):
    """{file name: bytes} for one game's subfolder; empty when the folder is absent."""
    if not os.path.isdir(folder):
        return {}
    with os.scandir(folder) as entries:
        return {entry.name: entry.stat().st_size for entry in entries if entry.is_file()}


def check(season):
    """Listed game ids, the ids missing each file type, and game folders nobody listed."""
    listed = pd.read_csv(os.path.join(GAME_IDS_DIR, f"game_id_{season}.csv"), dtype={"GAME_ID": str})
    game_ids = sorted(set(listed["GAME_ID"].str.zfill(10)))
    missing = {label: [] for label in FILES}
    for game_id in game_ids:
        game_dir = os.path.join(DATA_DIR, season, game_id)
        found = {sub: sizes_in(os.path.join(game_dir, sub)) for sub in {sub for sub, _ in FILES.values()}}
        for label, (sub, suffix) in FILES.items():
            if found[sub].get(game_id + suffix, 0) < MIN_BYTES:
                missing[label].append(game_id)
    season_dir = os.path.join(DATA_DIR, season)
    on_disk = set(os.listdir(season_dir)) if os.path.isdir(season_dir) else set()
    return game_ids, missing, sorted(on_disk - set(game_ids))


def report(season):
    game_ids, missing, unlisted = check(season)
    incomplete = set().union(*missing.values())
    print(f"{season}: {len(game_ids)} mac listede, {len(game_ids) - len(incomplete)} tanesi tam (10/10 dosya)")
    for label, ids in missing.items():
        if ids:
            shown = ", ".join(ids[:SHOWN_IDS]) + (" ..." if len(ids) > SHOWN_IDS else "")
            print(f"   eksik {label:<17} {len(ids):>5}  {shown}")
    if unlisted:
        print(f"   listede olmayan mac klasoru: {len(unlisted)}  {', '.join(unlisted[:SHOWN_IDS])}")


def main():
    seasons = sys.argv[1:] or sorted(os.path.basename(p)[len("game_id_"):-len(".csv")]
                                     for p in glob.glob(os.path.join(GAME_IDS_DIR, "game_id_*.csv")))
    for season in seasons:
        report(season)


if __name__ == "__main__":
    main()

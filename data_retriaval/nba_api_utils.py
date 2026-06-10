"""Shared helpers for the NBA data-retrieval scripts.

``format_season`` and ``read_game_ids`` were byte-for-byte identical across
``retrieve_nba_data.py`` and ``retrieve_box_scores.py``; they live here now as
the single source of truth.
"""

import os

import pandas as pd


def format_season(season):
    """
    Convert season format from "YYYY-YY" to "YYYY_YYYY" for directory structure
    """
    season_parts = season.split("-")
    season_format = f"{season_parts[0]}_{int(season_parts[0])+1}"
    return season_format


def read_game_ids(season):
    """
    Read game IDs from the corresponding CSV file

    Args:
        season (str): Season in format "YYYY-YY" (e.g., "2019-20")

    Returns:
        pd.DataFrame: DataFrame containing game IDs and metadata
    """
    formatted_season = format_season(season)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Game-ID CSVs live in the game_ids/ folder; fall back to the project root
    # for backward compatibility with older layouts.
    candidates = [
        os.path.join(project_root, "game_ids", f"game_id_{formatted_season}.csv"),
        os.path.join(project_root, f"game_id_{formatted_season}.csv"),
    ]
    csv_path = next((p for p in candidates if os.path.exists(p)), None)

    if csv_path is None:
        print(f"ERROR: Game IDs CSV file not found for season {season}. Looked in: {candidates}")
        return None

    try:
        games_df = pd.read_csv(csv_path)
        # Ensure we have unique game IDs
        games_df = games_df.drop_duplicates(subset=['GAME_ID'])
        return games_df
    except Exception as e:
        print(f"ERROR: Failed to read game IDs from {csv_path}: {e}")
        return None

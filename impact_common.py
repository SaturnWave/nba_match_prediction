"""Shared, stateless helper functions for NBA play-by-play impact scoring.

These utilities were previously copy-pasted across ``asasa.py`` and the scripts
in ``impact_score_calculation/``. They are collected here so there is a single
source of truth. Each function is pure (no module-level state), which makes them
safe to import from any of those scripts.
"""

import numpy as np
import pandas as pd


def is_clutch_time(clock_seconds, period):
    """Checks if the play is in the last 5 minutes of the 4th period or overtime."""
    return (period == 4 and clock_seconds <= 300) or period > 4


def is_last_2_minutes(clock_seconds, period):
    """Checks if the play is in the last 2 minutes of the 4th period or overtime."""
    return (period == 4 and clock_seconds <= 120) or period > 4


def get_score_margin(row):
    """Returns the absolute score margin between teams."""
    return abs(row['scoreHome'] - row['scoreAway']) if pd.notnull(row['scoreHome']) and pd.notnull(row['scoreAway']) else 0


def identify_scoring_run(data, current_idx, window=5):
    """Identifies if there's a scoring run by a team in the previous plays."""
    start_idx = max(0, current_idx - window)
    previous_plays = data.iloc[start_idx:current_idx]
    if not previous_plays.empty:
        team_counts = previous_plays['teamTricode'].value_counts()
        if not team_counts.empty and len(team_counts) > 0:
            return team_counts.index[0]
    return None


def categorize_shot_distance(distance):
    """Categorizes shot distance into bins for analysis."""
    if pd.isnull(distance):
        return None
    if distance <= 3:
        return "At Rim"
    elif distance <= 8:
        return "Paint"
    elif distance <= 16:
        return "Mid-Range"
    elif distance <= 24:
        return "Long 2"
    else:
        return "3-Point"


def calculate_expected_points(x, y, shot_value):
    """Calculate expected points based on shot location."""
    if pd.isnull(x) or pd.isnull(y) or pd.isnull(shot_value):
        return None

    # Convert coordinates to feet from basket
    distance = np.sqrt(x**2 + y**2) / 10  # Approximate conversion

    if shot_value == 3:  # 3-pointer
        if abs(x) > 220 and y < 90:  # Corner 3 coordinates
            return 1.1  # Corner 3 (higher percentage)
        else:
            return 0.9  # Above the break 3 (lower percentage)
    else:  # 2-pointer
        if distance < 5:
            return 1.6  # At rim
        elif distance < 10:
            return 0.9  # Paint non-restricted
        elif distance < 16:
            return 0.8  # Mid-range
        else:
            return 0.7  # Long 2 (inefficient)


def estimate_win_probability(row, home_score, away_score, period, time_remaining):
    """Simple win probability estimation based on score and time."""
    if pd.isnull(home_score) or pd.isnull(away_score) or pd.isnull(period) or pd.isnull(time_remaining):
        return 0.5  # Default to 50% if missing data

    lead = home_score - away_score
    total_seconds_left = (4 - min(period, 4)) * 720 + time_remaining  # Assuming 12 min periods
    seconds_factor = max(0.1, min(1, total_seconds_left / 2880))  # Normalize by total game seconds

    # Simple logistic model
    wp = 1 / (1 + np.exp(-lead * seconds_factor * 0.1))
    return wp


def calculate_team_possessions(df_traditional):
    """Estimates possessions for each team in the game."""
    team_possessions = {}

    for team in df_traditional['TEAM_ABBREVIATION'].unique():
        team_data = df_traditional[df_traditional['TEAM_ABBREVIATION'] == team]

        # Basic possession estimate
        fga = team_data['FGA'].sum()
        fta = team_data['FTA'].sum() * 0.44  # FTA coefficient
        to = team_data['TO'].sum()
        orb = team_data['OREB'].sum() * 0.2  # Offensive rebound coefficient

        possessions = fga + fta - orb + to
        team_possessions[team] = possessions

    return team_possessions

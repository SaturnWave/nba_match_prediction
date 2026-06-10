"""Unified per-play impact-scoring engine.

This is the single source of truth for the six ``calculate_*_impact`` functions.
It merges the three historical variants (``asasa.py``, ``impact_score.py`` and the
season-analysis scripts) into one set:

* The full-game DataFrame ``df`` is always passed explicitly (used for scoring-run
  and foul-trouble lookups) instead of relying on a module-level global.
* Optional ``df_defensive`` / ``df_player_track`` arguments add the
  tracking-based bonuses. When they are ``None`` (the default) no bonus is
  applied, so callers that do not have tracking data get the plain
  play-by-play score, identical to the old PBP-only / season behaviour.

All field access uses ``.get(...)`` so a row with a missing column degrades
gracefully instead of raising.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from impact_common import (
    is_clutch_time,
    is_last_2_minutes,
    get_score_margin,
    identify_scoring_run,
)


def calculate_block_impact(row, next_play, previous_plays, df, df_defensive=None):
    """Calculates enhanced impact value for blocks."""
    base_impact = 1.2

    # Original modifiers
    if next_play is not None and next_play.get('teamTricode') == row.get('teamTricode'):
        base_impact -= 0.2  # Block that stays with blocking team (reduced value)
    if next_play is not None and isinstance(next_play.get('description'), str) and 'Running' in next_play.get('description'):
        base_impact += 0.2  # Block leading to transition
    if next_play is not None and isinstance(next_play.get('description'), str) and 'Shot Clock' in next_play.get('description'):
        base_impact += 0.3  # Block causing shot clock violation

    # Multiple blocks demonstrating defensive dominance
    recent_blocks = [play for play in previous_plays[-3:]
                    if isinstance(play.get('description'), str) and 'BLOCK' in play.get('description')
                    and play.get('playerName') == row.get('playerName')]
    if len(recent_blocks) > 1:
        base_impact += 0.3

    # Clutch time blocks
    if is_last_2_minutes(row.get('clock_seconds'), row.get('period')) and get_score_margin(row) <= 3:
        base_impact += 0.5

    # Enhanced modifiers
    # Block location value (rim protection)
    if pd.notnull(row.get('shotDistance')) and row.get('shotDistance') <= 5:
        base_impact += 0.2  # Blocks at the rim

    # Block against a scoring run
    current_idx = df.index.get_loc(row.name)
    scoring_run_team = identify_scoring_run(df, current_idx)
    if scoring_run_team and scoring_run_team != row.get('teamTricode'):
        base_impact += 0.3  # Stopping opponent's momentum

    # Block results in change of possession
    if next_play is not None and next_play.get('teamTricode') != row.get('teamTricode'):
        base_impact += 0.2  # Block resulted in change of possession

    # Integration with defensive tracking data if available
    if df_defensive is not None and row.get('personId') is not None:
        player_def_data = df_defensive[df_defensive['personId'] == row['personId']]
        if not player_def_data.empty:
            # Add value for players who contest shots frequently
            contested_ratio = player_def_data['matchupFieldGoalsAttempted'].values[0] / max(1, player_def_data['MIN'].values[0]/60)
            if contested_ratio > 5:  # High contest rate
                base_impact += 0.2

    return base_impact

def calculate_steal_impact(row, next_play, previous_plays, df_defensive=None):
    """Calculates enhanced impact value for steals."""
    base_impact = 1.4

    # Original modifiers
    if isinstance(row.get('description'), str) and 'Backcourt' in row.get('description'):
        base_impact += 0.1  # Backcourt steal (higher pressure)
    if next_play is not None and next_play.get('actionType') == 'Made Shot':
        base_impact += 0.2  # Steal leading to immediate score

    # Multiple steals demonstrating defensive prowess
    recent_steals = [play for play in previous_plays[-5:]
                    if isinstance(play.get('description'), str) and 'STEAL' in play.get('description')
                    and play.get('playerName') == row.get('playerName')]
    if len(recent_steals) > 1:
        base_impact += 0.2

    # Game situation adjustments
    if is_clutch_time(row.get('clock_seconds'), row.get('period')):
        margin = get_score_margin(row)
        if margin > 20:
            base_impact = 1.0  # Reduced impact in blowouts
        elif margin > 10:
            base_impact = 1.1  # Slightly reduced impact
        else:
            base_impact = 1.5  # Increased impact in close games

    # Enhanced modifiers
    # Steal type classification
    if isinstance(row.get('description'), str):
        if 'Bad Pass' in row.get('description'):
            base_impact += 0.1  # Anticipation steal (passing lane)
        elif 'Lost Ball' in row.get('description'):
            base_impact += 0.3  # Active pickpocket (direct steal)

    # Steal leading to breakaway
    if next_play is not None and next_play.get('actionType') == 'Made Shot' and pd.notnull(next_play.get('shotDistance')) and next_play.get('shotDistance') <= 3:
        base_impact += 0.3  # Steal leading to easy basket

    # Steal in opponent's frontcourt
    if pd.notnull(row.get('xLegacy')) and pd.notnull(row.get('yLegacy')):
        # Check if steal is in opponent's half
        team_id = row.get('teamId')
        if (team_id == 1610612743 and row.get('xLegacy') < 0) or (team_id == 1610612744 and row.get('xLegacy') > 0):
            base_impact += 0.2  # Steal in opponent's frontcourt

    # Integration with defensive tracking data if available
    if df_defensive is not None and row.get('personId') is not None:
        player_def_data = df_defensive[df_defensive['personId'] == row['personId']]
        if not player_def_data.empty:
            # Add value for players who consistently generate steals
            if player_def_data['steals'].values[0] > 1:  # Multiple steals in game
                base_impact += 0.1 * player_def_data['steals'].values[0]  # Scale by steal count

    return base_impact

def calculate_rebound_impact(row, next_play, previous_plays, df_player_track=None):
    """Calculates enhanced impact value for rebounds."""
    is_offensive = isinstance(row.get('description'), str) and 'Off' in row.get('description')
    base_impact = 0.9 if is_offensive else 0.6

    # Original modifiers
    if any(isinstance(play.get('description'), str) and 'REBOUND' in play.get('description') for play in previous_plays[-2:]):
        base_impact += 0.2  # Multiple rebounds in sequence
    if next_play is not None and next_play.get('actionType') == 'Made Shot':
        base_impact += 0.2  # Rebound leading to score
    if is_last_2_minutes(row.get('clock_seconds'), row.get('period')) and get_score_margin(row) <= 3:
        base_impact += 0.3  # Critical late-game rebounds
    if is_clutch_time(row.get('clock_seconds'), row.get('period')):
        base_impact += 0.4  # Clutch time rebounds

    # Enhanced modifiers
    # Rebound after contested shot
    if any(isinstance(play.get('description'), str) and 'BLOCK' in play.get('description') for play in previous_plays[-1:]):
        base_impact += 0.3  # Rebound after blocked shot (higher difficulty)

    # Team context
    if is_offensive:
        # Check if team is trailing (offensive rebounds more valuable when behind)
        team_id = row.get('teamId')
        if (team_id == 1610612743 and row.get('scoreHome', 0) < row.get('scoreAway', 0)) or \
           (team_id == 1610612744 and row.get('scoreHome', 0) > row.get('scoreAway', 0)):
            base_impact += 0.2  # Offensive rebound while trailing

        # Putback attempt
        if next_play is not None and next_play.get('actionType') in ['Made Shot', 'Missed Shot'] and next_play.get('teamTricode') == row.get('teamTricode'):
            if pd.notnull(next_play.get('clock_seconds')) and row.get('clock_seconds') - next_play.get('clock_seconds') < 3:
                base_impact += 0.2  # Quick putback attempt
    else:  # Defensive rebound
        # Leading to fast break
        if next_play is not None and next_play.get('actionType') in ['Made Shot', 'Missed Shot'] and next_play.get('teamTricode') == row.get('teamTricode'):
            if pd.notnull(next_play.get('clock_seconds')) and row.get('clock_seconds') - next_play.get('clock_seconds') < 5:
                base_impact += 0.2  # Quick transition after defensive rebound

    # Shot clock context
    previous_shot = next((play for play in previous_plays[-2:] if play.get('actionType') in ['Made Shot', 'Missed Shot']), None)
    if previous_shot and pd.notnull(previous_shot.get('clock_seconds')):
        shot_clock_value = previous_shot.get('clock_seconds') % 24
        if shot_clock_value <= 4:
            base_impact += 0.2  # Rebound after end-of-shot-clock attempt (often more contested)

    # Integration with player tracking data if available
    if df_player_track is not None and row.get('personId') is not None:
        player_track_data = df_player_track[df_player_track['PLAYER_ID'] == row['personId']]
        if not player_track_data.empty:
            if is_offensive:
                # ORBC = Offensive Rebound Chances
                if pd.notnull(player_track_data['ORBC'].values[0]) and player_track_data['ORBC'].values[0] > 0:
                    oreb_chance_ratio = player_track_data['OREB'].values[0] / player_track_data['ORBC'].values[0]
                    base_impact += oreb_chance_ratio * 0.3  # Scale by success rate
            else:
                # DRBC = Defensive Rebound Chances
                if pd.notnull(player_track_data['DRBC'].values[0]) and player_track_data['DRBC'].values[0] > 0:
                    dreb_chance_ratio = player_track_data['DREB'].values[0] / player_track_data['DRBC'].values[0]
                    base_impact += dreb_chance_ratio * 0.2  # Scale by success rate

    return base_impact

def calculate_scoring_impact(row, previous_plays, df, df_player_track=None):
    """Calculates enhanced impact value for scoring plays."""
    base_impact = 3.0 if row.get('shotValue') == 3 else 2.0

    # Original modifiers
    if any(isinstance(play.get('description'), str) and 'Free Throw' in play.get('description') for play in previous_plays[:2]):
        base_impact += 0.3  # And-one plays
    if any(isinstance(play.get('description'), str) and 'Timeout' in play.get('description') for play in previous_plays[-3:]):
        base_impact += 0.2  # Scoring after timeout

    current_idx = df.index.get_loc(row.name)
    scoring_run_team = identify_scoring_run(df, current_idx)
    if scoring_run_team and scoring_run_team != row.get('teamTricode'):
        base_impact += 0.2  # Stopping opponent's run

    if any(isinstance(play.get('description'), str) and 'Start of' in play.get('description') for play in previous_plays[-3:]):
        base_impact += 0.1  # Period-starting baskets

    # Enhanced modifiers
    # Shot difficulty based on spatial data
    if pd.notnull(row.get('expected_points')):
        # Adjust impact based on expected value
        ep_modifier = row.get('expected_points')
        # Score higher than expected = more valuable
        base_impact *= ep_modifier

    # Shot difficulty based on shot description
    if isinstance(row.get('description'), str):
        # Special shot types
        if 'Fadeaway' in row.get('description'):
            base_impact += 0.2  # Difficult fadeaway shot
        elif 'Step Back' in row.get('description'):
            base_impact += 0.3  # Difficult step back
        elif 'Driving' in row.get('description') and 'Dunk' in row.get('description'):
            base_impact += 0.3  # Athletic driving dunk
        elif 'Alley Oop' in row.get('description'):
            base_impact += 0.4  # Highlight play
        elif 'Turnaround' in row.get('description'):
            base_impact += 0.2  # Difficult post move
        elif 'Pullup' in row.get('description'):
            base_impact += 0.1  # Pull-up jumper
        elif 'Bank' in row.get('description'):
            base_impact += 0.1  # Bank shot

    # Shot timing context
    if pd.notnull(row.get('clock_seconds')):
        shot_clock_value = row.get('clock_seconds') % 24
        if shot_clock_value <= 4:
            base_impact += 0.3  # End of shot clock (bailout shot)
        elif shot_clock_value <= 7:
            base_impact += 0.1  # Late shot clock

    # Score impact
    prev_margin = 0
    prev_score_play = next((play for play in previous_plays if pd.notnull(play.get('scoreHome')) and pd.notnull(play.get('scoreAway'))), None)
    if prev_score_play:
        team_id = row.get('teamId')
        home_team_id = 1610612743  # Denver Nuggets
        prev_diff = prev_score_play.get('scoreHome') - prev_score_play.get('scoreAway')
        curr_diff = row.get('scoreHome') - row.get('scoreAway')

        # Check if shot changed lead
        if (prev_diff <= 0 and curr_diff > 0) or (prev_diff >= 0 and curr_diff < 0):
            base_impact += 0.5  # Lead-changing basket
        # Check if shot tied game
        elif curr_diff == 0 and prev_diff != 0:
            base_impact += 0.4  # Game-tying basket
        # Check if shot reduced deficit to one possession
        elif (team_id == home_team_id and prev_diff < -3 and curr_diff >= -3) or \
             (team_id != home_team_id and prev_diff > 3 and curr_diff <= 3):
            base_impact += 0.3  # Cut to one possession

    # Integration with player tracking data if available
    if df_player_track is not None and row.get('personId') is not None:
        player_track_data = df_player_track[df_player_track['PLAYER_ID'] == row['personId']]
        if not player_track_data.empty:
            # Contested Field Goal %
            if pd.notnull(player_track_data['CFGM'].values[0]) and pd.notnull(player_track_data['CFGA'].values[0]) and player_track_data['CFGA'].values[0] > 0:
                cfg_pct = player_track_data['CFG_PCT'].values[0]
                # If the player is good at making contested shots, this shot is likely contested
                if cfg_pct > 0.5 and row.get('shotDistance', 0) > 5:
                    base_impact += 0.2  # Likely contested jumper from good contested shooter

    # Clutch scoring
    if is_clutch_time(row.get('clock_seconds'), row.get('period')):
        margin = get_score_margin(row)
        if margin <= 5:
            base_impact *= 1.3  # 30% boost for scoring in close clutch situations
        elif margin <= 10:
            base_impact *= 1.2  # 20% boost for scoring in moderate clutch situations

    return base_impact

def calculate_turnover_impact(row, next_play, previous_plays):
    """Calculates enhanced impact value for turnovers."""
    base_impact = -1.0 if is_clutch_time(row.get('clock_seconds'), row.get('period')) else -0.8

    # Enhanced modifiers
    # Turnover type
    if isinstance(row.get('description'), str):
        if 'Bad Pass' in row.get('description'):
            base_impact -= 0.2  # Decision error (worse)
        elif 'Lost Ball' in row.get('description'):
            base_impact -= 0.3  # Ball handling error (worse)
        elif 'Step Out of Bounds' in row.get('description') or 'Traveling' in row.get('description'):
            base_impact -= 0.1  # Unforced error (slightly better)
        elif 'Shot Clock' in row.get('description'):
            base_impact -= 0.3  # Team failure to get shot off
        elif 'Offensive Foul' in row.get('description'):
            base_impact -= 0.2  # Aggressive error
        elif 'Backcourt' in row.get('description'):
            base_impact -= 0.3  # Basic error

    # Turnover leading to opponent scoring
    if next_play is not None and next_play.get('actionType') == 'Made Shot' and next_play.get('teamTricode') != row.get('teamTricode'):
        time_diff = row.get('clock_seconds', 0) - next_play.get('clock_seconds', 0)
        if pd.notnull(time_diff) and time_diff < 5:
            base_impact -= 0.3  # Quick score off turnover
            if next_play.get('shotValue') == 3:
                base_impact -= 0.2  # Even worse if opponent hits a 3

    # Game context
    margin = get_score_margin(row)
    if margin <= 5 and row.get('period') >= 4:
        # Close late game
        base_impact *= 1.3  # 30% worse in close late games
    elif margin >= 15:
        # Blowout
        base_impact *= 0.7  # 30% less impactful in blowouts

    # Multiple turnovers
    recent_turnovers = [play for play in previous_plays[-5:]
                      if play.get('actionType') == 'Turnover'
                      and play.get('playerName') == row.get('playerName')]
    if len(recent_turnovers) >= 2:
        base_impact -= 0.2  # Compounding turnovers

    # Turnover after timeout (worse)
    if any(isinstance(play.get('description'), str) and 'Timeout' in play.get('description') for play in previous_plays[-3:]):
        base_impact -= 0.2  # Turnover after timeout

    return base_impact

def calculate_foul_impact(row, next_play, previous_plays, df):
    """Calculates enhanced impact value for fouls."""
    # Base value depends on foul type
    if isinstance(row.get('description'), str):
        if 'S.FOUL' in row.get('description'):
            base_impact = -0.7  # Shooting foul
        elif 'P.FOUL' in row.get('description'):
            base_impact = -0.3  # Personal foul
        elif 'OFF.FOUL' in row.get('description') or 'Offensive' in row.get('description'):
            base_impact = -0.6  # Offensive foul
        elif 'L.B.FOUL' in row.get('description'):
            base_impact = -0.4  # Loose ball foul
        elif 'T.FOUL' in row.get('description'):
            base_impact = -1.0  # Technical foul
        elif 'FLAGRANT' in row.get('description').upper():
            base_impact = -1.5  # Flagrant foul
        else:
            base_impact = -0.5  # Default foul value
    else:
        base_impact = -0.5  # Default if description missing

    # Enhanced modifiers
    # Foul trouble context
    foul_count = 1
    player_name = row.get('playerName')
    if player_name:
        # FIX: Need to handle previous fouls differently to avoid the str.get() error
        previous_fouls = []
        for idx in range(df.index.get_loc(row.name)):
            prev_row = df.iloc[idx]
            desc = prev_row.get('description', '')
            if isinstance(desc, str) and 'FOUL' in desc and prev_row.get('playerName') == player_name:
                previous_fouls.append(prev_row)
                
        foul_count += len(previous_fouls)

    # Scale impact based on foul count
    if foul_count == 2:
        base_impact *= 1.1  # 10% worse
    elif foul_count == 3:
        base_impact *= 1.2  # 20% worse
    elif foul_count == 4:
        base_impact *= 1.4  # 40% worse
    elif foul_count >= 5:
        base_impact *= 1.6  # 60% worse

    # Bonus situation
    if next_play is not None and isinstance(next_play.get('description'), str) and 'Free Throw' in next_play.get('description'):
        # Free throws without shooting foul means team in bonus
        if not isinstance(row.get('description'), str) or 'S.FOUL' not in row.get('description'):
            base_impact -= 0.2  # Worse for putting team in bonus

    # Game context
    if is_last_2_minutes(row.get('clock_seconds'), row.get('period')):
        margin = get_score_margin(row)
        if margin <= 3:
            base_impact *= 1.2  # 20% worse in close, late-game situations

    # Intentional foul strategy context (positive for trailing team)
    if is_last_2_minutes(row.get('clock_seconds'), row.get('period')):
        team_id = row.get('teamId')
        if (team_id == 1610612743 and row.get('scoreHome', 0) < row.get('scoreAway', 0)) or \
           (team_id == 1610612744 and row.get('scoreHome', 0) > row.get('scoreAway', 0)):
            # Trailing team fouling
            margin = get_score_margin(row)
            if 3 <= margin <= 7:
                base_impact *= 0.7  # 30% less negative (strategic foul)

    return base_impact

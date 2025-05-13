import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle, Rectangle, Arc
import time
from collections import defaultdict
import warnings
from datetime import datetime
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
import pickle

warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

# --- Global cache for previously processed games ---
GAME_CACHE_FILE = "game_impact_cache.pkl"
game_impact_cache = {}

# --- Data Loading and Preprocessing ---
def load_and_preprocess_pbp_file(filepath):
    """
    Loads and preprocesses a single play-by-play CSV file.
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Preprocessed play-by-play data
    """
    try:
        # Check if this game is in the cache by extracting game_id from filepath
        game_id = os.path.basename(filepath).split('pbp')[0]
        
        # If we have cached results for this game, return None to skip processing
        if game_id in game_impact_cache:
            return None
        
        # Extract game ID from filepath for reference
        df_pbp = pd.read_csv(filepath)
        
        # Early filtering: drop rows without player names (non-player events)
        df_pbp = df_pbp.dropna(subset=['playerName'])
        
        # Early filtering: focus only on relevant action types
        relevant_actions = ['Made Shot', 'Missed Shot', 'Rebound', 'Turnover', 'Foul', 'Free Throw']
        filter_mask = df_pbp['actionType'].isin(relevant_actions)
        
        # Also keep rows with descriptions containing specific keywords
        if 'description' in df_pbp.columns:
            description_mask = df_pbp['description'].astype(str).str.contains('STEAL|BLOCK', case=True, na=False)
            filter_mask = filter_mask | description_mask
            
        df_pbp = df_pbp[filter_mask].copy()
        
        # Add game_id column for tracking
        df_pbp['game_id'] = game_id
        
        # Extract game date from the data if available
        game_date = None
        if 'gameTimeUTC' in df_pbp.columns and not df_pbp['gameTimeUTC'].empty:
            try:
                game_date = pd.to_datetime(df_pbp['gameTimeUTC'].iloc[0]).strftime('%Y-%m-%d')
            except:
                # Try to extract date from game_id (format: 0022300001 - where 2023 is the season)
                try:
                    season_year = int("20" + game_id[2:4])
                    game_date = f"{season_year}-{season_year+1} season"
                except:
                    game_date = "Unknown date"
        else:
            game_date = "Unknown date"
        
        df_pbp['game_date'] = game_date
        
        # Extract home and away team names
        home_team = df_pbp['teamNameHome'].iloc[0] if 'teamNameHome' in df_pbp.columns and not df_pbp.empty else "Home team"
        away_team = df_pbp['teamNameAway'].iloc[0] if 'teamNameAway' in df_pbp.columns and not df_pbp.empty else "Away team"
        df_pbp['match_description'] = f"{away_team} @ {home_team}"
        
        # Convert 'clock' to total seconds from the start of the period
        df_pbp['clock_seconds'] = df_pbp['clock'].apply(
            lambda x: int(x.split('PT')[1].split('M')[0]) * 60 + float(x.split('M')[1].replace('S', '')) 
            if isinstance(x, str) and 'PT' in x else 0
        )

        # Add additional columns for analysis
        df_pbp['score_margin'] = df_pbp.apply(
            lambda row: abs(row['scoreHome'] - row['scoreAway']) 
            if pd.notnull(row['scoreHome']) and pd.notnull(row['scoreAway']) else 0, 
            axis=1
        )
        
        # Calculate clutch time flags in a vectorized way
        df_pbp['is_clutch'] = ((df_pbp['period'] == 4) & (df_pbp['clock_seconds'] <= 300)) | (df_pbp['period'] > 4)
        df_pbp['is_last_2min'] = ((df_pbp['period'] == 4) & (df_pbp['clock_seconds'] <= 120)) | (df_pbp['period'] > 4)

        # Create a shot distance bin for spatial analysis - only for shots
        shot_mask = df_pbp['actionType'].isin(['Made Shot', 'Missed Shot'])
        df_pbp.loc[shot_mask, 'shot_distance_bin'] = df_pbp.loc[shot_mask, 'shotDistance'].apply(
            lambda x: categorize_shot_distance(x) if pd.notnull(x) else None
        )

        # Create a column for expected points based on shot location - only for shots
        df_pbp.loc[shot_mask, 'expected_points'] = df_pbp.loc[shot_mask].apply(
            lambda row: calculate_expected_points(row['xLegacy'], row['yLegacy'], row['shotValue'])
            if pd.notnull(row.get('xLegacy', None)) and pd.notnull(row.get('yLegacy', None)) and pd.notnull(row.get('shotValue', None))
            else None, 
            axis=1
        )

        return df_pbp
        
    except Exception as e:
        print(f"Error preprocessing file {filepath}: {e}")
        return None

# --- Helper Functions ---
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

# --- Enhanced Impact Score Calculations ---
def calculate_block_impact(row, next_play, previous_plays, df):
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

    return base_impact

def calculate_steal_impact(row, next_play, previous_plays):
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

    return base_impact

def calculate_rebound_impact(row, next_play, previous_plays):
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

    return base_impact

def calculate_scoring_impact(row, previous_plays, df):
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

def calculate_enhanced_impact_score(df):
    """
    Calculates the enhanced impact score with contextual modifiers using play-by-play data.
    
    Args:
        df (pd.DataFrame): Play-by-play data
        
    Returns:
        pd.Series: Player impact scores
    """
    # If DataFrame is None (cached) or empty, return empty Series
    if df is None or df.empty:
        return pd.Series()
    
    player_impact = defaultdict(float)
    
    # Create a lookup dictionary for common checks to avoid repeated computation
    clutch_time_lookup = dict(zip(df.index, df['is_clutch']))
    last_2min_lookup = dict(zip(df.index, df['is_last_2min']))
    score_margin_lookup = dict(zip(df.index, df['score_margin']))

    for index, row in df.iterrows():
        if pd.isna(row.get('playerName')):
            continue

        player = row['playerName']
        
        # Get next play more efficiently
        next_idx = index + 1
        next_play = df.iloc[next_idx].to_dict() if next_idx < len(df) else None
        
        # Get previous plays more efficiently - limit to only what's needed
        prev_start = max(0, df.index.get_loc(index) - 5)
        prev_end = df.index.get_loc(index)
        previous_plays = [row.to_dict() for _, row in df.iloc[prev_start:prev_end].iterrows()]

        impact = 0  # Initialize

        # --- Base Impact (from play-by-play) with enhanced contextual modifiers ---
        # Use string methods for faster string checks
        desc = row.get('description', '')
        if isinstance(desc, str):
            if 'BLOCK' in desc:
                impact += calculate_block_impact(row, next_play, previous_plays, df)
            elif 'STEAL' in desc:
                impact += calculate_steal_impact(row, next_play, previous_plays)
            elif 'Foul' in desc:
                impact += calculate_foul_impact(row, next_play, previous_plays, df)

        action_type = row.get('actionType')
        if action_type == 'Rebound':
            impact += calculate_rebound_impact(row, next_play, previous_plays)
        elif action_type == 'Made Shot':
            impact += calculate_scoring_impact(row, previous_plays, df)
        elif action_type == 'Turnover':
            impact += calculate_turnover_impact(row, next_play, previous_plays)

        # --- Modifiers based on game context ---
        # Use pre-computed lookups for faster access
        if clutch_time_lookup.get(index, False):
            impact *= 1.5

        # Time Remaining (More Granular)
        if pd.notnull(row.get('clock_seconds')):
            time_remaining_factor = 1 + (1 / (row['clock_seconds'] + 1))
            impact *= time_remaining_factor

        player_impact[player] += impact

    return pd.Series(player_impact).sort_values(ascending=False)

def plot_impact_scores(impact_scores, title="Enhanced Impact Scores", top_n=15, save_filename=None):
    """Plots a bar chart of player impact scores."""
    plt.figure(figsize=(14, 8))

    # Get top N players
    if isinstance(impact_scores, pd.Series):
        top_impact = impact_scores.head(top_n)
    else:
        # Handle DataFrame case (for worst performers that might be negative)
        top_impact = impact_scores

    # Set colormap for visual appeal
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_impact)))

    ax = sns.barplot(x=top_impact.index, y=top_impact.values, palette=colors)

    # Add value labels on top of bars
    for i, v in enumerate(top_impact.values):
        ax.text(i, v + 0.5 if v >= 0 else v - 0.5, f"{v:.1f}", ha='center', fontsize=9)

    plt.title(title, fontsize=16, fontweight='bold', ha="center", color='black')
    plt.ylabel("Impact Score", fontsize=12)
    plt.xlabel("")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Use provided filename or generate one from title
    if save_filename:
        plt.savefig(save_filename)
    else:
        plt.savefig(f"{title.replace(' ', '_')}.png")
    
    plt.close()

# --- Process a single game function (for parallel processing) ---
def process_single_game(pbp_file):
    """Process a single game and return the impact scores and game metadata."""
    try:
        # Check if game is in cache
        game_id = os.path.basename(pbp_file).split('pbp')[0]
        if game_id in game_impact_cache:
            return game_impact_cache[game_id]
        
        # Load and preprocess data
        game_data = load_and_preprocess_pbp_file(pbp_file)
        
        # Skip if game data is None (already in cache) or empty
        if game_data is None or game_data.empty:
            return None
        
        # Extract game metadata
        game_id = game_data['game_id'].iloc[0]
        game_date = game_data['game_date'].iloc[0]
        match_description = game_data['match_description'].iloc[0]
        
        # Calculate impact scores
        impact_scores = calculate_enhanced_impact_score(game_data)
        
        # Create result object
        result = {
            'game_id': game_id,
            'game_date': game_date,
            'match_description': match_description,
            'impact_scores': impact_scores,
            'top_performers': [],
            'worst_performers': []
        }
        
        # Add top and worst performers
        if not impact_scores.empty:
            # Get top 3 performers
            top_players = impact_scores.head(3)
            for player, score in top_players.items():
                result['top_performers'].append({
                    'game_id': game_id,
                    'game_date': game_date,
                    'matchup': match_description,
                    'player_name': player,
                    'impact_score': score
                })
            
            # Get worst 3 performers
            worst_players = impact_scores.tail(3)
            for player, score in worst_players.items():
                result['worst_performers'].append({
                    'game_id': game_id,
                    'game_date': game_date,
                    'matchup': match_description,
                    'player_name': player,
                    'impact_score': score
                })
        
        # Add to cache
        game_impact_cache[game_id] = result
        
        return result
        
    except Exception as e:
        print(f"Error processing game {pbp_file}: {e}")
        return None

# --- Main Analysis Function ---
def analyze_season_data(use_multiprocessing=True, max_games=None, load_cache=True, save_cache=True):
    """
    Analyzes NBA season data to calculate impact scores for all games 
    and find the top and worst performing players across the season.
    
    Args:
        use_multiprocessing (bool): Whether to use multiprocessing
        max_games (int, optional): Maximum number of games to process (for testing)
        load_cache (bool): Whether to load cached results
        save_cache (bool): Whether to save results to cache
        
    Returns:
        tuple: (avg_impact_series, worst_avg_impact_series, top_performers_df, worst_performers_df)
    """
    print("Starting NBA season impact score analysis...")
    start_time = time.time()
    
    # Load cache if requested
    global game_impact_cache
    if load_cache and os.path.exists(GAME_CACHE_FILE):
        try:
            with open(GAME_CACHE_FILE, 'rb') as f:
                game_impact_cache = pickle.load(f)
            print(f"Loaded {len(game_impact_cache)} games from cache")
        except Exception as e:
            print(f"Error loading cache: {e}")
            game_impact_cache = {}
    
    # Path to the directory containing all the game data
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nba_data", "2023_2024")
    
    # Find all play-by-play CSV files
    print("Scanning for play-by-play data...")
    pbp_files = []
    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            game_dir = os.path.join(root, dir_name)
            pbp_dir = os.path.join(game_dir, "play_by_play")
            if os.path.exists(pbp_dir):
                for file in os.listdir(pbp_dir):
                    if file.endswith("pbp.csv"):
                        pbp_files.append(os.path.join(pbp_dir, file))
    
    # Limit number of games if specified
    if max_games is not None and max_games > 0:
        pbp_files = pbp_files[:max_games]
        
    print(f"Found {len(pbp_files)} play-by-play files")
    
    # Track all game results and performances
    all_game_results = []
    
    # Process games
    if use_multiprocessing and len(pbp_files) > 1:
        # Use multiprocessing for faster processing
        num_cores = max(1, mp.cpu_count() - 1)  # Leave one core free
        print(f"Using {num_cores} CPU cores for parallel processing")
        
        with mp.Pool(processes=num_cores) as pool:
            # Process games in parallel with progress bar
            all_game_results = list(tqdm(
                pool.imap(process_single_game, pbp_files),
                total=len(pbp_files),
                desc="Processing games"
            ))
    else:
        # Process games sequentially with progress bar
        for pbp_file in tqdm(pbp_files, desc="Processing games"):
            result = process_single_game(pbp_file)
            all_game_results.append(result)
    
    # Filter out None results
    all_game_results = [result for result in all_game_results if result is not None]
    
    # Save cache if requested
    if save_cache:
        try:
            with open(GAME_CACHE_FILE, 'wb') as f:
                pickle.dump(game_impact_cache, f)
            print(f"Saved {len(game_impact_cache)} games to cache")
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    # Consolidate results
    print("Consolidating results...")
    
    # Track player statistics
    player_game_counts = defaultdict(int)
    player_total_impact = defaultdict(float)
    game_top_performers = []
    game_worst_performers = []
    
    for result in all_game_results:
        # Add top and worst performers
        game_top_performers.extend(result['top_performers'])
        game_worst_performers.extend(result['worst_performers'])
        
        # Update player statistics
        for player, score in result['impact_scores'].items():
            player_total_impact[player] += score
            player_game_counts[player] += 1
    
    # Calculate average impact score per game for each player
    player_avg_impact = {}
    for player, total in player_total_impact.items():
        games_played = player_game_counts[player]
        if games_played >= 5:  # Only include players with enough games
            player_avg_impact[player] = total / games_played
    
    # Convert to Series for easier sorting
    avg_impact_series = pd.Series(player_avg_impact).sort_values(ascending=False)
    
    # Get bottom 10 players by average impact
    worst_avg_impact_series = pd.Series(player_avg_impact).sort_values(ascending=True).head(10)
    
    # Convert performance data to DataFrames for analysis
    top_performers_df = pd.DataFrame(game_top_performers)
    worst_performers_df = pd.DataFrame(game_worst_performers)
    
    # Find players with most top performances
    top_performer_counts = top_performers_df['player_name'].value_counts() if not top_performers_df.empty else pd.Series()
    
    # Find exceptional single-game performances (both best and worst)
    if not top_performers_df.empty:
        exceptional_games = top_performers_df.sort_values('impact_score', ascending=False).head(10)
    else:
        exceptional_games = pd.DataFrame()
        
    if not worst_performers_df.empty:
        worst_games = worst_performers_df.sort_values('impact_score', ascending=True).head(10)
    else:
        worst_games = pd.DataFrame()
    
    # Print and visualize results
    print("\n=== Season Analysis Results ===")
    
    print("\nTop 10 Players by Average Impact Score:")
    print(avg_impact_series.head(10))
    plot_impact_scores(avg_impact_series, "Top 10 Average Impact Score Per Game", 10, "Top_10_Average_Impact_Scores.png")
    
    print("\nWorst 10 Players by Average Impact Score:")
    print(worst_avg_impact_series)
    plot_impact_scores(worst_avg_impact_series, "Bottom 10 Average Impact Score Per Game", 10, "Bottom_10_Average_Impact_Scores.png")
    
    print("\nPlayers with Most Top Performances:")
    print(top_performer_counts.head(10))
    
    if not exceptional_games.empty:
        print("\nTop 10 Individual Game Performances:")
        print(exceptional_games[['game_date', 'game_id', 'matchup', 'player_name', 'impact_score']])
    
    if not worst_games.empty:
        print("\nWorst 10 Individual Game Performances:")
        print(worst_games[['game_date', 'game_id', 'matchup', 'player_name', 'impact_score']])
    
    # Save results to CSV
    avg_impact_series.to_csv("average_impact_scores.csv")
    worst_avg_impact_series.to_csv("worst_average_impact_scores.csv")
    
    if not top_performers_df.empty:
        top_performers_df.to_csv("game_top_performers.csv", index=False)
    
    if not worst_performers_df.empty:
        worst_performers_df.to_csv("game_worst_performers.csv", index=False)
    
    # Create a consolidated top/worst performances file
    if not exceptional_games.empty and not worst_games.empty:
        top_10_best = exceptional_games[['game_date', 'game_id', 'matchup', 'player_name', 'impact_score']].head(10)
        top_10_best['performance_type'] = 'Best'
        top_10_worst = worst_games[['game_date', 'game_id', 'matchup', 'player_name', 'impact_score']].head(10)
        top_10_worst['performance_type'] = 'Worst'
        
        combined_performances = pd.concat([top_10_best, top_10_worst])
        combined_performances.to_csv("top_and_worst_game_performances.csv", index=False)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nAnalysis completed in {elapsed_time:.2f} seconds!")
    
    return avg_impact_series, worst_avg_impact_series, top_performers_df, worst_performers_df

if __name__ == "__main__":
    try:
        # Run the season analysis with multiprocessing
        # Set max_games to None to process all games 
        # (or a small number like 10 for testing)
        top_avg_impact, worst_avg_impact, top_games, worst_games = analyze_season_data(
            use_multiprocessing=True,
            max_games=None,
            load_cache=True,
            save_cache=True
        )
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
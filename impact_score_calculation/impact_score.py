import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle, Rectangle, Arc
import os
from sklearn.preprocessing import MinMaxScaler

# --- Data Loading and Preprocessing ---
def load_and_preprocess_data():
    """Loads and preprocesses all available CSV files."""
    
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load dataframes using full paths
    df_pbp = pd.read_csv(os.path.join(script_dir, "detailed_play_by_play_0022400058_1.csv"))
    df_adv = pd.read_csv(os.path.join(script_dir, "0022400058_box_scores_advanced.csv"))
    df_match_detailed = pd.read_csv(os.path.join(script_dir, "0022400058_box_scores_matchups.csv"))
    df_traditional = pd.read_csv(os.path.join(script_dir, '0022400058_box_scores_traditional.csv'))
    
    # Load additional data sources
    df_defensive = pd.read_csv(os.path.join(script_dir, '0022400058_box_scores_defensive.csv'))
    df_player_track = pd.read_csv(os.path.join(script_dir, '0022400058_box_scores_player_track.csv'))

    # --- Preprocessing ---
    # 1. Play-by-Play Data
    # Convert 'clock' to total seconds from the start of the period.
    df_pbp['clock_seconds'] = df_pbp['clock'].apply(lambda x: int(x.split('PT')[1].split('M')[0]) * 60 + float(x.split('M')[1].replace('S', '')) if 'PT' in str(x) else 0)
    
    # Add additional columns for analysis
    df_pbp['score_margin'] = df_pbp.apply(lambda row: abs(row['scoreHome'] - row['scoreAway']) if pd.notnull(row['scoreHome']) and pd.notnull(row['scoreAway']) else 0, axis=1)
    df_pbp['is_clutch'] = df_pbp.apply(lambda row: is_clutch_time(row['clock_seconds'], row['period']), axis=1)
    df_pbp['is_last_2min'] = df_pbp.apply(lambda row: is_last_2_minutes(row['clock_seconds'], row['period']), axis=1)
    
    # Create a shot distance bin for spatial analysis
    df_pbp['shot_distance_bin'] = df_pbp['shotDistance'].apply(lambda x: categorize_shot_distance(x) if pd.notnull(x) else None)
    
    # Create a column for expected points based on shot location
    df_pbp['expected_points'] = df_pbp.apply(lambda row: 
        calculate_expected_points(row['xLegacy'], row['yLegacy'], row['shotValue']) 
        if pd.notnull(row['xLegacy']) and pd.notnull(row['yLegacy']) and pd.notnull(row['shotValue']) 
        else None, axis=1)

    # 2. Advanced Stats - handle missing data
    df_adv.fillna(0, inplace=True)

    # 3. Matchup data - Correctly create matchup_id for merging.
    df_match_detailed['matchup_id'] = df_match_detailed.apply(lambda row: tuple(sorted((row['personIdOff'], row['personIdDef']))), axis=1)

    # Add player names for merging/readability
    player_id_name_map = df_pbp[['personId', 'playerName']].dropna().drop_duplicates().set_index('personId')['playerName'].to_dict()
    df_match_detailed['playerNameOff'] = df_match_detailed['personIdOff'].map(player_id_name_map)
    df_match_detailed['playerNameDef'] = df_match_detailed['personIdDef'].map(player_id_name_map)
    
    # 4. Player tracking data cleanup
    df_player_track.fillna(0, inplace=True)
    
    # 5. Defensive data cleanup
    df_defensive.fillna(0, inplace=True)

    # 6- Merge df_pbp and df_adv
    df_merged = pd.merge(df_pbp, df_adv, left_on=['personId', 'teamId'], right_on=['PLAYER_ID', 'TEAM_ID'], how='left', suffixes=('', '_adv'))

    # 7- Correctly aggregate Matchup data (using only df_match_detailed)
    grouped_detailed = df_match_detailed.groupby('matchup_id').agg({
        'matchupMinutes': 'sum',
        'partialPossessions': 'sum',
        'playerPoints': 'sum',
        'matchupAssists': 'sum',
        'matchupTurnovers': 'sum',
        'matchupFieldGoalsMade': 'sum',
        'matchupFieldGoalsAttempted': 'sum',
        'matchupThreePointersMade': 'sum',
        'matchupThreePointersAttempted': 'sum',
        'playerNameOff': 'first',  # Keep player names
        'playerNameDef': 'first'
    }).reset_index()

    # Rename for clarity
    grouped_detailed.rename(columns={
        'matchupMinutes': 'totalMatchupMinutes',
        'partialPossessions' : 'totalPartialPossessions',
        'playerPoints' : 'totalPlayerPoints',
        'matchupAssists' : 'totalMatchupAssists',
        'matchupTurnovers' : 'totalMatchupTurnovers',
        'matchupFieldGoalsMade' : 'totalMatchupFieldGoalsMade',
        'matchupFieldGoalsAttempted' : 'totalMatchupFieldGoalsAttempted',
        'matchupThreePointersMade' : 'totalMatchupThreePointersMade',
        'matchupThreePointersAttempted' : 'totalMatchupThreePointersAttempted'
    }, inplace=True)

    df_matchup = grouped_detailed  # This is our main matchup data.
    
    # 8. Calculate possession estimates for normalization
    team_possessions = calculate_team_possessions(df_traditional)
    
    # 9. Create a player to team mapping for reference
    player_team_map = df_traditional[['PLAYER_NAME', 'TEAM_ABBREVIATION']].drop_duplicates().set_index('PLAYER_NAME')['TEAM_ABBREVIATION'].to_dict()
    
    return df_merged, df_matchup, df_match_detailed, df_pbp, df_defensive, df_player_track, team_possessions, player_team_map

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

# --- Enhanced Impact Score Calculations ---

def calculate_block_impact(row, next_play, previous_plays, df_defensive=None):
    """Calculates enhanced impact value for blocks."""
    base_impact = 1.2
    
    # Original modifiers
    if next_play is not None and next_play['teamTricode'] == row['teamTricode']:
        base_impact -= 0.2  # Block that stays with blocking team (reduced value)
    if next_play is not None and isinstance(next_play.get('description'), str) and 'Running' in next_play['description']:
        base_impact += 0.2  # Block leading to transition
    if next_play is not None and isinstance(next_play.get('description'), str) and 'Shot Clock' in next_play['description']:
        base_impact += 0.3  # Block causing shot clock violation
    
    # Multiple blocks demonstrating defensive dominance
    recent_blocks = [play for play in previous_plays[-3:] 
                    if isinstance(play.get('description'), str) and 'BLOCK' in play['description']
                    and play.get('playerName') == row.get('playerName')]
    if len(recent_blocks) > 1:
        base_impact += 0.3
    
    # Clutch time blocks
    if is_last_2_minutes(row['clock_seconds'], row['period']) and get_score_margin(row) <= 3:
        base_impact += 0.5
    
    # Enhanced modifiers
    # Block location value (rim protection)
    if pd.notnull(row.get('shotDistance')) and row['shotDistance'] <= 5:
        base_impact += 0.2  # Blocks at the rim
    
    # Block against a scoring run
    scoring_run_team = identify_scoring_run(df_pbp, df_pbp.index.get_loc(row.name))
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
    if isinstance(row.get('description'), str) and 'Backcourt' in row['description']:
        base_impact += 0.1  # Backcourt steal (higher pressure)
    if next_play is not None and next_play.get('actionType') == 'Made Shot':
        base_impact += 0.2  # Steal leading to immediate score
    
    # Multiple steals demonstrating defensive prowess
    recent_steals = [play for play in previous_plays[-5:] 
                    if isinstance(play.get('description'), str) and 'STEAL' in play['description']
                    and play.get('playerName') == row.get('playerName')]
    if len(recent_steals) > 1:
        base_impact += 0.2
    
    # Game situation adjustments
    if is_clutch_time(row['clock_seconds'], row['period']):
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
        if 'Bad Pass' in row['description']:
            base_impact += 0.1  # Anticipation steal (passing lane)
        elif 'Lost Ball' in row['description']:
            base_impact += 0.3  # Active pickpocket (direct steal)
    
    # Steal leading to breakaway
    if next_play is not None and next_play.get('actionType') == 'Made Shot' and pd.notnull(next_play.get('shotDistance')) and next_play['shotDistance'] <= 3:
        base_impact += 0.3  # Steal leading to easy basket
    
    # Steal in opponent's frontcourt
    if pd.notnull(row.get('xLegacy')) and pd.notnull(row.get('yLegacy')):
        # Check if steal is in opponent's half
        team_id = row.get('teamId')
        if (team_id == 1610612743 and row['xLegacy'] < 0) or (team_id == 1610612744 and row['xLegacy'] > 0):
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
    is_offensive = isinstance(row.get('description'), str) and 'Off' in row['description']
    base_impact = 0.9 if is_offensive else 0.6
    
    # Original modifiers
    if any(isinstance(play.get('description'), str) and 'REBOUND' in play['description'] for play in previous_plays[-2:]):
        base_impact += 0.2  # Multiple rebounds in sequence
    if next_play is not None and next_play.get('actionType') == 'Made Shot':
        base_impact += 0.2  # Rebound leading to score
    if is_last_2_minutes(row['clock_seconds'], row['period']) and get_score_margin(row) <= 3:
        base_impact += 0.3  # Critical late-game rebounds
    if is_clutch_time(row['clock_seconds'], row['period']):
        base_impact += 0.4  # Clutch time rebounds
    
    # Enhanced modifiers
    # Rebound after contested shot
    if any(isinstance(play.get('description'), str) and 'BLOCK' in play['description'] for play in previous_plays[-1:]):
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
            if pd.notnull(next_play.get('clock_seconds')) and row.get('clock_seconds') - next_play['clock_seconds'] < 3:
                base_impact += 0.2  # Quick putback attempt
    else:  # Defensive rebound
        # Leading to fast break
        if next_play is not None and next_play.get('actionType') in ['Made Shot', 'Missed Shot'] and next_play.get('teamTricode') == row.get('teamTricode'):
            if pd.notnull(next_play.get('clock_seconds')) and row.get('clock_seconds') - next_play['clock_seconds'] < 5:
                base_impact += 0.2  # Quick transition after defensive rebound
    
    # Shot clock context
    previous_shot = next((play for play in previous_plays[-2:] if play.get('actionType') in ['Made Shot', 'Missed Shot']), None)
    if previous_shot and pd.notnull(previous_shot.get('clock_seconds')):
        shot_clock_value = previous_shot['clock_seconds'] % 24
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

def calculate_scoring_impact(row, previous_plays, df_player_track=None):
    """Calculates enhanced impact value for scoring plays."""
    base_impact = 3.0 if row.get('shotValue') == 3 else 2.0
    
    # Original modifiers
    if any(isinstance(play.get('description'), str) and 'Free Throw' in play['description'] for play in previous_plays[:2]):
        base_impact += 0.3  # And-one plays
    if any(isinstance(play.get('description'), str) and 'Timeout' in play['description'] for play in previous_plays[-3:]):
        base_impact += 0.2  # Scoring after timeout
    
    scoring_run_team = identify_scoring_run(df_pbp, df_pbp.index.get_loc(row.name))
    if scoring_run_team and scoring_run_team != row.get('teamTricode'):
        base_impact += 0.2  # Stopping opponent's run
    
    if any(isinstance(play.get('description'), str) and 'Start of' in play['description'] for play in previous_plays[-3:]):
        base_impact += 0.1  # Period-starting baskets
    
    # Enhanced modifiers
    # Shot difficulty based on spatial data
    if pd.notnull(row.get('expected_points')):
        # Adjust impact based on expected value
        ep_modifier = row['expected_points']
        # Score higher than expected = more valuable
        base_impact *= ep_modifier
    
    # Shot difficulty based on shot description
    if isinstance(row.get('description'), str):
        # Special shot types
        if 'Fadeaway' in row['description']:
            base_impact += 0.2  # Difficult fadeaway shot
        elif 'Step Back' in row['description']:
            base_impact += 0.3  # Difficult step back
        elif 'Driving' in row['description'] and 'Dunk' in row['description']:
            base_impact += 0.3  # Athletic driving dunk
        elif 'Alley Oop' in row['description']:
            base_impact += 0.4  # Highlight play
        elif 'Turnaround' in row['description']:
            base_impact += 0.2  # Difficult post move
        elif 'Pullup' in row['description']:
            base_impact += 0.1  # Pull-up jumper
        elif 'Bank' in row['description']:
            base_impact += 0.1  # Bank shot
    
    # Shot timing context
    if pd.notnull(row.get('clock_seconds')):
        shot_clock_value = row['clock_seconds'] % 24
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
        prev_diff = prev_score_play['scoreHome'] - prev_score_play['scoreAway']
        curr_diff = row['scoreHome'] - row['scoreAway']
        
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
    if is_clutch_time(row['clock_seconds'], row['period']):
        margin = get_score_margin(row)
        if margin <= 5:
            base_impact *= 1.3  # 30% boost for scoring in close clutch situations
        elif margin <= 10:
            base_impact *= 1.2  # 20% boost for scoring in moderate clutch situations
    
    return base_impact

def calculate_turnover_impact(row, next_play, previous_plays):
    """Calculates enhanced impact value for turnovers."""
    base_impact = -1.0 if is_clutch_time(row['clock_seconds'], row['period']) else -0.8
    
    # Enhanced modifiers
    # Turnover type
    if isinstance(row.get('description'), str):
        if 'Bad Pass' in row['description']:
            base_impact -= 0.2  # Decision error (worse)
        elif 'Lost Ball' in row['description']:
            base_impact -= 0.3  # Ball handling error (worse)
        elif 'Step Out of Bounds' in row['description'] or 'Traveling' in row['description']:
            base_impact -= 0.1  # Unforced error (slightly better)
        elif 'Shot Clock' in row['description']:
            base_impact -= 0.3  # Team failure to get shot off
        elif 'Offensive Foul' in row['description']:
            base_impact -= 0.2  # Aggressive error
        elif 'Backcourt' in row['description']:
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
    if margin <= 5 and row['period'] >= 4:
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
    if any(isinstance(play.get('description'), str) and 'Timeout' in play['description'] for play in previous_plays[-3:]):
        base_impact -= 0.2  # Turnover after timeout
    
    return base_impact

def calculate_foul_impact(row, next_play, previous_plays):
    """Calculates enhanced impact value for fouls."""
    # Base value depends on foul type
    if isinstance(row.get('description'), str):
        if 'S.FOUL' in row['description']: 
            base_impact = -0.7  # Shooting foul
        elif 'P.FOUL' in row['description']:
            base_impact = -0.3  # Personal foul
        elif 'OFF.FOUL' in row['description'] or 'Offensive' in row['description']:
            base_impact = -0.6  # Offensive foul
        elif 'L.B.FOUL' in row['description']:
            base_impact = -0.4  # Loose ball foul
        elif 'T.FOUL' in row['description']:
            base_impact = -1.0  # Technical foul
        elif 'FLAGRANT' in row['description'].upper():
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
        previous_fouls = [play for play in df_pbp.loc[:row.name-1] 
                        if isinstance(play.get('description'), str) 
                        and 'FOUL' in play['description'] 
                        and play.get('playerName') == player_name]
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
    if next_play is not None and isinstance(next_play.get('description'), str) and 'Free Throw' in next_play['description']:
        # Free throws without shooting foul means team in bonus
        if not isinstance(row.get('description'), str) or 'S.FOUL' not in row['description']:
            base_impact -= 0.2  # Worse for putting team in bonus
    
    # Game context
    if is_last_2_minutes(row['clock_seconds'], row['period']):
        margin = get_score_margin(row)
        if margin <= 3:
            base_impact *= 1.2  # 20% worse in close, late-game situations
    
    # Intentional foul strategy context (positive for trailing team)
    if is_last_2_minutes(row['clock_seconds'], row['period']):
        team_id = row.get('teamId')
        if (team_id == 1610612743 and row.get('scoreHome', 0) < row.get('scoreAway', 0)) or \
           (team_id == 1610612744 and row.get('scoreHome', 0) > row.get('scoreAway', 0)):
            # Trailing team fouling
            margin = get_score_margin(row)
            if 3 <= margin <= 7:
                base_impact *= 0.7  # 30% less negative (strategic foul)
    
    return base_impact

def calculate_defensive_impact(row, df_defensive):
    """Calculate enhanced defensive impact using defensive tracking data."""
    base_impact = 0
    
    # Get defensive data for this player
    if pd.notnull(row.get('personId')):
        player_def_data = df_defensive[df_defensive['personId'] == row['personId']]
        
        if not player_def_data.empty:
            # Add value for contested shots (even if they didn't result in blocks)
            contested_shots = player_def_data['matchupFieldGoalsAttempted'].values[0]
            contest_impact = contested_shots * 0.1
            
            # Add value for forcing misses
            fg_att = player_def_data['matchupFieldGoalsAttempted'].values[0]
            fg_made = player_def_data['matchupFieldGoalsMade'].values[0]
            if fg_att > 0:
                forced_miss_impact = (fg_att - fg_made) * 0.3
                # Bonus for low FG% allowed
                if fg_att >= 5 and fg_made/fg_att < 0.4:
                    forced_miss_impact *= 1.3
            else:
                forced_miss_impact = 0
                
            # Add value for defending 3-point shots
            fg3_att = player_def_data['matchupThreePointersAttempted'].values[0]
            fg3_made = player_def_data['matchupThreePointersMade'].values[0]
            if fg3_att > 0:
                forced_3pt_miss_impact = (fg3_att - fg3_made) * 0.4
            else:
                forced_3pt_miss_impact = 0
                
            base_impact = contest_impact + forced_miss_impact + forced_3pt_miss_impact
    
    return base_impact

def add_player_tracking_impact(row, df_player_track):
    """Add impact from player tracking data."""
    tracking_impact = 0
    
    if pd.notnull(row.get('personId')):
        player_tracking = df_player_track[df_player_track['PLAYER_ID'] == row['personId']]
        if not player_tracking.empty:
            # Value for distance covered (hustle)
            if 'DIST' in player_tracking.columns:
                dist = player_tracking['DIST'].values[0]
                tracking_impact += min(dist * 0.05, 1.0)  # Cap at 1.0
                
            # Value for secondary assists
            if 'SAST' in player_tracking.columns:
                tracking_impact += player_tracking['SAST'].values[0] * 0.5
                
            # Value for potential assists that didn't convert
            if 'PASS' in player_tracking.columns and 'AST' in player_tracking.columns:
                passes = player_tracking['PASS'].values[0]
                assists = max(1, player_tracking['AST'].values[0])
                pass_to_ast_ratio = passes / assists
                if pass_to_ast_ratio > 5:  # Creating opportunities even if not converted
                    tracking_impact += 0.3
    
    return tracking_impact

def calculate_enhanced_impact_score(df, df_defensive=None, df_player_track=None, team_possessions=None, player_team_map=None):
    """Calculates the enhanced impact score with contextual modifiers and additional data sources."""
    player_impact = {}
    player_context = {}  # Store context for normalization

    for index, row in df.iterrows():
        if pd.isna(row.get('playerName')):
            continue

        player = row['playerName']
        if player not in player_impact:
            player_impact[player] = 0
            player_context[player] = {'plays': 0, 'team': None}
        
        # Track player's team for possession normalization
        if player_team_map and player in player_team_map:
            player_context[player]['team'] = player_team_map[player]
        
        player_context[player]['plays'] += 1
        
        next_play = df.iloc[index + 1] if index < len(df) - 1 else None
        previous_plays = df.iloc[max(0, index - 5):index].to_dict('records')

        impact = 0  # Initialize

        # --- Base Impact (from play-by-play) with enhanced contextual modifiers ---
        if isinstance(row.get('description'), str) and 'BLOCK' in row['description']:
            impact += calculate_block_impact(row, next_play, previous_plays, df_defensive)
        elif isinstance(row.get('description'), str) and 'STEAL' in row['description']:
            impact += calculate_steal_impact(row, next_play, previous_plays, df_defensive)
        elif row.get('actionType') == 'Rebound':
            impact += calculate_rebound_impact(row, next_play, previous_plays, df_player_track)
        elif row.get('actionType') == 'Made Shot':
            impact += calculate_scoring_impact(row, previous_plays, df_player_track)
        elif isinstance(row.get('description'), str) and 'Foul' in row['description']:
            impact += calculate_foul_impact(row, next_play, previous_plays)
        elif row.get('actionType') == 'Turnover':
            impact += calculate_turnover_impact(row, next_play, previous_plays)

        # --- Advanced Stat Adjustments ---
        if 'PIE' in row and not pd.isna(row['PIE']):
            impact += row['PIE'] * 2  # Scale PIE
        if 'E_NET_RATING' in row and not pd.isna(row['E_NET_RATING']):
            impact += row['E_NET_RATING'] * 0.05  # Smaller weight
        if 'USG_PCT' in row and not pd.isna(row['USG_PCT']):
            impact *= (1 + row['USG_PCT'] * 0.5)  # Scale usage impact

        # --- Add Defensive Impact ---
        if df_defensive is not None:
            impact += calculate_defensive_impact(row, df_defensive) * 0.5  # Lower weight for background defensive impact

        # --- Add Player Tracking Impact ---
        if df_player_track is not None:
            impact += add_player_tracking_impact(row, df_player_track) * 0.2  # Lower weight for tracking data

        # --- Modifiers based on game context ---
        # Clutch Time
        if is_clutch_time(row.get('clock_seconds', 0), row.get('period', 0)):
            impact *= 1.5

        # Time Remaining (More Granular)
        if pd.notnull(row.get('clock_seconds')):
            time_remaining_factor = 1 + (1 / (row['clock_seconds'] + 1))
            impact *= time_remaining_factor

        player_impact[player] += impact

    # --- Post-processing: Normalize by possessions ---
    if team_possessions and player_team_map:
        normalized_impact = {}
        for player, impact in player_impact.items():
            team = player_context[player]['team']
            if team and team in team_possessions:
                # Normalize to 100 possessions
                poss_factor = 100 / max(1, team_possessions[team])
                normalized_impact[player] = impact * poss_factor
            else:
                normalized_impact[player] = impact
        
        return pd.Series(normalized_impact).sort_values(ascending=False)
    
    return pd.Series(player_impact).sort_values(ascending=False)

def calculate_matchup_impact(df_matchup, df_advanced=None):
    """Calculates a matchup impact score with enhanced metrics."""
    matchup_impact = {}

    required_columns = ['playerNameOff', 'playerNameDef', 'totalMatchupFieldGoalsMade', 'totalMatchupFieldGoalsAttempted', 'totalMatchupMinutes']
    if not all(col in df_matchup.columns for col in required_columns):
        print("Missing required columns in matchup data.")
        return pd.Series(matchup_impact)

    for _, row in df_matchup.iterrows():
        # Initialize if not already in dictionary
        matchup_key = (row['playerNameOff'], row['playerNameDef'])
        reverse_key = (row['playerNameDef'], row['playerNameOff'])
        if matchup_key not in matchup_impact:
            matchup_impact[matchup_key] = 0
        if reverse_key not in matchup_impact:
            matchup_impact[reverse_key] = 0

        # Basic scoring impact
        points_scored = row['totalMatchupFieldGoalsMade'] * 2
        points_possible = row['totalMatchupFieldGoalsAttempted'] * 2
        
        # Add 3-point impact
        if 'totalMatchupThreePointersMade' in row and 'totalMatchupThreePointersAttempted' in row:
            points_scored += row['totalMatchupThreePointersMade'] * 1  # Additional point for 3s
            three_pt_efficiency = row['totalMatchupThreePointersMade'] / max(1, row['totalMatchupThreePointersAttempted'])
            points_scored += three_pt_efficiency * 2  # Bonus for 3pt efficiency
        
        # Scoring efficiency impact
        fg_pct = row['totalMatchupFieldGoalsMade'] / max(1, row['totalMatchupFieldGoalsAttempted'])
        efficiency_factor = 1.0
        if fg_pct > 0.55:  # High efficiency
            efficiency_factor = 1.3
        elif fg_pct < 0.35:  # Low efficiency
            efficiency_factor = 0.7
        
        scoring_impact = (points_scored - (0.5 * points_possible)) * efficiency_factor
        
        # Adjust for assists and turnovers
        if 'totalMatchupAssists' in row:
            scoring_impact += row['totalMatchupAssists'] * 0.8
        if 'totalMatchupTurnovers' in row:
            scoring_impact -= row['totalMatchupTurnovers'] * 0.5

        # Adjust for matchup minutes (avoid extreme values)
        minutes_factor = np.sqrt(row['totalMatchupMinutes']) if row['totalMatchupMinutes'] > 0 else 0
        scoring_impact *= minutes_factor
        
        # Adjust for possession intensity
        if 'totalPartialPossessions' in row:
            poss_per_minute = row['totalPartialPossessions'] / max(1, row['totalMatchupMinutes'])
            if poss_per_minute > 1.2:  # High intensity matchup
                scoring_impact *= 1.2

        # Add to offensive player, subtract from defensive
        matchup_impact[matchup_key] += scoring_impact
        matchup_impact[reverse_key] -= scoring_impact * 0.8  # Slightly reduced negative impact

    # Apply advanced stats modifier if available
    if df_advanced is not None:
        player_pie = {}
        for _, row in df_advanced.iterrows():
            if not pd.isna(row.get('NICKNAME')):
                player_pie[row['NICKNAME']] = row.get('PIE', 0)
        
        # Adjust matchup impact by PIE (Player Impact Estimate)
        adjusted_impact = {}
        for matchup, impact in matchup_impact.items():
            off_player, def_player = matchup
            off_player_first = off_player.split(' ')[0] if ' ' in off_player else off_player
            def_player_first = def_player.split(' ')[0] if ' ' in def_player else def_player
            
            pie_modifier = 1.0
            if off_player_first in player_pie:
                pie_modifier += player_pie[off_player_first] * 2
            if def_player_first in player_pie:
                pie_modifier += player_pie[def_player_first]
                
            adjusted_impact[matchup] = impact * pie_modifier
            
        matchup_impact = adjusted_impact

    return pd.Series(matchup_impact).sort_values(ascending=False)

# --- Visualization Functions ---

def create_shot_chart(df, team_tricode, title="Shot Chart"):
    """Creates a shot chart for a specific team."""
    shots = df[(df['actionType'].isin(['Made Shot', 'Missed Shot'])) & (df['teamTricode'] == team_tricode)].copy()
    
    # Add shot value and distance bin for better visualization
    shots['shot_type'] = shots['shotValue'].apply(lambda x: "3PT" if x == 3 else "2PT")
    
    plt.figure(figsize=(12, 11))
    draw_court()
    
    # Create scatter plot with different colors for 2PT and 3PT
    missed_2pt = shots[(shots['shot_type'] == "2PT") & (shots['actionType'] == 'Missed Shot')]
    made_2pt = shots[(shots['shot_type'] == "2PT") & (shots['actionType'] == 'Made Shot')]
    missed_3pt = shots[(shots['shot_type'] == "3PT") & (shots['actionType'] == 'Missed Shot')]
    made_3pt = shots[(shots['shot_type'] == "3PT") & (shots['actionType'] == 'Made Shot')]
    
    plt.scatter(missed_2pt['xLegacy'], missed_2pt['yLegacy'], marker='x', color='red', s=50, alpha=0.7, label='Missed 2PT')
    plt.scatter(made_2pt['xLegacy'], made_2pt['yLegacy'], marker='o', color='green', s=50, alpha=0.7, label='Made 2PT')
    plt.scatter(missed_3pt['xLegacy'], missed_3pt['yLegacy'], marker='x', color='orange', s=50, alpha=0.7, label='Missed 3PT')
    plt.scatter(made_3pt['xLegacy'], made_3pt['yLegacy'], marker='o', color='blue', s=50, alpha=0.7, label='Made 3PT')
    
    plt.title(f"{title} - {team_tricode}")
    plt.legend(loc='upper left')
    plt.xlim(-250, 250)
    plt.ylim(422.5, -47.5)
    plt.show()

def draw_court(ax=None, color='black', lw=2, outer_lines=False):
    """Draws a basketball court."""
    if ax is None:
        ax = plt.gca()
    hoop = Circle((0, 0), radius=7.5, linewidth=lw, color=color, fill=False)
    backboard = Rectangle((-30, -7.5), 60, -1, linewidth=lw, color=color)
    outer_box = Rectangle((-80, -47.5), 160, 190, linewidth=lw, color=color, fill=False)
    inner_box = Rectangle((-60, -47.5), 120, 190, linewidth=lw, color=color, fill=False)
    top_free_throw = Arc((0, 142.5), 120, 120, theta1=0, theta2=180, linewidth=lw, color=color, fill=False)
    bottom_free_throw = Arc((0, 142.5), 120, 120, theta1=180, theta2=0, linewidth=lw, color=color, linestyle='dashed')
    restricted = Arc((0, 0), 80, 80, theta1=0, theta2=180, linewidth=lw, color=color)
    corner_three_a = Rectangle((-220, -47.5), 0, 140, linewidth=lw, color=color)
    corner_three_b = Rectangle((220, -47.5), 0, 140, linewidth=lw, color=color)
    three_arc = Arc((0, 0), 475, 475, theta1=22, theta2=158, linewidth=lw, color=color)
    center_outer_arc = Arc((0, 422.5), 120, 120, theta1=180, theta2=0, linewidth=lw, color=color)
    center_inner_arc = Arc((0, 422.5), 40, 40, theta1=180, theta2=0, linewidth=lw, color=color)
    court_elements = [hoop, backboard, outer_box, inner_box, top_free_throw, bottom_free_throw, restricted, corner_three_a, corner_three_b, three_arc, center_outer_arc, center_inner_arc]
    if outer_lines:
        outer_lines = Rectangle((-250, -47.5), 500, 470, linewidth=lw, color=color, fill=False)
        court_elements.append(outer_lines)
    for element in court_elements:
        ax.add_patch(element)
    return ax

def plot_impact_scores(impact_scores, title="Enhanced Impact Scores", top_n=15):
    """Plots a bar chart of player impact scores."""
    plt.figure(figsize=(14, 8))
    
    # Get top N players
    top_impact = impact_scores.head(top_n)
    
    # Set colormap for visual appeal
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_impact)))
    
    ax = sns.barplot(x=top_impact.index, y=top_impact.values, palette=colors)
    
    # Add value labels on top of bars
    for i, v in enumerate(top_impact.values):
        ax.text(i, v + 0.5, f"{v:.1f}", ha='center', fontsize=9)
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.ylabel("Impact Score", fontsize=12)
    plt.xlabel("")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_cumulative_impact(df_pbp, player_name, df_defensive=None, df_player_track=None):
    """Plots cumulative impact score over the game with enhanced metrics."""
    player_data = df_pbp[df_pbp['playerName'] == player_name].copy()
    if player_data.empty:
        print(f"No data found for player: {player_name}")
        return

    player_data['impact'] = 0  # Initialize

    # Calculate impact for each play
    for index, row in player_data.iterrows():
        idx = df_pbp.index.get_loc(index)
        next_play = df_pbp.iloc[idx + 1] if idx < len(df_pbp) - 1 else None
        previous_plays = df_pbp.iloc[max(0, idx - 5):idx].to_dict('records')
        
        if isinstance(row['description'], str) and 'BLOCK' in row['description']:
            player_data.loc[index, 'impact'] = calculate_block_impact(row, next_play, previous_plays, df_defensive)
        elif isinstance(row['description'], str) and 'STEAL' in row['description']:
            player_data.loc[index, 'impact'] = calculate_steal_impact(row, next_play, previous_plays, df_defensive)
        elif row['actionType'] == 'Rebound':
            player_data.loc[index, 'impact'] = calculate_rebound_impact(row, next_play, previous_plays, df_player_track)
        elif row['actionType'] == 'Made Shot':
            player_data.loc[index, 'impact'] = calculate_scoring_impact(row, previous_plays, df_player_track)
        elif isinstance(row['description'], str) and 'Foul' in row['description']:
            player_data.loc[index, 'impact'] = calculate_foul_impact(row, next_play, previous_plays)
        elif row['actionType'] == 'Turnover':
            player_data.loc[index, 'impact'] = calculate_turnover_impact(row, next_play, previous_plays)
        
        # Add clutch time multiplier
        if is_clutch_time(row['clock_seconds'], row['period']):
            player_data.loc[index, 'impact'] *= 1.5

    # Calculate cumulative impact
    player_data['cumulative_impact'] = player_data['impact'].cumsum()
    
    # Create figure with subplots: main plot and period markers
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [4, 1]})
    
    # Create quarter markers
    quarters = sorted(player_data['period'].unique())
    quarter_colors = plt.cm.tab10(np.linspace(0, 1, len(quarters)))
    
    # Main impact plot with quarter changes highlighted
    for i, quarter in enumerate(quarters):
        quarter_data = player_data[player_data['period'] == quarter]
        if not quarter_data.empty:
            line, = ax1.plot(quarter_data.index, quarter_data['cumulative_impact'], 
                        marker='o', markersize=4, linestyle='-', linewidth=2,
                        label=f'Q{quarter}', color=quarter_colors[i])
            
            # Add period rectangles on bottom subplot
            min_idx = quarter_data.index.min()
            max_idx = quarter_data.index.max()
            ax2.axvspan(min_idx, max_idx, alpha=0.3, color=line.get_color(), label=f'Period {quarter}')
    
    # Add event labels for significant plays
    significant_plays = player_data[abs(player_data['impact']) > 2].copy()
    for idx, row in significant_plays.iterrows():
        event_desc = row['description'] if isinstance(row['description'], str) else row['actionType']
        if len(event_desc) > 30:
            event_desc = event_desc[:27] + "..."
        ax1.annotate(event_desc, 
                    xy=(idx, row['cumulative_impact']),
                    xytext=(0, 10 if row['impact'] > 0 else -20),
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'),
                    fontsize=8)
    
    # Styling
    ax1.set_title(f"Cumulative Impact Score for {player_name}", fontsize=16, fontweight='bold')
    ax1.set_ylabel("Cumulative Impact Score", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='upper left')
    
    # Remove y-axis from bottom subplot and set labels
    ax2.set_yticks([])
    ax2.set_xlabel("Play Sequence", fontsize=12)
    ax2.set_title("Game Periods", fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)
    plt.show()

def plot_matchup_matrix(df_matchup, min_minutes=3):
    """Plots a heatmap of matchup impact scores."""
    # Filter matchups by minimum minutes
    filtered_matchups = df_matchup[df_matchup['totalMatchupMinutes'] >= min_minutes].copy()
    
    # Create matrix format
    players_off = sorted(filtered_matchups['playerNameOff'].unique())
    players_def = sorted(filtered_matchups['playerNameDef'].unique())
    
    # Initialize matrix with zeros
    matchup_matrix = pd.DataFrame(0, index=players_def, columns=players_off)
    
    # Fill matrix with matchup data
    for _, row in filtered_matchups.iterrows():
        off_player = row['playerNameOff']
        def_player = row['playerNameDef']
        
        # Calculate impact score for this specific matchup
        points_scored = row['totalMatchupFieldGoalsMade'] * 2
        fg_pct = row['totalMatchupFieldGoalsMade'] / max(1, row['totalMatchupFieldGoalsAttempted'])
        
        # Add bonus for 3-pointers if available
        if 'totalMatchupThreePointersMade' in row:
            points_scored += row['totalMatchupThreePointersMade'] * 1
        
        # Scale by minutes and efficiency
        impact = points_scored * np.sqrt(row['totalMatchupMinutes']) * fg_pct
        
        # Store in matrix (defensive perspective)
        matchup_matrix.loc[def_player, off_player] = impact
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    mask = matchup_matrix == 0  # Mask zero values
    
    # Use diverging colormap for offensive vs defensive advantage
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    
    sns.heatmap(matchup_matrix, annot=True, fmt=".1f", cmap=cmap, center=0,
                mask=mask, linewidths=.5, cbar_kws={"shrink": .8},
                vmin=-matchup_matrix.max(), vmax=matchup_matrix.max())
    
    plt.title("Matchup Impact Matrix\n(Positive values = offensive advantage)", fontsize=16)
    plt.xlabel("Offensive Player", fontsize=14)
    plt.ylabel("Defensive Player", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

def visualize_player_breakdown(player_name, df_pbp, df_defensive=None, df_player_track=None):
    """Creates a detailed breakdown of a player's impact by play type."""
    player_data = df_pbp[df_pbp['playerName'] == player_name].copy()
    if player_data.empty:
        print(f"No data found for player: {player_name}")
        return
    
    # Initialize impact categories
    impact_categories = {
        'Scoring': 0,
        'Rebounds': 0,
        'Blocks': 0,
        'Steals': 0,
        'Turnovers': 0,
        'Fouls': 0,
        'Other': 0
    }
    
    # Calculate impact for each play
    for index, row in player_data.iterrows():
        idx = df_pbp.index.get_loc(index)
        next_play = df_pbp.iloc[idx + 1] if idx < len(df_pbp) - 1 else None
        previous_plays = df_pbp.iloc[max(0, idx - 5):idx].to_dict('records')
        
        impact = 0
        category = 'Other'
        
        if isinstance(row['description'], str) and 'BLOCK' in row['description']:
            impact = calculate_block_impact(row, next_play, previous_plays, df_defensive)
            category = 'Blocks'
        elif isinstance(row['description'], str) and 'STEAL' in row['description']:
            impact = calculate_steal_impact(row, next_play, previous_plays, df_defensive)
            category = 'Steals'
        elif row['actionType'] == 'Rebound':
            impact = calculate_rebound_impact(row, next_play, previous_plays, df_player_track)
            category = 'Rebounds'
        elif row['actionType'] == 'Made Shot':
            impact = calculate_scoring_impact(row, previous_plays, df_player_track)
            category = 'Scoring'
        elif isinstance(row['description'], str) and 'Foul' in row['description']:
            impact = calculate_foul_impact(row, next_play, previous_plays)
            category = 'Fouls'
        elif row['actionType'] == 'Turnover':
            impact = calculate_turnover_impact(row, next_play, previous_plays)
        
        # Add clutch time multiplier
        if is_clutch_time(row['clock_seconds'], row['period']):
            impact *= 1.5
            
        impact_categories[category] += impact
    
    # Create a breakdown visualization
    plt.figure(figsize=(10, 8))
    
    # Convert to DataFrame for easier plotting
    impact_df = pd.DataFrame({
        'Category': list(impact_categories.keys()),
        'Impact': list(impact_categories.values())
    })
    
    # Set colors based on positive/negative values
    colors = ['green' if x >= 0 else 'red' for x in impact_df['Impact']]
    
    # Create horizontal bar chart
    ax = sns.barplot(y='Category', x='Impact', data=impact_df, palette=colors, orient='h')
    
    # Add value labels
    for i, v in enumerate(impact_df['Impact']):
        ax.text(v + (0.5 if v >= 0 else -0.5), i, f"{v:.1f}", va='center')
    
    plt.title(f"Impact Breakdown for {player_name}", fontsize=16, fontweight='bold')
    plt.xlabel("Impact Score", fontsize=12)
    plt.ylabel("")
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.7)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# --- Main Execution Block ---

try:
    print("Loading and preprocessing data...")
    df_merged, df_matchup, df_match_detailed, df_pbp, df_defensive, df_player_track, team_possessions, player_team_map = load_and_preprocess_data()
    print("Data loaded successfully!")

    # Calculate enhanced impact scores
    print("\nCalculating enhanced impact scores...")
    impact_scores = calculate_enhanced_impact_score(
        df_merged.copy(), 
        df_defensive=df_defensive.copy(),
        df_player_track=df_player_track.copy(),
        team_possessions=team_possessions,
        player_team_map=player_team_map
    )
    print("Top 10 Enhanced Impact Scores:")
    print(impact_scores.head(10))

    # Calculate matchup impact scores
    print("\nCalculating matchup impact scores...")
    matchup_impact_scores = calculate_matchup_impact(df_matchup.copy(), df_advanced=df_adv.copy())
    print("Top 5 Matchup Impact Scores:")
    print(matchup_impact_scores.head(5))
    print("\nBottom 5 Matchup Impact Scores:")
    print(matchup_impact_scores.tail(5))

    # Visualizations
    print("\nCreating visualizations...")
    
    # Overall impact scores
    plot_impact_scores(impact_scores, title="Enhanced Impact Scores (Per 100 Possessions)")
    
    # Team shot charts
    team_tricodes = df_pbp['teamTricode'].dropna().unique()
    for tricode in team_tricodes:
        if tricode:
            create_shot_chart(df_pbp, tricode, title=f"Shot Chart - {tricode}")
    
    # Player impact breakdowns for top players
    top_players = impact_scores.head(3).index.tolist()
    for player in top_players:
        visualize_player_breakdown(player, df_pbp, df_defensive, df_player_track)
        plot_cumulative_impact(df_pbp, player, df_defensive, df_player_track)
    
    # Matchup matrix
    plot_matchup_matrix(df_match_detailed, min_minutes=5)

    print("\nAnalysis completed successfully!")
    
except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()
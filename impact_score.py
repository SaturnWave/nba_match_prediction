import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle, Rectangle, Arc
import os

# --- Data Loading and Preprocessing ---
def load_and_preprocess_data():
    """Loads and preprocesses the four CSV files."""
    
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load dataframes using full paths
    df_pbp = pd.read_csv(os.path.join(script_dir, "detailed_play_by_play_0022400058_1.csv"))
    df_adv = pd.read_csv(os.path.join(script_dir, "0022400058_box_scores_advanced.csv"))
    df_match_detailed = pd.read_csv(os.path.join(script_dir, "0022400058_box_scores_matchups.csv"))  # Corrected filename
    df_traditional = pd.read_csv(os.path.join(script_dir, '0022400058_box_scores_traditional.csv'))

    # --- Preprocessing ---
    # 1. Play-by-Play Data
    # Convert 'clock' to total seconds from the start of the period.
    df_pbp['clock_seconds'] = df_pbp['clock'].apply(lambda x: int(x.split('PT')[1].split('M')[0]) * 60 + float(x.split('M')[1].replace('S', '')) if 'PT' in x else 0)

    # 2. Advanced Stats - handle missing data
    df_adv.fillna(0, inplace=True)

    # 3. Matchup data - Correctly create matchup_id for merging.
    df_match_detailed['matchup_id'] = df_match_detailed.apply(lambda row: tuple(sorted((row['personIdOff'], row['personIdDef']))), axis=1)

    # Add player names for merging/readability
    player_id_name_map = df_pbp[['personId', 'playerName']].drop_duplicates().set_index('personId')['playerName'].to_dict()
    df_match_detailed['playerNameOff'] = df_match_detailed['personIdOff'].map(player_id_name_map)
    df_match_detailed['playerNameDef'] = df_match_detailed['personIdDef'].map(player_id_name_map)

    # 4- Merge df_pbp and df_adv
    df_merged = pd.merge(df_pbp, df_adv, left_on=['personId', 'teamId'], right_on=['PLAYER_ID', 'TEAM_ID'], how='left', suffixes=('', '_adv'))

    # 5- Correctly aggregate Matchup data (using only df_match_detailed)
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
    
    return df_merged, df_matchup, df_match_detailed, df_pbp # Return all the necessary dataframes

# --- Helper Functions --- (No changes needed here)

def is_clutch_time(clock_seconds, period):
    """Checks if the play is in the last 5 minutes of the 4th period."""
    return period == 4 and clock_seconds <= 300

def is_last_2_minutes(clock_seconds, period):
    return period == 4 and clock_seconds <= 120

def get_score_margin(row):
    return abs(row['scoreHome'] - row['scoreAway'])

def identify_scoring_run(data, current_idx, window=5):
    start_idx = max(0, current_idx - window)
    previous_plays = data.iloc[start_idx:current_idx]
    if not previous_plays.empty:
        team_counts = previous_plays['teamTricode'].value_counts()
        if not team_counts.empty:
            return team_counts.index[0]
    return None

# --- Impact Score Calculation ---

def calculate_block_impact(row, next_play, previous_plays):
    base_impact = 1.2
    if next_play is not None and next_play['teamTricode'] == row['teamTricode']:
        base_impact -= 0.2
    if next_play is not None and isinstance(next_play['description'], str) and 'Running' in next_play['description']:
        base_impact += 0.2
    if next_play is not None and isinstance(next_play['description'], str) and 'Shot Clock' in next_play['description']:
        base_impact += 0.3
    recent_blocks = [play for play in previous_plays[-3:]
                    if isinstance(play['description'], str) and 'BLOCK' in play['description']
                    and play['playerName'] == row['playerName']]
    if len(recent_blocks) > 1:
        base_impact += 0.3
    if is_last_2_minutes(row['clock_seconds'], row['period']) and get_score_margin(row) <= 3:
        base_impact += 0.5
    return base_impact

def calculate_steal_impact(row, next_play, previous_plays):
    base_impact = 1.4
    if isinstance(row['description'], str) and 'Backcourt' in row['description']:  # Example of using description
        base_impact += 0.1
    if next_play is not None and next_play['actionType'] == 'Made Shot':
        base_impact += 0.2
    recent_steals = [play for play in previous_plays[-5:]
                    if isinstance(play['description'], str) and 'STEAL' in play['description']
                    and play['playerName'] == row['playerName']]
    if len(recent_steals) > 1:
        base_impact += 0.2
    if is_clutch_time(row['clock_seconds'], row['period']):
        margin = get_score_margin(row)
        if margin > 20:
            base_impact = 1.0  # Reduced impact if large margin
        elif margin > 10:
            base_impact = 1.1  # Slightly reduced impact
        else:
            base_impact = 1.5  # Increased impact in close games
    return base_impact

def calculate_rebound_impact(row, next_play, previous_plays):
    is_offensive = isinstance(row['description'], str) and 'Off' in row['description']
    base_impact = 0.9 if is_offensive else 0.6
    if any(isinstance(play['description'], str) and 'REBOUND' in play['description'] for play in previous_plays[-2:]):
        base_impact += 0.2
    if next_play is not None and next_play['actionType'] == 'Made Shot':
        base_impact += 0.2
    if is_last_2_minutes(row['clock_seconds'], row['period']) and get_score_margin(row) <= 3:
        base_impact += 0.3
    if is_clutch_time(row['clock_seconds'], row['period']):
        base_impact += 0.4
    return base_impact

def calculate_scoring_impact(row, previous_plays):
    base_impact = 3.0 if row['shotValue'] == 3 else 2.0
    if any(isinstance(play['description'], str) and 'Free Throw' in play['description'] for play in previous_plays[:2]):
        base_impact += 0.3
    if any(isinstance(play['description'], str) and 'Timeout' in play['description'] for play in previous_plays[-3:]):
        base_impact += 0.2
    scoring_run_team = identify_scoring_run(df_pbp, df_pbp.index.get_loc(row.name))
    if scoring_run_team and scoring_run_team != row['teamTricode']:
        base_impact += 0.2
    if any(isinstance(play['description'], str) and 'Start of' in play['description'] for play in previous_plays[-3:]):
        base_impact += 0.1
    return base_impact

def calculate_enhanced_impact_score(df):
    """Calculates the enhanced impact score, incorporating advanced stats."""
    player_impact = {}

    for index, row in df.iterrows():
        if pd.isna(row['playerName']):
            continue

        player = row['playerName']
        if player not in player_impact:
            player_impact[player] = 0

        next_play = df.iloc[index + 1] if index < len(df) - 1 else None
        previous_plays = df.iloc[max(0, index - 5):index].to_dict('records')

        impact = 0  # Initialize

        # --- Base Impact (from play-by-play) ---
        if isinstance(row['description'], str) and 'BLOCK' in row['description']:
            impact += calculate_block_impact(row, next_play, previous_plays)
        elif isinstance(row['description'], str) and 'STEAL' in row['description']:
            impact += calculate_steal_impact(row, next_play, previous_plays)
        elif row['actionType'] == 'Rebound':
            impact += calculate_rebound_impact(row, next_play, previous_plays)
        elif row['actionType'] == 'Made Shot':
            impact += calculate_scoring_impact(row, previous_plays)
        elif isinstance(row['description'], str) and 'Foul' in row['description']:  # Using string check
            impact -= 0.7 if 'S.FOUL' in str(row['description']) else 0.3  # Differentiate fouls
        elif row['actionType'] == 'Turnover':
            impact -= 1.0 if is_clutch_time(row['clock_seconds'], row['period']) else 0.8

        # --- Advanced Stat Adjustments ---
        if 'PIE' in row and not pd.isna(row['PIE']):
            impact += row['PIE'] * 2  # Scale PIE.
        if 'E_NET_RATING' in row and not pd.isna(row['E_NET_RATING']):
            impact += row['E_NET_RATING'] * 0.05  # Smaller weight
        if 'USG_PCT' in row and not pd.isna(row['USG_PCT']):
            impact *= (1 + row['USG_PCT'])  # Higher usage = more impact


        # --- Modifiers based on game context ---
        # Clutch Time
        if is_clutch_time(row['clock_seconds'], row['period']):
            impact *= 1.5

        # Time Remaining (More Granular)
        time_remaining_factor = 1 + (1 / (row['clock_seconds'] + 1))
        impact *= time_remaining_factor

        player_impact[player] += impact

    return pd.Series(player_impact).sort_values(ascending=False)

def calculate_matchup_impact(df_matchup):
    """Calculates a matchup impact score."""
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
        scoring_impact = points_scored - (0.5 * points_possible)

        # Adjust for matchup minutes (avoid extreme values)
        minutes_factor = np.sqrt(row['totalMatchupMinutes']) if row['totalMatchupMinutes'] > 0 else 0
        scoring_impact *= minutes_factor

        # Add to offensive player, subtract from defensive
        matchup_impact[matchup_key] += scoring_impact
        matchup_impact[reverse_key] -= scoring_impact

    return pd.Series(matchup_impact).sort_values(ascending=False)

# --- Visualization Functions ---

def create_shot_chart(df, team_tricode, title="Shot Chart"):
    """Creates a shot chart for a specific team."""
    shots = df[(df['actionType'].isin(['Made Shot', 'Missed Shot'])) & (df['teamTricode'] == team_tricode)].copy()
    plt.figure(figsize=(12, 11))
    draw_court()
    plt.scatter(shots[shots['actionType'] == 'Missed Shot']['xLegacy'], shots[shots['actionType'] == 'Missed Shot']['yLegacy'], marker='x', color='red', label='Missed')
    plt.scatter(shots[shots['actionType'] == 'Made Shot']['xLegacy'], shots[shots['actionType'] == 'Made Shot']['yLegacy'], marker='o', color='green', label='Made')
    plt.title(f"{title} - {team_tricode}")  # Add team tricode
    plt.legend()
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

def plot_impact_scores(impact_scores, title="Enhanced Impact Scores"):
    """Plots a bar chart of player impact scores."""
    plt.figure(figsize=(12, 6))
    sns.barplot(x=impact_scores.index, y=impact_scores.values, palette="viridis")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Impact Score")
    plt.tight_layout()
    plt.show()

def plot_cumulative_impact(df_pbp, player_name):
    """Plots cumulative impact score over the game."""
    player_data = df_pbp[df_pbp['playerName'] == player_name].copy()
    if player_data.empty:
        print(f"No data found for player: {player_name}")
        return

    player_data['impact'] = 0  # Initialize

    # Calculate impact (simplified example)
    for index, row in player_data.iterrows():
        if row['actionType'] == 'Made Shot':
            player_data.loc[index, 'impact'] = row['shotValue']
        elif row['actionType'] == 'Rebound':
            player_data.loc[index, 'impact'] = 0.6
        elif isinstance(row['description'], str) and 'STEAL' in row['description']:
            player_data.loc[index, 'impact'] = 1.4
        elif isinstance(row['description'], str) and 'BLOCK' in row['description']:
             player_data.loc[index, 'impact'] = 1.2
        elif row['actionType'] == 'Turnover':
            player_data.loc[index, 'impact'] = -0.8
        elif 'Foul' in str(row['description']):
            player_data.loc[index, 'impact'] = -0.5

    player_data['cumulative_impact'] = player_data['impact'].cumsum()
    plt.figure(figsize=(12, 6))
    # Use clock_seconds for x-axis
    plt.plot(player_data['clock_seconds'], player_data['cumulative_impact'], label=player_name)
    plt.title(f"Cumulative Impact Score for {player_name}")
    plt.xlabel("Time (seconds from start of period)")
    plt.ylabel("Cumulative Impact Score")
    plt.legend()
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.show()

# --- Main Execution Block ---

try:
    print("Loading and preprocessing data...")
    df_merged, df_matchup, df_match_total, df_pbp = load_and_preprocess_data()
    print("Data loaded successfully!")

    impact_scores = calculate_enhanced_impact_score(df_merged.copy())
    print("\nEnhanced Impact Scores:")
    print(impact_scores)

    matchup_impact_scores = calculate_matchup_impact(df_matchup.copy())
    print("\nMatchup Impact Scores:")
    print(matchup_impact_scores)


    plot_impact_scores(impact_scores, title="Overall Enhanced Impact Scores")

    # Sort and display top/bottom matchup impacts
    matchup_impact_sorted = matchup_impact_scores.sort_values(ascending=False)
    print("\nTop 5 Positive Matchup Impacts:")
    print(matchup_impact_sorted.head(5))
    print("\nTop 5 Negative Matchup Impacts:")
    print(matchup_impact_sorted.tail(5))



    # Shot Charts for each team
    team_tricodes = df_pbp['teamTricode'].dropna().unique()
    for tricode in team_tricodes:
        if tricode:
          create_shot_chart(df_pbp, tricode, title=f"Shot Chart - {tricode}")


    # Cumulative impact for a couple of key players (example)
    plot_cumulative_impact(df_pbp, "Nikola Jokić")
    plot_cumulative_impact(df_pbp, "Stephen Curry")


except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()
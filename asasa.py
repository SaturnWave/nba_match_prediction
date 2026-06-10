import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle, Rectangle, Arc
import os

# Shared stateless helpers (de-duplicated into impact_common.py)
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from impact_common import (
    is_clutch_time,
    is_last_2_minutes,
    get_score_margin,
    identify_scoring_run,
    categorize_shot_distance,
    calculate_expected_points,
    estimate_win_probability,
)

# Unified per-play impact engine (de-duplicated into impact_engine.py at the project root).
from impact_engine import (
    calculate_block_impact,
    calculate_steal_impact,
    calculate_rebound_impact,
    calculate_scoring_impact,
    calculate_turnover_impact,
    calculate_foul_impact,
)



# --- Data Loading and Preprocessing ---
def load_and_preprocess_data():
    """Loads and preprocesses only the play-by-play CSV file."""

    # Sample-game CSVs live in the impact_score_files/ folder alongside this script.
    script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "impact_score_files")

    # Load only the play-by-play dataframe using the full path
    df_pbp = pd.read_csv(os.path.join(script_dir, "detailed_play_by_play_0022400058_1.csv"))

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

    return df_pbp


# --- Enhanced Impact Score Calculations ---







def calculate_enhanced_impact_score(df):
    """Calculates the enhanced impact score with contextual modifiers using only play-by-play data."""
    player_impact = {}

    for index, row in df.iterrows():
        if pd.isna(row.get('playerName')):
            continue

        player = row['playerName']
        if player not in player_impact:
            player_impact[player] = 0

        next_play = df.iloc[index + 1] if index < len(df) - 1 else None
        # Correctly handle previous_plays as a list of dictionaries
        previous_plays = df.iloc[max(0, index - 5):index].to_dict('records')


        impact = 0  # Initialize

        # --- Base Impact (from play-by-play) with enhanced contextual modifiers ---
        if isinstance(row.get('description'), str) and 'BLOCK' in row['description']:
            impact += calculate_block_impact(row, next_play, previous_plays, df)
        elif isinstance(row.get('description'), str) and 'STEAL' in row['description']:
            impact += calculate_steal_impact(row, next_play, previous_plays)
        elif row.get('actionType') == 'Rebound':
            impact += calculate_rebound_impact(row, next_play, previous_plays)
        elif row.get('actionType') == 'Made Shot':
            impact += calculate_scoring_impact(row, previous_plays, df)
        elif isinstance(row.get('description'), str) and 'Foul' in row['description']:
            impact += calculate_foul_impact(row, next_play, previous_plays, df)
        elif row.get('actionType') == 'Turnover':
            impact += calculate_turnover_impact(row, next_play, previous_plays)

        # --- Modifiers based on game context ---
        # Clutch Time
        if is_clutch_time(row.get('clock_seconds', 0), row.get('period', 0)):
            impact *= 1.5

        # Time Remaining (More Granular)
        if pd.notnull(row.get('clock_seconds')):
            time_remaining_factor = 1 + (1 / (row['clock_seconds'] + 1))
            impact *= time_remaining_factor

        player_impact[player] += impact

    return pd.Series(player_impact).sort_values(ascending=False)

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

    plt.title(title, fontsize=16, fontweight='bold', ha="right", color='black')
    plt.ylabel("Impact Score", fontsize=12)
    plt.xlabel("")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_cumulative_impact(df_pbp, player_name):
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
            player_data.loc[index, 'impact'] = calculate_block_impact(row, next_play, previous_plays, df_pbp)
        elif isinstance(row['description'], str) and 'STEAL' in row['description']:
            player_data.loc[index, 'impact'] = calculate_steal_impact(row, next_play, previous_plays)
        elif row['actionType'] == 'Rebound':
            player_data.loc[index, 'impact'] = calculate_rebound_impact(row, next_play, previous_plays)
        elif row['actionType'] == 'Made Shot':
            player_data.loc[index, 'impact'] = calculate_scoring_impact(row, previous_plays, df_pbp)
        elif isinstance(row['description'], str) and 'Foul' in row['description']:
            player_data.loc[index, 'impact'] = calculate_foul_impact(row, next_play, previous_plays, df_pbp)
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

def visualize_player_breakdown(player_name, df_pbp):
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
            impact = calculate_block_impact(row, next_play, previous_plays, df_pbp)
            category = 'Blocks'
        elif isinstance(row['description'], str) and 'STEAL' in row['description']:
            impact = calculate_steal_impact(row, next_play, previous_plays)
            category = 'Steals'
        elif row['actionType'] == 'Rebound':
            impact = calculate_rebound_impact(row, next_play, previous_plays)
            category = 'Rebounds'
        elif row['actionType'] == 'Made Shot':
            impact = calculate_scoring_impact(row, previous_plays, df_pbp)
            category = 'Scoring'
        elif isinstance(row['description'], str) and 'Foul' in row['description']:
            impact = calculate_foul_impact(row, next_play, previous_plays, df_pbp)
            category = 'Fouls'
        elif row['actionType'] == 'Turnover':
            impact = calculate_turnover_impact(row, next_play, previous_plays)
            category = 'Turnovers'

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
    df_pbp = load_and_preprocess_data()
    print("Data loaded successfully!")

    # Calculate enhanced impact scores
    print("\nCalculating enhanced impact scores...")
    impact_scores = calculate_enhanced_impact_score(df_pbp.copy())
    print("Top 10 Enhanced Impact Scores:")
    print(impact_scores.head(10))

    # Visualizations
    print("\nCreating visualizations...")

    # Overall impact scores
    plot_impact_scores(impact_scores, title="Enhanced Impact Scores (Play-by-Play Only)")

    # Team shot charts
    team_tricodes = df_pbp['teamTricode'].dropna().unique()
    for tricode in team_tricodes:
        if tricode:
            create_shot_chart(df_pbp, tricode, title=f"Shot Chart - {tricode}")

    # Player impact breakdowns for top players
    top_players = impact_scores.head(3).index.tolist()
    for player in top_players:
        visualize_player_breakdown(player, df_pbp)
        plot_cumulative_impact(df_pbp, player)

    print("\nAnalysis completed successfully!")

except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()

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
)

# Per-play impact functions shared with analyze_season_impact_1 / analyze_season_impact
# (de-duplicated into season_impact_engine.py).
from season_impact_engine import (
    calculate_block_impact,
    calculate_steal_impact,
    calculate_rebound_impact,
    calculate_scoring_impact,
    calculate_turnover_impact,
    calculate_foul_impact,
)

warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

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
        # Extract game ID from filepath for reference
        game_id = os.path.basename(filepath).split('pbp')[0]
        
        # Load the data
        df_pbp = pd.read_csv(filepath)
        
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
        
        df_pbp['is_clutch'] = df_pbp.apply(
            lambda row: is_clutch_time(row['clock_seconds'], row['period']), 
            axis=1
        )
        
        df_pbp['is_last_2min'] = df_pbp.apply(
            lambda row: is_last_2_minutes(row['clock_seconds'], row['period']), 
            axis=1
        )

        # Create a shot distance bin for spatial analysis
        df_pbp['shot_distance_bin'] = df_pbp['shotDistance'].apply(
            lambda x: categorize_shot_distance(x) if pd.notnull(x) else None
        )

        # Create a column for expected points based on shot location
        df_pbp['expected_points'] = df_pbp.apply(
            lambda row: calculate_expected_points(row['xLegacy'], row['yLegacy'], row['shotValue'])
            if pd.notnull(row.get('xLegacy', None)) and pd.notnull(row.get('yLegacy', None)) and pd.notnull(row.get('shotValue', None))
            else None, 
            axis=1
        )

        return df_pbp
        
    except Exception as e:
        print(f"Error preprocessing file {filepath}: {e}")
        return None

# --- Enhanced Impact Score Calculations ---

def calculate_enhanced_impact_score(df):
    """
    Calculates the enhanced impact score with contextual modifiers using play-by-play data.
    
    Args:
        df (pd.DataFrame): Play-by-play data
        
    Returns:
        pd.Series: Player impact scores
    """
    player_impact = {}

    for index, row in df.iterrows():
        if pd.isna(row.get('playerName')):
            continue

        player = row['playerName']
        if player not in player_impact:
            player_impact[player] = 0

        next_play = df.iloc[index + 1].to_dict() if index < len(df) - 1 else None
        # Correctly handle previous_plays as a list of dictionaries
        previous_plays = [row.to_dict() for _, row in df.iloc[max(0, index - 5):index].iterrows()]

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

# --- Main Analysis Function ---
def analyze_season_data():
    """
    Analyzes NBA season data to calculate impact scores for all games 
    and find the top and worst performing players across the season.
    """
    print("Starting NBA season impact score analysis...")
    
    # Path to the directory containing all the game data
    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "nba_data", "2023_2024")
    
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
    
    print(f"Found {len(pbp_files)} play-by-play files")
    
    # Track all game results and top performances
    all_game_scores = {}
    player_game_counts = defaultdict(int)
    player_total_impact = defaultdict(float)
    game_top_performers = []
    game_worst_performers = []
    
    # Process each game
    for i, pbp_file in enumerate(pbp_files):
        if i % 10 == 0:
            print(f"Processing game {i+1}/{len(pbp_files)}...")
        
        # Load and preprocess data
        game_data = load_and_preprocess_pbp_file(pbp_file)
        
        if game_data is not None and not game_data.empty:
            # Calculate impact scores for this game
            game_id = game_data['game_id'].iloc[0]
            game_date = game_data['game_date'].iloc[0]
            match_description = game_data['match_description'].iloc[0]
            
            try:
                impact_scores = calculate_enhanced_impact_score(game_data)
                
                # Store the results
                all_game_scores[game_id] = impact_scores
                
                # Track each player's total impact and game count
                for player, score in impact_scores.items():
                    player_total_impact[player] += score
                    player_game_counts[player] += 1
                
                # Track top performers for this game
                top_players = impact_scores.head(3)
                for player, score in top_players.items():
                    game_top_performers.append({
                        'game_id': game_id,
                        'game_date': game_date,
                        'matchup': match_description,
                        'player_name': player,
                        'impact_score': score
                    })
                
                # Track worst performers for this game (lowest impact scores)
                worst_players = impact_scores.tail(3)
                for player, score in worst_players.items():
                    game_worst_performers.append({
                        'game_id': game_id,
                        'game_date': game_date,
                        'matchup': match_description,
                        'player_name': player,
                        'impact_score': score
                    })
                
            except Exception as e:
                print(f"Error calculating impact scores for game {game_id}: {e}")
    
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
    top_performer_counts = top_performers_df['player_name'].value_counts()
    
    # Find exceptional single-game performances (both best and worst)
    exceptional_games = top_performers_df.sort_values('impact_score', ascending=False).head(10)
    worst_games = worst_performers_df.sort_values('impact_score', ascending=True).head(10)
    
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
    
    print("\nTop 10 Individual Game Performances:")
    print(exceptional_games[['game_date', 'game_id', 'matchup', 'player_name', 'impact_score']])
    
    print("\nWorst 10 Individual Game Performances:")
    print(worst_games[['game_date', 'game_id', 'matchup', 'player_name', 'impact_score']])
    
    # Save results to CSV
    avg_impact_series.to_csv("average_impact_scores.csv")
    worst_avg_impact_series.to_csv("worst_average_impact_scores.csv")
    top_performers_df.to_csv("game_top_performers.csv", index=False)
    worst_performers_df.to_csv("game_worst_performers.csv", index=False)
    
    # Create a consolidated top/worst performances file
    top_10_best = exceptional_games[['game_date', 'game_id', 'matchup', 'player_name', 'impact_score']].head(10)
    top_10_best['performance_type'] = 'Best'
    top_10_worst = worst_games[['game_date', 'game_id', 'matchup', 'player_name', 'impact_score']].head(10)
    top_10_worst['performance_type'] = 'Worst'
    
    combined_performances = pd.concat([top_10_best, top_10_worst])
    combined_performances.to_csv("top_and_worst_game_performances.csv", index=False)
    
    print("\nAnalysis completed successfully!")
    return avg_impact_series, worst_avg_impact_series, top_performers_df, worst_performers_df

if __name__ == "__main__":
    try:
        # Run the season analysis
        top_avg_impact, worst_avg_impact, top_games, worst_games = analyze_season_data()
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

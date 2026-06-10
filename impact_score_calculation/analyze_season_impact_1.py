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

# Unified per-play impact engine (de-duplicated into impact_engine.py at the project root).
from impact_engine import (
    calculate_block_impact,
    calculate_steal_impact,
    calculate_rebound_impact,
    calculate_scoring_impact,
    calculate_turnover_impact,
    calculate_foul_impact,
)

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

# --- Enhanced Impact Score Calculations ---

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

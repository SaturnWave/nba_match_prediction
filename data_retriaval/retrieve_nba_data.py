import pandas as pd
import time
import os
import json
import traceback
import datetime
from nba_api.stats.endpoints import PlayByPlayV3, BoxScorePlayerTrackV2

# ----------------------------------------
# Global Configuration
# ----------------------------------------
# Base directory for saving NBA data - Updated to absolute path
BASE_DIR = r"C:\Users\arcan\Desktop\Python\nba_new\impact_scores\nba_data"

# Default delay between API calls (will be adjusted automatically)
API_DELAY = 0.1  # starting with 0.1 seconds

# Delay tiers for retry attempts
RETRY_DELAYS = [0.3, 0.5, 1, 2, 3]

# Maximum number of retry attempts
MAX_RETRIES = 5

# Seasons to process (updated to include 2024-25)
SEASONS = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]

# ----------------------------------------
# Helper Functions
# ----------------------------------------
def format_game_id(game_id):
    """
    Format game ID to ensure it has the proper leading zeros
    """
    game_id_str = str(game_id)
    return f"00{game_id_str}" if not game_id_str.startswith('00') else game_id_str

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
    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                           f"game_id_{formatted_season}.csv")
    
    if not os.path.exists(csv_path):
        print(f"ERROR: Game IDs CSV file not found: {csv_path}")
        return None
    
    try:
        games_df = pd.read_csv(csv_path)
        # Ensure we have unique game IDs
        games_df = games_df.drop_duplicates(subset=['GAME_ID'])
        return games_df
    except Exception as e:
        print(f"ERROR: Failed to read game IDs from {csv_path}: {e}")
        return None

def update_api_delay(success=True):
    """
    Dynamic adjustment of API delay based on success/failure
    """
    global API_DELAY
    
    if not success:
        # Increase delay after a failure, max 5 seconds
        API_DELAY = min(5, API_DELAY * 2)
        print(f"Increased API delay to {API_DELAY} seconds due to errors")
    elif API_DELAY > 0.1 and success:
        # If we had success, potentially decrease delay
        API_DELAY = max(0.1, API_DELAY * 0.8)
        print(f"Decreased API delay to {API_DELAY} seconds")

def get_retry_delay(retry_count):
    """
    Get delay time based on the retry count using the tiered approach
    """
    if retry_count <= 0:
        return API_DELAY
    
    index = min(retry_count - 1, len(RETRY_DELAYS) - 1)
    return RETRY_DELAYS[index]

# ----------------------------------------
# Play-by-Play Data Retrieval
# ----------------------------------------
def retrieve_play_by_play(game_id):
    """
    Retrieves play-by-play data for a given game ID using the NBA API.
    
    Args:
        game_id (str): The NBA game ID to retrieve data for
        
    Returns:
        pd.DataFrame: Play-by-play data for the specified game
    """
    formatted_game_id = format_game_id(game_id)
    
    print(f"Retrieving play-by-play data for game ID: {formatted_game_id}")
    
    retries = 0
    while retries <= MAX_RETRIES:
        try:
            # Fetch play-by-play data with numeric values for periods
            pbp = PlayByPlayV3(
                game_id=formatted_game_id,
                start_period=1,
                end_period=10  # Covers all possible overtimes
            )
            
            # Convert to DataFrame - handle potential empty response
            data_frames = pbp.get_data_frames()
            if not data_frames or len(data_frames) == 0:
                raise Exception("API returned empty data")
                
            play_by_play_df = data_frames[0]
            
            print(f"Successfully retrieved {len(play_by_play_df)} play-by-play records for game {formatted_game_id}")
            update_api_delay(success=True)
            return play_by_play_df
            
        except Exception as e:
            retries += 1
            if retries <= MAX_RETRIES:
                wait_time = get_retry_delay(retries)
                print(f"Error retrieving data: {e}. Retrying in {wait_time} seconds (Attempt {retries}/{MAX_RETRIES})...")
                time.sleep(wait_time)
            else:
                print(f"Failed to retrieve play-by-play data for game {formatted_game_id} after {MAX_RETRIES} attempts: {e}")
                update_api_delay(success=False)
                return None

def save_play_by_play_data(df, game_id, season):
    """
    Saves play-by-play data to CSV with the specified directory structure.
    
    Directory structure:
    - nba_data/
        - YYYY_YYYY/
            - [game_id]/
                - play_by_play/
                    - [game_id]pbp.csv
    
    Args:
        df (pd.DataFrame): Play-by-play data to save
        game_id (str): Game ID for directory and filename
        season (str): Season in format "YYYY-YY" (e.g., "2019-20")
    
    Returns:
        str: Path to the saved file
    """
    # Format game_id to include leading zeros
    formatted_game_id = format_game_id(game_id)
    
    # Convert season format (YYYY-YY to YYYY_YYYY)
    season_format = format_season(season)
    
    # Construct the directory path
    season_dir = os.path.join(BASE_DIR, season_format)
    game_dir = os.path.join(season_dir, formatted_game_id)
    pbp_dir = os.path.join(game_dir, "play_by_play")
    
    # Create directories if they don't exist
    os.makedirs(pbp_dir, exist_ok=True)
    
    # Create filename and path
    filename = f"{formatted_game_id}pbp.csv"
    filepath = os.path.join(pbp_dir, filename)
    
    # Save to CSV
    df.to_csv(filepath, index=False)
    print(f"Play-by-play data saved to {filepath}")
    
    return filepath

def check_if_pbp_data_exists(game_id, season):
    """
    Check if play-by-play data already exists for the given game ID.
    
    Args:
        game_id (str): The game ID to check
        season (str): Season in format "YYYY-YY" (e.g., "2019-20")
    
    Returns:
        bool: True if data exists, False otherwise
    """
    formatted_game_id = format_game_id(game_id)
    
    # Convert season format (YYYY-YY to YYYY_YYYY)
    season_format = format_season(season)
    
    # Construct the expected file path
    season_dir = os.path.join(BASE_DIR, season_format)
    game_dir = os.path.join(season_dir, formatted_game_id)
    pbp_dir = os.path.join(game_dir, "play_by_play")
    filepath = os.path.join(pbp_dir, f"{formatted_game_id}pbp.csv")
    
    # Check if the file exists
    return os.path.exists(filepath)

# ----------------------------------------
# Player Tracking Data Retrieval
# ----------------------------------------
def retrieve_player_tracking_data(game_id):
    """
    Retrieves player tracking data for a given game ID using the NBA API.
    
    Args:
        game_id (str): The NBA game ID to retrieve data for
        
    Returns:
        tuple: (player_stats_df, team_stats_df) - DataFrames with player and team tracking data
    """
    formatted_game_id = format_game_id(game_id)
    
    print(f"Retrieving player tracking data for game ID: {formatted_game_id}")
    
    retries = 0
    while retries <= MAX_RETRIES:
        try:
            # Fetch player tracking data
            box_score_player_track_v2 = BoxScorePlayerTrackV2(game_id=formatted_game_id)
            
            # Convert response to dictionary
            box_score_player_track_v2_data = box_score_player_track_v2.get_dict()
            
            # Extract player tracking data
            player_stats = box_score_player_track_v2_data['resultSets'][0]['rowSet']
            team_stats = box_score_player_track_v2_data['resultSets'][1]['rowSet']
            
            # Get headers for column names
            player_headers = box_score_player_track_v2_data['resultSets'][0]['headers']
            team_headers = box_score_player_track_v2_data['resultSets'][1]['headers']
            
            # Convert to DataFrames
            player_stats_df = pd.DataFrame(player_stats, columns=player_headers)
            team_stats_df = pd.DataFrame(team_stats, columns=team_headers)
            
            # Add game_id column for reference
            player_stats_df['GAME_ID_CLEAN'] = formatted_game_id
            team_stats_df['GAME_ID_CLEAN'] = formatted_game_id
            
            print(f"Successfully retrieved player tracking data for game {formatted_game_id}")
            update_api_delay(success=True)
            return player_stats_df, team_stats_df
            
        except Exception as e:
            retries += 1
            if retries <= MAX_RETRIES:
                wait_time = get_retry_delay(retries)
                print(f"Error retrieving tracking data: {e}. Retrying in {wait_time} seconds (Attempt {retries}/{MAX_RETRIES})...")
                time.sleep(wait_time)
            else:
                print(f"Failed to retrieve player tracking data for game {formatted_game_id} after {MAX_RETRIES} attempts: {e}")
                update_api_delay(success=False)
                return None, None

def save_player_tracking_data(player_df, team_df, game_id, season):
    """
    Saves player tracking data to CSV with the specified directory structure.
    
    Directory structure:
    - nba_data/
        - YYYY_YYYY/
            - [game_id]/
                - player_tracking/
                    - [game_id]player_tracking.csv
                    - [game_id]team_tracking.csv
    
    Args:
        player_df (pd.DataFrame): Player tracking data
        team_df (pd.DataFrame): Team tracking data
        game_id (str): Game ID for directory and filename
        season (str): Season in format "YYYY-YY" (e.g., "2019-20")
    
    Returns:
        tuple: (player_filepath, team_filepath) - Paths to the saved files
    """
    # Format game_id to include leading zeros
    formatted_game_id = format_game_id(game_id)
    
    # Convert season format (YYYY-YY to YYYY_YYYY)
    season_format = format_season(season)
    
    # Construct the directory path
    season_dir = os.path.join(BASE_DIR, season_format)
    game_dir = os.path.join(season_dir, formatted_game_id)
    tracking_dir = os.path.join(game_dir, "player_tracking")
    
    # Create directories if they don't exist
    os.makedirs(tracking_dir, exist_ok=True)
    
    # Create filenames and paths
    player_filename = f"{formatted_game_id}player_tracking.csv"
    team_filename = f"{formatted_game_id}team_tracking.csv"
    
    player_filepath = os.path.join(tracking_dir, player_filename)
    team_filepath = os.path.join(tracking_dir, team_filename)
    
    # Save to CSV
    if player_df is not None:
        player_df.to_csv(player_filepath, index=False)
        print(f"Player tracking data saved to {player_filepath}")
    
    if team_df is not None:
        team_df.to_csv(team_filepath, index=False)
        print(f"Team tracking data saved to {team_filepath}")
    
    return player_filepath, team_filepath

def check_if_tracking_data_exists(game_id, season):
    """
    Check if player tracking data already exists for the given game ID.
    
    Args:
        game_id (str): The game ID to check
        season (str): Season in format "YYYY-YY" (e.g., "2019-20")
    
    Returns:
        bool: True if data exists, False otherwise
    """
    formatted_game_id = format_game_id(game_id)
    
    # Convert season format (YYYY-YY to YYYY_YYYY)
    season_format = format_season(season)
    
    # Construct the expected file path
    season_dir = os.path.join(BASE_DIR, season_format)
    game_dir = os.path.join(season_dir, formatted_game_id)
    tracking_dir = os.path.join(game_dir, "player_tracking")
    player_filepath = os.path.join(tracking_dir, f"{formatted_game_id}player_tracking.csv")
    
    # Check if the file exists
    return os.path.exists(player_filepath)

# ----------------------------------------
# Combined Processing
# ----------------------------------------
def process_season_data(season, data_types=["play_by_play", "tracking"]):
    """
    Process all games from a specific season to retrieve play-by-play and/or tracking data.
    Only process games that don't already have data.
    
    Args:
        season (str): Season in format "YYYY-YY" (e.g., "2019-20")
        data_types (list): Types of data to retrieve - "play_by_play", "tracking", or both
    """
    # Read game IDs for the season
    games_df = read_game_ids(season)
    
    if games_df is None:
        print(f"Skipping season {season} due to missing or invalid game ID file")
        return
    
    # Ensure we have unique game IDs
    game_ids = games_df['GAME_ID'].unique().tolist()
    total_games = len(game_ids)
    
    print(f"Found {total_games} unique games for {season} season")
    
    # Filter out games that already have data
    missing_pbp_game_ids = []
    missing_tracking_game_ids = []
    
    if "play_by_play" in data_types:
        for game_id in game_ids:
            if not check_if_pbp_data_exists(game_id, season):
                missing_pbp_game_ids.append(game_id)
        print(f"Found {len(missing_pbp_game_ids)} games without existing play-by-play data")
    
    if "tracking" in data_types:
        for game_id in game_ids:
            if not check_if_tracking_data_exists(game_id, season):
                missing_tracking_game_ids.append(game_id)
        print(f"Found {len(missing_tracking_game_ids)} games without existing tracking data")
    
    # Process each missing game
    processed_count = 0
    
    # Combined total of unique missing games
    all_missing_game_ids = list(set(missing_pbp_game_ids + missing_tracking_game_ids))
    total_missing = len(all_missing_game_ids)
    
    for i, game_id in enumerate(all_missing_game_ids):
        processed_count += 1
        # Calculate progress percentage
        progress = (processed_count / total_missing) * 100
        print(f"\nProcessing game {processed_count}/{total_missing} ({progress:.1f}%): Game ID {game_id}")
        
        # Step 1: Retrieve play-by-play data if needed
        if game_id in missing_pbp_game_ids and "play_by_play" in data_types:
            print("Retrieving play-by-play data...")
            pbp_df = retrieve_play_by_play(game_id)
            if pbp_df is not None:
                save_play_by_play_data(pbp_df, game_id, season)
        
        # Step 2: Retrieve tracking data if needed
        if game_id in missing_tracking_game_ids and "tracking" in data_types:
            print("Retrieving player tracking data...")
            player_df, team_df = retrieve_player_tracking_data(game_id)
            if player_df is not None and team_df is not None:
                save_player_tracking_data(player_df, team_df, game_id, season)
        
        # Add delay between games to avoid hitting API rate limits
        if i < len(all_missing_game_ids) - 1:  # No need to wait after the last game
            print(f"Waiting {API_DELAY} seconds before next request...")
            time.sleep(API_DELAY)
    
    print(f"\n{season} season processing completed!")

def process_all_seasons():
    """
    Process all specified seasons to retrieve both play-by-play and tracking data.
    """
    start_time = datetime.datetime.now()
    print(f"Starting NBA data retrieval at {start_time}")
    print(f"Base directory for data: {BASE_DIR}")
    print(f"Seasons to process: {', '.join(SEASONS)}")
    
    # Ensure base directory exists
    os.makedirs(BASE_DIR, exist_ok=True)
    
    # Process each season
    for season in SEASONS:
        print(f"\n{'='*80}")
        print(f"PROCESSING SEASON: {season}")
        print(f"{'='*80}\n")
        
        process_season_data(season)
    
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    print(f"\nAll seasons processed successfully!")
    print(f"Started: {start_time}")
    print(f"Finished: {end_time}")
    print(f"Total duration: {duration}")

if __name__ == "__main__":
    try:
        process_all_seasons()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()
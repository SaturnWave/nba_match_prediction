import pandas as pd
import time
import os
import json
import traceback
import datetime
from nba_api.stats.endpoints import BoxScoreTraditionalV2, BoxScoreAdvancedV2, BoxScoreDefensiveV2

# ----------------------------------------
# Global Configuration
# ----------------------------------------
# Base directory for saving NBA data - using absolute path
BASE_DIR = r"C:\Users\arcan\Desktop\Python\nba_new\impact_scores\nba_data"

# Initial delay between API calls
INITIAL_API_DELAY = 0.1  # seconds

# Progressive delay pattern for retries (in seconds)
RETRY_DELAYS = [0.1, 0.3, 0.5, 1.0, 2.0, 3.0]

# Current API delay (will be dynamically adjusted)
API_DELAY = INITIAL_API_DELAY

# Maximum number of retry attempts
MAX_RETRIES = 5  # Increased to match the number of retry delays

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
    Reset API delay after successful calls or maintain current delay
    """
    global API_DELAY
    
    if success:
        # Reset to initial delay after success
        API_DELAY = INITIAL_API_DELAY
        print(f"Reset API delay to {API_DELAY} seconds")

def get_retry_delay(retry_count):
    """
    Get the appropriate delay based on retry count using progressive pattern
    
    Args:
        retry_count (int): Current retry attempt (0-based)
        
    Returns:
        float: Delay in seconds to use for this retry
    """
    if retry_count < len(RETRY_DELAYS):
        return RETRY_DELAYS[retry_count]
    else:
        # If we've exceeded the defined delays, use the last one
        return RETRY_DELAYS[-1]

# ----------------------------------------
# Box Score Data Retrieval
# ----------------------------------------
def retrieve_traditional_box_score(game_id):
    """
    Retrieves traditional box score data for a given game ID using the NBA API.
    
    Args:
        game_id (str): The NBA game ID to retrieve data for
        
    Returns:
        tuple: (player_stats_df, team_stats_df) - DataFrames with player and team traditional box score data
    """
    formatted_game_id = format_game_id(game_id)
    
    print(f"Retrieving traditional box score data for game ID: {formatted_game_id}")
    
    retries = 0
    while retries <= MAX_RETRIES:
        try:
            # Fetch traditional box score data
            box_score = BoxScoreTraditionalV2(game_id=formatted_game_id)
            
            # Get data frames
            data_frames = box_score.get_data_frames()
            
            if not data_frames or len(data_frames) < 2:
                raise Exception("API returned incomplete data")
                
            player_stats_df = data_frames[0]  # Player box score
            team_stats_df = data_frames[1]    # Team box score
            
            print(f"Successfully retrieved traditional box score data for game {formatted_game_id}")
            update_api_delay(success=True)
            return player_stats_df, team_stats_df
            
        except Exception as e:
            wait_time = get_retry_delay(retries)
            print(f"Error retrieving traditional box score data: {e}. Retrying in {wait_time} seconds (Attempt {retries+1}/{MAX_RETRIES+1})...")
            time.sleep(wait_time)
            retries += 1
            
            if retries > MAX_RETRIES:
                print(f"Failed to retrieve traditional box score data for game {formatted_game_id} after {MAX_RETRIES+1} attempts: {e}")
                return None, None

def retrieve_advanced_box_score(game_id):
    """
    Retrieves advanced box score data for a given game ID using the NBA API.
    
    Args:
        game_id (str): The NBA game ID to retrieve data for
        
    Returns:
        tuple: (player_stats_df, team_stats_df) - DataFrames with player and team advanced box score data
    """
    formatted_game_id = format_game_id(game_id)
    
    print(f"Retrieving advanced box score data for game ID: {formatted_game_id}")
    
    retries = 0
    while retries <= MAX_RETRIES:
        try:
            # Fetch advanced box score data
            box_score = BoxScoreAdvancedV2(game_id=formatted_game_id)
            
            # Get data frames
            data_frames = box_score.get_data_frames()
            
            if not data_frames or len(data_frames) < 2:
                raise Exception("API returned incomplete data")
                
            player_stats_df = data_frames[0]  # Player box score
            team_stats_df = data_frames[1]    # Team box score
            
            print(f"Successfully retrieved advanced box score data for game {formatted_game_id}")
            update_api_delay(success=True)
            return player_stats_df, team_stats_df
            
        except Exception as e:
            wait_time = get_retry_delay(retries)
            print(f"Error retrieving advanced box score data: {e}. Retrying in {wait_time} seconds (Attempt {retries+1}/{MAX_RETRIES+1})...")
            time.sleep(wait_time)
            retries += 1
            
            if retries > MAX_RETRIES:
                print(f"Failed to retrieve advanced box score data for game {formatted_game_id} after {MAX_RETRIES+1} attempts: {e}")
                return None, None

def retrieve_defensive_box_score(game_id):
    """
    Retrieves defensive box score data for a given game ID using the NBA API.
    
    Args:
        game_id (str): The NBA game ID to retrieve data for
        
    Returns:
        tuple: (player_stats_df, team_stats_df) - DataFrames with player and team defensive box score data
    """
    formatted_game_id = format_game_id(game_id)
    
    print(f"Retrieving defensive box score data for game ID: {formatted_game_id}")
    
    retries = 0
    while retries <= MAX_RETRIES:
        try:
            # Fetch defensive box score data
            box_score = BoxScoreDefensiveV2(game_id=formatted_game_id)
            
            # Get data frames
            data_frames = box_score.get_data_frames()
            
            if not data_frames or len(data_frames) < 2:
                raise Exception("API returned incomplete data")
                
            player_stats_df = data_frames[0]  # Player box score
            team_stats_df = data_frames[1]    # Team box score
            
            print(f"Successfully retrieved defensive box score data for game {formatted_game_id}")
            update_api_delay(success=True)
            return player_stats_df, team_stats_df
            
        except Exception as e:
            wait_time = get_retry_delay(retries)
            
            # Check for the specific 'NoneType' error that seems to affect defensive box scores
            if "'NoneType' object has no attribute 'keys'" in str(e):
                print(f"ERROR: The defensive box score data is not available for game {formatted_game_id}. This appears to be a limitation of the NBA API.")
                # Log this game ID for future reference
                with open("defensive_box_score_unavailable_games.txt", "a") as f:
                    f.write(f"{formatted_game_id}\n")
                return None, None
                
            print(f"Error retrieving defensive box score data: {e}. Retrying in {wait_time} seconds (Attempt {retries+1}/{MAX_RETRIES+1})...")
            try:
                time.sleep(wait_time)
            except KeyboardInterrupt:
                print("\nOperation interrupted by user. Exiting gracefully...")
                return None, None
                
            retries += 1
            
            if retries > MAX_RETRIES:
                print(f"Failed to retrieve defensive box score data for game {formatted_game_id} after {MAX_RETRIES+1} attempts: {e}")
                return None, None

# ----------------------------------------
# Save Functions
# ----------------------------------------
def save_box_score_data(player_df, team_df, game_id, season, data_type):
    """
    Saves box score data to CSV with the specified directory structure.
    
    Directory structure:
    - nba_data/
        - YYYY_YYYY/
            - [game_id]/
                - box_scores/
                    - [game_id]box_score_traditional.csv or
                    - [game_id]box_score_advanced.csv or
                    - [game_id]box_score_defensive.csv
    
    Args:
        player_df (pd.DataFrame): Player box score data
        team_df (pd.DataFrame): Team box score data
        game_id (str): Game ID for directory and filename
        season (str): Season in format "YYYY-YY" (e.g., "2019-20")
        data_type (str): Type of box score data ("traditional", "advanced", or "defensive")
    
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
    box_scores_dir = os.path.join(game_dir, "box_scores")
    
    # Create directories if they don't exist
    os.makedirs(box_scores_dir, exist_ok=True)
    
    # Create filenames and paths
    player_filename = f"{formatted_game_id}box_score_{data_type}.csv"
    team_filename = f"{formatted_game_id}box_score_{data_type}_team.csv"
    
    player_filepath = os.path.join(box_scores_dir, player_filename)
    team_filepath = os.path.join(box_scores_dir, team_filename)
    
    # Save to CSV
    if player_df is not None:
        player_df.to_csv(player_filepath, index=False)
        print(f"Player {data_type} box score data saved to {player_filepath}")
    
    if team_df is not None:
        team_df.to_csv(team_filepath, index=False)
        print(f"Team {data_type} box score data saved to {team_filepath}")
    
    return player_filepath, team_filepath

def check_if_box_score_data_exists(game_id, season, data_type):
    """
    Check if box score data already exists for the given game ID.
    
    Args:
        game_id (str): The game ID to check
        season (str): Season in format "YYYY-YY" (e.g., "2019-20")
        data_type (str): Type of box score data ("traditional", "advanced", or "defensive")
    
    Returns:
        bool: True if data exists, False otherwise
    """
    formatted_game_id = format_game_id(game_id)
    
    # Convert season format (YYYY-YY to YYYY_YYYY)
    season_format = format_season(season)
    
    # Construct the expected file path
    season_dir = os.path.join(BASE_DIR, season_format)
    game_dir = os.path.join(season_dir, formatted_game_id)
    box_scores_dir = os.path.join(game_dir, "box_scores")
    player_filepath = os.path.join(box_scores_dir, f"{formatted_game_id}box_score_{data_type}.csv")
    
    # Check if the file exists
    return os.path.exists(player_filepath)

# ----------------------------------------
# Combined Processing
# ----------------------------------------
def process_season_data(season, data_types=["traditional", "advanced", "defensive"]):
    """
    Process all games from a specific season to retrieve different types of box score data.
    Only process games that don't already have data.
    
    Args:
        season (str): Season in format "YYYY-YY" (e.g., "2019-20")
        data_types (list): Types of data to retrieve - "traditional", "advanced", "defensive", or any combination
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
    missing_games = {data_type: [] for data_type in data_types}
    
    for data_type in data_types:
        for game_id in game_ids:
            if not check_if_box_score_data_exists(game_id, season, data_type):
                missing_games[data_type].append(game_id)
        print(f"Found {len(missing_games[data_type])} games without existing {data_type} box score data")
    
    # Create a unique set of all games missing any type of data
    all_missing_game_ids = set()
    for data_type in data_types:
        all_missing_game_ids.update(missing_games[data_type])
    
    total_missing = len(all_missing_game_ids)
    print(f"Total of {total_missing} unique games missing at least one type of box score data")
    
    # Process each missing game
    processed_count = 0
    
    for i, game_id in enumerate(all_missing_game_ids):
        try:
            processed_count += 1
            # Calculate progress percentage
            progress = (processed_count / total_missing) * 100
            print(f"\nProcessing game {processed_count}/{total_missing} ({progress:.1f}%): Game ID {game_id}")
            
            # Traditional Box Score
            if game_id in missing_games.get("traditional", []) and "traditional" in data_types:
                print("Retrieving traditional box score data...")
                player_df, team_df = retrieve_traditional_box_score(game_id)
                if player_df is not None and team_df is not None:
                    save_box_score_data(player_df, team_df, game_id, season, "traditional")
                time.sleep(API_DELAY)  # Add delay between API calls
            
            # Advanced Box Score
            if game_id in missing_games.get("advanced", []) and "advanced" in data_types:
                print("Retrieving advanced box score data...")
                player_df, team_df = retrieve_advanced_box_score(game_id)
                if player_df is not None and team_df is not None:
                    save_box_score_data(player_df, team_df, game_id, season, "advanced")
                time.sleep(API_DELAY)  # Add delay between API calls
            
            # Defensive Box Score
            if game_id in missing_games.get("defensive", []) and "defensive" in data_types:
                print("Retrieving defensive box score data...")
                player_df, team_df = retrieve_defensive_box_score(game_id)
                if player_df is not None and team_df is not None:
                    save_box_score_data(player_df, team_df, game_id, season, "defensive")
                
            # Add delay between games to avoid hitting API rate limits
            if i < len(all_missing_game_ids) - 1:  # No need to wait after the last game
                print(f"Waiting {API_DELAY} seconds before next game...")
                try:
                    time.sleep(API_DELAY)
                except KeyboardInterrupt:
                    print("\nOperation interrupted by user. Exiting gracefully...")
                    return
        except KeyboardInterrupt:
            print("\nOperation interrupted by user. Exiting gracefully...")
            return
    
    print(f"\n{season} season box score processing completed!")

def process_all_seasons():
    """
    Process all specified seasons to retrieve all types of box score data.
    """
    start_time = datetime.datetime.now()
    print(f"Starting NBA box score data retrieval at {start_time}")
    print(f"Base directory for data: {BASE_DIR}")
    print(f"Seasons to process: {', '.join(SEASONS)}")
    print(f"Initial API delay: {INITIAL_API_DELAY} seconds")
    print(f"Retry delay pattern: {RETRY_DELAYS}")
    
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
    except KeyboardInterrupt:
        print("\nOperation interrupted by user. Exiting gracefully...")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()
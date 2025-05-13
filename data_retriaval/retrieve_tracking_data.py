import pandas as pd
import time
import os
import json
from nba_api.stats.endpoints import BoxScorePlayerTrackV2
import traceback

# Define the base directory for all NBA data using the absolute path
BASE_DIR = r"C:\Users\arcan\Desktop\Python\nba_new\impact_scores\nba_data"

def retrieve_player_tracking_data(game_id, max_retries=3):
    """
    Retrieves player tracking data for a given game ID using the NBA API.
    
    Args:
        game_id (str or int): The NBA game ID to retrieve data for
        max_retries (int): Maximum number of retry attempts
        
    Returns:
        tuple: (player_stats_df, team_stats_df) - DataFrames with player and team tracking data
    """
    # Convert game_id to string if it's an integer
    game_id_str = str(game_id)
    
    # Format game_id to include leading zeros (proper NBA game ID format)
    formatted_game_id = f"00{game_id_str}" if not game_id_str.startswith('00') else game_id_str
    
    print(f"Retrieving player tracking data for game ID: {formatted_game_id}")
    
    retries = 0
    while retries <= max_retries:
        try:
            # Fetch player tracking data
            box_score_player_track_v2 = BoxScorePlayerTrackV2(game_id=formatted_game_id)
            
            # Convert response to dictionary
            box_score_player_track_v2_data = box_score_player_track_v2.get_dict()
            
            # Extract player tracking data
            player_stats = box_score_player_track_v2_data['resultSets'][0]['rowSet']
            team_stats = box_score_player_track_v2_data['resultSets'][1]['rowSet']
            
            # Get headers for columns
            player_headers = box_score_player_track_v2_data['resultSets'][0]['headers']
            team_headers = box_score_player_track_v2_data['resultSets'][1]['headers']
            
            # Convert to DataFrames
            player_stats_df = pd.DataFrame(player_stats, columns=player_headers)
            team_stats_df = pd.DataFrame(team_stats, columns=team_headers)
            
            # Add game_id column for reference
            player_stats_df['GAME_ID_CLEAN'] = formatted_game_id
            team_stats_df['GAME_ID_CLEAN'] = formatted_game_id
            
            print(f"Successfully retrieved player tracking data for game {formatted_game_id}")
            return player_stats_df, team_stats_df
            
        except Exception as e:
            retries += 1
            if retries <= max_retries:
                wait_time = 3 * retries  # Increase wait time with each retry
                print(f"Error retrieving tracking data: {e}. Retrying in {wait_time} seconds (Attempt {retries}/{max_retries})...")
                time.sleep(wait_time)
            else:
                print(f"Failed to retrieve player tracking data for game {formatted_game_id} after {max_retries} attempts: {e}")
                return None, None

def save_player_tracking_data(player_df, team_df, game_id, season="2024_2025", base_dir=None):
    """
    Saves player tracking data to CSV with the specified directory structure.
    
    Directory structure:
    - nba_data/
        - [season]/
            - [game_id]/
                - player_tracking/
                    - [game_id]player_tracking.csv
                    - [game_id]team_tracking.csv
    
    Args:
        player_df (pd.DataFrame): Player tracking data
        team_df (pd.DataFrame): Team tracking data
        game_id (str or int): Game ID for directory and filename
        season (str): Season folder name (default: "2024_2025")
        base_dir (str, optional): Base directory for data
    
    Returns:
        tuple: (player_filepath, team_filepath) - Paths to the saved files
    """
    # Convert game_id to string if it's an integer
    game_id_str = str(game_id)
    
    # Format game_id to include leading zeros
    formatted_game_id = f"00{game_id_str}" if not game_id_str.startswith('00') else game_id_str
    
    # Use the global BASE_DIR if base_dir is not provided
    if base_dir is None:
        nba_data_dir = BASE_DIR
    else:
        nba_data_dir = os.path.join(base_dir, "nba_data")
    
    # Construct the directory path with season parameter
    season_dir = os.path.join(nba_data_dir, season)
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

def check_if_tracking_data_exists(game_id, season="2024_2025", base_dir=None):
    """
    Check if player tracking data already exists for the given game ID.
    
    Args:
        game_id (str or int): The game ID to check
        season (str): Season folder name (default: "2024_2025")
        base_dir (str, optional): Base directory for data
    
    Returns:
        bool: True if data exists, False otherwise
    """
    # Convert game_id to string if it's an integer
    game_id_str = str(game_id)
    
    # Format game_id to include leading zeros
    formatted_game_id = f"00{game_id_str}" if not game_id_str.startswith('00') else game_id_str
    
    # Use the global BASE_DIR if base_dir is not provided
    if base_dir is None:
        nba_data_dir = BASE_DIR
    else:
        nba_data_dir = os.path.join(base_dir, "nba_data")
    
    # Construct the expected file path with season parameter
    season_dir = os.path.join(nba_data_dir, season)
    game_dir = os.path.join(season_dir, formatted_game_id)
    tracking_dir = os.path.join(game_dir, "player_tracking")
    player_filepath = os.path.join(tracking_dir, f"{formatted_game_id}player_tracking.csv")
    
    # Check if the file exists
    return os.path.exists(player_filepath)

def process_all_games_tracking_data(game_ids_path, season="2024_2025"):
    """
    Process all games from the CSV file of game IDs to retrieve player tracking data.
    Only process games that don't already have data.
    
    Args:
        game_ids_path (str): Path to the CSV file with game IDs
        season (str): Season folder name (default: "2024_2025")
    """
    # Read game IDs from CSV
    try:
        games_df = pd.read_csv(game_ids_path)
    except Exception as e:
        print(f"Error reading game IDs CSV file: {e}")
        return
    
    # Ensure we have unique game IDs
    games_df = games_df.drop_duplicates(subset=['GAME_ID'])
    game_ids = games_df['GAME_ID'].unique().tolist()
    
    print(f"Found {len(game_ids)} unique games in CSV")
    
    # Filter out games that already have tracking data
    missing_game_ids = []
    for game_id in game_ids:
        if not check_if_tracking_data_exists(game_id, season=season):
            missing_game_ids.append(game_id)
    
    print(f"Found {len(missing_game_ids)} games without existing tracking data")
    
    # Process each missing game
    successful_games = 0
    failed_games = 0
    
    for i, game_id in enumerate(missing_game_ids):
        print(f"Processing game {i+1}/{len(missing_game_ids)}: {game_id}")
        
        # Retrieve player tracking data
        player_df, team_df = retrieve_player_tracking_data(game_id)
        
        # Save the data if retrieved successfully
        if player_df is not None and team_df is not None:
            save_player_tracking_data(player_df, team_df, game_id, season=season)
            successful_games += 1
            # Add delay to avoid hitting API rate limits
            if i < len(missing_game_ids) - 1:  # No need to wait after the last game
                delay = 3
                print(f"Waiting {delay} seconds before next request...")
                time.sleep(delay)
        else:
            failed_games += 1
            # Continue to the next game if retrieval failed after retries
            if i < len(missing_game_ids) - 1:
                print("Continuing to next game...")
    
    print(f"\nProcessing completed: {successful_games} games successful, {failed_games} games failed")

def validate_game_ids(game_ids_path):
    """
    Validates game IDs from the CSV file to ensure they are in the correct format
    for the NBA API.
    
    Args:
        game_ids_path (str): Path to the CSV file with game IDs
    """
    try:
        games_df = pd.read_csv(game_ids_path)
        
        # Print sample game IDs for validation
        sample_game_ids = games_df['GAME_ID'].head(5).tolist()
        print(f"\nSample Game IDs from CSV: {sample_game_ids}")
        
        # Check data type of GAME_ID column
        game_id_dtype = games_df['GAME_ID'].dtype
        print(f"Game ID data type in CSV: {game_id_dtype}")
        
        # Convert a sample to properly formatted game ID
        for game_id in sample_game_ids:
            game_id_str = str(game_id)
            formatted_id = f"00{game_id_str}" if not game_id_str.startswith('00') else game_id_str
            print(f"Original: {game_id} -> Formatted: {formatted_id}")
            
    except Exception as e:
        print(f"Error validating game IDs: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    try:
        # Set season
        current_season = "2024_2025"
        
        # First check if game_id CSV exists in the data_retriaval directory
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"game_id_{current_season}.csv")
        
        # If not found, check in the parent directory (impact_scores)
        if not os.path.exists(csv_path):
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            csv_path = os.path.join(parent_dir, f"game_id_{current_season}.csv")
        
        if not os.path.exists(csv_path):
            print(f"Game IDs CSV file not found at either:")
            print(f"- {os.path.join(os.path.dirname(os.path.abspath(__file__)), f'game_id_{current_season}.csv')}")
            print(f"- {os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), f'game_id_{current_season}.csv')}")
            print(f"Please run game_data_retriaval.py first to generate the game IDs CSV file for {current_season}.")
            exit(1)
        
        print(f"Found game IDs CSV file at: {csv_path}")
        
        # Validate some game IDs from the CSV for debugging
        validate_game_ids(csv_path)
        
        # Process games to retrieve and save player tracking data
        print(f"\nStarting to process games for player tracking data for season {current_season}...")
        process_all_games_tracking_data(csv_path, season=current_season)
        
        print("\nProcess completed successfully!")
        
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()
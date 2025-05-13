import pandas as pd
import time
import os
import json
from nba_api.stats.endpoints import LeagueGameFinder, PlayByPlayV3

# Define the base directory for all NBA data using the absolute path
BASE_DIR = r"C:\Users\arcan\Desktop\Python\nba_new\impact_scores\nba_data"

def get_regular_season_game_ids(season="2024-25"):
    """
    Retrieves all game IDs from the specified NBA regular season.
    
    Args:
        season (str): Season in format "YYYY-YY" (e.g., "2023-24")
        
    Returns:
        list: List of game IDs
    """
    print(f"Retrieving game IDs for {season} regular season...")
    
    try:
        # Use LeagueGameFinder to get regular season games
        game_finder = LeagueGameFinder(
            season_nullable=season,
            season_type_nullable="Regular Season",
            league_id_nullable="00"  # NBA
        )
        
        games_df = game_finder.get_data_frames()[0]
        
        # Extract unique game IDs and remove duplicates
        game_ids = games_df['GAME_ID'].unique().tolist()
        
        print(f"Found {len(game_ids)} unique games for {season} regular season")
        return game_ids, games_df
    
    except Exception as e:
        print(f"Error retrieving game IDs: {e}")
        return [], None

def save_game_ids_to_csv(game_ids_df, season, output_dir=None):
    """
    Saves the game IDs DataFrame to a CSV file.
    
    Args:
        game_ids_df (pd.DataFrame): DataFrame with game IDs and metadata
        season (str): Season in format "YYYY-YY" (e.g., "2023-24")
        output_dir (str, optional): Directory to save the file.
    
    Returns:
        str: Path to the saved file
    """
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Remove duplicates based on GAME_ID
    unique_games_df = game_ids_df.drop_duplicates(subset=['GAME_ID'])
    
    print(f"Removed {len(game_ids_df) - len(unique_games_df)} duplicate game IDs")
    print(f"Saving {len(unique_games_df)} unique game IDs to CSV")
    
    # Create season format for filename (YYYY_YYYY)
    season_parts = season.split("-")
    season_format = f"{season_parts[0]}_{int(season_parts[0])+1}"
    
    # Create filename
    filename = f"game_id_{season_format}.csv"
    filepath = os.path.join(output_dir, filename)
    
    # Save to CSV
    unique_games_df.to_csv(filepath, index=False)
    print(f"Game IDs saved to {filepath}")
    
    return filepath

def retrieve_play_by_play(game_id, max_retries=3):
    """
    Retrieves play-by-play data for a given game ID using the NBA API.
    
    Args:
        game_id (str): The NBA game ID to retrieve data for
        max_retries (int): Maximum number of retry attempts
        
    Returns:
        pd.DataFrame: Play-by-play data for the specified game
    """
    # Format game_id to include leading zeros (proper NBA game ID format)
    formatted_game_id = f"00{game_id}"
    
    print(f"Retrieving play-by-play data for game ID: {formatted_game_id}")
    
    retries = 0
    while retries <= max_retries:
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
            return play_by_play_df
            
        except Exception as e:
            retries += 1
            if retries <= max_retries:
                wait_time = 3 * retries  # Increase wait time with each retry
                print(f"Error retrieving data: {e}. Retrying in {wait_time} seconds (Attempt {retries}/{max_retries})...")
                time.sleep(wait_time)
            else:
                print(f"Failed to retrieve play-by-play data for game {formatted_game_id} after {max_retries} attempts: {e}")
                return None

def save_play_by_play_data(df, game_id, season, base_dir=None):
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
        season (str): Season in format "YYYY-YY" (e.g., "2023-24")
        base_dir (str, optional): Base directory for data
    
    Returns:
        str: Path to the saved file
    """
    # Format game_id to include leading zeros
    formatted_game_id = f"00{game_id}"
    
    # Use the global BASE_DIR if base_dir is not provided
    if base_dir is None:
        nba_data_dir = BASE_DIR
    else:
        nba_data_dir = os.path.join(base_dir, "nba_data")
    
    # Convert season format (YYYY-YY to YYYY_YYYY)
    season_parts = season.split("-")
    season_format = f"{season_parts[0]}_{int(season_parts[0])+1}"
    
    # Construct the directory path
    season_dir = os.path.join(nba_data_dir, season_format)
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

def check_if_data_exists(game_id, season, base_dir=None):
    """
    Check if play-by-play data already exists for the given game ID.
    
    Args:
        game_id (str): The game ID to check
        season (str): Season in format "YYYY-YY" (e.g., "2023-24")
        base_dir (str, optional): Base directory for data
    
    Returns:
        bool: True if data exists, False otherwise
    """
    # Format game_id to include leading zeros
    formatted_game_id = f"00{game_id}"
    
    # Use the global BASE_DIR if base_dir is not provided
    if base_dir is None:
        nba_data_dir = BASE_DIR
    else:
        nba_data_dir = os.path.join(base_dir, "nba_data")
    
    # Convert season format (YYYY-YY to YYYY_YYYY)
    season_parts = season.split("-")
    season_format = f"{season_parts[0]}_{int(season_parts[0])+1}"
    
    # Construct the expected file path
    season_dir = os.path.join(nba_data_dir, season_format)
    game_dir = os.path.join(season_dir, formatted_game_id)
    pbp_dir = os.path.join(game_dir, "play_by_play")
    filepath = os.path.join(pbp_dir, f"{formatted_game_id}pbp.csv")
    
    # Check if the file exists
    return os.path.exists(filepath)

def process_all_games(game_ids_path, season):
    """
    Process all games from the CSV file of game IDs.
    Only process games that don't already have data.
    
    Args:
        game_ids_path (str): Path to the CSV file with game IDs
        season (str): Season in format "YYYY-YY" (e.g., "2023-24")
    """
    # Read game IDs from CSV
    games_df = pd.read_csv(game_ids_path)
    
    # Ensure we have unique game IDs
    games_df = games_df.drop_duplicates(subset=['GAME_ID'])
    game_ids = games_df['GAME_ID'].unique().tolist()
    
    print(f"Found {len(game_ids)} unique games in CSV")
    
    # Filter out games that already have data
    missing_game_ids = []
    for game_id in game_ids:
        if not check_if_data_exists(game_id, season):
            missing_game_ids.append(game_id)
    
    print(f"Found {len(missing_game_ids)} games without existing data")
    
    # Process each missing game
    for i, game_id in enumerate(missing_game_ids):
        print(f"Processing game {i+1}/{len(missing_game_ids)}: {game_id}")
        
        # Retrieve play-by-play data
        pbp_df = retrieve_play_by_play(game_id)
        
        # Save the data if retrieved successfully
        if pbp_df is not None:
            save_play_by_play_data(pbp_df, game_id, season)
            # Smaller delay if successful
            if i < len(missing_game_ids) - 1:  # No need to wait after the last game
                print(f"Waiting 3 seconds before next request...")
                time.sleep(3)
        else:
            # Continue to the next game if retrieval failed after retries
            if i < len(missing_game_ids) - 1:
                print("Continuing to next game...")

def get_season_formatted_filename(season):
    """
    Convert season format from "YYYY-YY" to "YYYY_YYYY" for filenames.
    
    Args:
        season (str): Season in format "YYYY-YY" (e.g., "2023-24")
    
    Returns:
        str: Season in format "YYYY_YYYY" (e.g., "2023_2024")
    """
    season_parts = season.split("-")
    season_format = f"{season_parts[0]}_{int(season_parts[0])+1}"
    return season_format

if __name__ == "__main__":
    # Focus on retrieving 2024-25 season data
    seasons = ["2024-25"]
    
    # Process the 2024-25 season
    for season in seasons:
        print(f"\n{'='*80}")
        print(f"PROCESSING SEASON: {season}")
        print(f"{'='*80}\n")
        
        season_format = get_season_formatted_filename(season)
        
        # Define the exact target CSV path first
        target_csv_path = r"C:\Users\arcan\Desktop\Python\nba_new\impact_scores\game_id_2024_2025.csv"
        
        # Alternative paths
        data_retriaval_csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"game_id_{season_format}.csv")
        parent_csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), f"game_id_{season_format}.csv")
        
        # First check if the CSV exists at the specific target path
        if os.path.exists(target_csv_path):
            game_ids_path = target_csv_path
            print(f"Using existing game IDs from {target_csv_path}")
        # Then check the parent directory
        elif os.path.exists(parent_csv_path):
            game_ids_path = parent_csv_path
            print(f"Using existing game IDs from {parent_csv_path}")
        # Finally check the data_retriaval directory
        elif os.path.exists(data_retriaval_csv_path):
            game_ids_path = data_retriaval_csv_path
            print(f"Using existing game IDs from {data_retriaval_csv_path}")
        else:
            # Step 1: Retrieve all game IDs for the season
            game_ids, games_df = get_regular_season_game_ids(season)
            
            if len(game_ids) > 0:
                # Step 2: Save game IDs to CSV (save to both directories)
                game_ids_path = save_game_ids_to_csv(games_df, season)
                
                # Also save to parent directory - this will be the target_csv_path
                parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                save_game_ids_to_csv(games_df, season, output_dir=parent_dir)
                print(f"Game IDs also saved to {os.path.join(parent_dir, f'game_id_{season_format}.csv')}")
            else:
                print(f"Failed to retrieve game IDs for {season}. Exiting.")
                exit(1)
        
        # Step 3: Process games to retrieve and save play-by-play data
        print(f"\nStarting to process games for {season} play-by-play data...")
        process_all_games(game_ids_path, season)
        
        print(f"\n{season} processing completed!")
    
    print("\nSeason 2024-2025 processed successfully!")
    
    # Uncomment below to process additional seasons if needed
    # additional_seasons = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24"]
    # for season in additional_seasons:
    #     print(f"\n{'='*80}")
    #     print(f"PROCESSING SEASON: {season}")
    #     print(f"{'='*80}\n")
    #     # ...process season here...

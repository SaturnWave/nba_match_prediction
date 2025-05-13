import pandas as pd
import time
import os
import json
import traceback
from nba_api.stats.endpoints import PlayByPlayV3, BoxScorePlayerTrackV2, LeagueGameFinder

# Define the base directory for all NBA data using the absolute path
BASE_DIR = r"C:\Users\arcan\Desktop\Python\nba_new\impact_scores\nba_data"

# --- General Helper Functions ---
def format_game_id(game_id):
    """Format game ID to ensure it has the proper NBA API format with leading zeros."""
    game_id_str = str(game_id)
    return f"00{game_id_str}" if not game_id_str.startswith('00') else game_id_str

def get_season_formatted(season):
    """Convert season format from 'YYYY-YY' to 'YYYY_YYYY' for directory and file naming."""
    season_parts = season.split('-')
    if len(season_parts) == 2:
        year1 = season_parts[0]
        year2 = int(year1[:2] + season_parts[1])
        season_format = f"{year1}_{year2}"
    else:
        # Handle directly provided format like "2023_2024"
        season_format = season
    return season_format

def calculate_progress_percentage(current, total):
    """Calculate and format progress percentage."""
    return f"{(current / total * 100):.1f}%" if total > 0 else "0.0%"

def create_season_dir(season, base_dir=None):
    """
    Create the season directory structure if it doesn't exist.
    
    Args:
        season (str): Season format YYYY_YYYY
        base_dir (str, optional): Base directory path. Defaults to script location.
    
    Returns:
        str: Path to the season directory
    """
    # Use the global BASE_DIR if base_dir is not provided
    if base_dir is None:
        nba_data_dir = BASE_DIR
    else:
        nba_data_dir = os.path.join(base_dir, "nba_data")
    
    season_dir = os.path.join(nba_data_dir, season)
    
    # Create the directory if it doesn't exist
    os.makedirs(season_dir, exist_ok=True)
    
    return season_dir

# --- Play by Play Data Functions ---
def retrieve_play_by_play(game_id, max_retries=3, delay=0.1):
    """
    Retrieves play-by-play data for a given game ID using the NBA API.
    
    Args:
        game_id (str or int): The NBA game ID to retrieve data for
        max_retries (int): Maximum number of retry attempts
        delay (float): Initial delay in seconds between retries
        
    Returns:
        pd.DataFrame: Play-by-play data for the specified game
    """
    # Format game_id to include leading zeros
    formatted_game_id = format_game_id(game_id)
    
    print(f"Retrieving play-by-play data for game ID: {formatted_game_id}")
    
    retries = 0
    current_delay = delay
    
    while retries <= max_retries:
        try:
            # Fetch play-by-play data
            pbp = PlayByPlayV3(
                game_id=formatted_game_id,
                start_period=1,
                end_period=10  # Covers all possible overtimes
            )
            
            # Convert to DataFrame
            data_frames = pbp.get_data_frames()
            if not data_frames or len(data_frames) == 0:
                raise Exception("API returned empty data")
                
            play_by_play_df = data_frames[0]
            
            print(f"Successfully retrieved {len(play_by_play_df)} play-by-play records for game {formatted_game_id}")
            return play_by_play_df
            
        except Exception as e:
            retries += 1
            # Calculate wait time with progressive delay strategy
            if retries <= max_retries:
                print(f"Error retrieving play-by-play data: {e}. Retrying in {current_delay} seconds (Attempt {retries}/{max_retries})...")
                time.sleep(current_delay)
                # Increase delay for next retry using the progressive strategy
                if retries == 1:
                    current_delay = 0.3
                elif retries == 2:
                    current_delay = 0.5
                else:
                    current_delay = min(current_delay * 2, 3.0)  # Max delay of 3 seconds
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
        game_id (str or int): Game ID for directory and filename
        season (str): Season in format "YYYY_YYYY"
        base_dir (str, optional): Base directory for data
    
    Returns:
        str: Path to the saved file
    """
    # Format game_id to include leading zeros
    formatted_game_id = format_game_id(game_id)
    
    # Use the global BASE_DIR if base_dir is not provided
    if base_dir is None:
        nba_data_dir = BASE_DIR
    else:
        nba_data_dir = os.path.join(base_dir, "nba_data")
    
    # Construct the directory path
    season_dir = os.path.join(nba_data_dir, season)
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

def check_if_pbp_data_exists(game_id, season, base_dir=None):
    """
    Check if play-by-play data already exists for the given game ID.
    
    Args:
        game_id (str or int): The game ID to check
        season (str): Season in format "YYYY_YYYY"
        base_dir (str, optional): Base directory for data
    
    Returns:
        bool: True if data exists, False otherwise
    """
    formatted_game_id = format_game_id(game_id)
    
    # Use the global BASE_DIR if base_dir is not provided
    if base_dir is None:
        nba_data_dir = BASE_DIR
    else:
        nba_data_dir = os.path.join(base_dir, "nba_data")
    
    # Construct the expected file path
    season_dir = os.path.join(nba_data_dir, season)
    game_dir = os.path.join(season_dir, formatted_game_id)
    pbp_dir = os.path.join(game_dir, "play_by_play")
    filepath = os.path.join(pbp_dir, f"{formatted_game_id}pbp.csv")
    
    # Check if the file exists
    return os.path.exists(filepath)

# --- Player Tracking Data Functions ---
def retrieve_player_tracking_data(game_id, max_retries=3, delay=0.1):
    """
    Retrieves player tracking data for a given game ID using the NBA API.
    
    Args:
        game_id (str or int): The NBA game ID to retrieve data for
        max_retries (int): Maximum number of retry attempts
        delay (float): Initial delay in seconds between retries
        
    Returns:
        tuple: (player_stats_df, team_stats_df) - DataFrames with player and team tracking data
    """
    # Format game_id to include leading zeros
    formatted_game_id = format_game_id(game_id)
    
    print(f"Retrieving player tracking data for game ID: {formatted_game_id}")
    
    retries = 0
    current_delay = delay
    
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
            # Calculate wait time with progressive delay strategy
            if retries <= max_retries:
                print(f"Error retrieving tracking data: {e}. Retrying in {current_delay} seconds (Attempt {retries}/{max_retries})...")
                time.sleep(current_delay)
                # Increase delay for next retry using the progressive strategy
                if retries == 1:
                    current_delay = 0.3
                elif retries == 2:
                    current_delay = 0.5
                else:
                    current_delay = min(current_delay * 2, 3.0)  # Max delay of 3 seconds
            else:
                print(f"Failed to retrieve player tracking data for game {formatted_game_id} after {max_retries} attempts: {e}")
                return None, None

def save_player_tracking_data(player_df, team_df, game_id, season, base_dir=None):
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
        game_id (str or int): Game ID for directory and filename
        season (str): Season in format "YYYY_YYYY"
        base_dir (str, optional): Base directory for data
    
    Returns:
        tuple: (player_filepath, team_filepath) - Paths to the saved files
    """
    # Format game_id to include leading zeros
    formatted_game_id = format_game_id(game_id)
    
    # Use the global BASE_DIR if base_dir is not provided
    if base_dir is None:
        nba_data_dir = BASE_DIR
    else:
        nba_data_dir = os.path.join(base_dir, "nba_data")
    
    # Construct the directory path
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

def check_if_tracking_data_exists(game_id, season, base_dir=None):
    """
    Check if player tracking data already exists for the given game ID.
    
    Args:
        game_id (str or int): The game ID to check
        season (str): Season in format "YYYY_YYYY"
        base_dir (str, optional): Base directory for data
    
    Returns:
        bool: True if data exists, False otherwise
    """
    formatted_game_id = format_game_id(game_id)
    
    # Use the global BASE_DIR if base_dir is not provided
    if base_dir is None:
        nba_data_dir = BASE_DIR
    else:
        nba_data_dir = os.path.join(base_dir, "nba_data")
    
    # Construct the expected file path
    season_dir = os.path.join(nba_data_dir, season)
    game_dir = os.path.join(season_dir, formatted_game_id)
    tracking_dir = os.path.join(game_dir, "player_tracking")
    player_filepath = os.path.join(tracking_dir, f"{formatted_game_id}player_tracking.csv")
    
    # Check if the file exists
    return os.path.exists(player_filepath)

# --- Main Processing Functions ---
def process_season_games(csv_path, season, data_type='both', max_retries=3, delay=0.1):
    """
    Process all games from a CSV file of game IDs for a specific season.
    
    Args:
        csv_path (str): Path to the CSV file with game IDs
        season (str): Season in format 'YYYY-YY' or 'YYYY_YYYY'
        data_type (str): Type of data to retrieve ('play_by_play', 'tracking', or 'both')
        max_retries (int): Maximum number of retry attempts for API calls
        delay (float): Initial delay in seconds between retries
        
    Returns:
        tuple: (successful_games, failed_games, total_games)
    """
    # Format season consistently
    season_formatted = get_season_formatted(season)
    
    print(f"\n{'='*80}")
    print(f"PROCESSING {season_formatted} SEASON - {data_type.upper()} DATA")
    print(f"{'='*80}")
    
    # Create season directory
    create_season_dir(season_formatted)
    
    # Read game IDs from CSV
    try:
        games_df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading game IDs CSV file: {e}")
        return 0, 0, 0
    
    # Ensure we have unique game IDs
    games_df = games_df.drop_duplicates(subset=['GAME_ID'])
    game_ids = games_df['GAME_ID'].unique().tolist()
    
    print(f"Found {len(game_ids)} unique games in CSV")
    
    # Filter out games that already have data
    pbp_missing = []
    tracking_missing = []
    
    if data_type in ['play_by_play', 'both']:
        for game_id in game_ids:
            if not check_if_pbp_data_exists(game_id, season_formatted):
                pbp_missing.append(game_id)
        print(f"Found {len(pbp_missing)} games without existing play-by-play data ({calculate_progress_percentage(len(pbp_missing), len(game_ids))} needed)")
    
    if data_type in ['tracking', 'both']:
        for game_id in game_ids:
            if not check_if_tracking_data_exists(game_id, season_formatted):
                tracking_missing.append(game_id)
        print(f"Found {len(tracking_missing)} games without existing tracking data ({calculate_progress_percentage(len(tracking_missing), len(game_ids))} needed)")
    
    # Process play-by-play data
    pbp_successful = 0
    pbp_failed = 0
    
    if data_type in ['play_by_play', 'both'] and pbp_missing:
        print(f"\nRetrieving play-by-play data for {len(pbp_missing)} games...")
        
        # Adaptive delay based on API response
        current_delay = delay
        consecutive_errors = 0
        
        for i, game_id in enumerate(pbp_missing):
            progress = calculate_progress_percentage(i+1, len(pbp_missing))
            print(f"Processing game {i+1}/{len(pbp_missing)} ({progress}): {game_id}")
            
            # Retrieve play-by-play data
            pbp_df = retrieve_play_by_play(game_id, max_retries, current_delay)
            
            # Save the data if retrieved successfully
            if pbp_df is not None and not pbp_df.empty:
                save_play_by_play_data(pbp_df, game_id, season_formatted)
                pbp_successful += 1
                consecutive_errors = 0  # Reset consecutive errors
                
                # Add delay to avoid hitting API rate limits
                if i < len(pbp_missing) - 1:
                    time.sleep(current_delay)
            else:
                pbp_failed += 1
                consecutive_errors += 1
                
                # Adjust delay based on consecutive errors using exact progression
                if consecutive_errors >= 3:
                    # Exact progression: 0.1 -> 0.3 -> 0.5 -> 1.0 -> 2.0 -> 3.0
                    if current_delay < 0.1 or current_delay == 0.1:
                        current_delay = 0.3
                    elif current_delay == 0.3:
                        current_delay = 0.5
                    elif current_delay == 0.5:
                        current_delay = 1.0
                    elif current_delay == 1.0:
                        current_delay = 2.0
                    elif current_delay == 2.0 or current_delay > 2.0:
                        current_delay = 3.0
                    print(f"Increasing delay to {current_delay}s due to consecutive errors")
                    consecutive_errors = 0  # Reset counter after adjustment
    
    # Process tracking data
    tracking_successful = 0
    tracking_failed = 0
    
    if data_type in ['tracking', 'both'] and tracking_missing:
        print(f"\nRetrieving player tracking data for {len(tracking_missing)} games...")
        
        # Adaptive delay based on API response
        current_delay = delay
        consecutive_errors = 0
        
        for i, game_id in enumerate(tracking_missing):
            progress = calculate_progress_percentage(i+1, len(tracking_missing))
            print(f"Processing game {i+1}/{len(tracking_missing)} ({progress}): {game_id}")
            
            # Retrieve player tracking data
            player_df, team_df = retrieve_player_tracking_data(game_id, max_retries, current_delay)
            
            # Save the data if retrieved successfully
            if player_df is not None and team_df is not None:
                save_player_tracking_data(player_df, team_df, game_id, season_formatted)
                tracking_successful += 1
                consecutive_errors = 0  # Reset consecutive errors
                
                # Add delay to avoid hitting API rate limits
                if i < len(tracking_missing) - 1:
                    time.sleep(current_delay)
            else:
                tracking_failed += 1
                consecutive_errors += 1
                
                # Adjust delay based on consecutive errors using exact progression
                if consecutive_errors >= 3:
                    # Exact progression: 0.1 -> 0.3 -> 0.5 -> 1.0 -> 2.0 -> 3.0
                    if current_delay < 0.1 or current_delay == 0.1:
                        current_delay = 0.3
                    elif current_delay == 0.3:
                        current_delay = 0.5
                    elif current_delay == 0.5:
                        current_delay = 1.0
                    elif current_delay == 1.0:
                        current_delay = 2.0
                    elif current_delay == 2.0 or current_delay > 2.0:
                        current_delay = 3.0
                    print(f"Increasing delay to {current_delay}s due to consecutive errors")
                    consecutive_errors = 0  # Reset counter after adjustment
    
    # Calculate and return results
    successful_games = pbp_successful + tracking_successful
    failed_games = pbp_failed + tracking_failed
    total_games = len(pbp_missing) + len(tracking_missing)
    
    print(f"\n{season_formatted} SUMMARY:")
    if data_type in ['play_by_play', 'both']:
        print(f"Play-by-Play: {pbp_successful} successful, {pbp_failed} failed")
    if data_type in ['tracking', 'both']:
        print(f"Tracking: {tracking_successful} successful, {tracking_failed} failed")
    print(f"Total: {successful_games} successful, {failed_games} failed, {total_games} total")
    
    return successful_games, failed_games, total_games

def process_all_seasons(seasons=['2019_2020', '2020_2021', '2021_2022', '2022_2023'], 
                      data_type='both', max_retries=3, initial_delay=0.1):
    """
    Process multiple seasons to retrieve play-by-play and tracking data.
    
    Args:
        seasons (list): List of seasons in format 'YYYY_YYYY'
        data_type (str): Type of data to retrieve ('play_by_play', 'tracking', or 'both')
        max_retries (int): Maximum number of retry attempts for API calls
        initial_delay (float): Initial delay in seconds between retries
    """
    print(f"Starting to process {len(seasons)} seasons for {data_type} data...")
    
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    
    # Keep track of results
    total_successful = 0
    total_failed = 0
    total_games = 0
    
    # Track API errors to adjust delay 
    current_delay = initial_delay
    consecutive_errors = 0
    
    # Process each season
    for season in seasons:
        try:
            # Construct path to the game IDs CSV
            csv_filename = f"game_id_{season}.csv"
            csv_path = os.path.join(base_dir, csv_filename)
            
            if not os.path.exists(csv_path):
                print(f"CSV file not found: {csv_path}")
                continue
            
            # Process the season's games
            successful, failed, games = process_season_games(
                csv_path, season, data_type, max_retries, current_delay)
            
            # Update totals
            total_successful += successful
            total_failed += failed
            total_games += games
            
            # Adjust delay based on error rate for the next season (using exact progression)
            if failed > successful and games > 10:
                # If more failures than successes, increase delay
                consecutive_errors += 1
                if consecutive_errors >= 2:
                    # Exact progression: 0.1 -> 0.3 -> 0.5 -> 1.0 -> 2.0 -> 3.0
                    if current_delay < 0.1 or current_delay == 0.1:
                        current_delay = 0.3
                    elif current_delay == 0.3:
                        current_delay = 0.5
                    elif current_delay == 0.5:
                        current_delay = 1.0
                    elif current_delay == 1.0:
                        current_delay = 2.0
                    elif current_delay == 2.0 or current_delay > 2.0:
                        current_delay = 3.0
                    print(f"Increased delay to {current_delay} seconds for next season due to high failure rate")
                    consecutive_errors = 0
            else:
                consecutive_errors = 0
                
        except Exception as e:
            print(f"Error processing season {season}: {e}")
            traceback.print_exc()
    
    # Print overall summary
    print("\n" + "="*80)
    print(f"OVERALL PROCESSING SUMMARY:")
    print(f"Total games processed: {total_games}")
    print(f"Successful retrievals: {total_successful} ({calculate_progress_percentage(total_successful, total_games)})")
    print(f"Failed retrievals: {total_failed} ({calculate_progress_percentage(total_failed, total_games)})")
    print("="*80)

# --- Main Execution Block ---
if __name__ == "__main__":
    # Define the seasons to process
    seasons = ['2019_2020', '2020_2021', '2021_2022', '2022_2023', '2023_2024']
    
    # Ask user what type of data to retrieve
    print("What type of data would you like to retrieve?")
    print("1. Play-by-play data only")
    print("2. Player tracking data only")
    print("3. Both play-by-play and tracking data (default)")
    
    try:
        choice = input("Enter your choice (1-3): ").strip()
        
        if choice == "1":
            data_type = "play_by_play"
        elif choice == "2":
            data_type = "tracking"
        else:
            data_type = "both"
            
        # Ask user for API delay
        try:
            delay_options = {
                "1": 0.1,  # Very low delay (starting point)
                "2": 0.3,  # Low delay
                "3": 0.5,  # Medium delay
                "4": 1.0,  # Standard delay
                "5": 2.0,  # High delay
                "6": 3.0   # Very high delay
            }
            
            print("\nSelect initial API delay:")
            print("1. 0.1 seconds (very aggressive, may hit rate limits)")
            print("2. 0.3 seconds (aggressive)")
            print("3. 0.5 seconds (moderate)")
            print("4. 1.0 seconds (conservative)")
            print("5. 2.0 seconds (very conservative)")
            print("6. 3.0 seconds (extremely conservative)")
            
            delay_choice = input("Enter your choice (1-6, default=1): ").strip() or "1"
            delay = delay_options.get(delay_choice, 0.1)
            
        except:
            delay = 0.1
        
        # Process all seasons
        print(f"\nStarting data retrieval with {delay}s initial delay...")
        process_all_seasons(seasons, data_type, max_retries=3, initial_delay=delay)
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting...")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()
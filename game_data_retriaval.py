import pandas as pd
import time
import os
from nba_api.stats.endpoints import PlayByPlayV3, LeagueGameFinder

def retrieve_play_by_play(game_id):
    """
    Retrieves play-by-play data for a given game ID using the NBA API.
    
    Args:
        game_id (str): The NBA game ID to retrieve data for
        
    Returns:
        pd.DataFrame: Play-by-play data for the specified game
    """
    print(f"Retrieving play-by-play data for game ID: {game_id}")
    
    try:
        # Fetch play-by-play data with numeric values for periods
        pbp = PlayByPlayV3(
            game_id=game_id,
            start_period=1,  # Using integers instead of constants
            end_period=10    # Covers all possible overtimes
        )
        
        # Convert to DataFrame
        play_by_play_df = pbp.get_data_frames()[0]
        
        print(f"Successfully retrieved {len(play_by_play_df)} play-by-play records")
        return play_by_play_df
        
    except Exception as e:
        print(f"Error retrieving play-by-play data: {e}")
        return None

def save_to_csv(df, game_id, output_dir=None):
    """
    Saves the DataFrame to a CSV file.
    
    Args:
        df (pd.DataFrame): DataFrame to save
        game_id (str): Game ID for the filename
        output_dir (str, optional): Directory to save the file. Defaults to script directory.
    
    Returns:
        str: Path to the saved file
    """
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename
    filename = f"detailed_play_by_play_{game_id}_2.csv"
    filepath = os.path.join(output_dir, filename)
    
    # Save to CSV
    df.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")
    
    return filepath

def retrieve_season_games(season):
    """
    Retrieves all game IDs for a specific NBA season.
    
    Args:
        season (str): Season in the format 'YYYY-YY' (e.g., '2019-20')
        
    Returns:
        pd.DataFrame: DataFrame containing game information including game IDs
    """
    print(f"Retrieving game IDs for the {season} NBA season...")
    
    try:
        # Use LeagueGameFinder to get all games for the season
        game_finder = LeagueGameFinder(
            season_nullable=season,
            league_id_nullable='00',  # NBA
            season_type_nullable='Regular Season'
        )
        
        # Get the DataFrame
        games_df = game_finder.get_data_frames()[0]
        
        # Filter to get unique game IDs (each game appears twice, once for each team)
        games_df = games_df.drop_duplicates(subset=['GAME_ID'])
        
        print(f"Found {len(games_df)} games for the {season} season")
        return games_df
        
    except Exception as e:
        print(f"Error retrieving games for season {season}: {e}")
        return None

def save_game_ids_to_csv(games_df, season, output_dir=None):
    """
    Saves the game IDs for a season to a CSV file.
    
    Args:
        games_df (pd.DataFrame): DataFrame containing game information
        season (str): Season string for the filename
        output_dir (str, optional): Directory to save the file. Defaults to script directory.
        
    Returns:
        str: Path to the saved file
    """
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Format season for filename (e.g., '2019-20' becomes '2019_2020')
    season_formatted = season.replace('-', '_')
    if len(season_formatted.split('_')[1]) == 2:
        # Convert '2019_20' to '2019_2020'
        year1, year2 = season_formatted.split('_')
        season_formatted = f"{year1}_{int(year1[:2] + year2):d}"
    
    # Create filename
    filename = f"game_id_{season_formatted}.csv"
    filepath = os.path.join(output_dir, filename)
    
    # Select only necessary columns
    if 'GAME_ID' in games_df.columns:
        # Keep GAME_ID and some metadata columns for reference
        columns_to_keep = ['GAME_ID', 'GAME_DATE', 'MATCHUP', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID']
        columns_to_keep = [col for col in columns_to_keep if col in games_df.columns]
        games_df = games_df[columns_to_keep]
    
    # Save to CSV
    games_df.to_csv(filepath, index=False)
    print(f"Game IDs for {season} season saved to {filepath}")
    
    return filepath

if __name__ == "__main__":
    # Define the seasons to retrieve, now including 2024-25
    seasons = ['2024-25', '2019-20', '2020-21', '2021-22', '2022-23', '2023-24']
    
    # Output directory is the root folder of the project
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Retrieve game IDs for each season and save to separate CSV files
    for season in seasons:
        print(f"\n{'='*50}\nProcessing {season} season\n{'='*50}")
        
        # Retrieve games for the season
        games_df = retrieve_season_games(season)
        
        if games_df is not None and not games_df.empty:
            # Save game IDs to CSV
            csv_path = save_game_ids_to_csv(games_df, season, output_dir)
            
            print(f"Successfully processed {season} season. Data saved to {csv_path}")
        else:
            print(f"Failed to retrieve games for {season} season")
        
        # Add a delay to avoid hitting API rate limits
        if season != seasons[-1]:  # Skip delay after the last season
            delay = 3
            print(f"Waiting {delay} seconds before processing the next season...")
            time.sleep(delay)
    
    print("\nAll seasons processed successfully!")
    
    # Original play-by-play retrieval code for reference
    # GAME_ID = "0022400058"
    # play_by_play_df = retrieve_play_by_play(GAME_ID)
    # if play_by_play_df is not None:
    #     save_to_csv(play_by_play_df, GAME_ID)
    # else:
    #     print("Failed to retrieve play-by-play data.")

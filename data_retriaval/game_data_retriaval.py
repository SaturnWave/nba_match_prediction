import pandas as pd
import time
import os
from nba_api.stats.endpoints import PlayByPlayV3

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

if __name__ == "__main__":
    # Define the game ID
    GAME_ID = "0022400058"
    
    # Add a small delay to avoid hitting API rate limits
    time.sleep(1)
    
    # Retrieve the play-by-play data
    play_by_play_df = retrieve_play_by_play(GAME_ID)
    
    # Save to CSV if data was retrieved successfully
    if play_by_play_df is not None:
        save_to_csv(play_by_play_df, GAME_ID)
    else:
        print("Failed to retrieve play-by-play data.")

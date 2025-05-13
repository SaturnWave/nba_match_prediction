import pandas as pd
import time
import os
import traceback
from nba_api.stats.endpoints import BoxScorePlayerTrackV2

def retrieve_player_tracking(game_id, output_dir=None, max_retries=5):
    """
    Retrieve player tracking data for a specified game ID with improved error handling
    
    Args:
        game_id (str): NBA Game ID
        output_dir (str, optional): Directory to save CSV files. If None, current directory is used.
        max_retries (int): Maximum number of retry attempts if API call fails
    
    Returns:
        tuple: (player_stats_df, team_stats_df) DataFrames containing player and team tracking data
    """
    # Format game ID to ensure proper format (with leading zeros)
    game_id_str = str(game_id)
    formatted_game_id = f"00{game_id_str}" if not game_id_str.startswith('00') else game_id_str
    
    print(f"Retrieving player tracking data for game ID: {formatted_game_id}")
    
    # Set up retry parameters
    retry_delays = [0.5, 1, 2, 3, 5]  # Increasing delays between retries
    retries = 0
    
    while retries <= max_retries:
        try:
            # Fetch player tracking data
            box_score_player_track_v2 = BoxScorePlayerTrackV2(game_id=formatted_game_id)
            
            # Convert response to dictionary
            box_score_player_track_v2_data = box_score_player_track_v2.get_dict()
            
            # Verify we have valid data
            if not box_score_player_track_v2_data or 'resultSets' not in box_score_player_track_v2_data:
                raise ValueError("API returned empty or invalid response")
            
            # Extract player tracking data
            player_stats = box_score_player_track_v2_data['resultSets'][0]['rowSet']
            team_stats = box_score_player_track_v2_data['resultSets'][1]['rowSet']
            
            # Get headers
            player_headers = box_score_player_track_v2_data['resultSets'][0]['headers']
            team_headers = box_score_player_track_v2_data['resultSets'][1]['headers']
            
            # Convert to DataFrames
            player_stats_df = pd.DataFrame(player_stats, columns=player_headers)
            team_stats_df = pd.DataFrame(team_stats, columns=team_headers)
            
            # Save to CSV files if output directory is specified
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                player_file = os.path.join(output_dir, f"{formatted_game_id}_player_tracking.csv")
                team_file = os.path.join(output_dir, f"{formatted_game_id}_team_tracking.csv")
                
                player_stats_df.to_csv(player_file, index=False)
                team_stats_df.to_csv(team_file, index=False)
                
                print(f"Player tracking data saved to {player_file}")
                print(f"Team tracking data saved to {team_file}")
            
            print(f"Successfully retrieved player tracking data for game {formatted_game_id}")
            return player_stats_df, team_stats_df
            
        except Exception as e:
            retries += 1
            if retries <= max_retries:
                delay = retry_delays[min(retries-1, len(retry_delays)-1)]
                print(f"Error retrieving tracking data: {e}. Retrying in {delay} seconds (Attempt {retries}/{max_retries})...")
                time.sleep(delay)
            else:
                print(f"Failed to retrieve player tracking data for game {formatted_game_id} after {max_retries} attempts: {e}")
                traceback.print_exc()
                return None, None

if __name__ == "__main__":
    # Example usage with a sample game ID
    game_id = '0022400058'  # Sample Game ID
    output_dir = './tracking_data'
    
    player_df, team_df = retrieve_player_tracking(game_id, output_dir)
    
    if player_df is not None:
        print(f"Retrieved {len(player_df)} player tracking records")
    if team_df is not None:
        print(f"Retrieved {len(team_df)} team tracking records")
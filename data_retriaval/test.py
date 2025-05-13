import pandas as pd
import os
from nba_api.stats.endpoints import BoxScoreTraditionalV2, BoxScoreAdvancedV2

# Define the base directory for all NBA data using the absolute path
BASE_DIR = r"C:\Users\arcan\Desktop\Python\nba_new\impact_scores\nba_data"

# Replace with a 2024-2025 season game ID
game_id = '0022400058'  # This is already using a 2024-2025 game ID format (the leading '00224' indicates 2024-25 season)
season = "2024_2025"

# Create directory structure if it doesn't exist
game_dir = os.path.join(BASE_DIR, season, game_id)
box_score_dir = os.path.join(game_dir, "box_scores")
os.makedirs(box_score_dir, exist_ok=True)

# Retrieve the traditional boxscore data
traditional_data = BoxScoreTraditionalV2(game_id).player_stats.get_data_frame()

# Save the traditional boxscore data to a CSV file in the proper directory
traditional_csv_filename = f'{game_id}_traditional_boxscore.csv'
traditional_filepath = os.path.join(box_score_dir, traditional_csv_filename)
traditional_data.to_csv(traditional_filepath, index=False)
print(f"Traditional boxscore data for game ID {game_id} has been saved to {traditional_filepath}")

# Retrieve the advanced boxscore data
advanced_data = BoxScoreAdvancedV2(game_id).player_stats.get_data_frame()

# Save the advanced boxscore data to a CSV file in the proper directory
advanced_csv_filename = f'{game_id}_advanced_boxscore.csv'
advanced_filepath = os.path.join(box_score_dir, advanced_csv_filename)
advanced_data.to_csv(advanced_filepath, index=False)
print(f"Advanced boxscore data for game ID {game_id} has been saved to {advanced_filepath}")

# The commented defensive boxscore code can be updated similarly
# import pandas as pd
# from nba_api.stats.endpoints import BoxScoreDefensiveV2
#
# # Retrieve the defensive data
# defensive_data = BoxScoreDefensiveV2(game_id).player_stats.get_data_frame()
#
# # Check if data was retrieved successfully
# if not defensive_data.empty:
#     # Save the data to a CSV file in the proper directory
#     defensive_csv_filename = f'{game_id}_defensive_data.csv'
#     defensive_filepath = os.path.join(box_score_dir, defensive_csv_filename)
#     defensive_data.to_csv(defensive_filepath, index=False)
#     print(f"Defensive data for game ID {game_id} has been saved to {defensive_filepath}")
# else:
#     print(f"No defensive data found for game ID {game_id}")
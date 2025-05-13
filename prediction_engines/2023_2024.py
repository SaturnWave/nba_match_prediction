import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split # Will replace with chronological
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error, mean_squared_error
import lightgbm as lgb
import pickle
from tqdm import tqdm
import warnings
from collections import defaultdict # Added for ImpactScoreCalculator

warnings.filterwarnings('ignore')

# --- GameDataLoader Class (Keep as is from your script) ---
class GameDataLoader:
    def __init__(self, data_dir="nba_data", game_ids_dir="game_ids"):
        self.data_dir = os.path.abspath(data_dir)
        self.game_ids_dir = os.path.abspath(game_ids_dir)

    def get_game_ids_for_season(self, season_str):
        filepath = os.path.join(self.game_ids_dir, f"game_id_{season_str}.csv")
        if not os.path.exists(filepath):
            print(f"Warning: Game ID file not found for season {season_str} at {filepath}")
            return pd.DataFrame()
        try:
            return pd.read_csv(filepath)
        except Exception as e:
            print(f"Error reading game ID file {filepath}: {e}")
            return pd.DataFrame()

    def load_game_data(self, game_id, season_str):
        game_id_str = str(game_id).zfill(10)
        season_game_dir = os.path.join(self.data_dir, season_str, game_id_str)
        
        if not os.path.exists(season_game_dir):
             # Try with raw game_id if formatted one not found (due to your _load_season_data logic)
            alt_game_dir = os.path.join(self.data_dir, season_str, str(game_id))
            if os.path.exists(alt_game_dir):
                season_game_dir = alt_game_dir
                game_id_str = str(game_id) # Use raw if that's the folder name
            else:
                # print(f"Warning: Game directory not found: {season_game_dir} or {alt_game_dir}")
                return None

        data = {"game_id": game_id_str, "season": season_str}
        pbp_file_name_options = [f"{game_id_str}pbp.csv", f"{str(game_id)}pbp.csv"] # Try both formats
        
        pbp_path_found = None
        for fname in pbp_file_name_options:
            pbp_path = os.path.join(season_game_dir, "play_by_play", fname)
            if os.path.exists(pbp_path):
                pbp_path_found = pbp_path
                break
        
        if pbp_path_found:
            try:
                data['pbp'] = pd.read_csv(pbp_path_found)
            except Exception as e:
                print(f"Error reading PBP for {game_id}: {e}")
                return None
        else:
            # print(f"Warning: PBP data not found for {game_id} in {season_str}")
            return None

        box_score_types = ["traditional", "advanced", "defensive", 
                           "traditional_team", "advanced_team", "defensive_team", "matchups"]
        box_scores_dir = os.path.join(season_game_dir, "box_scores")

        if os.path.exists(box_scores_dir):
            for bs_type in box_score_types:
                bs_file_name_options = [f"{game_id_str}box_score_{bs_type}.csv", f"{str(game_id)}box_score_{bs_type}.csv"]
                bs_path_found = None
                for fname in bs_file_name_options:
                    bs_path = os.path.join(box_scores_dir, fname)
                    if os.path.exists(bs_path):
                        bs_path_found = bs_path
                        break
                if bs_path_found:
                    try:
                        data[f'box_{bs_type}'] = pd.read_csv(bs_path_found)
                    except Exception as e:
                        # print(f"Warning: Error reading box score {bs_type} for {game_id}: {e}")
                        data[f'box_{bs_type}'] = None 
        return data

    def preprocess_pbp_data(self, df_pbp):
        if df_pbp is None or df_pbp.empty: return None # Changed from df_pbp to None
        df_pbp_copy = df_pbp.copy()
        df_pbp_copy['clock_seconds'] = df_pbp_copy['clock'].apply(
            lambda x: int(x.split('PT')[1].split('M')[0]) * 60 + float(x.split('M')[1].replace('S', ''))
            if isinstance(x, str) and 'PT' in x and 'M' in x and 'S' in x
            else 0.0 # Ensure float for consistency
        )
        df_pbp_copy['scoreHome'] = pd.to_numeric(df_pbp_copy['scoreHome'], errors='coerce').fillna(0).astype(float)
        df_pbp_copy['scoreAway'] = pd.to_numeric(df_pbp_copy['scoreAway'], errors='coerce').fillna(0).astype(float)
        df_pbp_copy['period'] = pd.to_numeric(df_pbp_copy['period'], errors='coerce').fillna(0).astype(int)
        df_pbp_copy['score_margin'] = abs(df_pbp_copy['scoreHome'] - df_pbp_copy['scoreAway'])
        df_pbp_copy['is_clutch'] = ((df_pbp_copy['period'] == 4) & (df_pbp_copy['clock_seconds'] <= 300)) | (df_pbp_copy['period'] > 4)
        return df_pbp_copy

# --- ImpactScoreCalculator Helper Functions (Keep these outside or as static methods) ---
def is_clutch_time(clock_seconds, period): # Already defined in NBAPredictor, ensure consistency or remove duplicate
    if pd.isna(period) or pd.isna(clock_seconds): return False
    return (int(period) == 4 and float(clock_seconds) <= 300) or int(period) > 4

def get_score_margin_for_row(row_dict): # Renamed
    home = row_dict.get('scoreHome', np.nan)
    away = row_dict.get('scoreAway', np.nan)
    if pd.notnull(home) and pd.notnull(away):
        return abs(float(home) - float(away))
    return 0.0

# --- ImpactScoreCalculator Class (Keep as is from your script, but ensure helpers are accessible) ---
class ImpactScoreCalculator: # Simplified for brevity - use your full version
    def __init__(self, game_pbp_df, game_box_traditional_df=None, game_box_advanced_df=None, 
                 game_box_defensive_df=None, game_player_tracking_df=None):
        self.df_pbp = game_pbp_df
        self.df_traditional = game_box_traditional_df
        # ... (rest of your __init__)

    def _create_player_team_map_for_game(self):
        if self.df_pbp is not None and not self.df_pbp.empty:
            pt_map = self.df_pbp[['playerName', 'teamTricode']].dropna().drop_duplicates()
            return pt_map.set_index('playerName')['teamTricode'].to_dict()
        return {}
        
    def _calculate_team_possessions_for_game(self):
        team_possessions = {}
        if self.df_traditional is not None and not self.df_traditional.empty:
            for team_abb in self.df_traditional['TEAM_ABBREVIATION'].unique():
                team_data = self.df_traditional[self.df_traditional['TEAM_ABBREVIATION'] == team_abb]
                if not team_data.empty:
                    fga = team_data['FGA'].sum()
                    fta_factor = team_data['FTA'].sum() * 0.44 
                    to = team_data['TO'].sum()
                    orb = team_data['OREB'].sum() 
                    possessions = fga - orb + to + fta_factor
                    team_possessions[team_abb] = possessions
        return team_possessions
    
    # --- Include ALL your _calculate_..._impact and _identify_scoring_run methods here ---
    # Make sure they use self.df_pbp and other game-specific dataframes (self.df_traditional, etc.)
    def _identify_scoring_run(self, current_action_num_or_idx, window=5):
        current_pos = -1
        # Ensure self.df_pbp is sorted by actionNumber if using it for lookup
        # A safer way if actionNumber can have gaps or isn't strictly sequential is to use original index
        # For now, assuming actionNumber is somewhat usable or fallback to index
        action_number_col = 'actionNumber' if 'actionNumber' in self.df_pbp.columns else self.df_pbp.index.name
        if action_number_col == 'actionNumber':
            matches = self.df_pbp[self.df_pbp['actionNumber'] == current_action_num_or_idx]
            if not matches.empty:
                current_pos = self.df_pbp.index.get_loc(matches.index[0])
        elif isinstance(current_action_num_or_idx, (int, np.integer)) and current_action_num_or_idx in self.df_pbp.index:
             current_pos = self.df_pbp.index.get_loc(current_action_num_or_idx)
        
        if current_pos == -1: return None

        start_pos = max(0, current_pos - window)
        previous_plays = self.df_pbp.iloc[start_pos:current_pos]
        if not previous_plays.empty:
            team_counts = previous_plays['teamTricode'].value_counts()
            if not team_counts.empty: return team_counts.index[0]
        return None

    def _calculate_block_impact(self, row_dict, next_play_dict, previous_plays_list): # Simplified
        base_impact = 1.2
        # scoring_run_team = self._identify_scoring_run(row_dict.get('actionNumber', row_dict.get('_original_index')))
        # if scoring_run_team and scoring_run_team != row_dict.get('teamTricode'): base_impact += 0.3
        return base_impact
    def _calculate_steal_impact(self, r,n,p): return 1.4 # Simplified
    def _calculate_rebound_impact(self, r,n,p): return 0.7 # Simplified
    def _calculate_scoring_impact(self, r,p): # Simplified
        # scoring_run_team = self._identify_scoring_run(r.get('actionNumber', r.get('_original_index')))
        # if scoring_run_team and scoring_run_team != r.get('teamTricode'): return (3.0 if r.get('shotValue')==3 else 2.0) + 0.2
        return (3.0 if r.get('shotValue')==3 else 2.0)
    def _calculate_turnover_impact(self, r,n,p): return -0.8 # Simplified
    def _calculate_foul_impact(self, r,n,p): return -0.5 # Simplified

    def _calculate_single_play_impact(self, row_dict, next_play_dict, previous_plays_list):
        impact = 0
        desc = row_dict.get('description', '')
        action_type = row_dict.get('actionType')

        if isinstance(desc, str) and 'BLOCK' in desc: impact = self._calculate_block_impact(row_dict, next_play_dict, previous_plays_list)
        elif isinstance(desc, str) and 'STEAL' in desc: impact = self._calculate_steal_impact(row_dict, next_play_dict, previous_plays_list)
        elif action_type == 'Rebound': impact = self._calculate_rebound_impact(row_dict, next_play_dict, previous_plays_list)
        elif action_type == 'Made Shot': impact = self._calculate_scoring_impact(row_dict, previous_plays_list)
        elif action_type == 'Turnover' and not (isinstance(desc, str) and 'Foul' in desc): impact = self._calculate_turnover_impact(row_dict, next_play_dict, previous_plays_list)
        elif isinstance(desc, str) and 'Foul' in desc: impact = self._calculate_foul_impact(row_dict, next_play_dict, previous_plays_list)
        
        if is_clutch_time(row_dict.get('clock_seconds'), row_dict.get('period')): impact *= 1.5
        clock_secs = row_dict.get('clock_seconds')
        if pd.notnull(clock_secs) and clock_secs > 0: impact *= (1 + (1 / (clock_secs + 1)))
        return impact

    def calculate_player_impact_scores_for_game(self): # Renamed
        if self.df_pbp is None or self.df_pbp.empty: return pd.Series(dtype=float), {}

        player_impact_dict = defaultdict(float)
        df_pbp_sorted = self.df_pbp.sort_values(by='actionNumber', ascending=True).reset_index(drop=True)
        df_pbp_sorted['_original_index'] = df_pbp_sorted.index

        for index, row_series in df_pbp_sorted.iterrows():
            row_dict = row_series.to_dict()
            if pd.isna(row_dict.get('playerName')): continue
            player = row_dict['playerName']
            next_play_series = df_pbp_sorted.iloc[index + 1] if index < len(df_pbp_sorted) - 1 else None
            next_play_dict = next_play_series.to_dict() if next_play_series is not None else None
            prev_start_idx = max(0, index - 5)
            previous_plays_list = [r.to_dict() for _, r in df_pbp_sorted.iloc[prev_start_idx:index].iterrows()]
            impact = self._calculate_single_play_impact(row_dict, next_play_dict, previous_plays_list)
            player_impact_dict[player] += impact
        
        home_team_tricode = self.df_pbp['teamTricodeHome'].iloc[0] if 'teamTricodeHome' in self.df_pbp.columns and not self.df_pbp.empty and pd.notna(self.df_pbp['teamTricodeHome'].iloc[0]) else None
        away_team_tricode = self.df_pbp['teamTricodeAway'].iloc[0] if 'teamTricodeAway' in self.df_pbp.columns and not self.df_pbp.empty and pd.notna(self.df_pbp['teamTricodeAway'].iloc[0]) else None
        home_total_impact, away_total_impact = 0.0, 0.0
        normalized_player_impact = {}

        for player, impact_val in player_impact_dict.items():
            team_abb = self.player_team_map.get(player)
            norm_impact = impact_val
            if team_abb and team_abb in self.team_possessions and self.team_possessions.get(team_abb, 0) > 0:
                norm_impact = impact_val * (100 / self.team_possessions[team_abb])
            normalized_player_impact[player] = norm_impact
            if team_abb == home_team_tricode: home_total_impact += impact_val 
            elif team_abb == away_team_tricode: away_total_impact += impact_val
        
        team_impact_scores = {'home_impact': home_total_impact, 'away_impact': away_total_impact}
        return pd.Series(normalized_player_impact).sort_values(ascending=False), team_impact_scores


# --- FeatureEngineer Class (Keep as is from your script) ---
class FeatureEngineer: # Simplified for brevity - use your full version
    def __init__(self): pass
    def _calculate_rolling_stats(self, team_games_history, stat_columns, windows): # Simplified
        team_stats = pd.DataFrame(index=team_games_history.index)
        for stat in stat_columns:
            if stat in team_games_history.columns:
                for w in windows: team_stats[f'L{w}_{stat}'] = team_games_history[stat].rolling(window=w, min_periods=1).mean().shift(1)
        if 'won' in team_games_history.columns:
            for w in windows: team_stats[f'L{w}_win_pct'] = team_games_history['won'].rolling(window=w, min_periods=1).mean().shift(1)
        return team_stats
    def _calculate_streaks(self, team_games_history): # Simplified
        if 'won' not in team_games_history.columns or team_games_history.empty: return pd.Series(0, index=team_games_history.index, name='streak')
        streaks = []
        current_streak = 0
        for won_status in team_games_history['won'].shift(1).fillna(0.5): 
            if won_status == 1: current_streak = max(1, current_streak + 1)
            elif won_status == 0: current_streak = min(-1, current_streak - 1)
            else: current_streak = 0 
            streaks.append(current_streak)
        return pd.Series(streaks, index=team_games_history.index, name='streak')

    def engineer_features_for_dataset(self, all_games_df):
        if all_games_df.empty: return pd.DataFrame()
        df = all_games_df.sort_values('game_date').copy()
        
        stats_to_roll_base = ['score', 'score_allowed', 'point_margin', 'FGM', 'FGA', 'FG_PCT', 
                              'FG3M', 'FG3A', 'FG3_PCT', 'REB', 'AST', 'TO', 'home_impact_score_agg', 'away_impact_score_agg']
        
        engineered_features_list = []
        unique_team_ids = pd.concat([df['home_team_id'], df['away_team_id']]).dropna().unique().astype(int) # Ensure int

        for team_id_float in tqdm(unique_team_ids, desc="Engineering team features", leave=False):
            team_id = int(team_id_float) # Convert to int for comparison
            is_home = (df['home_team_id'] == team_id)
            is_away = (df['away_team_id'] == team_id)
            
            team_games_list = []
            home_games_temp = df[is_home].copy()
            if not home_games_temp.empty:
                home_games_temp['team_id_context'] = team_id
                home_games_temp['won'] = home_games_temp['home_win']
                home_games_temp['score'] = home_games_temp['home_score']
                home_games_temp['score_allowed'] = home_games_temp['away_score']
                home_games_temp['point_margin'] = home_games_temp['score'] - home_games_temp['score_allowed']
                for stat_base in stats_to_roll_base:
                    if f'home_{stat_base}' in home_games_temp.columns: home_games_temp[stat_base] = home_games_temp[f'home_{stat_base}']
                    elif stat_base == 'impact_score_agg': home_games_temp[stat_base] = home_games_temp['home_impact_score_agg'] # Special handling
                team_games_list.append(home_games_temp)

            away_games_temp = df[is_away].copy()
            if not away_games_temp.empty:
                away_games_temp['team_id_context'] = team_id
                away_games_temp['won'] = 1 - away_games_temp['home_win']
                away_games_temp['score'] = away_games_temp['away_score']
                away_games_temp['score_allowed'] = away_games_temp['home_score']
                away_games_temp['point_margin'] = away_games_temp['score'] - away_games_temp['score_allowed']
                for stat_base in stats_to_roll_base:
                    if f'away_{stat_base}' in away_games_temp.columns: away_games_temp[stat_base] = away_games_temp[f'away_{stat_base}']
                    elif stat_base == 'impact_score_agg': away_games_temp[stat_base] = away_games_temp['away_impact_score_agg'] # Special handling
                team_games_list.append(away_games_temp)

            if not team_games_list: continue
            
            # Select only necessary columns before concat to avoid type issues from missing columns
            cols_to_select = ['game_id', 'game_date', 'season', 'team_id_context', 'won', 'score', 'score_allowed', 'point_margin']
            for tg_df in team_games_list:
                existing_stat_cols = [s for s in stats_to_roll_base if s in tg_df.columns]
                cols_to_select_current = cols_to_select + existing_stat_cols
                # Ensure all selected columns actually exist in tg_df
                tg_df_filtered = tg_df[[col for col in cols_to_select_current if col in tg_df.columns]]
                team_games_list[team_games_list.index(tg_df)] = tg_df_filtered # Replace with filtered

            team_all_games = pd.concat(team_games_list).sort_values('game_date').drop_duplicates(subset=['game_id'])
            
            current_team_stats_to_roll = [s for s in stats_to_roll_base if s in team_all_games.columns]
            rolling_stats_df = self._calculate_rolling_stats(team_all_games, current_team_stats_to_roll, windows=[3, 5, 10])
            streak_s = self._calculate_streaks(team_all_games)
            
            season_avg_stats = pd.DataFrame(index=team_all_games.index)
            for stat in current_team_stats_to_roll:
                 season_avg_stats[f'season_avg_{stat}'] = team_all_games.groupby('season')[stat].expanding().mean().reset_index(level=0, drop=True).shift(1)
            if 'won' in team_all_games.columns:
                season_avg_stats['season_win_pct'] = team_all_games.groupby('season')['won'].expanding().mean().reset_index(level=0, drop=True).shift(1)

            team_engineered_stats = pd.concat([rolling_stats_df, streak_s, season_avg_stats], axis=1)
            team_engineered_stats['game_id'] = team_all_games['game_id']
            team_engineered_stats['team_id_for_features'] = team_id
            engineered_features_list.append(team_engineered_stats)
        
        if not engineered_features_list: return pd.DataFrame()
        all_team_engineered_features = pd.concat(engineered_features_list).reset_index(drop=True)

        df = pd.merge(df, all_team_engineered_features.add_prefix('home_'), 
                      left_on=['game_id', 'home_team_id'], 
                      right_on=['home_game_id', 'home_team_id_for_features'], how='left')
        df = pd.merge(df, all_team_engineered_features.add_prefix('away_'), 
                      left_on=['game_id', 'away_team_id'], 
                      right_on=['away_game_id', 'away_team_id_for_features'], how='left')
        
        cols_to_drop = [col for col in df.columns if col.endswith(('_game_id', '_team_id_for_features')) and col != 'game_id']
        df = df.drop(columns=cols_to_drop, errors='ignore')

        df['h2h_L5_home_wins_count'] = 0.0 
        df['h2h_L5_avg_point_diff'] = 0.0
        
        for idx, game_row_series in tqdm(df.iterrows(), total=len(df), desc="Engineering H2H features", leave=False):
            game_row = game_row_series.to_dict()
            home_id = game_row.get('home_team_id')
            away_id = game_row.get('away_team_id')
            current_date = game_row.get('game_date')
            if pd.isna(home_id) or pd.isna(away_id) or pd.isna(current_date): continue

            past_h2h_games = df[((df['home_team_id'] == home_id) & (df['away_team_id'] == away_id) |
                                 (df['home_team_id'] == away_id) & (df['away_team_id'] == home_id)) &
                                (df['game_date'] < current_date)].sort_values('game_date', ascending=False).head(5)

            if not past_h2h_games.empty:
                home_wins_count, point_diff_sum = 0, 0.0
                for _, h2h_game_series in past_h2h_games.iterrows():
                    h2h_game = h2h_game_series.to_dict()
                    h_score, a_score = h2h_game.get('home_score',0), h2h_game.get('away_score',0)
                    if h2h_game.get('home_team_id') == home_id: 
                        if h2h_game.get('home_win'): home_wins_count +=1
                        point_diff_sum += (h_score - a_score)
                    else: 
                        if not h2h_game.get('home_win'): home_wins_count +=1
                        point_diff_sum += (a_score - h_score)
                df.loc[idx, 'h2h_L5_home_wins_count'] = home_wins_count
                df.loc[idx, 'h2h_L5_avg_point_diff'] = point_diff_sum / len(past_h2h_games) if len(past_h2h_games) > 0 else 0.0
        
        # Dynamically generate diff features based on what was created
        created_home_rolling_cols = [col for col in df.columns if col.startswith('home_L') or col.startswith('home_season_avg_')]
        
        for home_col in created_home_rolling_cols:
            base_stat_name = home_col.replace('home_L3_', '').replace('home_L5_', '').replace('home_L10_', '').replace('home_season_avg_', '')
            away_col_L3 = f'away_L3_{base_stat_name}'
            away_col_L5 = f'away_L5_{base_stat_name}'
            away_col_L10 = f'away_L10_{base_stat_name}'
            away_col_season = f'away_season_avg_{base_stat_name}'

            # Determine the correct away_col based on the prefix of home_col
            if home_col.startswith('home_L3_') and away_col_L3 in df.columns: away_col = away_col_L3
            elif home_col.startswith('home_L5_') and away_col_L5 in df.columns: away_col = away_col_L5
            elif home_col.startswith('home_L10_') and away_col_L10 in df.columns: away_col = away_col_L10
            elif home_col.startswith('home_season_avg_') and away_col_season in df.columns: away_col = away_col_season
            else: continue # Skip if corresponding away col not found

            diff_col_name = home_col.replace('home_', 'diff_')
            df[diff_col_name] = df[home_col] - df[away_col]
        
        if 'home_streak' in df.columns and 'away_streak' in df.columns:
            df['diff_streak'] = df['home_streak'] - df['away_streak']
        return df

# --- NBAPredictor Class (Main Logic - From your script, with modifications for paths and integration) ---
class NBAPredictor:
    def __init__(self, base_data_dir, game_ids_dir, model_dir="models"): # Expect absolute paths
        self.base_data_dir = base_data_dir
        self.game_ids_dir = game_ids_dir
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

        self.loader = GameDataLoader(data_dir=self.base_data_dir, game_ids_dir=self.game_ids_dir)
        self.feature_engineer = FeatureEngineer()
        
        self.all_games_master_df = None
        self.final_modeling_dataset_df = None
        
        self.models = {}
        self.feature_columns = []
        self.validation_metrics = {}
        
        self.team_id_map = {} 
        self.team_name_to_id_map = {}
        self.team_abbreviation_to_id_map = {}

    def _build_team_id_maps_from_data(self, df_with_team_info_list):
        # (Same as provided in previous refined `predictor.py`)
        if not df_with_team_info_list: return
        all_teams_info = []
        for df_item in df_with_team_info_list: # Iterate if it's a list of DFs
            df = df_item # If it's already a DF
            if isinstance(df_item, list) and df_item: # If it's a list containing a DF
                df = df_item[0]

            if df is not None and not df.empty and \
               all(col in df.columns for col in ['TEAM_ID', 'TEAM_NAME', 'TEAM_ABBREVIATION']):
                all_teams_info.append(df[['TEAM_ID', 'TEAM_NAME', 'TEAM_ABBREVIATION']])
        if not all_teams_info: return
        combined_teams_info = pd.concat(all_teams_info).drop_duplicates()
        for _, row in combined_teams_info.iterrows():
            try:
                team_id = int(row['TEAM_ID']) 
                self.team_id_map[team_id] = {'name': row['TEAM_NAME'], 'abbreviation': row['TEAM_ABBREVIATION']}
                self.team_name_to_id_map[row['TEAM_NAME']] = team_id
                self.team_abbreviation_to_id_map[row['TEAM_ABBREVIATION']] = team_id
            except ValueError: pass


    def _get_team_id(self, team_identifier):
        # (Same as provided)
        if pd.isna(team_identifier): return None
        if isinstance(team_identifier, (int, np.integer)): return team_identifier
        if isinstance(team_identifier, float) and not np.isnan(team_identifier): return int(team_identifier)
        
        team_id_str = str(team_identifier).upper()
        team_id = self.team_abbreviation_to_id_map.get(team_id_str)
        if team_id is None: team_id = self.team_name_to_id_map.get(str(team_identifier)) # Try original case for name
        return team_id
        
    def _extract_initial_game_features(self, game_id_val, game_date_val, matchup_str, pbp_data, 
                                       box_trad=None, box_adv_team=None, season_val=None):
        # (Same as provided)
        home_team_abbr, away_team_abbr = "UNK_H", "UNK_A"
        if ' vs. ' in matchup_str:
            parts = matchup_str.split(' vs. ')
            pbp_home_tricode = pbp_data['teamTricodeHome'].iloc[0] if 'teamTricodeHome' in pbp_data.columns and not pbp_data.empty and pd.notna(pbp_data['teamTricodeHome'].iloc[0]) else None
            if pbp_home_tricode: # Prioritize PBP for home team identification
                if pbp_home_tricode == parts[1]: away_team_abbr, home_team_abbr = parts[0], parts[1]
                elif pbp_home_tricode == parts[0]: away_team_abbr, home_team_abbr = parts[1], parts[0]
                else: away_team_abbr, home_team_abbr = parts[0], parts[1] # Fallback
            else: away_team_abbr, home_team_abbr = parts[0], parts[1] 
        elif ' @ ' in matchup_str: 
            parts = matchup_str.split(' @ ')
            if len(parts) == 2: away_team_abbr, home_team_abbr = parts[0], parts[1]
        else:
            parts = matchup_str.split(' ')
            if len(parts) >= 2: away_team_abbr, home_team_abbr = parts[0], parts[-1]

        home_team_id = self._get_team_id(home_team_abbr)
        away_team_id = self._get_team_id(away_team_abbr)

        final_row = pbp_data.iloc[-1] if not pbp_data.empty else pd.Series(dtype='object')
        home_score = pd.to_numeric(final_row.get('scoreHome'), errors='coerce')
        away_score = pd.to_numeric(final_row.get('scoreAway'), errors='coerce')
        home_score = 0 if pd.isna(home_score) else home_score
        away_score = 0 if pd.isna(away_score) else away_score


        features = {
            'game_id': str(game_id_val).zfill(10), 'game_date': pd.to_datetime(game_date_val),
            'season': season_val,
            'home_team': home_team_abbr, 'away_team': away_team_abbr,
            'home_team_id': home_team_id, 'away_team_id': away_team_id,
            'home_score': home_score, 'away_score': away_score,
            'point_diff': home_score - away_score,
            'home_win': 1 if home_score > away_score else 0,
            'total_score': home_score + away_score
        }
        
        for prefix, team_abbr_for_lookup, actual_team_id_for_lookup in [('home', home_team_abbr, home_team_id), ('away', away_team_abbr, away_team_id)]:
            team_bs = None
            if box_trad is not None and not box_trad.empty:
                team_bs = box_trad[box_trad['TEAM_ABBREVIATION'] == team_abbr_for_lookup]
                if team_bs.empty and actual_team_id_for_lookup is not None and 'TEAM_ID' in box_trad.columns:
                    team_bs = box_trad[box_trad['TEAM_ID'] == actual_team_id_for_lookup]
            if team_bs is not None and not team_bs.empty:
                for stat in ['FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TO', 'PF']:
                    features[f'{prefix}_{stat}'] = team_bs[stat].sum() if stat in team_bs else 0
                features[f'{prefix}_FG_PCT'] = features[f'{prefix}_FGM'] / features[f'{prefix}_FGA'] if features.get(f'{prefix}_FGA', 0) > 0 else 0
                features[f'{prefix}_FG3_PCT'] = features[f'{prefix}_FG3M'] / features[f'{prefix}_FG3A'] if features.get(f'{prefix}_FG3A', 0) > 0 else 0
                features[f'{prefix}_FT_PCT'] = features[f'{prefix}_FTM'] / features[f'{prefix}_FTA'] if features.get(f'{prefix}_FTA', 0) > 0 else 0
            
            team_adv_data = None
            if box_adv_team is not None and not box_adv_team.empty:
                team_adv_data = box_adv_team[box_adv_team['TEAM_ABBREVIATION'] == team_abbr_for_lookup]
                if team_adv_data.empty and actual_team_id_for_lookup is not None and 'TEAM_ID' in box_adv_team.columns:
                     team_adv_data = box_adv_team[box_adv_team['TEAM_ID'] == actual_team_id_for_lookup]
            if team_adv_data is not None and not team_adv_data.empty:
                for stat in ['PACE', 'OFF_RATING', 'DEF_RATING', 'TS_PCT']: # Ensure these are in your _team.csv
                     features[f'{prefix}_{stat}'] = team_adv_data[stat].iloc[0] if stat in team_adv_data.columns and not team_adv_data.empty else np.nan
        return features

    def load_and_prepare_data(self, seasons_for_context=["2022_2023", "2023_2024"]):
        print(f"Loading and preparing data for seasons: {seasons_for_context}")
        
        sample_box_trad_list = []
        sample_box_adv_team_list = [] # To also get team abbreviations from advanced team files

        for season_str_map in seasons_for_context:
            game_ids_df_map = self.loader.get_game_ids_for_season(season_str_map)
            if not game_ids_df_map.empty:
                for i in range(min(10, len(game_ids_df_map))): # Try more games for robust map
                    sample_game_id = str(game_ids_df_map.iloc[i]['GAME_ID']).zfill(10)
                    sample_game_bundle = self.loader.load_game_data(sample_game_id, season_str_map)
                    if sample_game_bundle:
                        if sample_game_bundle.get('box_traditional') is not None:
                            sample_box_trad_list.append(sample_game_bundle.get('box_traditional'))
                        if sample_game_bundle.get('box_advanced_team') is not None: # Check for team advanced
                             sample_box_adv_team_list.append(sample_game_bundle.get('box_advanced_team'))
        
        # Build maps from both traditional and advanced team box scores
        combined_map_dfs = []
        if sample_box_trad_list: combined_map_dfs.extend(sample_box_trad_list)
        if sample_box_adv_team_list: combined_map_dfs.extend(sample_box_adv_team_list)
        
        if combined_map_dfs: self._build_team_id_maps_from_data(combined_map_dfs)

        all_games_data_list = []
        for season_str in seasons_for_context:
            game_ids_df = self.loader.get_game_ids_for_season(season_str)
            if game_ids_df.empty: continue

            for _, game_meta in tqdm(game_ids_df.iterrows(), total=len(game_ids_df), desc=f"Processing {season_str}", leave=False):
                game_id_raw = str(game_meta['GAME_ID'])
                game_id_for_loader = game_id_raw.zfill(10) if len(game_id_raw) <= 8 else game_id_raw # Handle existing 10-digit
                
                game_bundle = self.loader.load_game_data(game_id_for_loader, season_str) # Use potentially formatted ID
                
                if not game_bundle or game_bundle.get('pbp') is None: continue
                
                pbp_data = self.loader.preprocess_pbp_data(game_bundle['pbp'])
                if pbp_data is None or pbp_data.empty: continue

                initial_features = self._extract_initial_game_features(
                    game_id_for_loader, game_meta['GAME_DATE'], game_meta['MATCHUP'],
                    pbp_data, game_bundle.get('box_traditional'), 
                    game_bundle.get('box_advanced_team'), # Pass TEAM advanced box score
                    season_str
                )
                
                if pd.isna(initial_features['home_team_id']) or pd.isna(initial_features['away_team_id']): continue

                calculator = ImpactScoreCalculator(
                    game_pbp_df=pbp_data, 
                    game_box_traditional_df=game_bundle.get('box_traditional'),
                    game_box_advanced_df=game_bundle.get('box_advanced'), 
                    game_box_defensive_df=game_bundle.get('box_defensive'),
                    game_player_tracking_df=game_bundle.get('player_tracking_player')
                )
                _, team_impact_scores = calculator.calculate_player_impact_scores_for_game()
                
                initial_features['home_impact_score_agg'] = team_impact_scores.get('home_impact', 0)
                initial_features['away_impact_score_agg'] = team_impact_scores.get('away_impact', 0)
                initial_features['impact_score_diff'] = initial_features['home_impact_score_agg'] - initial_features['away_impact_score_agg']
                all_games_data_list.append(initial_features)
        
        if not all_games_data_list:
            print("No games were processed successfully to build master_df.")
            self.all_games_master_df = pd.DataFrame()
            self.final_modeling_dataset_df = pd.DataFrame()
            return self

        self.all_games_master_df = pd.DataFrame(all_games_data_list)
        self.all_games_master_df['game_date'] = pd.to_datetime(self.all_games_master_df['game_date'])
        self.all_games_master_df = self.all_games_master_df.sort_values('game_date').reset_index(drop=True)
        
        print("Engineering time-aware features for all loaded games...")
        self.final_modeling_dataset_df = self.feature_engineer.engineer_features_for_dataset(
            self.all_games_master_df.copy() 
        )
        self.final_modeling_dataset_df = self.final_modeling_dataset_df.fillna(0) 

        print(f"Data preparation complete. Final modeling dataset has {self.final_modeling_dataset_df.shape[0]} games and {self.final_modeling_dataset_df.shape[1]} features.")
        return self

    def train_models(self, train_season="2022_2023", target_predict_season="2023_2024"):
        if self.final_modeling_dataset_df is None or self.final_modeling_dataset_df.empty:
            print("Run load_and_prepare_data() first.")
            return self

        targets_to_predict = ['home_win', 'point_diff', 'total_score']
        # More careful exclusion: keep IDs if used as categorical, exclude actual scores from features
        cols_to_exclude = ['game_id', 'game_date', 'home_team', 'away_team', 
                           'home_score', 'away_score', 'season'] + targets_to_predict 
                           # Potentially keep 'home_team_id', 'away_team_id' if using them as categorical features
                           # For now, let's assume they are not direct features if we use team-specific rolling stats
        
        self.feature_columns = [col for col in self.final_modeling_dataset_df.columns if col not in cols_to_exclude]
        # One final check to ensure no target-like columns accidentally slipped in
        self.feature_columns = [f for f in self.feature_columns if not any(target_part in f for target_part in ['_win', '_score', '_diff']) or f.startswith(('L','season_avg_', 'diff_L', 'diff_season_avg_', 'h2h_'))]
        self.feature_columns = [f for f in self.feature_columns if 'home_team_id' not in f and 'away_team_id' not in f] # Remove specific ID columns if not used as cat features

        if not self.feature_columns: print("No feature columns identified for training."); return self

        train_df_context_season = self.final_modeling_dataset_df[self.final_modeling_dataset_df['season'] == train_season]
        predict_season_df = self.final_modeling_dataset_df[self.final_modeling_dataset_df['season'] == target_predict_season].copy()
        
        if not predict_season_df.empty:
             predict_season_df = predict_season_df.sort_values('game_date')
             split_idx_target = int(len(predict_season_df) * 0.75) 
             train_df_target_early = predict_season_df.iloc[:split_idx_target]
             test_data = predict_season_df.iloc[split_idx_target:]
             train_data = pd.concat([train_df_context_season, train_df_target_early] if not train_df_context_season.empty else [train_df_target_early])
        elif not train_df_context_season.empty:
             print(f"No data for target season {target_predict_season}. Using {train_season} for train/test.")
             split_idx_context = int(len(train_df_context_season) * 0.8)
             train_data = train_df_context_season.iloc[:split_idx_context]
             test_data = train_df_context_season.iloc[split_idx_context:]
        else: print("No data available for training or testing."); return self
        
        if train_data.empty or test_data.empty: print("Not enough data for train/test splits."); return self

        X_train = train_data[self.feature_columns].copy()
        X_test = test_data[self.feature_columns].copy()
        
        for col in X_train.columns: # Ensure numeric types
            X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
            X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
        X_train = X_train.fillna(0); X_test = X_test.fillna(0)

        print(f"Training data shape: {X_train.shape}, Testing data shape: {X_test.shape}")

        for target_name in targets_to_predict:
            print(f"\nTraining model for {target_name}...")
            y_train = train_data[target_name].astype(float) 
            y_test = test_data[target_name].astype(float)

            model_params = {'random_state': 42, 'n_estimators': 150, 'learning_rate': 0.05, 'num_leaves': 31, 'verbose': -1, 'n_jobs': -1}

            if target_name == 'home_win':
                model = lgb.LGBMClassifier(**model_params)
                model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=20, verbose=False) # eval_metric can be 'logloss' or 'auc'
                preds, probs = model.predict(X_test), model.predict_proba(X_test)[:, 1]
                self.validation_metrics[target_name] = {'accuracy': accuracy_score(y_test, preds), 'auc': roc_auc_score(y_test, probs)}
                print(f"  Accuracy: {self.validation_metrics[target_name]['accuracy']:.4f}, AUC: {self.validation_metrics[target_name]['auc']:.4f}")
            else: 
                model = lgb.LGBMRegressor(**model_params)
                model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=20, verbose=False) # eval_metric can be 'rmse' or 'mae'
                preds = model.predict(X_test)
                self.validation_metrics[target_name] = {'mae': mean_absolute_error(y_test, preds), 'rmse': np.sqrt(mean_squared_error(y_test, preds))}
                print(f"  MAE: {self.validation_metrics[target_name]['mae']:.4f}, RMSE: {self.validation_metrics[target_name]['rmse']:.4f}")
            
            self.models[target_name] = model
            with open(os.path.join(self.model_dir, f"{target_name}_model.pkl"), 'wb') as f: pickle.dump(model, f)
            
            if hasattr(model, 'feature_importances_') and model.feature_importances_ is not None :
                feature_imp_df = pd.DataFrame({'feature': X_train.columns, 'importance': model.feature_importances_})
                feature_imp_df = feature_imp_df.sort_values('importance', ascending=False).head(20)
                if not feature_imp_df.empty:
                    plt.figure(figsize=(10,max(8, len(feature_imp_df)*0.4))) # Adjust height for num features
                    sns.barplot(x='importance', y='feature', data=feature_imp_df)
                    plt.title(f'Top 20 Feature Importances for {target_name}')
                    plt.tight_layout(); plt.savefig(os.path.join(self.model_dir, f"{target_name}_feature_importance.png")); plt.close()
        
        print("Models trained and saved.")
        return self

    def predict_specific_game(self, home_team_identifier, away_team_identifier, game_date_str):
        if not self.models: print("Models not trained."); return None
        if self.all_games_master_df is None: print("Historical game data not processed."); return None

        home_team_id = self._get_team_id(home_team_identifier)
        away_team_id = self._get_team_id(away_team_identifier)
        game_date = pd.to_datetime(game_date_str)

        if home_team_id is None or away_team_id is None:
            print(f"Could not identify teams: {home_team_identifier}, {away_team_identifier}")
            return None
        
        year, month = game_date.year, game_date.month
        predict_season_str = f"{year-1}_{year}" if month < 7 else f"{year}_{year+1}"

        game_info_for_fe = {
            'game_id': f"predict_{home_team_id}_vs_{away_team_id}_{game_date_str.replace('-', '')}",
            'game_date': game_date,
            'home_team': self.team_id_map.get(home_team_id, {}).get('abbreviation', str(home_team_identifier)),
            'away_team': self.team_id_map.get(away_team_id, {}).get('abbreviation', str(away_team_identifier)),
            'home_team_id': home_team_id, 'away_team_id': away_team_id, 'season': predict_season_str, 
            'home_score': np.nan, 'away_score': np.nan, 'home_win': np.nan, 'point_diff': np.nan, 'total_score': np.nan,
             # Add placeholders for other base stats if FeatureEngineer expects them from all_games_master_df
            'home_FGM': np.nan, 'away_FGM': np.nan, # ... and so on for all stats_to_roll_base
            'home_impact_score_agg': np.nan, 'away_impact_score_agg': np.nan # Also for impact
        }
        # Add all other base stats expected by FeatureEngineer as NaN
        stats_to_roll_base = ['score', 'score_allowed', 'point_margin', 'FGM', 'FGA', 'FG_PCT', 
                              'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT',
                              'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TO', 'PF',
                              'impact_score_agg']
        for prefix in ['home', 'away']:
            for stat_base in stats_to_roll_base:
                if f'{prefix}_{stat_base}' not in game_info_for_fe: # If not already set (like _score)
                    game_info_for_fe[f'{prefix}_{stat_base}'] = np.nan


        historical_data_for_fe = self.all_games_master_df[self.all_games_master_df['game_date'] < game_date].copy()
        if historical_data_for_fe.empty:
            print(f"No historical data found before {game_date_str} to make a prediction. Using zeros for historical features.")
            # Create a DataFrame of one row with all feature columns set to 0
            # This is a fallback, predictions will be very naive.
            X_predict_df = pd.DataFrame(0, index=[0], columns=self.feature_columns)
        else:
            temp_df_for_engineering = pd.concat([historical_data_for_fe, pd.DataFrame([game_info_for_fe])]).sort_values('game_date').reset_index(drop=True)
            engineered_df_for_prediction = self.feature_engineer.engineer_features_for_dataset(temp_df_for_engineering)
            game_features_to_predict_row = engineered_df_for_prediction[engineered_df_for_prediction['game_id'] == game_info_for_fe['game_id']]

            if game_features_to_predict_row.empty: print("Could not engineer features."); return None
            
            X_predict_df = pd.DataFrame(columns=self.feature_columns) # Ensure columns match training
            X_predict_df = pd.concat([X_predict_df, game_features_to_predict_row[self.feature_columns]], ignore_index=True)
        
        for col in X_predict_df.columns: X_predict_df[col] = pd.to_numeric(X_predict_df[col], errors='coerce')
        X_predict_df = X_predict_df.fillna(0)
        X_predict = X_predict_df[self.feature_columns] # Final reorder/selection

        predictions = {'home_team': game_info_for_fe['home_team'], 'away_team': game_info_for_fe['away_team'], 'game_date': game_date_str}
        
        for target_name, model in self.models.items():
            if X_predict.empty: predictions[target_name] = np.nan; continue
            try:
                if target_name == 'home_win':
                    pred_proba = model.predict_proba(X_predict)[:, 1]
                    predictions['home_win_prob'] = pred_proba[0]
                    predictions['predicted_winner'] = game_info_for_fe['home_team'] if pred_proba[0] > 0.5 else game_info_for_fe['away_team']
                else:
                    predictions[target_name] = model.predict(X_predict)[0]
            except Exception as e:
                print(f"Error predicting {target_name}: {e}"); predictions[target_name] = np.nan
        return predictions

if __name__ == '__main__':
    USER_PROJECT_ROOT = r"C:\Users\arcan\Desktop\Python\nba_new\impact_scores"
    PROJECT_ROOT = USER_PROJECT_ROOT
    
    BASE_DATA_DIR_ABS = os.path.join(PROJECT_ROOT, "nba_data")
    GAME_IDS_DIR_ABS = os.path.join(PROJECT_ROOT, "game_ids") # Corrected path
    MODEL_DIR_ABS = os.path.join(PROJECT_ROOT, "models")

    print(f"--- Path Configuration ---")
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"BASE_DATA_DIR_ABS: {BASE_DATA_DIR_ABS}")
    print(f"GAME_IDS_DIR_ABS: {GAME_IDS_DIR_ABS}")
    print(f"MODEL_DIR_ABS: {MODEL_DIR_ABS}")
    print(f"--- End Path Configuration ---")

    predictor = NBAPredictor(BASE_DATA_DIR_ABS, GAME_IDS_DIR_ABS, MODEL_DIR_ABS)
    
    predictor.load_and_prepare_data(seasons_for_context=["2022_2023", "2023_2024"])
    
    if predictor.final_modeling_dataset_df is not None and not predictor.final_modeling_dataset_df.empty:
        predictor.train_models(train_season="2022_2023", target_predict_season="2023_2024")
        
        print("\n--- Validation Metrics ---")
        if predictor.validation_metrics:
            for target, metrics_dict in predictor.validation_metrics.items(): # Iterate through dict
                print(f"Target: {target}")
                for metric_name, value in metrics_dict.items(): # Iterate through inner dict
                    print(f"  {metric_name}: {value:.4f}")
        else:
            print("No validation metrics to display.")

        print("\n--- Example Game Prediction ---")
        if predictor.team_abbreviation_to_id_map: 
            home_team_example = "GSW" 
            away_team_example = "LAL" 
            game_to_predict_date = "2024-04-09" 
            
            if predictor._get_team_id(home_team_example) and predictor._get_team_id(away_team_example):
                print(f"Attempting to predict: {home_team_example} vs {away_team_example} on {game_to_predict_date}")
                
                specific_game_prediction = predictor.predict_specific_game(
                    home_team_identifier=home_team_example,
                    away_team_identifier=away_team_example,
                    game_date_str=game_to_predict_date
                )
                if specific_game_prediction:
                    print("Prediction Results:")
                    for key, value in specific_game_prediction.items():
                        if isinstance(value, float): print(f"  {key}: {value:.3f}")
                        else: print(f"  {key}: {value}")
                else:
                    print(f"Could not make prediction for {home_team_example} vs {away_team_example} on {game_to_predict_date}.")
            else:
                print(f"Could not find team IDs for {home_team_example} or {away_team_example}. Check team maps.")
        else:
            print("Team ID map not populated. Cannot run example prediction without team identifiers.")
    else:
        print("Failed to prepare modeling dataset. Cannot train or predict.")

    print("\n--- Script Execution Complete ---")
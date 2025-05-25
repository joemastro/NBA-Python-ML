import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import json
import logging
from datetime import datetime, timedelta
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, teamgamelog, boxscoretraditionalv2
from nba_api.stats.library.parameters import SeasonAll, SeasonType
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler


## We scrape on game-by-game data with the intention of predicting the stats of the next game
## (obv, but AI usually starts with seasonal data -- so this should be specified)
## With how this is setup we are only scraping the current 2024-2025 data 
## -> crtl+F 2024 and update it for when next year's season rolls around 
## I had this bias bc for ex: Luka was on the Mavs last season, but is now on the Lakers
## We do not have it setup to scrape for playoff data or all-star data, 
## figuring out how to include playoffs next would be nice, but does not seem super necessary
## We do not need any all-star data -> games are exhibitionist and not serious



# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("nba_scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NBADataScraper:
    def __init__(self, season="2024-25", season_type=SeasonType.regular):
        """
        Initialize the NBA Data Scraper.
        
        Args:
            season (str): Season to scrape data for (e.g., "2024-25")
            season_type: Type of season (regular, playoffs, etc.)
        """
        self.season = season
        self.season_type = season_type
        
        # Create directories for storing data if they don't exist
        self.data_dir = "nba_data"
        self.player_data_dir = os.path.join(self.data_dir, "players")
        self.team_data_dir = os.path.join(self.data_dir, "teams")
        self.boxscore_data_dir = os.path.join(self.data_dir, "boxscores")
        
        for directory in [self.data_dir, self.player_data_dir, self.team_data_dir, self.boxscore_data_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                logger.info(f"Created directory: {directory}")
        
        # Try to get all teams data
        try:
            self.teams_data = teams.get_teams()
            logger.info(f"Successfully loaded data for {len(self.teams_data)} teams")
        except Exception as e:
            self.teams_data = []
            logger.error(f"Failed to load teams data: {str(e)}")
            
        # Map team ID to team abbr for easier reference
        self.team_id_to_abbr = {}
        for team in self.teams_data:
            self.team_id_to_abbr[team['id']] = team['abbreviation']
            
        # Track requests to avoid hitting rate limits
        self.last_request_time = datetime.now()
        self.request_delay = 1  # seconds between requests
            
    def _rate_limit_request(self):
        """Apply rate limiting to avoid getting blocked by the API."""
        current_time = datetime.now()
        elapsed = (current_time - self.last_request_time).total_seconds()
        
        if elapsed < self.request_delay:
            sleep_time = self.request_delay - elapsed
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
            
        self.last_request_time = datetime.now()
        
    def get_player_id_by_name(self, player_name):
        """
        Get a player's ID by their name.
        
        Args:
            player_name (str): Full or partial name of the player
            
        Returns:
            int: Player ID if found, None otherwise
        """
        try:
            # Search for player by name
            player_matches = players.find_players_by_full_name(player_name)
            
            if not player_matches:
                # Try partial name search if full name search fails
                player_matches = players.find_players_by_first_name(player_name.split()[0])
                if not player_matches and len(player_name.split()) > 1:
                    player_matches = players.find_players_by_last_name(player_name.split()[-1])
            
            if player_matches:
                if len(player_matches) > 1:
                    logger.warning(f"Multiple players found for '{player_name}': {[p['full_name'] for p in player_matches]}")
                    logger.warning(f"Using the first match: {player_matches[0]['full_name']}")
                
                player_id = player_matches[0]['id']
                logger.info(f"Found player ID {player_id} for '{player_name}'")
                return player_id
            else:
                logger.error(f"No player found with name '{player_name}'")
                return None
                
        except Exception as e:
            logger.error(f"Error finding player ID for '{player_name}': {str(e)}")
            return None
            
    def get_team_id_by_name(self, team_name):
        """
        Get a team's ID by their name.
        
        Args:
            team_name (str): Full or partial name of the team
            
        Returns:
            int: Team ID if found, None otherwise
        """
        try:
            # Search for team by full name or abbreviation
            team_matches = teams.find_teams_by_full_name(team_name)
            
            if not team_matches:
                # Try abbreviation search
                team_matches = teams.find_teams_by_abbreviation(team_name)
                
            if not team_matches:
                # Try nickname search
                team_matches = teams.find_teams_by_nickname(team_name)
                
            if team_matches:
                team_id = team_matches[0]['id']
                logger.info(f"Found team ID {team_id} for '{team_name}'")
                return team_id
            else:
                logger.error(f"No team found with name '{team_name}'")
                return None
                
        except Exception as e:
            logger.error(f"Error finding team ID for '{team_name}': {str(e)}")
            return None
            
    def get_player_game_log(self, player_id, save=True):
        """
        Get game-by-game statistics for a player.
        
        Args:
            player_id (int): The player's ID
            save (bool): Whether to save the data to a CSV file
            
        Returns:
            pandas.DataFrame: DataFrame containing player's game log
        """
        try:
            logger.info(f"Fetching game log for player ID {player_id} for season {self.season}")
            self._rate_limit_request()
            
            # Get the player game log
            player_game_log = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=self.season,
                season_type_all_star=self.season_type
            )
            
            # Convert to DataFrame
            df = player_game_log.get_data_frames()[0]
            
            if df.empty:
                logger.warning(f"No game data found for player ID {player_id} in {self.season} season")
                return pd.DataFrame()
                
            # Add a player_id column for reference
            df['PLAYER_ID'] = player_id
            
            # Get player name for reference
            try:
                player_info = players.find_player_by_id(player_id)
                player_name = player_info['full_name'] if player_info else f"Unknown_{player_id}"
            except:
                player_name = f"Unknown_{player_id}"
                
            # Calculate additional stats that might be useful for ML
            df['DOUBLE_DOUBLE'] = ((df['PTS'] >= 10).astype(int) + 
                                  (df['REB'] >= 10).astype(int) + 
                                  (df['AST'] >= 10).astype(int) + 
                                  (df['STL'] >= 10).astype(int) + 
                                  (df['BLK'] >= 10).astype(int) >= 2).astype(int)
            
            df['TRIPLE_DOUBLE'] = ((df['PTS'] >= 10).astype(int) + 
                                  (df['REB'] >= 10).astype(int) + 
                                  (df['AST'] >= 10).astype(int) + 
                                  (df['STL'] >= 10).astype(int) + 
                                  (df['BLK'] >= 10).astype(int) >= 3).astype(int)
            
            # Calculate shooting percentages where not already provided
            if 'FG_PCT' not in df.columns and 'FGM' in df.columns and 'FGA' in df.columns:
                df['FG_PCT'] = df['FGM'] / df['FGA']
                
            if 'FG3_PCT' not in df.columns and 'FG3M' in df.columns and 'FG3A' in df.columns:
                df['FG3_PCT'] = df['FG3M'] / df['FG3A']
                
            # Save to CSV if requested
            if save:
                file_path = os.path.join(self.player_data_dir, f"{player_name.replace(' ', '_')}_games.csv")
                df.to_csv(file_path, index=False)
                logger.info(f"Saved player game log to {file_path}")
                
            logger.info(f"Successfully fetched {len(df)} games for player ID {player_id}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching game log for player ID {player_id}: {str(e)}")
            return pd.DataFrame()
            
    def get_team_game_log(self, team_id, save=True):
        """
        Get game-by-game statistics for a team.
        
        Args:
            team_id (int): The team's ID
            save (bool): Whether to save the data to a CSV file
            
        Returns:
            pandas.DataFrame: DataFrame containing team's game log
        """
        try:
            logger.info(f"Fetching game log for team ID {team_id} for season {self.season}")
            self._rate_limit_request()
            
            # Get the team game log
            team_game_log = teamgamelog.TeamGameLog(
                team_id=team_id,
                season=self.season,
                season_type_all_star=self.season_type
            )
            
            # Convert to DataFrame
            df = team_game_log.get_data_frames()[0]
            
            if df.empty:
                logger.warning(f"No game data found for team ID {team_id} in {self.season} season")
                return pd.DataFrame()
            
            # Add a team_id column for reference
            df['TEAM_ID'] = team_id
            
            # Get team abbreviation for reference
            team_abbr = self.team_id_to_abbr.get(team_id, f"Unknown_{team_id}")
                
            # Save to CSV if requested
            if save:
                file_path = os.path.join(self.team_data_dir, f"{team_abbr}_games.csv")
                df.to_csv(file_path, index=False)
                logger.info(f"Saved team game log to {file_path}")
                
            logger.info(f"Successfully fetched {len(df)} games for team ID {team_id}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching game log for team ID {team_id}: {str(e)}")
            return pd.DataFrame()
            
    def get_box_score(self, game_id, save=True):
        """
        Get the box score for a specific game.
        
        Args:
            game_id (str): The ID of the game
            save (bool): Whether to save the data to a CSV file
            
        Returns:
            dict: Dictionary containing player and team box scores
        """
        try:
            logger.info(f"Fetching box score for game ID {game_id}")
            self._rate_limit_request()
            
            # Get the box score
            box_score = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
            
            # Get data frames
            player_stats = box_score.player_stats.get_data_frame()
            team_stats = box_score.team_stats.get_data_frame()
            
            if player_stats.empty or team_stats.empty:
                logger.warning(f"No box score data found for game ID {game_id}")
                return {'player_stats': pd.DataFrame(), 'team_stats': pd.DataFrame()}
                
            # Save to CSV if requested
            if save:
                player_file_path = os.path.join(self.boxscore_data_dir, f"game_{game_id}_player_stats.csv")
                team_file_path = os.path.join(self.boxscore_data_dir, f"game_{game_id}_team_stats.csv")
                
                player_stats.to_csv(player_file_path, index=False)
                team_stats.to_csv(team_file_path, index=False)
                
                logger.info(f"Saved player box score to {player_file_path}")
                logger.info(f"Saved team box score to {team_file_path}")
                
            logger.info(f"Successfully fetched box score for game ID {game_id}")
            return {'player_stats': player_stats, 'team_stats': team_stats}
            
        except Exception as e:
            logger.error(f"Error fetching box score for game ID {game_id}: {str(e)}")
            return {'player_stats': pd.DataFrame(), 'team_stats': pd.DataFrame()}
            
    def get_recent_games_for_all_teams(self, days_back=30):
        """
        Get recent games for all teams.
        
        Args:
            days_back (int): Number of days to look back for games
            
        Returns:
            dict: Dictionary mapping team IDs to their game data
        """
        result = {}
        
        for team in self.teams_data:
            team_id = team['id']
            team_name = team['full_name']
            
            logger.info(f"Fetching recent games for {team_name} (ID: {team_id})")
            
            try:
                df = self.get_team_game_log(team_id)
                if not df.empty:
                    result[team_id] = df
                
                # Avoid hitting rate limits
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error fetching games for {team_name}: {str(e)}")
                
        return result
    
    def visualize_player_comparison(self, player1_name, player2_name, stat_column):
        """
        Create a visualization comparing two players' statistics.
        
        Args:
            player1_name (str): Name of the first player
            player2_name (str): Name of the second player
            stat_column (str): The statistic to compare (e.g., 'PTS', 'AST')
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        try:
            # Get player IDs
            player1_id = self.get_player_id_by_name(player1_name)
            player2_id = self.get_player_id_by_name(player2_name)
            
            if not player1_id or not player2_id:
                if not player1_id:
                    logger.error(f"Could not find player ID for {player1_name}")
                if not player2_id:
                    logger.error(f"Could not find player ID for {player2_name}")
                return None
                
            # Get player data
            player1_data = self.get_player_game_log(player1_id)
            player2_data = self.get_player_game_log(player2_id)
            
            if player1_data.empty or player2_data.empty:
                if player1_data.empty:
                    logger.error(f"No data found for {player1_name}")
                if player2_data.empty:
                    logger.error(f"No data found for {player2_name}")
                return None
                
            # Check if the stat column exists
            if stat_column not in player1_data.columns or stat_column not in player2_data.columns:
                logger.error(f"Statistic '{stat_column}' not found in player data")
                available_stats = list(set(player1_data.columns) & set(player2_data.columns))
                logger.info(f"Available statistics: {available_stats}")
                return None
                
            # Create figure and axis
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Sort by date
            player1_data = player1_data.sort_values('GAME_DATE')
            player2_data = player2_data.sort_values('GAME_DATE')
            
            # Plot data
            ax.plot(range(len(player1_data)), player1_data[stat_column], 'b-', label=player1_name)
            ax.plot(range(len(player2_data)), player2_data[stat_column], 'r-', label=player2_name)
            
            # Add rolling average (last 5 games)
            window = min(5, len(player1_data), len(player2_data))
            if window > 1:
                player1_rolling = player1_data[stat_column].rolling(window=window).mean()
                player2_rolling = player2_data[stat_column].rolling(window=window).mean()
                
                ax.plot(range(len(player1_data)), player1_rolling, 'b--', alpha=0.7, 
                        label=f"{player1_name} (5-game avg)")
                ax.plot(range(len(player2_data)), player2_rolling, 'r--', alpha=0.7,
                        label=f"{player2_name} (5-game avg)")
            
            # Add labels and title
            ax.set_xlabel('Game Number')
            ax.set_ylabel(stat_column)
            ax.set_title(f'{stat_column} Comparison: {player1_name} vs {player2_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add season average line
            player1_avg = player1_data[stat_column].mean()
            player2_avg = player2_data[stat_column].mean()
            
            ax.axhline(y=player1_avg, color='b', linestyle=':', alpha=0.5,
                       label=f"{player1_name} Avg: {player1_avg:.1f}")
            ax.axhline(y=player2_avg, color='r', linestyle=':', alpha=0.5,
                       label=f"{player2_name} Avg: {player2_avg:.1f}")
            
            ax.legend()
            
            # Save figure
            output_dir = os.path.join(self.data_dir, "visualizations")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            file_path = os.path.join(output_dir, f"{player1_name.replace(' ', '_')}_vs_{player2_name.replace(' ', '_')}_{stat_column}.png")
            plt.savefig(file_path)
            logger.info(f"Saved visualization to {file_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            return None
            
    def visualize_team_comparison(self, team1_name, team2_name, stat_column):
        """
        Create a visualization comparing two teams' statistics.
        
        Args:
            team1_name (str): Name of the first team
            team2_name (str): Name of the second team
            stat_column (str): The statistic to compare (e.g., 'PTS', 'AST', 'REB')
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        try:
            # Get team IDs
            team1_id = self.get_team_id_by_name(team1_name)
            team2_id = self.get_team_id_by_name(team2_name)
            
            if not team1_id or not team2_id:
                if not team1_id:
                    logger.error(f"Could not find team ID for {team1_name}")
                if not team2_id:
                    logger.error(f"Could not find team ID for {team2_name}")
                return None
                
            # Get team data
            team1_data = self.get_team_game_log(team1_id)
            team2_data = self.get_team_game_log(team2_id)
            
            if team1_data.empty or team2_data.empty:
                if team1_data.empty:
                    logger.error(f"No data found for {team1_name}")
                if team2_data.empty:
                    logger.error(f"No data found for {team2_name}")
                return None
                
            # Check if the stat column exists
            if stat_column not in team1_data.columns or stat_column not in team2_data.columns:
                logger.error(f"Statistic '{stat_column}' not found in team data")
                available_stats = list(set(team1_data.columns) & set(team2_data.columns))
                logger.info(f"Available statistics: {available_stats}")
                return None
                
            # Create figure and axis
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Sort by date
            team1_data = team1_data.sort_values('GAME_DATE')
            team2_data = team2_data.sort_values('GAME_DATE')
            
            # Plot data
            ax.plot(range(len(team1_data)), team1_data[stat_column], 'b-', label=team1_name)
            ax.plot(range(len(team2_data)), team2_data[stat_column], 'r-', label=team2_name)
            
            # Add rolling average (last 5 games)
            window = min(5, len(team1_data), len(team2_data))
            if window > 1:
                team1_rolling = team1_data[stat_column].rolling(window=window).mean()
                team2_rolling = team2_data[stat_column].rolling(window=window).mean()
                
                ax.plot(range(len(team1_data)), team1_rolling, 'b--', alpha=0.7, 
                        label=f"{team1_name} (5-game avg)")
                ax.plot(range(len(team2_data)), team2_rolling, 'r--', alpha=0.7,
                        label=f"{team2_name} (5-game avg)")
            
            # Add labels and title
            ax.set_xlabel('Game Number')
            ax.set_ylabel(stat_column)
            ax.set_title(f'Team {stat_column} Comparison: {team1_name} vs {team2_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add season average line
            team1_avg = team1_data[stat_column].mean()
            team2_avg = team2_data[stat_column].mean()
            
            ax.axhline(y=team1_avg, color='b', linestyle=':', alpha=0.5,
                       label=f"{team1_name} Avg: {team1_avg:.1f}")
            ax.axhline(y=team2_avg, color='r', linestyle=':', alpha=0.5,
                       label=f"{team2_name} Avg: {team2_avg:.1f}")
            
            ax.legend()
            
            # Save figure
            output_dir = os.path.join(self.data_dir, "visualizations")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            file_path = os.path.join(output_dir, f"{team1_name.replace(' ', '_')}_vs_{team2_name.replace(' ', '_')}_{stat_column}.png")
            plt.savefig(file_path)
            logger.info(f"Saved visualization to {file_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            return None
            
    def spot_check_game(self, game_id):
        """
        Perform a spot check on a specific game to verify data accuracy.
        
        Args:
            game_id (str): The ID of the game to check
            
        Returns:
            bool: True if spot check passes, False otherwise
        """
        try:
            logger.info(f"Performing spot check on game ID {game_id}")
            
            # Get box score for the game
            box_score = self.get_box_score(game_id, save=False)
            
            if not box_score['player_stats'].empty and not box_score['team_stats'].empty:
                player_stats = box_score['player_stats']
                team_stats = box_score['team_stats']
                
                # Check if the sum of player points equals team points
                home_team_id = team_stats['TEAM_ID'].iloc[0]
                away_team_id = team_stats['TEAM_ID'].iloc[1]
                
                home_team_pts = team_stats.loc[team_stats['TEAM_ID'] == home_team_id, 'PTS'].values[0]
                away_team_pts = team_stats.loc[team_stats['TEAM_ID'] == away_team_id, 'PTS'].values[0]
                
                home_player_pts = player_stats.loc[player_stats['TEAM_ID'] == home_team_id, 'PTS'].sum()
                away_player_pts = player_stats.loc[player_stats['TEAM_ID'] == away_team_id, 'PTS'].sum()
                
                # Check if they match
                home_check = abs(home_team_pts - home_player_pts) < 0.1
                away_check = abs(away_team_pts - away_player_pts) < 0.1
                
                if home_check and away_check:
                    logger.info(f"Spot check passed for game ID {game_id}")
                    return True
                else:
                    logger.warning(f"Spot check failed for game ID {game_id}")
                    logger.warning(f"Home team: reported {home_team_pts}, calculated {home_player_pts}")
                    logger.warning(f"Away team: reported {away_team_pts}, calculated {away_player_pts}")
                    return False
            else:
                logger.warning(f"No data available for spot check on game ID {game_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error during spot check for game ID {game_id}: {str(e)}")
            return False
        
    def predict_next_game_points(self, player_name, visualize=True):
        """
        Simple XGBoost example to predict a player's next game points.
        
        Args:
            player_name (str): Name of the player to predict for
            visualize (bool): Whether to show feature importance plot
            
        Returns:
            dict: Prediction results and model metrics
        """
        try:
            # Get player data
            player_id = self.get_player_id_by_name(player_name)
            if not player_id:
                return {"error": f"Player {player_name} not found"}
                
            df = self.get_player_game_log(player_id, save=False)
            if df.empty:
                return {"error": f"No data found for {player_name}"}
                
            # Sort by date and prepare data
            df = df.sort_values('GAME_DATE')
            
            # Create features (using simple rolling averages)
            features = [
                'PTS', 'REB', 'AST', 'FG_PCT', 'MIN', 
                'FGA', 'FG3A', 'FTA', 'FGM', 'FG3M', 'FTM',
                'STL', 'PLUS_MINUS'
            ]
            
            # Create lagged features (previous game stats)
            for feature in features:
                df[f'prev_{feature}'] = df[feature].shift(1)
                
            # Create rolling averages (last 3 games)
            for feature in features:
                df[f'rolling3_{feature}'] = df[feature].rolling(3).mean().shift(1)
                
            # Target is next game's points
            df['target'] = df['PTS'].shift(-1)
            
            # Remove rows with missing values
            df = df.dropna()
            
            if len(df) < 10:
                return {"error": "Not enough data to build model"}
                
            # Split data
            X = df[[col for col in df.columns if col.startswith('prev_') or col.startswith('rolling3_')]]
            y = df['target']
            
            # Sequential split based on time order (data is already sorted by GAME_DATE)
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train simple XGBoost model
            model = XGBRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=42
            )
            
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            train_preds = model.predict(X_train_scaled)
            test_preds = model.predict(X_test_scaled)
            
            # Calculate metrics
            train_mae = mean_absolute_error(y_train, train_preds)
            test_mae = mean_absolute_error(y_test, test_preds)
            
            # Get feature importance
            importance = model.feature_importances_
            feat_importance = dict(zip(X.columns, importance))
            
            # Visualize feature importance
            if visualize:
                plt.figure(figsize=(10, 6))
                plt.barh(list(feat_importance.keys()), list(feat_importance.values()))
                plt.title(f'Feature Importance for {player_name} Points Prediction')
                plt.xlabel('Importance Score')
                plt.tight_layout()
                
                output_dir = os.path.join(self.data_dir, "visualizations")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                    
                file_path = os.path.join(output_dir, f"{player_name.replace(' ', '_')}_feature_importance.png")
                plt.savefig(file_path)
                plt.close()
            
            # Prepare next game prediction
            last_game = X.iloc[-1:].values
            last_game_scaled = scaler.transform(last_game)
            next_game_pred = model.predict(last_game_scaled)[0]
            
            return {
                "player": player_name,
                "train_mae": round(train_mae, 2),
                "test_mae": round(test_mae, 2),
                "next_game_prediction": round(next_game_pred, 1),
                "last_5_games_actual": df['PTS'].tail(5).tolist(),
                "feature_importance": feat_importance
            }
            
        except Exception as e:
            logger.error(f"Error in predict_next_game_points: {str(e)}")
            return {"error": str(e)}

        

# Example usage function
def demo_usage():
    """
    Demonstrate how to use the NBADataScraper class.
    """
    try:
        # Initialize the scraper
        scraper = NBADataScraper(season="2024-25")
        logger.info("NBA Data Scraper initialized successfully")
        
        # Get data for a specific player (e.g., Luka Doncic)
        player_name = "Luka Doncic"
        player_id = scraper.get_player_id_by_name(player_name)
        
        if player_id:
            logger.info(f"Getting game log for {player_name}")
            player_data = scraper.get_player_game_log(player_id)
            
            if not player_data.empty:
                logger.info(f"Successfully retrieved {len(player_data)} games for {player_name}")
                logger.info(f"Last 5 games:")
                logger.info(player_data.head().to_string())
                
                # Example visualization
                another_player = "Stephen Curry"
                logger.info(f"Creating visualization comparing {player_name} and {another_player}")
                scraper.visualize_player_comparison(player_name, another_player, "PTS")
            else:
                logger.error(f"Failed to retrieve game data for {player_name}")
                
        # Get data for a specific team (e.g., Los Angeles Lakers)
        team_name = "Los Angeles Lakers"
        team_id = scraper.get_team_id_by_name(team_name)
        
        if team_id:
            logger.info(f"Getting game log for {team_name}")
            team_data = scraper.get_team_game_log(team_id)
            
            if not team_data.empty:
                logger.info(f"Successfully retrieved {len(team_data)} games for {team_name}")
                logger.info(f"Last 5 games:")
                logger.info(team_data.head().to_string())
                
                # Example team visualization
                another_team = "Oklahoma City Thunder"
                logger.info(f"Creating visualization comparing {team_name} and {another_team}")
                scraper.visualize_team_comparison(team_name, another_team, "PTS")
            else:
                logger.error(f"Failed to retrieve game data for {team_name}")
                
            # If we have game IDs, we can spot check a game
            if not team_data.empty and 'GAME_ID' in team_data.columns:
                game_id = team_data['GAME_ID'].iloc[0]
                logger.info(f"Performing spot check on game ID {game_id}")
                spot_check_result = scraper.spot_check_game(game_id)
                logger.info(f"Spot check {'passed' if spot_check_result else 'failed'}")
                
        logger.info("Demo completed successfully")
        
    except Exception as e:
        logger.error(f"Error in demo: {str(e)}")


## To compare different players, just update  player_name and another_player
##  with their full proper names and same goes for team_name and another_team
## i.e. put in the full proper name ex: Karl-Anthony Towns is the proper name or Los Angeles Lakers
## Luka Doncic's name or names with special characters have been normalized afaik, 
## so entering Luka Doncic, Nikola Jokic, Bogdan Bovanivic(idk how to spell his name) as the name should work

if __name__ == "__main__":
    logger.info("Starting NBA Data Scraper")
    
    # Initialize scraper once
    scraper = NBADataScraper(season="2024-25")
    
    # Example 1: Basic scraping demo
    #demo_usage()
    
    # Example 2: XGBoost prediction
    try:
        player_to_predict = "Luka Doncic" 
        
        logger.info(f"\nRunning XGBoost prediction for {player_to_predict}")
        prediction_result = scraper.predict_next_game_points(player_to_predict)
        
        if "error" not in prediction_result:
            logger.info(f"\nPrediction Results for {player_to_predict}:")
            logger.info(f"- Train MAE: {prediction_result['train_mae']}")
            logger.info(f"- Test MAE: {prediction_result['test_mae']}")
            logger.info(f"- Next game points prediction: {prediction_result['next_game_prediction']}")
            logger.info(f"- Last 5 games actual points: {prediction_result['last_5_games_actual']}")
            logger.info("\nTop 5 important features:")
            for feat, imp in sorted(prediction_result['feature_importance'].items(), key=lambda x: -x[1])[:5]:
                logger.info(f"{feat}: {imp:.3f}")
        else:
            logger.error(f"Prediction failed: {prediction_result['error']}")
    except Exception as e:
        logger.error(f"Error in prediction demo: {str(e)}")
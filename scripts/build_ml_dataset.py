
"""
The purpose of this script is to create feature engineered variables (e.g. per-game stats)
to be used in the ML model. These variables are saved in the mlfeatures table.
"""

### SETUP ###

import pandas as pd
import numpy as np
from copy import deepcopy
from pathlib import Path
from scripts.helper import create_database_connection, execute_query

PATH_DB = Path('data/raw/nhl.db')
PATH_DATA_PROCESSED = Path('data/processed')
PATH_QUERIES = Path('queries')
WEIGHT_PREV_SEASON = 10 # consider previous season to be equivalent to this many games
SEASONS = [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020] ## seasons to train model on

## FUNCTIONS ##

def weight_season_stats(df, previous, current, games_played, weight_previous_season):
       """
       Computes a weighted average for a statistic including the previous year as equivalent
       to weight_previous_season games from the current year.

       :param df: dataframe including game-by-game data
       :param previous: column containing previous year's stat
       :param current: column containing this year's stat
       :param games_played: column containing number of games played so far this year
       :param weight_previous_season: weight of previous season
       :return: weighted stat
       """

       return (weight_previous_season * df[previous] + df[current]) / (df[games_played] + weight_previous_season)

### DATA PROCESSING ###

conn, cursor = create_database_connection(PATH_DB)

df_all_seasons = pd.DataFrame({})
for season in SEASONS:

       print(season)

       ## query the data for the season
       df_season = pd.read_sql(f"SELECT * FROM boxscore_processed WHERE season = {season}", conn)
       df_season_prev = pd.read_sql(f"SELECT * FROM boxscore_processed WHERE season = {season-1}", conn)

       ### determine whether this season has any new teams
       teams_all = df_season['home_franchise_id'].unique()
       teams_prev = df_season_prev['home_franchise_id'].unique()
       teams_new = teams_all[[team not in teams_prev for team in teams_all]]

       ### get the previous season's stats for each team
       df_season_previous = pd.DataFrame({})
       for team in teams_prev:

              query_team = f'SELECT * FROM boxscore_processed_team WHERE season = {season - 1} AND franchise_id = {team}'
              df_team = pd.read_sql(query_team, conn)
              n_games_played = np.max(df_team['games_played_after'])

              goals_for_per_game = df_team.loc[n_games_played-1, ['goals_for_after']].values[0] / n_games_played
              goals_against_per_game = df_team.loc[n_games_played-1, ['goals_against_after']].values[0] / n_games_played
              shots_for_per_game = df_team.loc[n_games_played-1, ['shots_for_after']].values[0] / n_games_played
              shots_against_per_game = df_team.loc[n_games_played-1, ['shots_against_after']].values[0] / n_games_played
              wins_per_game = df_team.loc[n_games_played-1, ['wins_after']].values[0] / n_games_played


              df_team_previous = pd.DataFrame({'franchise_id' : [team],
                                               'previous_goals_for_per_game' : [goals_for_per_game],
                                               'previous_goals_against_per_game' : [goals_against_per_game],
                                               'previous_shots_for_per_game' : [shots_for_per_game],
                                               'previous_shots_against_per_game' : [shots_against_per_game],
                                               'previous_wins_per_game' : [wins_per_game]})
              df_season_previous = pd.concat([df_season_previous, df_team_previous])


       ### impute the mean for teams that didn't exist last season
       means = df_season_previous.mean()
       df_missing = pd.DataFrame(columns=means.index)
       for i, missing_team in enumerate(teams_new):
              df_missing.loc[i] = means.values
              df_missing['franchise_id'][i] = missing_team
       df_season_previous = pd.concat([df_season_previous, df_missing])

       ### add previous season info to this season's dataset
       df_season = pd.merge(df_season, df_season_previous, left_on = 'home_franchise_id', right_on = 'franchise_id')
       df_season = pd.merge(df_season, df_season_previous, left_on = 'away_franchise_id', right_on = 'franchise_id', suffixes = ('_home','_away'))


       ## set up the start of the dataset
       df = deepcopy(df_season[['game_id', 'game_type', 'season', 'datetime', 'away_franchise_id', 'home_franchise_id']])
       df['home_win'] = df_season['winner'] == 'home'  ## y-variable to be predicted

       ## ADD FEATURES
       ## wins per game
       df['wins_per_game_home'] = weight_season_stats(df_season, 'previous_wins_per_game_home', 'home_season_wins', 'home_season_games_played', WEIGHT_PREV_SEASON)
       df['wins_per_game_away'] = weight_season_stats(df_season, 'previous_wins_per_game_away', 'away_season_wins', 'away_season_games_played', WEIGHT_PREV_SEASON)

       ## goals for per game
       df['goals_for_per_game_home'] = weight_season_stats(df_season, 'previous_goals_for_per_game_home', 'home_season_goals_for', 'home_season_games_played', WEIGHT_PREV_SEASON)
       df['goals_for_per_game_away'] = weight_season_stats(df_season, 'previous_goals_for_per_game_away', 'away_season_goals_for', 'away_season_games_played', WEIGHT_PREV_SEASON)

       ## goals against per game
       df['goals_against_per_game_home'] = weight_season_stats(df_season, 'previous_goals_against_per_game_home', 'home_season_goals_against', 'home_season_games_played', WEIGHT_PREV_SEASON)
       df['goals_against_per_game_away'] = weight_season_stats(df_season, 'previous_goals_against_per_game_away', 'away_season_goals_against', 'away_season_games_played', WEIGHT_PREV_SEASON)

       ## shots for per game
       df['shots_for_per_game_home'] = weight_season_stats(df_season, 'previous_shots_for_per_game_home', 'home_season_shots_for', 'home_season_games_played', WEIGHT_PREV_SEASON)
       df['shots_for_per_game_away'] = weight_season_stats(df_season, 'previous_shots_for_per_game_away', 'away_season_shots_against', 'away_season_games_played', WEIGHT_PREV_SEASON)

       ## shots against per game
       df['shots_against_per_game_home'] = weight_season_stats(df_season, 'previous_shots_against_per_game_home', 'home_season_shots_against', 'home_season_games_played', WEIGHT_PREV_SEASON)
       df['shots_against_per_game_away'] = weight_season_stats(df_season, 'previous_shots_against_per_game_away', 'away_season_shots_for', 'away_season_games_played', WEIGHT_PREV_SEASON)

       ## create the table if it doesn't exist
       execute_query(PATH_QUERIES / 'create_table_mlfeatures', cursor)

       ## query the table in order to skip any existing game
       df_mlfeatures = pd.read_sql_query(f'SELECT * from mlfeatures WHERE season = {season}', conn)
       for row in df.iterrows():
              if row[1]['game_id'] not in df_mlfeatures['game_id'].values:
                     game_dict = dict(row[1])
                     execute_query(PATH_QUERIES / 'insert_or_ignore_entry', cursor,
                                   values=tuple(game_dict.values()),
                                   replacements={'xtablex' : 'mlfeatures',
                                                 'xkeysx': ', '.join(list(game_dict.keys())),
                                                 'xvaluesx': ', '.join(['?'] * len(game_dict.keys()))})
       conn.commit()




### SAVE TRAIN SET TO CSV ###

TRAINING_SEASONS = [2011, 2012, 2013, 2014, 2015, 2016] ## seasons to train model on

season_string = ', '.join([str(season) for season in TRAINING_SEASONS])
df_all_seasons = pd.read_sql_query(f'SELECT * from mlfeatures WHERE season IN ({season_string})', conn)


## save
df_all_seasons.to_csv(PATH_DATA_PROCESSED/'dataset_2021_03_01_wins_only.csv', index = False)






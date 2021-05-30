

## SETUP ##

import pickle
from requests import get
import sqlite3
from pathlib import Path
import pandas as pd; pd.options.display.max_columns = None
import numpy as np
from time import sleep
from copy import deepcopy
from datetime import datetime, date

from scripts.helper import create_database_connection#convert_team_stats
from scripts.download_game_data import add_boxscore_to_db
from scripts.process_feed_data import process_season_feed_data

PATH_DB = Path('../data/raw/nhl.db')
PATH_MODEL = Path('../models/2021_04_20_logreg_win_percentage_only.pickle')

## FUNCTIONS ##

def extract_dates_games(today):

    ## including this to look up franchise id for each team
    teams_url = 'https://statsapi.web.nhl.com/api/v1/teams'
    teams = get(teams_url)
    teams = teams.json()

    ## create the url for the given date
    if isinstance(today, date):
        today = datetime.strftime(today, '%Y-%m-%d')
    today_url = f'https://statsapi.web.nhl.com/api/v1/schedule?date={today}'

    ## read the page
    data = get(today_url)
    data = data.json()

    ## re-organize the game info into a dataframe
    df_games = pd.DataFrame({})
    if data['totalGames'] > 0:
        games = data['dates'][0]['games']
        for game in games:
            df_new = pd.DataFrame({'game_id': [game['gamePk']],
                                   'home_id': [team['franchise']['franchiseId'] for team in teams['teams'] if team['id'] == game['teams']['home']['team']['id']],
                                   'away_id': [team['franchise']['franchiseId'] for team in teams['teams'] if team['id'] == game['teams']['away']['team']['id']],
                                   'home_team': [game['teams']['home']['team']['name']],
                                   'away_team': [game['teams']['away']['team']['name']]})
            df_games = pd.concat([df_games, df_new])

    return df_games


## SCRIPT ##

### create a variation of the download_game_data script that only checks games not already in db
## and run that here for this season
## then run process_feed_data and build_ml_dataset for this season

WEIGHT_PREV_SEASON = 10

## find today's games
today = date.today()
df_games = extract_dates_games(today)

## determine which games have already been completed and make sure they are downloaded
## using the latest id because the earliest id could be a re-scheduled game
latest_game_id = df_games['game_id'].max()
latest_game_code = int(str(latest_game_id)[-4:])
season = int(str(latest_game_id)[:4])

## load in this season's data
conn, cursor = create_database_connection(PATH_DB)

# df_features = pd.read_sql_query(f'SELECT * from mlfeatures WHERE season = {season}', conn)
## prob: mlfeatures doesn't have the after-match numbers
## also, it would be good to confirm that I'm not leaking current game stats into these

## add these cols in
df_games['goals_for_per_game_home'] = np.nan
df_games['goals_against_per_game_home'] = np.nan
df_games['shots_for_per_game_home'] = np.nan
df_games['shots_against_per_game_home'] = np.nan
df_games['wins_per_game_home'] = np.nan

df_games['goals_for_per_game_away'] = np.nan
df_games['goals_against_per_game_away'] = np.nan
df_games['shots_for_per_game_away'] = np.nan
df_games['shots_against_per_game_away'] = np.nan
df_games['wins_per_game_away'] = np.nan
for franchise_id in df_games['away_id']:

    ## need to change this to a function that is also used to build the model
    df_team = pd.read_sql(f"SELECT * FROM boxscore_processed_team WHERE season = {season} AND franchise_id = {franchise_id}", conn)
    df_team_prev = pd.read_sql(f"SELECT * FROM boxscore_processed_team WHERE season = {season-1} AND franchise_id = {franchise_id}", conn)

    n_games_played_prev = df_team_prev['games_played_after'].max()
    goals_for_per_game_prev = df_team_prev.loc[n_games_played_prev - 1, ['goals_for_after']].values[0] / n_games_played_prev
    goals_against_per_game_prev = df_team_prev.loc[n_games_played_prev - 1, ['goals_against_after']].values[0] / n_games_played_prev
    shots_for_per_game_prev = df_team_prev.loc[n_games_played_prev - 1, ['shots_for_after']].values[0] / n_games_played_prev
    shots_against_per_game_prev = df_team_prev.loc[n_games_played_prev - 1, ['shots_against_after']].values[0] / n_games_played_prev
    wins_per_game_prev = df_team_prev.loc[n_games_played_prev - 1, ['wins_after']].values[0] / n_games_played_prev

    n_games_played = df_team['games_played_after'].max()
    goals_for_per_game = df_team.loc[n_games_played - 1, ['goals_for_after']].values[0] / n_games_played
    goals_against_per_game = df_team.loc[n_games_played - 1, ['goals_against_after']].values[0] / n_games_played
    shots_for_per_game = df_team.loc[n_games_played - 1, ['shots_for_after']].values[0] / n_games_played
    shots_against_per_game = df_team.loc[n_games_played - 1, ['shots_against_after']].values[0] / n_games_played
    wins_per_game = df_team.loc[n_games_played - 1, ['wins_after']].values[0] / n_games_played

    goals_for_per_game = (goals_for_per_game * n_games_played + WEIGHT_PREV_SEASON * goals_for_per_game_prev) / (WEIGHT_PREV_SEASON + n_games_played)
    goals_against_per_game = (goals_against_per_game * n_games_played + WEIGHT_PREV_SEASON * goals_against_per_game_prev) / (WEIGHT_PREV_SEASON + n_games_played)
    shots_for_per_game = (shots_for_per_game * n_games_played + WEIGHT_PREV_SEASON * shots_for_per_game_prev) / (WEIGHT_PREV_SEASON + n_games_played)
    shots_against_per_game = (shots_against_per_game * n_games_played + WEIGHT_PREV_SEASON * shots_against_per_game_prev) / (WEIGHT_PREV_SEASON + n_games_played)
    wins_per_game = (wins_per_game * n_games_played + WEIGHT_PREV_SEASON * wins_per_game_prev) / (WEIGHT_PREV_SEASON + n_games_played)

    df_games.loc[df_games['away_id'] == franchise_id,'goals_for_per_game_away'] = goals_for_per_game
    df_games.loc[df_games['away_id'] == franchise_id, 'goals_against_per_game_away'] = goals_against_per_game
    df_games.loc[df_games['away_id'] == franchise_id, 'shots_for_per_game_away'] = shots_for_per_game
    df_games.loc[df_games['away_id'] == franchise_id, 'shots_against_per_game_away'] = shots_against_per_game
    df_games.loc[df_games['away_id'] == franchise_id, 'wins_per_game_away'] = wins_per_game


for franchise_id in df_games['home_id']:

    ## need to change this to a function that is also used to build the model
    df_team = pd.read_sql(f"SELECT * FROM boxscore_processed_team WHERE season = {season} AND franchise_id = {franchise_id}", conn)
    df_team_prev = pd.read_sql(f"SELECT * FROM boxscore_processed_team WHERE season = {season-1} AND franchise_id = {franchise_id}", conn)

    n_games_played_prev = df_team_prev['games_played_after'].max()
    goals_for_per_game_prev = df_team_prev.loc[n_games_played_prev - 1, ['goals_for_after']].values[0] / n_games_played_prev
    goals_against_per_game_prev = df_team_prev.loc[n_games_played_prev - 1, ['goals_against_after']].values[0] / n_games_played_prev
    shots_for_per_game_prev = df_team_prev.loc[n_games_played_prev - 1, ['shots_for_after']].values[0] / n_games_played_prev
    shots_against_per_game_prev = df_team_prev.loc[n_games_played_prev - 1, ['shots_against_after']].values[0] / n_games_played_prev
    wins_per_game_prev = df_team_prev.loc[n_games_played_prev - 1, ['wins_after']].values[0] / n_games_played_prev

    n_games_played = df_team['games_played_after'].max()
    goals_for_per_game = df_team.loc[n_games_played - 1, ['goals_for_after']].values[0] / n_games_played
    goals_against_per_game = df_team.loc[n_games_played - 1, ['goals_against_after']].values[0] / n_games_played
    shots_for_per_game = df_team.loc[n_games_played - 1, ['shots_for_after']].values[0] / n_games_played
    shots_against_per_game = df_team.loc[n_games_played - 1, ['shots_against_after']].values[0] / n_games_played
    wins_per_game = df_team.loc[n_games_played - 1, ['wins_after']].values[0] / n_games_played

    goals_for_per_game = (goals_for_per_game * n_games_played + WEIGHT_PREV_SEASON * goals_for_per_game_prev) / (WEIGHT_PREV_SEASON + n_games_played)
    goals_against_per_game = (goals_against_per_game * n_games_played + WEIGHT_PREV_SEASON * goals_against_per_game_prev) / (WEIGHT_PREV_SEASON + n_games_played)
    shots_for_per_game = (shots_for_per_game * n_games_played + WEIGHT_PREV_SEASON * shots_for_per_game_prev) / (WEIGHT_PREV_SEASON + n_games_played)
    shots_against_per_game = (shots_against_per_game * n_games_played + WEIGHT_PREV_SEASON * shots_against_per_game_prev) / (WEIGHT_PREV_SEASON + n_games_played)
    wins_per_game = (wins_per_game * n_games_played + WEIGHT_PREV_SEASON * wins_per_game_prev) / (WEIGHT_PREV_SEASON + n_games_played)

    df_games.loc[df_games['home_id'] == franchise_id,'goals_for_per_game_home'] = goals_for_per_game
    df_games.loc[df_games['home_id'] == franchise_id, 'goals_against_per_game_home'] = goals_against_per_game
    df_games.loc[df_games['home_id'] == franchise_id, 'shots_for_per_game_home'] = shots_for_per_game
    df_games.loc[df_games['home_id'] == franchise_id, 'shots_against_per_game_home'] = shots_against_per_game
    df_games.loc[df_games['home_id'] == franchise_id, 'wins_per_game_home'] = wins_per_game


model = pickle.load(open(PATH_MODEL, 'rb'))

away_probs = model.predict_proba(df_games[['wins_per_game_home','wins_per_game_away']])[:,0] # home, away
home_probs = model.predict_proba(df_games[['wins_per_game_home','wins_per_game_away']])[:,1] # home, away
df = deepcopy(df_games[['game_id', 'home_id', 'away_id', 'home_team', 'away_team']])
df['home_prob'] = np.round(100 * home_probs, 1)
df['away_prob'] = np.round(100 * away_probs, 1)

#
# ## df with each of these for home and away
#
# # new end
# # old start
# if __name__ == "__main__":
#
#     ## find today's games
#     today = date.today()
#     df_games = extract_dates_games(today)
#
#     ## determine which games have already been completed and make sure they are downloaded
#     ## using the latest id because the earliest id could be a re-scheduled game
#     latest_game_id = df_games['game_id'].max()
#     latest_game_code = int(str(latest_game_id)[-4:])
#     season = int(str(latest_game_id)[:4])
#
#     ## load in this season's data
#     conn = sqlite3.connect(str(PATH_DB))
#     cursor = conn.cursor()
#     df_previous_games = pd.read_sql_query(f'SELECT * from boxscore WHERE season = {season}', conn)
#
#     ## add in any missing games ---
#     ######### instead of this, just rerun the download_game_Data, process_feed_data, and build_ml_dataset scripts
#     ### then get stats from build_ml_dataset for the latest games? need to .. hmm
#     if str(latest_game_id)[5] == '2': ## ie. regular season
#         i = 0
#         while i < latest_game_code:
#             i += 1
#             id = str(int(latest_game_id) - latest_game_code + i)
#             if id not in df_previous_games['game_id'].values:
#                 add_boxscore_to_db(season, 2, id[-4:], conn, cursor); sleep(1)
#
#     if str(latest_game_id)[5] == '3': ## ie. playoffs, check all reg season games
#         i = 0
#         while i < (82 * 32 / 2): ## presumed number of games in a season
#             i += 1
#             id = str(int(latest_game_id) - latest_game_code + i - 10000) # 10k is to adjust from p/o to reg season
#             print(id, id not in df_previous_games['game_id'].values)
#             if id not in df_previous_games['game_id'].values:
#                 add_boxscore_to_db(season, 2, id[-4:], conn, cursor); sleep(1)
#
#
#
#
#
# #### build dataset to input
#
# conn, cursor = create_database_connection(PATH_DB)
# process_season_feed_data(season, conn, cursor) ## process the season in case it isn't already processed
# df_season = pd.read_sql_query(f'SELECT * from boxscore_processed WHERE season = {season}', conn) ## the processed season is in this table


# this has the cumulative stats but still need to do the feature engineering







### just stealing from process_feed_data.py for now....
## so refactor process_feed_data before this..
# process_season_feed_data(season, conn, cursor)

# conn = sqlite3.connect(str(PATH_DB))
# df_season = pd.read_sql_query(f'SELECT * from boxscore WHERE season = {current_season}', conn)
# conn.close()

#
# teams = list(df_games['home_id'].values) + list(df_games['away_id'].values)
#
# stats = {'team_id' : teams, 'win_percentage': []}
# for team in teams:
#
#     df_team = df_season[(df_season['home_id'] == team)|(df_season['away_id'] == team)]
#     df_team = df_team.sort_values('game_id').reset_index(drop = True) #### this needs to be sorted by date, not id (see 2020)
#
#     ## get the game stats for the given team
#     home_bool = df_team['home_id'] == team; away_bool = df_team['away_id'] == team
#     df_team_results = deepcopy(df_team[['game_id']])
#     df_team_results.loc[home_bool, 'win'] = df_team.loc[home_bool, 'winner'] == 'home'
#     df_team_results.loc[away_bool, 'win'] = df_team.loc[away_bool, 'winner'] == 'away'
#
#     vars = ['shots', 'goals', 'pim', 'shots', 'pp_goals', 'pp_attempts', 'fo_percent', 'blocks', 'takeaways', 'giveaways', 'hits']
#     for var in vars:
#         df_team_results = convert_team_stats(var, df_team, df_team_results, homeaway_bools = (home_bool, away_bool))
#
#     ## compute cumulative wins, losses, and games played
#     df_team_results['games_played_before'] = np.arange(df_team_results.shape[0])
#     df_team_results['games_played_after'] = df_team_results['games_played_before'] + 1
#
#     df_team_results['wins_after'] = df_team_results['win'].cumsum()
#     df_team_results['wins_before'] = df_team_results['wins_after'] - df_team_results['win']
#
#     df_team_results['losses_before'] = df_team_results['games_played_before'] - df_team_results['wins_before']
#     df_team_results['losses_after'] = df_team_results['games_played_after'] - df_team_results['wins_after']
#
#     df_team_results['team_id'] = team
#
#     wins = df_team_results['wins_after'].values[-1]
#     gp = df_team_results['games_played_after'].values[-1]
#
#     weight_prev = 10
#
#     ## previous season
#     df = pd.read_csv(f'data/processed/temp_{current_season - 1}_season_team{team}.csv')
#     wins_per_game_previous = int(df.loc[np.max(df['games_played_before']), ['wins_after']].values[0]) / df.loc[np.max(df['games_played_before']), ['games_played_after']].values[0]
#
#     ## compute the stats
#
#     stats['win_percentage'] += [(weight_prev * wins_per_game_previous + wins) / (weight_prev + gp)] #### need to take into account last year's record here!
#
# df_stats = pd.DataFrame(stats)
#
# df_games = pd.merge(df_games, df_stats, left_on = 'home_id', right_on = 'team_id')
# df_games = pd.merge(df_games, df_stats, left_on = 'away_id', right_on = 'team_id', suffixes = ('_home', '_away'))


#### run prediction
lr = pickle.load(open('models/2021_02_16_baseline_winperc_logreg.pickle', 'rb'))
X = df_games[['win_percentage_home', 'win_percentage_away']]

predictions = lr.predict_proba(X)

results = deepcopy(df_games[['game_id','home_team']]) ## just putting the preductions into a readable format for me
results['home_id'] = df_games['home_id']
results['home_prob'] = np.round(100 * predictions[:,1])
results['away_prob'] = np.round(100 * predictions[:,0])
results['away_id'] = df_games['away_id']
results['away_team'] = df_games['away_team'] ## note: these "probs" may not be well-calibrated

results.to_csv('app/predictions.csv', index = False)
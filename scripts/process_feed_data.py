
"""
This script processes the downloaded raw live feed data for a given season into a format
more suitable for building ML models (e.g. computes cumulative stats for each team
over the season).
"""

## SETUP ##

from pathlib import Path
import pandas as pd
import numpy as np
from copy import deepcopy
import re

from scripts.helper import create_database_connection, execute_query

PATH_DB = Path('data/raw/nhl.db')
PATH_QUERIES = Path('queries')

SEASONS_TO_PROCESS = np.arange(2010, 2021)
STATS_TO_PROCESS = ['shots', 'goals', 'pim', 'pp_goals', 'pp_attempts', 'fo_percent', 'blocks', 'takeaways', 'giveaways', 'hits']

## FUNCTIONS ##

def compute_cumulative_stats(stat, df_games, home_bool = None, team_id = None):

    """
    Given a dataframe containing each of a team's games, determines a team's stat for and against for each game,
    as well as cumulatively across all games.

    :param stat: Name of the statistic to be cumulated. df_games should have columns named "home_{stat}" and "away_{stat}"
    :param df_games: Dataframe containing raw feed data for each of the team's games
    :param home_bools: optional array or series of booleans indicating the games for which the team played at home
    :param team_id: optional int with the team's franchise id
    :return: returns a dataframe containing the teams stats for and against for each game as well as cumulatively
    """

    ## set up a df to add data to
    df_team_results = deepcopy(df_games[['game_id']])

    ## determine when the team is home /away
    if home_bool is None:
        if team_id is None:
            raise ValueError('Need to input either homeaway_bools or team_id')
        else:
            home_bool = df_games['home_franchise_id'] == team_id
            away_bool = ~home_bool
    else:
        away_bool = ~home_bool

    ## add in the stats FOR the given team
    df_team_results[f'{stat}_for'] = np.nan
    df_team_results.loc[home_bool, f'{stat}_for'] = df_games.loc[home_bool, f'home_{stat}']
    df_team_results.loc[away_bool, f'{stat}_for'] = df_games.loc[away_bool, f'away_{stat}']

    ## add in the stats AGAINST the given team
    df_team_results[f'{stat}_against'] = np.nan
    df_team_results.loc[home_bool, f'{stat}_against'] = df_games.loc[home_bool, f'away_{stat}']
    df_team_results.loc[away_bool, f'{stat}_against'] = df_games.loc[away_bool, f'home_{stat}']

    ## compute seasons stats as of end of each game
    df_team_results[f'{stat}_for_after'] = df_team_results[f'{stat}_for'].cumsum()
    df_team_results[f'{stat}_against_after'] = df_team_results[f'{stat}_against'].cumsum()

    ## compute seasons stats as of start of each game
    df_team_results[f'{stat}_for_before'] = df_team_results[f'{stat}_for_after'] - df_team_results[f'{stat}_for']
    df_team_results[f'{stat}_against_before'] = df_team_results[f'{stat}_against_after'] - df_team_results[f'{stat}_against']

    return df_team_results

def process_season_feed_data(season, conn, cursor):

    """
    Processes raw live feed data (from db conn input) for a given season into a format
    more suitable for building ML models, and saves back to the db.

    :param season: year in which the season started
    :param conn:
    :param cursor:
    :return: no return
    """

    query_str = f"SELECT * from boxscore WHERE season = {season} AND game_type IN (2,3)" ## ignore pre-season (game_type = 1)
    df_season = pd.read_sql_query(query_str, conn)
    all_teams = df_season['home_franchise_id'].unique()

    for team in all_teams:

        ## get all of team's games, sorted by date
        df_team = deepcopy(df_season[(df_season['home_franchise_id'] == team)|(df_season['away_franchise_id'] == team)])
        df_team = df_team.sort_values('datetime').reset_index(drop = True)

        ## get the game stats for the given team
        home_bool = df_team['home_franchise_id'] == team
        away_bool = ~home_bool

        ## set up a df to which to add the team's results
        df_team_results = deepcopy(df_team[['game_id', 'datetime', 'season']])
        df_team_results['franchise_id'] = team

        ## cumulative count of games played before / after each game
        df_team_results['games_played_before'] = np.arange(df_team_results.shape[0])
        df_team_results['games_played_after'] = df_team_results['games_played_before'] + 1

        ## check whether the team won each game
        df_team_results.loc[home_bool, 'win'] = df_team.loc[home_bool, 'winner'] == 'home'
        df_team_results.loc[away_bool, 'win'] = df_team.loc[away_bool, 'winner'] == 'away'
        df_team_results['win'] = df_team_results['win'].astype('int')

        ## cumulative count of wins before / after each game
        df_team_results['wins_after'] = df_team_results['win'].cumsum()
        df_team_results['wins_before'] = df_team_results['wins_after'] - df_team_results['win']

        ## cumulative count of losses before / after each game
        df_team_results['losses_before'] = df_team_results['games_played_before'] - df_team_results['wins_before']
        df_team_results['losses_after'] = df_team_results['games_played_after'] - df_team_results['wins_after']

        ## add in each of the other stats
        for stat in STATS_TO_PROCESS:
            df_team_next_result = compute_cumulative_stats(stat, df_team, home_bool = home_bool)
            df_team_results = pd.merge(df_team_results, df_team_next_result, on = 'game_id')

        ## add processed team data to db
        execute_query(PATH_QUERIES/'create_table_boxscore_processed_team', cursor)
        execute_query(PATH_QUERIES/'insert_or_ignore_entry', cursor,
                      replacements = {'xtablex' : 'boxscore_processed_team',
                                      'xkeysx' : ', '.join(list(df_team_results.iloc[0].to_dict().keys())),
                                      'xvaluesx' : ', '.join([str(tuple(row[1].to_dict().values())) for row in df_team_results.iterrows()])})
        conn.commit()

        ## add season stats (as of start of each game) back to df_season
        stats_to_add = [re.sub('_before','',stat) for stat in df_team_results.columns if '_before' in stat]
        for stat in stats_to_add:
            if f'home_season_{stat}' not in df_season.columns:
                df_season[f'home_season_{stat}'] = np.nan
            if f'away_season_{stat}' not in df_season.columns:
                df_season[f'away_season_{stat}'] = np.nan
            df_season.loc[df_season['game_id'].isin(df_team.loc[home_bool, 'game_id']),f'home_season_{stat}'] = df_team_results.loc[home_bool, f'{stat}_before'].values
            df_season.loc[df_season['game_id'].isin(df_team.loc[away_bool, 'game_id']),f'away_season_{stat}'] = df_team_results.loc[away_bool, f'{stat}_before'].values

    ## add processed season data to db
    execute_query(PATH_QUERIES / 'create_table_boxscore_processed', cursor)
    execute_query(PATH_QUERIES/'insert_or_ignore_entry', cursor,
                  replacements = {'xtablex' : 'boxscore_processed',
                                  'xkeysx' : ', '.join(list(df_season.iloc[0].to_dict().keys())),
                                  'xvaluesx' : ', '.join([str(tuple(row[1].to_dict().values())) for row in df_season.iterrows()])})
    conn.commit()

## SCRIPT ##

if __name__ == "__main__":
    conn, cursor = create_database_connection(PATH_DB)
    for season in SEASONS_TO_PROCESS:
        process_season_feed_data(season, conn, cursor)


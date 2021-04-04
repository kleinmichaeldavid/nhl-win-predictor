
"""
This script downloads boxscore data for all regular season and playoff games
from 2010 to 2019, saving the data in a SQLite3 db located at PATH_DB.
"""

## SETUP ##

from requests import get
from pathlib import Path
import re
import sqlite3
import pandas as pd
from time import sleep

PATH_DB = Path('data/raw/nhl.db')
PATH_QUERIES = Path('queries')


## FUNCTIONS ##

def create_database_connection(path):

    """
    Given a path to a sqlite3 db, returns a conn and cursor to that db.
    """

    ## connect to the database
    conn = sqlite3.connect(str(path))
    cursor = conn.cursor()

    return conn, cursor

def execute_query(query_path, cursor, values = None, replacements = None):

    """
    Executes a query script.

    :param query_path: path to a text file containing a sql query
    :param cursor: cursor for db
    :param values: values used for inserts
    :param replacements: dict with keys to be replaced by values in the query string
    :return: no output
    """

    ## read in the query string
    f = open(query_path)
    query = f.read()
    f.close()

    ## clean up the query
    query = re.sub('\n', ' ', query)
    query = re.sub('\t', ' ', query)

    ## add in any variables if necessary
    if replacements is not None:
        for replacement in replacements.keys():
            query = re.sub(replacement, replacements[replacement], query)

    ## execute the query, passing in values if they exist
    if values is None:
        cursor.execute(query)
    else:
        cursor.execute(query, values)

def generate_game_url(season, season_segment, game_number):

    """
    Generates the url for a particular nhl game. The url is a concatenation
    of the year, season type, and game number.

    Inputs:
    :param season: the year in which the season started
    :param season_segment: 1 = preseason, 2 = regular season, 3 = playoffs
    :param game_number:
    :return: tuple containing the game code (i.e. segment of the url coding for the particular game) and the url
    """

    ## the game code is just a concatenation of the year (in which the season started)
    ## the season segment (zero padded to len 2, 0 = preseason, 1 = regular, 2 = playoffs),
    ## and the game number (zero padded to len 4)
    game_code = str(season) + str(season_segment).zfill(2) + str(game_number).zfill(4)
    game_url = f"https://statsapi.web.nhl.com/api/v1/game/{game_code}/feed/live"

    return game_code, game_url

def download_page(url):

    """
    Downloads a webpage given a url and returns a json object.

    :param url: url for the page to be downloaded
    :return: json object containing the page's data
    """

    data = get(url)
    data = data.json()

    return data

def determine_game_winner(dict_live):

    """
    Given a dict containing data from a live feed, determines the game winner.
    This is done by checking which team scored more goals, or, if there was
    a shootout, which team scored more goals in the shootout.

    :param dict_live: dict containing live feed data
    :return: string 'home' or 'away'
    """

    if dict_live['liveData']['linescore']['hasShootout']:
        shootout = 1
        away_goals = dict_live['liveData']['linescore']['shootoutInfo']['away']['scores']
        home_goals = dict_live['liveData']['linescore']['shootoutInfo']['home']['scores']
    else:
        shootout = 0
        away_goals = dict_live['liveData']['boxscore']['teams']['away']['teamStats']['teamSkaterStats']['goals']
        home_goals = dict_live['liveData']['boxscore']['teams']['home']['teamStats']['teamSkaterStats']['goals']

    winner = 'home' if home_goals > away_goals else 'away'

    return winner

def extract_boxscore_data(dict_live):

    """
    Extracts the relevant subset of live feed data from a live feed dict.

    :param dict_live: dict containing live feed data
    :return: dict containing desired subset of data
    """

    ## components that will be used for multiple data points
    home_skater_stats = dict_live['liveData']['boxscore']['teams']['home']['teamStats']['teamSkaterStats']
    away_skater_stats = dict_live['liveData']['boxscore']['teams']['away']['teamStats']['teamSkaterStats']
    team_data = dict_live['gameData']['teams']

    dict_data = {
        'game_id': str(dict_live['gamePk']),
        'game_type': int(str(dict_live['gamePk'])[4:6]),
        'season': int(str(dict_live['gamePk'])[:4]),
        'datetime': dict_live['gameData']['datetime']['dateTime'],
        'venue_name': dict_live['gameData']['venue']['name'],
        'venue_link': dict_live['gameData']['venue']['link'],
        'away_id': team_data['away']['id'],
        'away_franchise_id': team_data['away']['franchiseId'],
        'away_name': team_data['away']['name'],
        'home_id': team_data['home']['id'],
        'home_franchise_id': team_data['home']['franchiseId'],
        'home_name': team_data['home']['name'],
        'winner': determine_game_winner(dict_live),
        'shootout': int(dict_live['liveData']['linescore']['hasShootout']),
        'away_goals': away_skater_stats['goals'],
        'away_pim': away_skater_stats['pim'],
        'away_shots': away_skater_stats['shots'],
        'away_pp_goals': away_skater_stats['powerPlayGoals'],
        'away_pp_attempts': away_skater_stats['powerPlayOpportunities'],
        'away_fo_percent': away_skater_stats['faceOffWinPercentage'],
        'away_blocks': away_skater_stats['blocked'],
        'away_takeaways': away_skater_stats['takeaways'],
        'away_giveaways': away_skater_stats['giveaways'],
        'away_hits': away_skater_stats['hits'],
        'home_goals': home_skater_stats['goals'],
        'home_pim': home_skater_stats['pim'],
        'home_shots': home_skater_stats['shots'],
        'home_pp_goals': home_skater_stats['powerPlayGoals'],
        'home_pp_attempts': home_skater_stats['powerPlayOpportunities'],
        'home_fo_percent': home_skater_stats['faceOffWinPercentage'],
        'home_blocks': home_skater_stats['blocked'],
        'home_takeaways': home_skater_stats['takeaways'],
        'home_giveaways': home_skater_stats['giveaways'],
        'home_hits': home_skater_stats['hits']
    }

    return dict_data

def add_boxscore_to_db(season, season_segment, game_number, conn, cursor):

    """

    :param season: start year for a particular season
    :param season_segment: 1 = preseason, 2 = regular season, 3 = playoffs
    :param game_number:  number representing a particular game
    :param conn:  conn for the db
    :param cursor: cursor for the db
    :return: returns 1 if the game has been completed and 0 otherwise
    """

    game_code, game_url = generate_game_url(season, season_segment, game_number)
    data = download_page(game_url)

    if 'message' in data:
        print(f'Game {game_code} does not exist.')
        return 0 ## this occurs when the game can't be found (e.g. non-existent game code)

    elif data['gameData']['status']['detailedState'] != 'Final':
        print(f'Game {game_code} has not yet been completed.')
        return 0 ## this occurs when the game exists but hasn't yet been completed

    elif (pd.read_sql("SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'boxscore'", conn).shape[0] > 0):
        if (pd.read_sql_query(f'SELECT * from boxscore WHERE game_id = {game_code}', conn).shape[0] > 0):
            print(f'Game {game_code} already exists in the database.')
            return 1 ## this occurs when the game already exists in the database

        else: ## in this case we actually add the game
            dict_data = extract_boxscore_data(data)

            ## add data
            execute_query(PATH_QUERIES / 'create_table_boxscore', cursor)
            execute_query(PATH_QUERIES/'insert_entry_boxscore', cursor,
                          values = tuple(dict_data.values()),
                          replacements = {'xkeysx' : ', '.join(list(dict_data.keys())), 'xvaluesx' : ', '.join(['?'] * len(dict_data.keys()))})
            conn.commit()

            return 1


## SCRIPT ##

if __name__ == "__main__":

    conn, cursor = create_database_connection(PATH_DB)

    print('downloading regular season game data...')
    for season in [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]:
        print(season)
        cont = True
        game = 0
        while cont:
            sleep(1)
            game += 1
            cont = add_boxscore_to_db(season, 2, game, conn, cursor) ## True if the game exists / is complete, False otherwise

    print('downloading post-season game data...')
    for season in [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]:
        print(season)
        for round in [1,2,3,4]:
            series = 0
            cont = True
            while cont:
                series += 1
                game_number = 0
                while cont:
                    sleep(1)
                    game_number += 1
                    game_num = '0' + str(round) + str(series) + str(game_number) ## game id more complex for playoffs than reg seson
                    cont = add_boxscore_to_db(season, 3, game_num, conn, cursor)
                    if cont: print(season, game_num)
                if game_number > 1:
                    cont = True

    conn.close()



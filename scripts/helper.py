
import sqlite3

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
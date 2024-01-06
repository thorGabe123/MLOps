from typing import Iterator

import click
import sqlite3
import pandas as pd
import numpy as np
from pandas import DataFrame


def get_data_from_database(database_dir: str, table: str):
    """
    Function that extracts data from database

    Args:
        database_dir (string): directory to the database containing the raw data
        table (string): The table to extract data from in the database

    Returns:
        body_text (Numpy): Numpy containing the body text from the selected table in the database
    """

    conn = sqlite3.connect(database_dir)
    query = f'select body_text from {table}'

    body_text = pd.read_sql_query(query, conn)
    body_text = body_text.to_numpy()

    conn.close()

    return body_text


if __name__ == '__main__':
    # Get the data and process it
    body_text = get_data_from_database("data/raw/wikibooks.sqlite", "en")
    np.save("data/raw/body_text.npy", body_text)
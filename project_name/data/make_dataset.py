import click
import sqlite3
import pandas as pd


def get_data_from_database(database_dir: str, table: str):
    """
    Function that extract data from database

    Args:
        database_dir (string): directory to the database containing the raw data
        table (string): The table to extract data from in the database

    Returns:
        table (DataFrame): DataFrame containing data from the selected table in the database
    """

    conn = sqlite3.connect(database_dir)
    query = f'SELECT * FROM {table}'

    table = pd.read_sql_query(query, conn)
    conn.close()

    return table


if __name__ == '__main__':
    # Get the data and process it
    table = get_data_from_database("data/raw/wikibooks.sqlite", "en")
    table.to_csv('data/processed/data.csv', index=False)
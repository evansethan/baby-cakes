import sqlite3
import pandas as pd


def check_correlations(db_path="propaganda.db", table_name="df_tweets_HiQualProp"):
    # Connect to the SQLite database and read in table
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()

    # Find/return the correlation matrix
    correlation_matrix = df.corr(numeric_only=True)
    print(correlation_matrix)
    return correlation_matrix


check_correlations()

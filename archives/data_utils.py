import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_features_and_data(db="data/tweets_features.db", table="tweets"):
    # Load data
    conn = sqlite3.connect(db)
    df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    conn.close()

    # Features and labels
    features = [
        "toxicity",
        "subjectivity",
        "profanity",
        "punctuation",
        "capitalization",
        "repetition",
        "keywords",
    ]
    X = df[features]
    y = df["is_extremist"]

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )  # 30% test data

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    return X_train_scaled, X_test_scaled, y_train, y_test

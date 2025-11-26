import sqlite3
import pandas as pd
import numpy as np
from archives.data_utils import load_features_and_data
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    recall_score,
)


def logreg_variations(X, y, penalty="l2", C=1.0, cv=5, scoring=["accuracy", "recall"]):
    logreg = LogisticRegression(
        penalty=penalty, C=C, solver="liblinear", class_weight="balanced", max_iter=1000
    )
    scores = cross_validate(
        logreg, X, y, cv=cv, return_train_score=True, scoring=scoring
    )
    return scores


def model(db="data/tweets_features.db", table="tweets"):
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

    # Cross-validate over different C values
    C_list = [1.0, 0.1, 0.01, 0.001, 0.0001, 1e-20]
    print("Looking at accuracy for different values of C for cross validation:")
    for C in C_list:
        scores = logreg_variations(X_train_scaled, y_train, C=C)
        mean_accuracy = np.mean(scores["test_accuracy"])
        mean_recall = np.mean(scores["test_recall"])
        print(f"C={C}, accuracy={mean_accuracy:.4f}, recall={mean_recall}")

    # Train the Logistic Regression model on the training data with the chosen C value
    logreg = LogisticRegression(
        penalty="l2",
        C=1.0, # chose 1.0
        solver="liblinear",
        class_weight="balanced",
        max_iter=1000,
    )
    logreg.fit(X_train_scaled, y_train)

    # Predict on the test data
    y_pred = logreg.predict(X_test_scaled)

    # Test Confusion Matrix and classification report
    cm = confusion_matrix(y_test, y_pred)
    print()
    print("Test  Confusion Matrix:")
    print(cm)
    print()
    print("Test Classification Report:")
    print(classification_report(y_test, y_pred))
    print()

    # Look at data averages
    print("Label averages", df["is_extremist"].value_counts(normalize=True))

    # After fitting the model
    print("Model Coefficients:")
    feature_importance = pd.DataFrame(
        {"Feature": features, "Coefficient": logreg.coef_[0]}
    ).sort_values(by="Coefficient", ascending=False)
    print(feature_importance)

    ## Output predictions
    # Get testing data
    df_test = df.loc[X_test.index]

    # Keep relevant columns
    columns_to_keep = [
        "text",
        #"toxicity",
        "subjectivity",
        "profanity",
        "punctuation",
        "capitalization",
        #"repetition",
        "keywords",
    ]
    df_test = df_test[columns_to_keep].copy()
    df_test["true_label"] = y_test.values
    df_test["predicted_label"] = y_pred

    # Split into false positive and negative
    false_positives = df_test[
        (df_test["true_label"] == 0) & (df_test["predicted_label"] == 1)
    ]
    false_negatives = df_test[
        (df_test["true_label"] == 1) & (df_test["predicted_label"] == 0)
    ]

    # Save them to the database
    conn = sqlite3.connect(db)
    false_positives.to_sql("false_positives", conn, if_exists="replace", index=False)
    false_negatives.to_sql("false_negatives", conn, if_exists="replace", index=False)
    conn.close()


# Run it
model()
import pandas as pd
import sqlite3
import re
from textblob import TextBlob
from profanity_check import predict_prob
from detoxify import Detoxify

"""
# Create a dummy sklearn.externals module with joblib in it
sklearn_externals = types.ModuleType("sklearn.externals")
sklearn_externals.joblib = joblib
sys.modules["sklearn.externals"] = sklearn_externals
"""


def polarity(text):
    return TextBlob(text).sentiment[0]


def subjectivity(text):
    return TextBlob(text).sentiment[1]


def profanity(text):
    return predict_prob([text])[0]


def toxicity(text):
    return Detoxify("original").predict(text)["toxicity"]


def clean_text(text):
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove mentions
    text = re.sub(r"@\w+", "", text)

    # Remove '#' but keep the word
    text = re.sub(r"#", "", text)

    # Remove shortcode emojis
    text = re.sub(r":[^\s:]+:", "", text)

    # Remove emojis and non-ASCII characters
    text = re.sub(r"[^\x00-\x7F]+", "", text)

    # Keep letters, numbers, spaces, and basic punctuation: ! ? . , ' "
    text = re.sub(r"[^a-zA-Z0-9\s!?,.'\"]", "", text)

    # Remove HTTPURL
    text = re.sub(r"HTTPURL", "", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def create_table(df):
    # Open connection
    con = sqlite3.connect("propaganda.db")
    cur = con.cursor()

    # Create table
    df.to_sql("df_tweets_HiQualProp", con, if_exists="replace", index=False)

    # Get sample output
    cur.execute("""
        SELECT * 
        FROM df_tweets_HiQualProp
        LIMIT 5;
    """)
    con.commit()

    # Print output
    resp = cur.fetchall()
    for row in resp:
        print(row)
    con.close()


def create_db():
    # Create df
    df = pd.read_csv("df_tweets_HiQualProp.csv", encoding="utf-8")

    # Clean up text
    df["cleaned_text"] = df["text_normalized"].apply(clean_text)

    # Toxicity
    toxicity_scores = []
    for i, text in enumerate(df["cleaned_text"]):
        score = toxicity(text)
        toxicity_scores.append(score)
        print(f"{i + 1}/{len(df)}: Processed")

    df["toxicity"] = toxicity_scores

    # Polarity and subjectivity
    df["polarity"] = df["cleaned_text"].apply(polarity)
    print("Retrieved polarity")

    df["subjectivity"] = df["cleaned_text"].apply(subjectivity)
    print("Retrieved subjectivity")

    # Profanity
    df["profanity"] = df["cleaned_text"].apply(profanity)
    print("Retrieved profanity")

    # Keep relevant rows
    df.drop(
        columns=[
            "text",
            "labels_weak1",
            "labels_weak2",
            "labels_weak3",
            "text_normalized",
        ],
        inplace=True,
    )

    # Create database
    create_table(df)


create_db()

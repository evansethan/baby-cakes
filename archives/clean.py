import pandas as pd
import sqlite3
import re
from pathlib import Path

raw_extremist = Path("data") / "extremist_1500.csv"
raw_norm = Path("data") / "normal_1500.csv"

cln_extremist = Path("data") / "extremist_cleaned.csv"
cln_norm = Path("data") / "normal_cleaned.csv"

db_basic = Path("data") / "tweets_basic.db"


def extract_key(row):
    '''
    Helper for json function
    '''
    return row["screen_name"]


def clean_text(text):
    '''
    Cleans the text of a tweet
    '''
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove '@' but keep the mention
    text = re.sub(r"@", "", text)

    # Remove shortcode emojis
    text = re.sub(r":[^\s:]+:", "", text)

    # Remove html artifacts
    text = re.sub(r"&amp", "", text)
                  
    # Remove emojis and non-ASCII characters
    text = re.sub(r"[^\x00-\x7F]+", "", text)

    # Keep letters, numbers, spaces, hashtags, and basic punctuation: ! ? . , ' "
    text = re.sub(r"[^a-zA-Z0-9\s!?,.#'\"]", "", text)

    # Remove HTTPURL
    text = re.sub(r"HTTPURL", "", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_normal(input_csv_path=raw_norm, output_csv_path=cln_norm):

    # create dataframe
    df = pd.read_csv(input_csv_path, encoding="utf-8")
    
    df.columns = ['username', 'text_unclean']

    # clean tweet text
    df["text"] = df["text_unclean"].apply(clean_text)

    # add label
    df['is_extremist'] = 0

    # df.to_csv(output_csv_path, index=False) # save to csv (optional)

    return df


def clean_extremist(input_csv_path=raw_extremist, output_csv_path=cln_extremist):

    # Read CSV file
    df = pd.read_csv(input_csv_path)

    # clean tweet text
    df["text"] = df["text"].apply(clean_text)

    # add label
    df['is_extremist'] = 1

    # df.to_csv(output_csv_path, index=False) # save to csv (optional)

    return df


def build_basic_db():
    '''
    Builds basic Sqlite database (only raw data/labels, no extracted features)
    '''
    # Clean and load to df
    df = pd.concat([clean_extremist(), clean_normal()])

    # Shuffle the data to avoid class ordering bias
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Create db
    conn = sqlite3.connect(db_basic)
    df.to_sql('tweets', conn, if_exists='replace', index=False)
    conn.close()


if __name__ == "__main__":
    build_basic_db()
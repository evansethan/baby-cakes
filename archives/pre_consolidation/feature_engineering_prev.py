import pandas as pd
import sqlite3
import re
from pathlib import Path
from textblob import TextBlob
from profanity_check import predict_prob
from detoxify import Detoxify
from collections import Counter
from archives.clean import clean_nazi, clean_normal, db_basic
from archives.keywords import KEYWORDS, STOPWORDS


detox_model = Detoxify("original")
db_features = Path("data") / "tweets_features.db"


# First build features
def profanity(text):
    """
    Returns the probability that a given text contains profanity.

    Uses the profanity_check library's predict_prob function, which
    takes an array of strings and returns a probability score for profanity.

    Args:
        text (str): The input text to evaluate.

    Returns:
        float: Probability that the text contains profanity (range 0–1).
    """
    return predict_prob([text])[0]


def keywords(text):
    
    # Lowercase tweet, remove stop words, count length
    text_tokens = [word for word in (text.lower()).split() if word not in STOPWORDS]
    total_tokens = len(text_tokens)
    
    # Set to keep track of tokens that are part of keywords
    counted_indices = set()

    for keyword in KEYWORDS:
        kw_tokens = (keyword.lower()).split()
        n = len(kw_tokens)

        for i in range(len(text_tokens) - n + 1):
            if text_tokens[i:i+n] == kw_tokens:
                # Check if any of the tokens in this span were already counted
                if any((i + j) in counted_indices for j in range(n)):
                    continue
                for j in range(n):
                    counted_indices.add(i + j)

    return (len(counted_indices) / total_tokens) if total_tokens else 0


def subjectivity(text):
    """
    Returns the subjectivity score of a given text.

    Uses TextBlob's sentiment property, where the subjectivity score
    ranges from 0 (very objective) to 1 (very subjective).

    Args:
        text (str): The input text to evaluate.

    Returns:
        float: Subjectivity score (range 0–1).
    """
    return TextBlob(text).sentiment[1]


def toxicity(text):
    """
    Returns the probability that a given text contains toxic content.

    Uses the Detoxify library's predict method, where toxic content is defined as
    "rude, disrespectful, or unreasonable" language likely to discourage participation
    or open dialogue.

    Args:
        text (str): The input text to evaluate.

    Returns:
        float: Probability that the text is toxic (range 0–1).
    """
    return detox_model.predict(text)["toxicity"]


def excess_punctuation(text):
    """
    Calculates the ratio of exclamation points and question marks
    to the total number of words in a given text.

    Parameters:
    text (str): The input string to analyze.
    """
    # Count words
    words = len(re.findall(r"\b\w+\b", text))

    # Count exclamation points
    exclamations = len(re.findall(r"!", text))

    # Count question marks
    questions = len(re.findall(r"\?", text))

    return (exclamations + questions) / words


def excess_capitalization(text):
    """
    Calculates the ratio of fully capitalized words to total words in
    the input text exceeds.

    A fully capitalized word is defined as a word with at least two consecutive
    uppercase letters and no lowercase letters (e.g., 'WARNING', 'HELP').

    Parameters:
    text (str): The input string to analyze.
    """
    # Count words
    words = len(re.findall(r"\b\w+\b", text))

    # Count all uppercase words
    uppercase_words = len(re.findall(r"\b[A-Z]{2,}\b", text))
    print(uppercase_words)

    return uppercase_words / words


def excess_repetition(text):
    """
    Analyzes the frequency of non-stop words in a text and calculates the ratio of
    the total number of repeated occurrences of non-stop words to the total number
    of non-stop words in the text.

    Parameters:
    text (str): The input string to analyze, typically a sentence or passage of text.

    Returns:
    float: The ratio of the total occurrences of repeated non-stop words to the total
           number of non-stop words in the text. If no non-stop words are present,
           the function returns 0.
    """

    # Extract words and normalize to lowercase
    words = re.findall(r"\b\w+\b", text.lower())

    # Filter out stop words
    non_stop_words = [word for word in words if word not in STOPWORDS]

    if not non_stop_words:
        return 0

    # Count word frequencies
    word_counts = Counter(non_stop_words)
    repeated_words = sum(word_counts.values())

    # Calculate repetition ratio
    ratio = repeated_words / len(non_stop_words)

    return ratio


def build_db(from_start=False):

    if from_start:
        df = pd.concat([clean_nazi(), clean_normal()])
    else:
        conn = sqlite3.connect(db_basic)
        df = pd.read_sql_query("SELECT * FROM tweets", conn)
        conn.close()

    # Add toxicity feature
    toxicity_scores = []
    for i, text in enumerate(df["text"]):
        score = toxicity(text)
        toxicity_scores.append(score)
        print(f"{i + 1}/{len(df)}: Processed")
    df["toxicity"] = toxicity_scores

    # Add keywords feature
    df["keywords"] = df["text"].apply(keywords)
    print("Retrieved keywords")

    # Add subjectivity feature
    df["subjectivity"] = df["text"].apply(subjectivity)
    print("Retrieved subjectivity")

    # # Add profanity feature
    df["profanity"] = df["text"].apply(profanity)
    print("Retrieved profanity")

    # Add punctuation feature
    df["punctuation"] = df["text"].apply(excess_punctuation)
    print("Retrieved punctuation scores")

    # Add capitalization feature
    df["capitalization"] = df["text"].apply(excess_capitalization)
    print("Retrieved capitalization scores")

    # Add repetition feature
    df["repetition"] = df["text"].apply(excess_repetition)
    print("Retrieved repetition scores")

    conn = sqlite3.connect(db_features)
    df.to_sql('tweets', conn, if_exists='replace', index=False)
    conn.close()


if __name__ == "__main__":
    build_db()
import pandas as pd
import sqlite3
from pathlib import Path
from nltk.stem import PorterStemmer
from rapidfuzz import process
from textblob import TextBlob
from detoxify import Detoxify
from functools import lru_cache
from archives.clean import clean_extremist, clean_normal, db_basic
from archives.keywords import KEYWORDS, STOPWORDS
from tqdm import tqdm
from collections import Counter
import re
import string

db_features = Path("data") / "tweets_features.db"
stemmer = PorterStemmer()
detox_model = Detoxify("original")

# helper function for word stemming ("birds" --> "bird") in keywords function
# (caching stores words that have already been processed, saves a ton of time)
@lru_cache(maxsize=10000)
def normalize(word):
    return stemmer.stem(word.lower())


# This divides KEYWORDS into single and multi- keywords groupings,
# so single keywords can be processed separately.
# We could make seperate files or objects instead of doing this. This was more of a "quick fix".
def preprocess_keywords():
    print("preprocessing keywords...")
    single_keywords = set()
    multi_keywords = set()

    for keyword in KEYWORDS:
        tokens = tuple(normalize(w) for w in keyword.lower().split())
        if len(tokens) == 1:
            single_keywords.add(tokens[0])
        else:
            multi_keywords.add(tokens)

    all_keywords_list = list(single_keywords.union(multi_keywords))
    print("done preprocessing keywords")
    return (all_keywords_list, single_keywords, multi_keywords)


# helper function for loading single word lists (technically sets) from text files
# this may be preferable, as keywords.py may get unwieldy
def load_word_list(filepath):
    """
    Loads a set of unique words from a .txt file
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return set(line.strip().lower() for line in f if line.strip())

# Load bad words from txt file
BAD_WORDS = load_word_list("profanity_list.txt")


def subjectivity(text):
    """
    Returns the subjectivity score of a given text.

    Uses TextBlob's sentiment property, where the subjectivity score
    ranges from 0 (very objective) to 1 (very subjective).

    Args:
        text (str): The input text to evaluate.

    Returns:
        float: Subjectivity score (range 0 - 1).
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


# we can now customize list of profane words (in profanity_list.txt)
# list includes all words from the better_profanity library.
# we'll want to remove keywords from the .txt file (like slurs)
def profanity(text, bad_words=BAD_WORDS):
    words = text.lower().split()
    return any(word in bad_words for word in words)


def excess_punctuation(text):
    """
    Calculates the ratio of exclamation points and question marks
    to the total number of words in a given text.

    Parameters:
    text (str): The input string to analyze.
    """
    # Count words
    words = len(re.findall(r"\b\w+\b", text))
    if words == 0:
        return 0

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
    if words == 0:
        return 0

    # Count all uppercase words
    uppercase_words = len(re.findall(r"\b[A-Z]{2,}\b", text))

    return uppercase_words / words



def excess_repetition(text):
    """
    Calculates the ratio of repeated occurrences of non-stop words to the total number
    of non-stop words in the input text.

    Parameters:
    text (str): The input string to analyze.

    Returns:
    float: Ratio of repeated (extra) non-stop word occurrences to total non-stop words.
           Returns 0 if there are no non-stop words.
    """

    # Extract words and normalize to lowercase
    words = re.findall(r"\b\w+\b", text.lower())

    # Filter out stop words
    non_stop_words = [word for word in words if word not in STOPWORDS]

    if not non_stop_words:
        return 0

    # Count word frequencies
    word_counts = Counter(non_stop_words)

    # Count repeated occurrences (i.e., total - unique)
    repeated_count = sum(count - 1 for count in word_counts.values() if count > 1)

    # Calculate repetition ratio
    ratio = repeated_count / len(non_stop_words)

    return ratio


def keywords(text, fuzzy_threshold=85, enable_fuzzy=True):
    """
    Analyze a given text and compute the proportion of tokens that match predefined keywords.

    This function tokenizes the input text, normalizes the tokens, and then attempts to match 
    them against a set of predefined keywords. The matching supports:
    - Exact match for single-word keywords.
    - Optional fuzzy matching using Levenshtein distance via the `fuzzywuzzy` library.
    - Exact match for multi-word (n-gram) keywords.

    Parameters:
        text (str): The input text to analyze.
        fuzzy_threshold (int, optional): Minimum score (0–100) required for fuzzy matches. 
                                         Only used if `enable_fuzzy` is True. Default is 85.
        enable_fuzzy (bool, optional): Whether to use fuzzy matching for single keywords. Default is True.

    Returns:
        float: The proportion of matched tokens (based on exact and fuzzy matching) relative to total valid tokens.
               Returns 0 if no valid tokens are found after preprocessing.
    """
    # Normalize text: split, make lowercase and remove punctuation.
    words = text.lower().translate(str.maketrans('', '', string.punctuation)).split()

    # Remove stopwords
    text_tokens = [w for w in words if w not in STOPWORDS]

    # Word stemming using normalize() function and nltk library ("birds" --> "bird")
    norm_tokens = [normalize(w) for w in text_tokens]

    # Count the tokens we're working with
    total_tokens = len(norm_tokens)
    counted_indices = set()
    
    # Pre process keywords 
    all_keywords_list, single_keywords, multi_keywords = preprocess_keywords()

    # Match on single keywords set
    for i, token in enumerate(norm_tokens):
        # Token match woo!!
        if token in single_keywords:
            counted_indices.add(i)
        # If fuzzy is enabled, do a fuzzy match on all tokens (this is slow.) ("b1rd --> bird")
        elif enable_fuzzy:
            result = process.extractOne(token, all_keywords_list, score_cutoff=fuzzy_threshold)
            if result:
                match, _, _ = result
                counted_indices.add(i)

    # Now, match on multi-word keywords set:
    # First, get maximum multi-keyword length (I'd say we should avoid max_len > 3 words)
    max_len = max(len(k) for k in multi_keywords) if multi_keywords else 0

    # For every n-gram from 2-gram up to max-gram (we could simplify this if we want)
    for n in range(2, max_len + 1):

        # For every token in normalized list of tokens,
        for i in range(len(norm_tokens) - n + 1):

            # (This part avoids inflating the match ratio by skipping potential
            # matches that have already been counted in the single word search)
            # Example: "I love Jeb Bush!" shouldn't count both "Jeb" and "Jeb Bush"
            if any((i + j) in counted_indices for j in range(n)):
                continue

            # Get the next n-gram from the normalized list of tokens
            span = tuple(norm_tokens[i:i+n])

            # Compare with multi-keywords and add to counted_indices if it's a match
            if span in multi_keywords:
                # Multi-keyword match woo!
                for j in range(n):
                    # Adds each word's index (this is why we need to skip potential matches earlier)
                    counted_indices.add(i + j)

    # Return ratio of matched tokens to total tokens
    return (len(counted_indices) / total_tokens) if total_tokens else 0


def build_db(from_start=False):

    # This loads from the raw data files and takes a long time (only needs to be done once)
    if from_start:
        print("loading raw data files (you sure about this?)")
        df = pd.concat([clean_extremist(), clean_normal()])

    # This processes the cleaned sqlite database instead once it's created
    else:
        print("accessing db")
        conn = sqlite3.connect(db_basic)
        df = pd.read_sql_query("SELECT * FROM tweets", conn)
        conn.close()

    # Add toxicity feature (very slow)
    tqdm.pandas(desc="Scoring toxicity")
    df["toxicity"] = df["text"].progress_apply(toxicity)
    print("Retrieved toxicity")

    # Add subjectivity feature (7 min)
    tqdm.pandas(desc="Scoring subjectivity")
    df["subjectivity"] = df["text"].progress_apply(subjectivity)
    print("Retrieved subjectivity")

    # Add profanity feature (fast)
    tqdm.pandas(desc="Scoring profanity")
    df["profanity"] = df["text"].progress_apply(profanity)
    print("Retrieved profanity")

    # Add punctuation feature (fast, abt 20 sec)
    tqdm.pandas(desc="Scoring punctuation")
    df["punctuation"] = df["text"].progress_apply(excess_punctuation)
    print("Retrieved punctuation scores")

    # Add capitalization feature (fast, abt 20 sec)
    tqdm.pandas(desc="Scoring capitalization")
    df["capitalization"] = df["text"].progress_apply(excess_capitalization)
    print("Retrieved capitalization scores")

    # Add repetition feature (About 1 min)
    tqdm.pandas(desc="Scoring repetition")
    df["repetition"] = df["text"].progress_apply(excess_repetition)
    print("Retrieved repetition scores")

    # Add keywords feature (abt 4 min)
    tqdm.pandas(desc="Scoring keywords")
    df["keywords"] = df["text"].progress_apply(lambda text: keywords(text, enable_fuzzy=False))
    print("Retrieved keywords")

    conn = sqlite3.connect(db_features)
    df.to_sql('tweets', conn, if_exists='replace', index=False)
    conn.close()


if __name__ == "__main__":
    build_db()
    print("all done!")
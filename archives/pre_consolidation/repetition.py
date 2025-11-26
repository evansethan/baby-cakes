import re
from collections import Counter
from numpy import mean
from itertools import combinations


def excess_punctuation(text, ratio_threshold=0.4):
    """
    Determines whether the ratio of exclamation points and question marks
    to the total number of words in a given text exceeds a specified threshold.

    Parameters:
    text (str): The input string to analyze.
    ratio_threshold (float): The punctuation-to-word ratio threshold. If the
                             ratio of (! and ?) to total words is greater than
                             or equal to this value, the function returns 1.
    Returns:
    int: 1 if the punctuation ratio exceeds or equals the threshold, 0 otherwise.
    """
    # Count words
    words = len(re.findall(r"\b\w+\b", text))

    # Count exclamation points
    exclamations = len(re.findall(r"!", text))

    # Count question marks
    questions = len(re.findall(r"\?", text))

    if (exclamations + questions) / words >= ratio_threshold:
        return 1
    else:
        return 0


def excess_capitalization(text, ratio_threshold=0.3):
    """
    Determines whether the ratio of fully capitalized words to total words in
    the input text exceeds a given threshold.

    A fully capitalized word is defined as a word with at least two consecutive
    uppercase letters and no lowercase letters (e.g., 'WARNING', 'HELP').

    Parameters:
    text (str): The input string to analyze.
    ratio_threshold (float): The capitalization-to-word ratio threshold. If the
                             ratio of fully capitalized words to total words is
                             greater than or equal to this value, the function returns 1.

    Returns:
    int: 1 if the capitalization ratio is greater than or equal to the threshold,
         0 otherwise.
    """
    # Count words
    words = len(re.findall(r"\b\w+\b", text))

    # Count all uppercase words
    uppercase_words = len(re.findall(r"\b[A-Z]{2,}\b", text))
    print(uppercase_words)

    if uppercase_words / words >= ratio_threshold:
        return 1
    else:
        return 0


def excess_repetition(text, ratio_threshold=0.3):
    """
    Checks for repeated non-stop words in a string and determines if the most
    repeated word's frequency relative to total non-stop word count exceeds a
    given threshold.

    Parameters:
    text (str): The input string to analyze.
    ratio_threshold (float): The repetition-to-word ratio threshold. If the most
                             repeated word appears with a ratio greater than or
                             equal to this value, the function returns 1.

    Returns:
    int: 1 if the most repeated word's ratio >= ratio_threshold, 0 otherwise.
    """

    # Basic English stop words # LOOK IN PYTHON ASSIGNMENTS FOR MORE
    stop_words = set(
        [
            "the",
            "is",
            "in",
            "at",
            "on",
            "and",
            "a",
            "an",
            "to",
            "of",
            "it",
            "for",
            "with",
            "that",
            "this",
            "as",
            "by",
            "was",
            "are",
            "be",
            "from",
            "or",
            "but",
        ]
    )

    # Extract words and normalize to lowercase
    words = re.findall(r"\b\w+\b", text.lower())

    # Filter out stop words
    non_stop_words = [word for word in words if word not in stop_words]

    if not non_stop_words:
        return 0

    # Count word frequencies
    word_counts = Counter(non_stop_words)
    most_common_word, most_common_count = word_counts.most_common(1)[0]

    # Calculate repetition ratio
    ratio = most_common_count / len(non_stop_words)
    print(ratio)

    return 1 if ratio >= ratio_threshold else 0

import re
import pandas as pd


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


def clean_data(input_csv_path, output_csv_path):
    # Create dataframe & apply cleaning function to all data
    df = pd.read_csv(input_csv_path, encoding="utf-8")
    df["cleaned_text"] = df["text_normalized"].apply(clean_text)
    # Save cleaned version for next step
    df.to_csv(output_csv_path, index=False)
    print(f"Saved cleaned data to {output_csv_path}")


if __name__ == "__main__":
    clean_data("df_tweets_HiQualProp.csv", "df_tweets_HiQualProp_cleaned.csv")

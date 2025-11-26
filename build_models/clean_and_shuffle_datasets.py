import json
import pandas as pd
import random
import re

def clean_text(text):
    """
    Cleans text with various regex substitutions

    - Removes text starting with http or www
    - Removes "@" 
    - Removes custom emoji codes 
    - Removes (&amp)
    - Removes non-ASCII characters
    - Removes non-alphanumeric characters except !?,.'" and whitespace
    - Removes placeholder token 'HTTPURL'.
    - Cleans whitespace 

    Inputs:
        text (str): Raw text

    Returns:
        (str): Cleaned version of the input text.
    """
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@", "", text)
    text = re.sub(r":[^\s:]+:", "", text)
    text = re.sub(r"&amp", "", text)
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s!?,.'\"]", "", text)
    text = re.sub(r"HTTPURL", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def import_stormchaser_data(path: str):
    """
    Load stormchaser (mostly extremist) data from specified JSON, 
    and fetch username and cleaned text
    
    Inputs:
        path: str with path to stormchaser data 
    
    Outputs:
        clean_data: list of dictionaries with user, text keys
    """    
    # Load in data 
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    clean_data = []
    for post in data:
        # For each post in data, fetch username and text, cleaning text 
        clean_post = {"user": post.get("user", {}).get("screen_name")}
        full_text = post.get("full_text")
        if full_text:
            # Use full text if available, otherwise use limited text field
            clean_post["text"] = clean_text(full_text)
        else:
            clean_post["text"] = clean_text(post.get("text", ""))
            
        clean_data.append(clean_post)
            
    return clean_data


def import_sentiment140_data(path):
    """
    Load Sentiment140 data from specified CSV file,
    and fetch user and cleaned text
    
    Inputs:
        path: str with path to sentiment140 data 
    
    Outputs:
        clean_data: list of dictionaries with user, text keys
    """
    # Load in data 
    columns = ['_1', '_2', '_3', '_4', 'user', 'text'] 
    df = pd.read_csv(path, names=columns, encoding="ISO-8859-1")
    df = df[['user', 'text']]
    
    # Clean text 
    df['text'] = df['text'].apply(clean_text)

    # Convert to list of dicts
    return df.to_dict(orient="records")

def save_shuffled_dataset(data: list[dict], output_filename: str):
    """
    Load in dataset (list of dicts), randomize order, output as CSV file
    """
    # Shuffle the list in place
    random.shuffle(data)
    
    # Save as CSV file 
    pd.DataFrame(data).to_csv(output_filename, index=False)


if __name__ == "__main__":
    # Import data
    stormchaser_data = import_stormchaser_data("data/stormchaser_data.json")
    sentiment140_data = import_sentiment140_data("data/sentiment140_data.csv")
    
    # Shuffle and save data
    save_shuffled_dataset(stormchaser_data, "data/stormchaser_shuffled.csv")
    save_shuffled_dataset(sentiment140_data, "data/sentiment140_shuffled.csv")
    
    
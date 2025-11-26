import httpx
import time
import pandas as pd 
import os 
from build_models.clean_and_shuffle_datasets import clean_text

# Global variables for scrape 
API_KEY = os.getenv("API_KEY")
QUERY = os.getenv("QUERY_TERM")
MAX_TWEETS = 250
API_URL = "https://api.twitterapi.io/twitter/tweet/advanced_search"
HEADERS = {"X-API-Key": API_KEY}

def fetch_tweets():
    """
    Paginates through tweets from Twitterapi.io up to a maximum tweets number 

    Returns:
        tweets: A list of cleaned tweet dictionaries 
    """
    
    # Initialize empty list of tweets, cursor
    tweets = []
    cursor = ""

    while len(tweets) < MAX_TWEETS:
        params = {
            "query": QUERY,
            "queryType": "Latest",
            "cursor": cursor
        }

        # Get data, add to tweets list 
        response = httpx.get(API_URL, headers=HEADERS, params=params)
        data = response.json()
        batch = data.get("tweets", [])

        # For each tweet, extract user and clean text
        for tweet in batch:
            username = tweet.get("author", {}).get("userName", "")
            text = clean_text(tweet.get("text", ""))
            tweets.append({"user": username, "text": text})

        # Stop searching if reached end of tweets 
        if not data.get("has_next_page") or not data.get("next_cursor"):
            break

        # Move onto next cursor
        cursor = data["next_cursor"]
        
        # Avoid getting blocked by the API
        time.sleep(3)

    return tweets[:MAX_TWEETS]

if __name__ == "__main__":
    filepath = "data/addional_data.csv"
    tweets = fetch_tweets()
    
    # Convert list of tweet dictionaries to csv file 
    pd.DataFrame(tweets).to_csv(filepath, index=False)

# Dataset Preparation

## Step 1: Download and Shuffle Datasets
Please download the following datasets manually and save them with the listed filenames in the `data/` directory.

- **Stormchaser Dataset**  
  Source: [Islet, S. (2017, November 14). Nazi Tweets [Data set]. Kaggle.](https://www.kaggle.com/datasets/saraislet/nazi-tweets)  
  Save as: `stormchaser_data.json`  

- **Sentiment140 Dataset**  
  Source: [Kazanova. (2017). Sentiment140 dataset with 1.6 million tweets [Data set]. Kaggle.](https://www.kaggle.com/datasets/kazanova/sentiment140)
  Save as: `sentiment140_data.csv`

Once these files are downloaded, run `clean_and_shuffle_datasets.py` to clean and shuffle them.

## Step 2 (Optional): Scrape Additional Data
If you would like to include additional tweets from recent years, run `scrape_additional_data.py`, using [appropriate values for the scraping global variables as needed.](https://docs.twitterapi.io/api-reference/endpoint/tweet_advanced_search) 

## Step 3: Label Data
You should now have the following files in your data directory:
 - `stormchaser_shuffled.csv`
 - `sentiment140_shuffled.csv`
 - `(Additional Optional Scraped CSV files)`

At this point, you are ready to begin labeling. Using the files above, create a final file in the data directory called `manually_labeled_data.csv` with the following sample format:
| user    | text    | extreme |
| ------- | ------  | ------- |
| "user1" | "text1" | 0       |
| "user2" | "text2" | 1       |
| "user3" | "text3" | 0       |

Alternatively, you may reach out to our team to ask for our cleaned, labeled dataset. We are unable to share it on GitHub.


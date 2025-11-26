# Propaganda Detection Model

## Members
- Evan Fantozzi, evanfantozzi@uchicago.edu
- Grace Kluender, graceek@uchicago.edu
- Ethan Evans, ethane@uchicago.edu


## Proposal
Our proposal from Milestone 1 remains the same. We aim to build a binary classification model that identifies whether a tweet is political propaganda based upon whether it contains text that signals the use of propaganda techniques, including fear-based language, name calling, repetition, excessive use of capitalization and punctuation, and flag-waving.

## Data

We plan to use an HQP dataset, a human-annotated collection of 29,596 tweets related to Russia, China, and India. The HQP dataset is the first large-scale collection for online propaganda detection that is fully human-annotated. It reports Area Under the Curve (AUC) scores reaching over 92% when using these annotations.

We plan to use both lexical and semantic features extracted from each tweet. So far, these include toxicity (using Detoxify), polarity and subjectivity (using TextBlob), profanity (using profanity_check), and punctuation/capitalization patterns using basic text analysis. In addition, we aim to add new features such as emotionality (via NRClex), repetition, and propaganda techniques like slogan use, flag-waving, and war rhetoric, using the Empath library.

Below is the link where you can find this dataset and the paper, which details the development process for the dataset.

[human-annotated dataset of 29,596 tweets related to Russia, China, and India](https://arxiv.org/abs/2304.14931) 

Explore the dataset. Print or plot basic statistics, minimum and maximum values, etc.

## Data Exploration
We explored this data using sqlite3 and skimpy. To see the in-depth exploration,
see exploration.ipynb. Below are some top line stats:
- 15.32% of our labeled data is classified as a "propaganda" tweet
- The mean toxicity score (probability that a tweet contains "toxic" content) across our datasest is 0.1194
- The average polarity score across our dataset is 0.01963
    - NOTE: The polarity score ranges from -1.0 to 1.0, where -1.0 indicates negative sentiment and 1.0 indicates 
            positive sentiment.
- The average subjectivity score across our dataset is 0.3554
    - NOTE: The subjectivity score ranges from 0.0 to 1.0, where 0.0 is very objective and 1.0 is very subjective.
- The average profanity score across our dataset is 0.1188
    - NOTE: The profanity score is the probability that the tweet contains profanity in it
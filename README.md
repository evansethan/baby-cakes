# BabyCakes: Hate Speech Detector

Our team developed a supervised binary classification model which predicts whether a social media post (e.g., a tweet) uses far-right hate speech.

To train and run the model yourself, you can build out your own dataset or contact us.

## Dataset Preparation
See Data.md in the data folder

## Keyword Lists Preparation
See KeywordLists.md in build_models/keyword_lists


Once all files are obtained, from the main directory run:


## uv sync
This ensures you have the libraries you need. Ony need to run once.

## python3 build_models/create_model_pkls.py 
This learns the model and saves it to a file. Only should be run once.

### python3 app.py
This opens the interactive user interface.

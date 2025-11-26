# babycakes

1) Run combine.py on the 4 csv files to combine and label them. this will create tweets.csv

2) Run run.py to build and deploy the model. After building it once, uncomment "if __name__ == "__main__:" in learn.py to avoid retraining the model over and over. the model is saved in the workspace as model.pkl
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import FunctionTransformer
from sentence_transformers import SentenceTransformer
import joblib

# Import insider terms 
with open('models/insider_terms.txt', 'r', encoding='utf-8') as file:
    INSIDER_TERMS = file.readlines()

# --- MiniLM Transformer ---
class MiniLMTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.model.encode(X['text'].tolist(), convert_to_numpy=True)
    

# --- Insider Terms ---
class InsiderTermsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, terms=None):
        self.terms = [term.lower() for term in (terms or INSIDER_TERMS)]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        texts = X['text'].str.lower().tolist()
        features = []

        for text in texts:
            features.append([int(term in text) for term in self.terms])

        return np.array(features)
    

class Pre2018Selector(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[['pre_2018']].values.astype(float)


def select_text_column(X):
    return X['text']


def create_model(dirty=False):
    # Load data
    df = pd.read_csv('models/tweets.csv')
    
    # Basic cleaning if needed
    if dirty:
        df = df.dropna(subset=['text'])
        df = df[df['text'].str.strip() != '']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        df[['text']], df['extreme'], test_size=0.2, random_state=42
    )

    # Combine features
    combined_features = FeatureUnion([
        ('tfidf', Pipeline([
            ('selector', FunctionTransformer(select_text_column, validate=False)),
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2)))
        ])),
        ('minilm', MiniLMTransformer()),
        ('insider_terms', InsiderTermsTransformer()),
        #('pre_2018', Pre2018Selector())
    ])

    # Full pipeline with non-linear SVM
    model = Pipeline([
        ('features', combined_features),
        ('clf', SVC(kernel='rbf', class_weight='balanced', probability=True))
    ])

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("\nClassification Report (TF-IDF + MiniLM + RBF-SVM):\n")
    print(classification_report(y_test, y_pred, target_names=['normal', 'extreme']))

    # Save trained model
    joblib.dump(model, 'models/model.pkl')
    
if __name__ == "__main__":
    create_model(dirty=True)
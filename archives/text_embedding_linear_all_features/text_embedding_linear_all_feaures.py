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
from textblob import TextBlob
from detoxify import Detoxify
import joblib

# Import insider terms 
with open('../models/insider_terms.txt', 'r', encoding='utf-8') as file:
    INSIDER_TERMS = file.readlines()

# Import bad words
with open('../profanity_list.txt', 'r', encoding='utf-8') as file:
    BAD_WORDS = file.readlines()

class SubjectivityTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([TextBlob(text).sentiment.subjectivity for text in X['text']]).reshape(-1, 1)


class ToxicityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.model = Detoxify('original')

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([self.model.predict(text)['toxicity'] for text in X['text']]).reshape(-1, 1)


class ProfanityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, bad_words=None):
        self.bad_words = set(bad_words) if bad_words else set()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        result = []
        for text in X['text'].str.lower():
            words = text.split()
            result.append(int(any(word in self.bad_words for word in words)))
        return np.array(result).reshape(-1, 1)

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
    df = pd.read_csv('tweets.csv')
    
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
        ('pre_2018', Pre2018Selector()),
        ('subjectivity', SubjectivityTransformer()),
        ('toxicity', ToxicityTransformer())
        ('profanity', ProfanityTransformer(bad_words=BAD_WORDS))
    ])

    # Full pipeline
    model = Pipeline([
        ('features', combined_features),
        ('clf', LogisticRegression(class_weight='balanced', max_iter=1000))
    ])

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("\nClassification Report (TF-IDF + MiniLM):\n")
    print(classification_report(y_test, y_pred, target_names=['normal', 'extreme']))

    # Save trained model
    joblib.dump(model, 'model.pkl')

#if __name__ == "__main__":
create_model(dirty=True)
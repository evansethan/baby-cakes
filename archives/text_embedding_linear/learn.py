import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import FunctionTransformer
from sentence_transformers import SentenceTransformer
import joblib

# --- MiniLM Transformer ---
class MiniLMTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X is a DataFrame with a 'text' column
        return self.model.encode(X['text'].tolist(), convert_to_numpy=True)


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
        ('minilm', MiniLMTransformer())
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
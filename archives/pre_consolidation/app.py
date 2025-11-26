from flask import Flask, request, render_template_string
import pickle
from archives.feature_engineering import (
    keywords,
    toxicity,
    subjectivity,
    profanity,
    excess_punctuation,
    excess_capitalization,
    excess_repetition,
    normalize,
    preprocess_keywords,
)
from nltk.stem import PorterStemmer

app = Flask(__name__)

# Initialize stemmer
stemmer = PorterStemmer()

# Preprocess keywords once
all_keywords_list, single_keywords, multi_keywords = preprocess_keywords()

# Load trained model once
with open("nonlinear_model.pkl", "rb") as f:
    model = pickle.load(f)


def extract_features(text):
    """
    Extract all features needed for model prediction from input text.
    Returns a dict with feature_name: feature_value pairs.
    """

    features = {
        "toxicity": toxicity(text),
        "subjectivity": subjectivity(text),
        "profanity": profanity(text),
        "punctuation": excess_punctuation(text),
        "capitalization": excess_capitalization(text),
        "repetition": excess_repetition(text),
        "keywords": keywords(text),
    }
    return features

# HTML template
HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <title>Far-Right Extremist Detector</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 700px; margin: auto; padding: 2em; }
    textarea { width: 100%; height: 150px; }
    button { padding: 0.5em 1em; font-size: 1em; }
    .feature-score { margin: 0.2em 0; }
  </style>
</head>
<body>
  <h1>Far-Right Extremist Detector</h1>
  <form method="POST">
    <textarea name="user_input" placeholder="Enter text to analyze...">{{ user_input }}</textarea><br>
    <button type="submit">Analyze</button>
  </form>

  {% if error %}
    <p style="color: red;">{{ error }}</p>
  {% endif %}

  {% if features %}
    <h2>Feature Scores</h2>
    <ul>
    {% for k, v in features.items() %}
      <li class="feature-score"><strong>{{ k.capitalize() }}:</strong> {{ "%.4f"|format(v) }}</li>
    {% endfor %}
    </ul>

    <h2>Model Prediction</h2>
    <p><strong>Predicted Class:</strong> {{ prediction }}</p>
    <p><strong>Class Probabilities:</strong></p>
    <ul>
    {% for prob in probabilities %}
      <li>Class {{ loop.index0 }}: {{ "%.4f"|format(prob) }}</li>
    {% endfor %}
    </ul>
  {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def main():
    features = None
    prediction = None
    probabilities = None
    user_input = ""
    error = None

    if request.method == "POST":
        user_input = request.form.get("user_input", "").strip()
        if not user_input:
            error = "Please enter some text to analyze."
        else:
            # Extract features
            features = extract_features(user_input)

            # Prepare feature vector for prediction (model expects list of feature values)
            feature_vector = [[features[k] for k in features]]

            # Predict
            prediction = model.predict(feature_vector)[0]
            probabilities = model.predict_proba(feature_vector)[0].tolist()

    return render_template_string(
        HTML_TEMPLATE,
        user_input=user_input,
        features=features,
        prediction=prediction,
        probabilities=probabilities,
        error=error,
    )


if __name__ == "__main__":
    app.run(debug=True)

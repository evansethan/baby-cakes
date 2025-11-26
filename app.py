from flask import Flask, request, render_template
import joblib
import pandas as pd
from nltk.stem import PorterStemmer
from build_models.create_model_pkls import (
    MiniLMTransformer, 
    InsiderTermsTransformer, 
    SubjectivityTransformer,
    ToxicityTransformer,
    ProfanityTransformer,
    select_text_column
)

app = Flask(__name__)

# Initialize stemmer
stemmer = PorterStemmer()

# Load trained pipeline model
model = joblib.load("build_models/model_pkls/model_svc.pkl") # Using SVC model 

@app.route("/", methods=["GET", "POST"])
def main():
    probability = None
    prediction = None
    user_input = ""
    error = None

    if request.method == "POST":
        user_input = request.form.get("user_input", "").strip()
        
        # Confirm user input 
        if not user_input:
            error = "Please enter some text to analyze."
        else:
            # Run model on input
            input_df = pd.DataFrame({'text': [user_input]})
            
            # Get probability of extremist text
            probability = model.predict_proba(input_df)[0].tolist()[1]
            
            if probability < 0.2:
                prediction = "Very unlikely to be extremist"
            elif probability < 0.4:
                prediction = "Unlikely to be extremist"
            elif probability < 0.6:
                prediction = "May be extremist"
            elif probability < 0.8:
                prediction = "Likely to be extremist"
            else:
                prediction = "Very likely to be extremist"
        

    return render_template(
        "app.html",
        user_input=user_input,
        prediction=prediction,
        probability=probability,
        error=error,
    )

if __name__ == "__main__":
    app.run(debug=True)
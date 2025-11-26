import pandas as pd
import joblib
import learn

def input_tweet(path):

    model = joblib.load(path)

    while True:
        user_tweet = input("\nEnter a tweet to classify (or type 'exit' to quit):\n> ")
        if user_tweet.lower() == 'exit':
            break
        elif not user_tweet.strip():
            print("Please enter a non-empty tweet.")
            continue

        sample = pd.DataFrame({'text': [user_tweet]})

        prediction = model.predict(sample)[0]
        proba = model.predict_proba(sample)[0][1]

        label = "extreme" if prediction == 1 else "normal"
        print(f"\n→ This tweet is classified as: **{label}** (probability: {proba:.2f})")


input_tweet('model.pkl')

# sample strings (problem ones included):

# "all hail kek! blood and soil, brothers!"
# "hitler had some great ideas!"
# "donald trump will purify our nation and drive out the degenerates"
# "I WILL END INFLATION, I WILL STOP THE INVASION OF CRIMINALS COMING INTO OUR COUNTRY, and I WILL BRING BACK THE AMERICAN DREAM"
# "NO TAX ON TIPS. NO TAX ON OVERTIME. NO TAX ON SENIORS' SOCIAL SECURITY."
# "donald trump is a good president in my opinion."
# "donald trump is a bad president in my opinion."
# "richard b spencer is a major figure in the far-right."
# "i am an american citizen and i like living here"
# "I woke up on a white beach with an american flag towel"
# "I really like cats hehe :)"

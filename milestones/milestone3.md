# Propaganda Detection Model

## Members
- Evan Fantozzi, evanfantozzi@uchicago.edu
- Grace Kluender, graceek@uchicago.edu
- Ethan Evans, ethane@uchicago.edu

## Proposal
We aim to build a binary classification model that identifies whether a tweet is political propaganda. Our model will scan a piece of text for propaganda techniques, including use of toxic language, repetition, excessive use of capitalization and punctuation, and run the output through a logistic regression machine learning model to determine if the text is propaganda. We plan to train our model on differing datasets to determine which training data will lead to the best model, and we also plan to incorporate other features not included in our current model, like hyperbolic keywords and nationalistic language.

# Which linear model did you pick? Why?
We chose a logistic regression model. Our problem is a supervised binary classification problem, and logistic regression models the probability of a binary outcome, so it is suitable for a problem like ours. Additionally, we chose the logistic regression model in particular because it is computationally efficient and relatively easy to understand which features are contributing to the prediction. It is also typical for text classification problems to utilize logistic regression as the baseline model. Therefore, we figured it would be a good starting point before trying out more complex models. Finally, the logistic model allows us to add regularizers, which can help with managing overfitting. This is especially important for our high-dimensional text data.

# Which regularization term did you pick? Why did you pick that regularizer for your model?
We decided to go with L2 regularization because it helps reduce overfitting by shrinking all the coefficients evenly, without removing any features. That’s important for our project, since propaganda can show up in subtle ways across multiple features—so we didn’t want to risk dropping any that might still carry useful information. We also tested L1 regularization, which pushes some coefficients to zero, but found that both regularizers gave us similar results. That actually gave us more confidence in sticking with L2, since it tends to be more stable and leads to smoother optimization. Additionally, we tested out a number of lambda values for our regularizer, see below:

| Lambda  | Accuracy | Recall              |
| ------- | -------- | ------------------- |
| 1.0     | 0.7063   | 0.3643767412140575  |
| 0.1     | 0.7063   | 0.3640567412140575  |
| 0.01    | 0.7071   | 0.3650167412140575  |
| 0.001   | 0.7090   | 0.36725571884984026 |
| 0.0001  | 0.7120   | 0.37333162939297126 |
| 1e-20   | 0.8491   | 0.0                 |

As discussed further below, we prioritize recall in selecting our lambda value, and use a value of 0.0001. (It appears that the e-20 lambda value results in a model that predicts non-propaganda 100% of the time.) As we adapt our model, we expect our lambda value may change.

# How do you plan on describing your linear model and its output? What do small or large weights mean?
Our model outputs the probability that a tweet is propaganda. Each feature in the model is assigned a weight (coefficient) that represents how strongly it contributes to the prediction. A large weight means that the feature contributes more strongly toward the likelihood the tweet is classified as propaganda, while a small weight indicates that a feature has little influence on the outcome. In the case of negative weights, the relationship between features and classification is inverse. So far, most of our features have relatively small weights. This suggests that either the features we’re using (e.g., punctuation, repetition, capitalization) aren’t strong predictors of propaganda in the current dataset, or that our training data isn't capturing enough variation. Therefore, we may need to rethink what data we use or incorporate features that capture more nuanced language like hyperbole, emotional tone, or nationalistic keywords.

# What metrics will you use to analyze the performance of your model?
As discussed, we consider all metrics, including accuracy, precision and recall. However, since false negatives are the most important to avoid (We don't want to identify propaganda as non-propaganda), we prioritize recall.
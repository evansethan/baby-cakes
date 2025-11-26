# Team Extremist Detectors: Grace Kluender, Ethan Evans, Evan Fantozzi

# Milestone 5

Our team aims to develop a supervised binary classification model, which predicts whether a social media post (e.g., a tweet) is far-right extremist. We have already implemented a linear model—a logistic regression model—but now have also implemented a non-linear model with the goal of capturing more complex patterns and interactions between features that may better distinguish extremist content from non-extremist content. 

# Support Vector Machine Model
Our team decided to employ an SVM model with a nonlinear kernel because SVM models work particularly well on high-dimensional data, like our engineered text analysis features. Additionally, SVM models have a number of hyperparmeters that can be tuned to the needs of the model.

To determine which nonlinear kernel to select, we ran a 5-fold cross validation to compare each model's accuracy, recall, and f1 scores. We found that the Radial Basis Function (RBF) kernel resulted in the highest recall and accuracy. Therefore, we will implement an SVM model with an RBF kernel.

## Explain How SVM with RBF Kernel is Non-Linear
The SVM model can be linear or non-linear, depending on the kernel that is used. Because our team is employing the non-linear RBF kernel, the SVM model becomes non-linear. Simply put, the RBF kernel function is a mathematical function that converts a low-dimensional input space into a higher-dimensional feature space, where a linear decision boundary can be found. This transformation allows the SVM to construct complex, non-linear decision boundaries in the original input space.

The RBF kernel works by finding the dot products and squares of all the features in the dataset and then performing the classification using the basic idea of Linear SVM. The RBF kernel computes the similarity between pairs of data points using the following formula:

$$
K(x, y) = \exp\left(-\gamma \|x - y\|^2\right)
$$

where x and y are data points and γ is a parameter that controls the width of the kernel.

In sum: by using the RBF kernel, the SVM is able to separate data that is not linearly separable in the original feature space, making it an effective tool for non-linear classification.

## Interpreting the Model's Results
We interpret the model using standard classification metrics: accuracy, precision, recall, and F1 score. Since the cost of failing to classify extremist content is high, we prioritize recall. We also will examine the confusion matrix to understand the nature of false positives and false negatives. 

## Our Team's Twist
We incorporated two major twists to improve model performance and interpretability:
1) First Twist: Tuning Lesser-Known SVM Hyperparameters Through Grid Search
    - C = The regularization parameter determines how much we penalize misclassifications on the training data. We experimented with the following values: [1.0, 0.1, 0.01, 0.001, 0.0001, 1e-20]
    - class_weight='balanced' or not: This parameter adjusts the penalty for misclassifying underrepresented classes. This may prove to be critical for our dataset where extremist tweets are slightly less common.
    - gamma: This parameter controls the influence of individual training points. We experimented with values like 0.01, 0.1, and 1, beyond the usual 'scale' or 'auto'.
    - shrinking: The shrinking parameter disables the shrinking heuristic, which can improve optimization time and stability on small datasets. We experimented with shrinking turned on and off.
*NOTE: While our primary metric for analysis is recall, when we optimized recall in the grid search, the "best" set of parameters was really the set that resulted in a totally overfit model that predicted all samples as the majority class (0 = not extremist). To address this, we ran grid search on the recall_weighted metric, which accounts for class imbalance by weighting classes according to their frequency. 

RESULTS:
 - We found that the best combination of C, gamma, class_weight, and shrinking parameters for recall (our primary metric) was: 
{'C': 10, 'class_weight': 'balanced', 'gamma': 0.1, 'shrinking': True}
- And this resulted in a weighted recall score of 0.9033

- We then evaluated our model on the test set and got the following results: 
              precision    recall  f1-score   support

           0       0.86      0.95      0.90       434
           1       0.95      0.86      0.90       466

    accuracy                           0.90       900
   macro avg       0.90      0.90      0.90       900
weighted avg       0.90      0.90      0.90       900

- You can see these results for yourself by running the nonlinear_model.py script.

2) Second Twist: Feature Selection Using SelectKBest
- Our team was concerned that some of our text analysis features (e.g., punctuation score, capitalization score) might not provide relevant information to our model, potentially adding noise and reducing interpretability. To address this, we decided to use SelectKBest with mutual information to identify the most informative features before training our final RBF SVM model. We currently have 7 features in our model, so we ran the model on the top 5 features, then on the top 6 features, and finally on all 7 features. 

RESULTS: 
    - Top 5 features selected: ['toxicity', 'punctuation', 'capitalization', 'repetition', 'keywords']
    Best weighted recall score: 0.8985
              precision    recall  f1-score   support

           0       0.84      0.94      0.89       434
           1       0.94      0.83      0.88       466

    accuracy                           0.88       900
   macro avg       0.89      0.89      0.88       900
weighted avg       0.89      0.88      0.88       900

* We then ran it again, but this time choosing the top 6 features to see if the model performance improved:
- Top 6 features selected: ['toxicity', 'subjectivity', 'punctuation', 'capitalization', 'repetition', 'keywords']
Best weighted recall score: 0.9052
              precision    recall  f1-score   support

           0       0.86      0.95      0.90       434
           1       0.95      0.86      0.90       466

    accuracy                           0.90       900
   macro avg       0.91      0.90      0.90       900
weighted avg       0.91      0.90      0.90       900

* The results with these 6 features give us the same results as when we include the final excluded feature: profanity
- This indicates that the profanity feature may not be very informative in our model. 

from sklearn.svm import SVC
from archives.data_utils import load_features_and_data
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import pickle

# Load in split & scaled training and test data
X_train, X_test, y_train, y_test = load_features_and_data()

# Use SelectKBest to find top 5 features on entire training set
selector = SelectKBest(score_func=mutual_info_classif, k=7)
X_train_selected = selector.fit_transform(X_train, y_train)

# Use same selector so only top 5 features in test set
X_test_selected = selector.transform(X_test)

# Define list of nonlinear kernel functions to test
kernel_functions = ['poly', 'rbf', 'sigmoid']

# use 5-fold cross validation to select the kernel with best recall (our primary metric)
for kernel in kernel_functions:
    svm_classifier = SVC(kernel=kernel)
    scores = cross_validate(
        svm_classifier, 
        X_train_selected, 
        y_train, 
        cv=5, 
        scoring=['recall', 'accuracy', 'f1'],  # include recall, accuracy, and f1 as scoring metrics
        return_train_score=False  # We only care about test scores here
    )
    
    # Print out the mean scores for each metric
    print(f"{kernel} kernel:")
    print(f"  Mean recall = {scores['test_recall'].mean():.4f}")
    print(f"  Mean accuracy = {scores['test_accuracy'].mean():.4f}")
    print(f"  Mean F1 score = {scores['test_f1'].mean():.4f}")
    print()

# We find that the RBF kernel results in the highest mean recall, accuracy, and F1 score,
# So we will employ the RBF kernel.

# Now we want to tune our hyperparameters using GridSearch
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'gamma': [0.001, 0.01, 0.1, 1],
    'class_weight': [None, 'balanced'],
    'shrinking': [True, False]         
}
# Initialize RBF SVM
svm_rbf = SVC(kernel='rbf', probability=True)

# Run grid search with 5-fold CV, optimizing for recall
grid_search = GridSearchCV(
    svm_rbf,
    param_grid,
    cv=5,
    scoring='recall_weighted',
    return_train_score=True,
    n_jobs=-1, # use all available CPI cores in parallel during training & evaluation
    verbose=2,
    refit=True
)

# Fit model with best hyperparameter set
grid_search.fit(X_train_selected, y_train)

# Best parameters and score
print("Best parameters from GridSearchCV:")
print(grid_search.best_params_)
print(f"Best weighted recall score: {grid_search.best_score_:.4f}")

# Save the best trained model to a file
with open("nonlinear_model.pkl", "wb") as f:
    pickle.dump(grid_search.best_estimator_, f)

# Evaluate on the test set
y_pred = grid_search.best_estimator_.predict(X_test_selected)
print(classification_report(y_test, y_pred))

# Original Set of Features (UPDATE WHEN TF IDF AND EMBEDDINGS ADDED)
feature_names = [
    "toxicity",
    "subjectivity",
    "profanity",
    "punctuation",
    "capitalization",
    "repetition",
    "keywords",
]
mask = selector.get_support()
top5_features = [name for name, selected in zip(feature_names, mask) if selected]
print("Top 5 features selected:", top5_features)
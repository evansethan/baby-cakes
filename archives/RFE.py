from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from archives.data_utils import load_features_and_data
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.metrics import classification_report

# Load data
X_train, X_test, y_train, y_test = load_features_and_data()

# Initialize the base estimator for RFE from best grid search params in nonlinear_model.py
# {'C': 10, 'class_weight': 'balanced', 'gamma': 0.1, 'shrinking': True}
base_svm = SVC(kernel='rbf', C=10, gamma=0.1 , class_weight='balanced', shrinking=True) 

# Define RFE selector - select 5 out of our 7 features (UPDATE THIS WHEN WE HAVE MORE FEATURES !!)
rfe_selector = RFE(estimator=base_svm, n_features_to_select=5, step=1)  

# Fit RFE on training data to select features
rfe_selector = rfe_selector.fit(X_train, y_train)

# Transform training and test data to selected features only
X_train_rfe = rfe_selector.transform(X_train)
X_test_rfe = rfe_selector.transform(X_test)

# Now do grid search or cross-validation on the reduced feature set
svm = SVC()

param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'gamma': [0.001, 0.01, 0.1, 1],
    'class_weight': [None, 'balanced'],
    'shrinking': [True, False]
}

grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='recall_weighted', n_jobs=-1)
grid_search.fit(X_train_rfe, y_train)

# Predict on test set with best estimator
y_pred = grid_search.best_estimator_.predict(X_test_rfe)

# Evaluate
print(classification_report(y_test, y_pred))

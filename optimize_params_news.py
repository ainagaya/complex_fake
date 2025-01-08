from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200, 500],            # Number of trees
    'max_depth': [10, 20, 30, None],           # Tree depth
    'min_samples_split': [2, 5, 10],           # Minimum samples to split a node
    'min_samples_leaf': [1, 2, 4],             # Minimum samples at a leaf node
    'max_features': ['sqrt', 'log2']           # Number of features to consider for split
}

# Initialize RandomForestClassifier
rf = RandomForestClassifier(random_state=42)

# Set up GridSearchCV
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,                                      # 5-fold cross-validation
    scoring='accuracy',                        # Metric for evaluation
    verbose=2,                                 # Show progress
    n_jobs=-1                                  # Use all CPU cores
)

# Fit the grid search on training data
grid_search.fit(X_train_vectorized, y_train)

# Get best parameters and best score
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validated Accuracy:", grid_search.best_score_)

# Evaluate the best model on test data
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test_vectorized)

# Classification report
print("\nTest Set Performance:")
print(classification_report(y_test, y_pred))

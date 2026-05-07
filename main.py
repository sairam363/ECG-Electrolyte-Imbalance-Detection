import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import numpy as np
import joblib

# Load the PTXLB dataset (update the path to your dataset)
# Example: df = pd.read_csv("path_to_ptxlb_data.csv")

# Example of how to create a dataframe for demonstration (replace with your actual dataset loading code)
# Example: Let's assume the target column is called 'target' and features are in the rest of the columns
# For now, we will generate a sample dataframe as an example.
np.random.seed(42)
df = pd.DataFrame({
    'feature1': np.random.randn(1000),
    'feature2': np.random.randn(1000),
    'feature3': np.random.randn(1000),
    'feature4': np.random.randn(1000),
    'target': np.random.choice([0, 1], size=1000, p=[0.9, 0.1])  # Imbalanced dataset (90% class 0, 10% class 1)
})

# Check class distribution
print("Original class distribution:\n", df['target'].value_counts())

# Split the data into features (X) and target (y)
X = df.drop('target', axis=1)  # Features
y = df['target']  # Target variable

# Split data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE (Oversampling) to handle class imbalance
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Feature selection (Optional: Select best features based on statistical tests)
selector = SelectKBest(score_func=f_classif, k='all')
X_selected = selector.fit_transform(X_resampled, y_resampled)

# Get the feature scores to check the relevance
selected_features = selector.scores_
print("Feature scores:", selected_features)

# Initialize RandomForestClassifier with class weights to handle imbalance
model = RandomForestClassifier(class_weight='balanced', random_state=42)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_selected, y_resampled)

# Get the best model after hyperparameter tuning
best_model = grid_search.best_estimator_

# Evaluate the model on the test set
X_test_selected = selector.transform(X_test)  # Apply the same feature selection to the test set
y_pred = best_model.predict(X_test_selected)

# Confusion matrix and evaluation metrics
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

# Display results
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nAccuracy:", accuracy)
print("F1 Score (Weighted):", f1)


# Save the best model after training
joblib.dump(best_model, 'model.pkl')


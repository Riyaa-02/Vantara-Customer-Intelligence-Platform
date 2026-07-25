# ==========================================================
# Day 6 - Baseline Churn Models
# Logistic Regression and Decision Tree
# ==========================================================

import os
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ==========================================================
# Create Required Directories
# ==========================================================

os.makedirs("models", exist_ok=True)
os.makedirs("outputs/reports", exist_ok=True)


# ==========================================================
# Load Datasets
# ==========================================================

print("\nLoading processed datasets...")

X_train = pd.read_csv("data/final/X_train.csv")
X_validation = pd.read_csv("data/final/X_validation.csv")
X_test = pd.read_csv("data/final/X_test.csv")

y_train = pd.read_csv(
    "data/final/y_train.csv"
).squeeze()

y_validation = pd.read_csv(
    "data/final/y_validation.csv"
).squeeze()

y_test = pd.read_csv(
    "data/final/y_test.csv"
).squeeze()


print("Training Set   :", X_train.shape)
print("Validation Set :", X_validation.shape)
print("Testing Set    :", X_test.shape)


# ==========================================================
# Evaluation Function
# ==========================================================

def evaluate_model(
    model_name,
    model,
    X_data,
    y_data
):

    predictions = model.predict(X_data)

    probabilities = (
        model.predict_proba(X_data)[:, 1]
    )

    results = {
        "Model": model_name,
        "Accuracy": accuracy_score(
            y_data,
            predictions
        ),
        "Precision": precision_score(
            y_data,
            predictions
        ),
        "Recall": recall_score(
            y_data,
            predictions
        ),
        "F1_Score": f1_score(
            y_data,
            predictions
        ),
        "ROC_AUC": roc_auc_score(
            y_data,
            probabilities
        )
    }

    return results


# ==========================================================
# Logistic Regression
# ==========================================================

print("\nTraining Logistic Regression...")

logistic_model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)

logistic_model.fit(
    X_train,
    y_train
)

print(
    "Logistic Regression training completed."
)


# ==========================================================
# Decision Tree
# ==========================================================

print("\nTraining Decision Tree...")

decision_tree_model = DecisionTreeClassifier(
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42
)

decision_tree_model.fit(
    X_train,
    y_train
)

print(
    "Decision Tree training completed."
)


# ==========================================================
# Validation Evaluation
# ==========================================================

print(
    "\nEvaluating baseline models "
    "on validation dataset..."
)

validation_results = []

validation_results.append(
    evaluate_model(
        "Logistic Regression",
        logistic_model,
        X_validation,
        y_validation
    )
)

validation_results.append(
    evaluate_model(
        "Decision Tree",
        decision_tree_model,
        X_validation,
        y_validation
    )
)


validation_df = pd.DataFrame(
    validation_results
)

print(
    "\nValidation Model Comparison"
)

print("-" * 70)

print(
    validation_df.round(4).to_string(
        index=False
    )
)


# ==========================================================
# Test Evaluation
# ==========================================================

print(
    "\nEvaluating baseline models "
    "on test dataset..."
)

test_results = []

test_results.append(
    evaluate_model(
        "Logistic Regression",
        logistic_model,
        X_test,
        y_test
    )
)

test_results.append(
    evaluate_model(
        "Decision Tree",
        decision_tree_model,
        X_test,
        y_test
    )
)


test_df = pd.DataFrame(
    test_results
)

print(
    "\nTest Model Comparison"
)

print("-" * 70)

print(
    test_df.round(4).to_string(
        index=False
    )
)


# ==========================================================
# Save Results
# ==========================================================

validation_df.to_csv(
    "outputs/reports/"
    "baseline_validation_results.csv",
    index=False
)

test_df.to_csv(
    "outputs/reports/"
    "baseline_test_results.csv",
    index=False
)


# ==========================================================
# Save Models
# ==========================================================

print("\nSaving trained models...")

joblib.dump(
    logistic_model,
    "models/logistic_regression_model.pkl"
)

joblib.dump(
    decision_tree_model,
    "models/decision_tree_model.pkl"
)

print("Models saved successfully.")


# ==========================================================
# Completion
# ==========================================================

print(
    "\nDay 6 Baseline Models "
    "completed successfully."
)
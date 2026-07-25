# ==========================================================
# Day 7 - Random Forest and XGBoost
# Cross-Validated Hyperparameter Tuning
# ==========================================================

import os
import time
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from xgboost import XGBClassifier


# ==========================================================
# Create Required Directories
# ==========================================================

os.makedirs("models", exist_ok=True)
os.makedirs("outputs/reports", exist_ok=True)


# ==========================================================
# Load Data
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

    return {
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


# ==========================================================
# Random Forest Hyperparameter Search
# ==========================================================

print("\nTuning Random Forest...")

rf_model = RandomForestClassifier(
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf_parameters = {
    "n_estimators": [
        100,
        200,
        300,
        500
    ],
    "max_depth": [
        None,
        10,
        20,
        30
    ],
    "min_samples_split": [
        2,
        5,
        10
    ],
    "min_samples_leaf": [
        1,
        2,
        4
    ],
    "max_features": [
        "sqrt",
        "log2"
    ]
}


rf_search = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=rf_parameters,
    n_iter=20,
    scoring="roc_auc",
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=1
)


rf_start = time.time()

rf_search.fit(
    X_train,
    y_train
)

rf_time = time.time() - rf_start

best_rf_model = rf_search.best_estimator_


print("\nBest Random Forest Parameters:")
print(rf_search.best_params_)

print(
    "Best Random Forest CV ROC-AUC:",
    round(
        rf_search.best_score_,
        4
    )
)

print(
    "Random Forest Training Time:",
    round(
        rf_time,
        2
    ),
    "seconds"
)


# ==========================================================
# XGBoost Hyperparameter Search
# ==========================================================

print("\nTuning XGBoost...")

xgb_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)


xgb_parameters = {
    "n_estimators": [
        100,
        200,
        300,
        500
    ],
    "max_depth": [
        3,
        5,
        7,
        10
    ],
    "learning_rate": [
        0.01,
        0.05,
        0.1,
        0.2
    ],
    "subsample": [
        0.7,
        0.8,
        1.0
    ],
    "colsample_bytree": [
        0.7,
        0.8,
        1.0
    ]
}


xgb_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=xgb_parameters,
    n_iter=20,
    scoring="roc_auc",
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=1
)


xgb_start = time.time()

xgb_search.fit(
    X_train,
    y_train
)

xgb_time = time.time() - xgb_start

best_xgb_model = xgb_search.best_estimator_


print("\nBest XGBoost Parameters:")
print(xgb_search.best_params_)

print(
    "Best XGBoost CV ROC-AUC:",
    round(
        xgb_search.best_score_,
        4
    )
)

print(
    "XGBoost Training Time:",
    round(
        xgb_time,
        2
    ),
    "seconds"
)


# ==========================================================
# Validation Evaluation
# ==========================================================

print(
    "\nEvaluating tuned models "
    "on validation dataset..."
)

validation_results = []

validation_results.append(
    evaluate_model(
        "Random Forest",
        best_rf_model,
        X_validation,
        y_validation
    )
)

validation_results.append(
    evaluate_model(
        "XGBoost",
        best_xgb_model,
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

print("-" * 75)

print(
    validation_df.round(4).to_string(
        index=False
    )
)


# ==========================================================
# Test Evaluation
# ==========================================================

print(
    "\nEvaluating tuned models "
    "on test dataset..."
)

test_results = []

rf_test_results = evaluate_model(
    "Random Forest",
    best_rf_model,
    X_test,
    y_test
)

rf_test_results[
    "Training_Time_Seconds"
] = round(
    rf_time,
    2
)

test_results.append(
    rf_test_results
)


xgb_test_results = evaluate_model(
    "XGBoost",
    best_xgb_model,
    X_test,
    y_test
)

xgb_test_results[
    "Training_Time_Seconds"
] = round(
    xgb_time,
    2
)

test_results.append(
    xgb_test_results
)


test_df = pd.DataFrame(
    test_results
)

print(
    "\nTest Model Comparison"
)

print("-" * 90)

print(
    test_df.round(4).to_string(
        index=False
    )
)


# ==========================================================
# Save Models
# ==========================================================

print("\nSaving tuned models...")

joblib.dump(
    best_rf_model,
    "models/"
    "random_forest_tuned_model.pkl"
)

joblib.dump(
    best_xgb_model,
    "models/"
    "xgboost_tuned_model.pkl"
)

print("Tuned models saved successfully.")


# ==========================================================
# Save Comparison Results
# ==========================================================

validation_df.to_csv(
    "outputs/reports/"
    "day7_validation_comparison.csv",
    index=False
)

test_df.to_csv(
    "outputs/reports/"
    "day7_test_comparison.csv",
    index=False
)


# ==========================================================
# Save Best Parameters
# ==========================================================

parameter_results = pd.DataFrame(
    [
        {
            "Model": "Random Forest",
            "Best_Parameters": str(
                rf_search.best_params_
            ),
            "Best_CV_ROC_AUC": (
                rf_search.best_score_
            )
        },
        {
            "Model": "XGBoost",
            "Best_Parameters": str(
                xgb_search.best_params_
            ),
            "Best_CV_ROC_AUC": (
                xgb_search.best_score_
            )
        }
    ]
)


parameter_results.to_csv(
    "outputs/reports/"
    "day7_best_parameters.csv",
    index=False
)


# ==========================================================
# Completion
# ==========================================================

print(
    "\nDay 7 Random Forest and XGBoost "
    "tuning completed successfully."
)
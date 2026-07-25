# ==========================================================
# Day 8 - LightGBM and SVM Comparison
# ==========================================================

import os
import time
import joblib
import pandas as pd

from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
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
# LightGBM Hyperparameter Tuning
# ==========================================================

print("\nTuning LightGBM...")

lightgbm_model = LGBMClassifier(
    random_state=42,
    class_weight="balanced",
    verbosity=-1,
    n_jobs=1,
    deterministic=True,
    force_col_wise=True
)

lightgbm_parameters = {
    "n_estimators": [
        100,
        200,
        300,
        500
    ],
    "learning_rate": [
        0.01,
        0.05,
        0.1,
        0.2
    ],
    "num_leaves": [
        15,
        31,
        63
    ],
    "max_depth": [
        -1,
        5,
        10,
        20
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

lightgbm_search = RandomizedSearchCV(
    estimator=lightgbm_model,
    param_distributions=lightgbm_parameters,
    n_iter=20,
    scoring="roc_auc",
    cv=5,
    random_state=42,
    n_jobs=1,
    verbose=1
)

lightgbm_start = time.time()

lightgbm_search.fit(
    X_train,
    y_train
)

lightgbm_time = (
    time.time() - lightgbm_start
)

best_lightgbm_model = (
    lightgbm_search.best_estimator_
)

print("\nBest LightGBM Parameters:")
print(lightgbm_search.best_params_)

print(
    "Best LightGBM CV ROC-AUC:",
    round(
        lightgbm_search.best_score_,
        4
    )
)

print(
    "LightGBM Training Time:",
    round(
        lightgbm_time,
        2
    ),
    "seconds"
)


# ==========================================================
# SVM Hyperparameter Tuning
# ==========================================================

print("\nTuning SVM...")

svm_model = SVC(
    probability=True,
    class_weight="balanced",
    random_state=42
)

svm_parameters = {
    "C": [
        0.1,
        1,
        10,
        100
    ],
    "kernel": [
        "rbf",
        "linear"
    ],
    "gamma": [
        "scale",
        "auto"
    ]
}

svm_search = RandomizedSearchCV(
    estimator=svm_model,
    param_distributions=svm_parameters,
    n_iter=10,
    scoring="roc_auc",
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

svm_start = time.time()

svm_search.fit(
    X_train,
    y_train
)

svm_time = (
    time.time() - svm_start
)

best_svm_model = (
    svm_search.best_estimator_
)

print("\nBest SVM Parameters:")
print(svm_search.best_params_)

print(
    "Best SVM CV ROC-AUC:",
    round(
        svm_search.best_score_,
        4
    )
)

print(
    "SVM Training Time:",
    round(
        svm_time,
        2
    ),
    "seconds"
)


# ==========================================================
# Validation Evaluation
# ==========================================================

print(
    "\nEvaluating models "
    "on validation dataset..."
)

validation_results = []

validation_results.append(
    evaluate_model(
        "LightGBM",
        best_lightgbm_model,
        X_validation,
        y_validation
    )
)

validation_results.append(
    evaluate_model(
        "SVM",
        best_svm_model,
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
    "\nEvaluating models "
    "on test dataset..."
)

test_results = []

lightgbm_test_results = evaluate_model(
    "LightGBM",
    best_lightgbm_model,
    X_test,
    y_test
)

lightgbm_test_results[
    "Training_Time_Seconds"
] = round(
    lightgbm_time,
    2
)

test_results.append(
    lightgbm_test_results
)


svm_test_results = evaluate_model(
    "SVM",
    best_svm_model,
    X_test,
    y_test
)

svm_test_results[
    "Training_Time_Seconds"
] = round(
    svm_time,
    2
)

test_results.append(
    svm_test_results
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

print("\nSaving trained models...")

joblib.dump(
    best_lightgbm_model,
    "models/"
    "lightgbm_tuned_model.pkl"
)

joblib.dump(
    best_svm_model,
    "models/"
    "svm_tuned_model.pkl"
)

print("Models saved successfully.")


# ==========================================================
# Save Results
# ==========================================================

validation_df.to_csv(
    "outputs/reports/"
    "day8_validation_comparison.csv",
    index=False
)

test_df.to_csv(
    "outputs/reports/"
    "day8_test_comparison.csv",
    index=False
)


# ==========================================================
# Save Best Parameters
# ==========================================================

parameter_results = pd.DataFrame(
    [
        {
            "Model": "LightGBM",
            "Best_Parameters": str(
                lightgbm_search.best_params_
            ),
            "Best_CV_ROC_AUC": (
                lightgbm_search.best_score_
            )
        },
        {
            "Model": "SVM",
            "Best_Parameters": str(
                svm_search.best_params_
            ),
            "Best_CV_ROC_AUC": (
                svm_search.best_score_
            )
        }
    ]
)

parameter_results.to_csv(
    "outputs/reports/"
    "day8_best_parameters.csv",
    index=False
)


# ==========================================================
# Completion
# ==========================================================

print(
    "\nDay 8 LightGBM and SVM "
    "comparison completed successfully."
)
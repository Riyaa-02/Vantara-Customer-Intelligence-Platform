# ==========================================================
# VANTARA CUSTOMER INTELLIGENCE PLATFORM
# DAY 8 - CUSTOMER LIFETIME VALUE REGRESSION
# ==========================================================

import os
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==========================================================
# Configuration
# ==========================================================

RANDOM_SEED = 42

CUTOFF_DATE = pd.Timestamp("2011-06-01")
CLV_HORIZON_DAYS = 180

FEATURE_PATH = (
    "data/processed/customer_features.csv"
)

TRANSACTION_PATH = (
    "data/interim/cleaned_data.csv"
)

MODEL_PATH = (
    "models/clv_regression_model.pkl"
)

RESULTS_PATH = (
    "outputs/reports/"
    "day8_clv_regression_results.csv"
)

DATASET_PATH = (
    "data/final/"
    "customer_clv_dataset.csv"
)


os.makedirs("models", exist_ok=True)
os.makedirs("outputs/reports", exist_ok=True)
os.makedirs("data/final", exist_ok=True)


# ==========================================================
# 1. Load Data
# ==========================================================

print("\nLoading customer features...")

features = pd.read_csv(
    FEATURE_PATH
)

transactions = pd.read_csv(
    TRANSACTION_PATH
)

transactions["InvoiceDate"] = pd.to_datetime(
    transactions["InvoiceDate"]
)

transactions["Total"] = (
    transactions["Quantity"]
    * transactions["Price"]
)

print(
    "Customer Feature Shape:",
    features.shape
)

print(
    "Transaction Shape:",
    transactions.shape
)


# ==========================================================
# 2. Create Future CLV Target
# ==========================================================

print("\nCreating future CLV target...")

future_end_date = (
    CUTOFF_DATE
    + pd.Timedelta(
        days=CLV_HORIZON_DAYS
    )
)

future_transactions = transactions[
    (
        transactions["InvoiceDate"]
        >= CUTOFF_DATE
    )
    &
    (
        transactions["InvoiceDate"]
        < future_end_date
    )
].copy()


future_clv = (
    future_transactions
    .groupby("Customer ID")["Total"]
    .sum()
    .rename("Future_CLV")
)


print(
    "Future CLV Window:",
    CUTOFF_DATE.date(),
    "to",
    future_end_date.date()
)


# ==========================================================
# 3. Merge Historical Features With Future Target
# ==========================================================

clv_data = features.merge(
    future_clv,
    on="Customer ID",
    how="left"
)

# Customers with no future purchases
# have zero observed future value.
clv_data["Future_CLV"] = (
    clv_data["Future_CLV"]
    .fillna(0)
)


# ==========================================================
# 4. Handle Extreme Future Revenue Values
# ==========================================================

print("\nHandling extreme CLV values...")

lower_limit = 0

upper_limit = (
    clv_data["Future_CLV"]
    .quantile(0.99)
)

clv_data["Future_CLV"] = (
    clv_data["Future_CLV"]
    .clip(
        lower=lower_limit,
        upper=upper_limit
    )
)

print(
    "Future CLV 99th Percentile:",
    round(
        upper_limit,
        2
    )
)


# ==========================================================
# 5. Prepare Features
# ==========================================================

print("\nPreparing regression features...")

drop_columns = [
    "Customer ID",
    "Churn",
    "Predicted_CLV",
    "Future_CLV"
]


X = clv_data.drop(
    columns=drop_columns,
    errors="ignore"
)

y = clv_data[
    "Future_CLV"
]


# One-hot encode Country
X = pd.get_dummies(
    X,
    columns=["Country"],
    drop_first=True
)


# Ensure numeric feature matrix
X = X.astype(float)


print(
    "Regression Feature Shape:",
    X.shape
)

print(
    "Target Shape:",
    y.shape
)


# ==========================================================
# 6. Train / Validation / Test Split
# ==========================================================

print("\nSplitting regression dataset...")

X_train, X_temp, y_train, y_temp = (
    train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_SEED
    )
)

X_validation, X_test, y_validation, y_test = (
    train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_SEED
    )
)


print(
    "Training Set   :",
    X_train.shape
)

print(
    "Validation Set :",
    X_validation.shape
)

print(
    "Testing Set    :",
    X_test.shape
)


# ==========================================================
# 7. Random Forest Regression Tuning
# ==========================================================

print(
    "\nTuning Random Forest "
    "CLV Regressor..."
)


clv_model = RandomForestRegressor(
    random_state=RANDOM_SEED,
    n_jobs=-1
)


parameter_grid = {

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
        "log2",
        0.7
    ]
}


search = RandomizedSearchCV(

    estimator=clv_model,

    param_distributions=parameter_grid,

    n_iter=20,

    scoring=(
        "neg_mean_squared_error"
    ),

    cv=5,

    random_state=RANDOM_SEED,

    n_jobs=-1,

    verbose=1
)


start_time = time.time()


search.fit(
    X_train,
    y_train
)


training_time = (
    time.time()
    - start_time
)


best_model = (
    search.best_estimator_
)


print(
    "\nBest CLV Model Parameters:"
)

print(
    search.best_params_
)

print(
    "Training Time:",
    round(
        training_time,
        2
    ),
    "seconds"
)


# ==========================================================
# 8. Evaluation Function
# ==========================================================

def evaluate_regression(
    dataset_name,
    model,
    X_data,
    y_data
):

    predictions = model.predict(
        X_data
    )

    mae = mean_absolute_error(
        y_data,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_data,
            predictions
        )
    )

    r2 = r2_score(
        y_data,
        predictions
    )

    return {

        "Dataset":
            dataset_name,

        "MAE":
            mae,

        "RMSE":
            rmse,

        "R2":
            r2
    }


# ==========================================================
# 9. Validation Evaluation
# ==========================================================

print(
    "\nEvaluating CLV model "
    "on validation dataset..."
)


validation_results = (
    evaluate_regression(

        "Validation",

        best_model,

        X_validation,

        y_validation
    )
)


print("\nValidation Results")
print("-" * 45)

print(
    "MAE  :",
    round(
        validation_results["MAE"],
        2
    )
)

print(
    "RMSE :",
    round(
        validation_results["RMSE"],
        2
    )
)

print(
    "R2   :",
    round(
        validation_results["R2"],
        4
    )
)


# ==========================================================
# 10. Final Test Evaluation
# ==========================================================

print(
    "\nEvaluating CLV model "
    "on test dataset..."
)


test_results = (
    evaluate_regression(

        "Test",

        best_model,

        X_test,

        y_test
    )
)


print("\nTest Results")
print("-" * 45)

print(
    "MAE  :",
    round(
        test_results["MAE"],
        2
    )
)

print(
    "RMSE :",
    round(
        test_results["RMSE"],
        2
    )
)

print(
    "R2   :",
    round(
        test_results["R2"],
        4
    )
)


# ==========================================================
# 11. Save Model
# ==========================================================

print("\nSaving CLV regression model...")

joblib.dump(
    best_model,
    MODEL_PATH
)

print(
    "Model saved successfully."
)


# ==========================================================
# 12. Save Model Features
# ==========================================================

joblib.dump(
    list(X.columns),
    "models/clv_feature_names.pkl"
)


# ==========================================================
# 13. Generate CLV Predictions
# ==========================================================

print(
    "\nGenerating predicted CLV "
    "for all customers..."
)


clv_data[
    "ML_Predicted_CLV"
] = best_model.predict(
    X
)


# Prevent negative predicted value
clv_data[
    "ML_Predicted_CLV"
] = (
    clv_data[
        "ML_Predicted_CLV"
    ]
    .clip(lower=0)
)


clv_data.to_csv(
    DATASET_PATH,
    index=False
)


# ==========================================================
# 14. Save Results
# ==========================================================

results_df = pd.DataFrame(
    [
        validation_results,
        test_results
    ]
)


results_df[
    "Training_Time_Seconds"
] = round(
    training_time,
    2
)


results_df.to_csv(
    RESULTS_PATH,
    index=False
)


# ==========================================================
# 15. Summary
# ==========================================================

print(
    "\nCLV Regression Summary"
)

print("-" * 50)

print(
    results_df
    .round(4)
    .to_string(
        index=False
    )
)


print(
    "\nSaved:"
)

print(
    MODEL_PATH
)

print(
    RESULTS_PATH
)

print(
    DATASET_PATH
)


print(
    "\nDay 8 CLV Regression "
    "completed successfully."
)
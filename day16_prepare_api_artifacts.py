"""
======================================================================
VANTARA CUSTOMER INTELLIGENCE PLATFORM
DAY 16 : PREPARE API PREPROCESSING ARTIFACTS
======================================================================
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# Configuration
# ============================================================

RANDOM_SEED = 42

SOURCE_DATA_PATH = "data/processed/customer_features.csv"
EXISTING_TEST_PATH = "data/final/X_test.csv"

SCALER_OUTPUT_PATH = "models/ann_input_scaler.pkl"
NUMERIC_FEATURES_OUTPUT_PATH = "models/ann_numeric_features.pkl"
COUNTRIES_OUTPUT_PATH = "models/ann_country_values.pkl"
FEATURE_NAMES_OUTPUT_PATH = "models/ann_feature_names.pkl"


# ============================================================
# Create Required Directory
# ============================================================

os.makedirs("models", exist_ok=True)


# ============================================================
# Load Original Customer Features
# ============================================================

print("=" * 70)
print("DAY 16 : PREPARING API ARTIFACTS")
print("=" * 70)

print("\nLoading original customer feature dataset...")

df = pd.read_csv(SOURCE_DATA_PATH)

print("Dataset shape:", df.shape)


# ============================================================
# Apply Same Day 5 Cleaning
# ============================================================

if "Historical_Revenue" in df.columns:
    df.drop(
        columns=["Historical_Revenue"],
        inplace=True
    )

numeric_columns = df.select_dtypes(
    include=["int64", "float64"]
).columns

for column in numeric_columns:
    df[column] = df[column].fillna(
        df[column].median()
    )

categorical_columns = df.select_dtypes(
    include=["object"]
).columns

for column in categorical_columns:
    df[column] = df[column].fillna(
        df[column].mode()[0]
    )


# ============================================================
# Save Supported Country Values
# ============================================================

country_values = sorted(
    df["Country"]
    .astype(str)
    .unique()
    .tolist()
)

joblib.dump(
    country_values,
    COUNTRIES_OUTPUT_PATH
)


# ============================================================
# Apply Same Country Encoding
# ============================================================

df = pd.get_dummies(
    df,
    columns=["Country"],
    drop_first=True
)


# ============================================================
# Separate Features and Target
# ============================================================

X = df.drop(
    columns=["Churn"]
)

y = df["Churn"]

if "Customer ID" in X.columns:
    X = X.drop(
        columns=["Customer ID"]
    )


# ============================================================
# Apply Exact Same Dataset Split
# ============================================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=RANDOM_SEED,
    stratify=y
)

X_validation, X_test, y_validation, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=RANDOM_SEED,
    stratify=y_temp
)


# ============================================================
# Identify and Save Numerical Features
# ============================================================

numeric_features = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

print("\nNumerical feature count:", len(numeric_features))

for feature in numeric_features:
    print("-", feature)


# ============================================================
# Fit Correct ANN Input Scaler
# ============================================================

scaler = StandardScaler()

X_train.loc[:, numeric_features] = scaler.fit_transform(
    X_train[numeric_features]
)

X_validation.loc[:, numeric_features] = scaler.transform(
    X_validation[numeric_features]
)

X_test.loc[:, numeric_features] = scaler.transform(
    X_test[numeric_features]
)


# ============================================================
# Load Saved ANN Feature Order
# ============================================================

ann_feature_names = joblib.load(
    FEATURE_NAMES_OUTPUT_PATH
)

X_train = X_train.reindex(
    columns=ann_feature_names,
    fill_value=0
)

X_validation = X_validation.reindex(
    columns=ann_feature_names,
    fill_value=0
)

X_test = X_test.reindex(
    columns=ann_feature_names,
    fill_value=0
)


# ============================================================
# Validate Against Existing Day 5 X_test
# ============================================================

print("\nValidating reconstructed preprocessing...")

existing_test = pd.read_csv(
    EXISTING_TEST_PATH
)

existing_test = existing_test.reindex(
    columns=ann_feature_names
)

reconstructed_values = (
    X_test.astype("float32").to_numpy()
)

existing_values = (
    existing_test.astype("float32").to_numpy()
)

maximum_difference = np.max(
    np.abs(
        reconstructed_values - existing_values
    )
)

print(
    "Maximum difference:",
    maximum_difference
)

if not np.allclose(
    reconstructed_values,
    existing_values,
    rtol=1e-5,
    atol=1e-6
):
    raise ValueError(
        "Validation failed. Reconstructed preprocessing "
        "does not match the saved X_test dataset."
    )


# ============================================================
# Save Correct API Artifacts
# ============================================================

joblib.dump(
    scaler,
    SCALER_OUTPUT_PATH
)

joblib.dump(
    numeric_features,
    NUMERIC_FEATURES_OUTPUT_PATH
)


# ============================================================
# Completion
# ============================================================

print("\nValidation passed successfully.")

print("\nSaved API artifacts:")

print("-", SCALER_OUTPUT_PATH)
print("-", NUMERIC_FEATURES_OUTPUT_PATH)
print("-", COUNTRIES_OUTPUT_PATH)
print("-", FEATURE_NAMES_OUTPUT_PATH)

print("\n" + "=" * 70)
print("DAY 16 API ARTIFACT PREPARATION COMPLETED SUCCESSFULLY")
print("=" * 70)
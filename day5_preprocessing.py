"""
======================================================================
VANTARA CUSTOMER INTELLIGENCE PLATFORM
DAY 5 : DATA PREPROCESSING
======================================================================
"""

import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("=" * 70)
print("DAY 5 : DATA PREPROCESSING")
print("=" * 70)

# ============================================================
# Configuration
# ============================================================

RANDOM_SEED = 42

os.makedirs("data/final", exist_ok=True)
os.makedirs("docs", exist_ok=True)

# ============================================================
# Load Feature Dataset
# ============================================================

df = pd.read_csv("data/processed/customer_features.csv")

print(f"\nLoaded {len(df):,} customers")
print(f"Columns : {len(df.columns)}")

# ============================================================
# Remove Duplicate Features
# ============================================================

print("\nRemoving duplicate features...")

if "Historical_Revenue" in df.columns:
    df.drop(columns=["Historical_Revenue"], inplace=True)
    print("Removed: Historical_Revenue")

# ============================================================
# Missing Values
# ============================================================

print("\nChecking missing values...")

missing = df.isnull().sum()

print(missing[missing > 0])

numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns

for col in numeric_columns:
    df[col] = df[col].fillna(df[col].median())

categorical_columns = df.select_dtypes(include=["object"]).columns

for col in categorical_columns:
    df[col] = df[col].fillna(df[col].mode()[0])

print("Missing values handled.")

# ============================================================
# Encode Country
# ============================================================

print("\nEncoding categorical features...")

df = pd.get_dummies(
    df,
    columns=["Country"],
    drop_first=True
)

print("Country encoded successfully.")

# # ============================================================
# Split Features & Target
# ============================================================

# Separate features and target
X = df.drop(columns=["Churn"])
y = df["Churn"]

# Customer ID is an identifier, not a predictive feature.
# Remove it from the ML feature set.
if "Customer ID" in X.columns:
    X = X.drop(columns=["Customer ID"])
    print("Removed: Customer ID (Identifier only)")

print("\nFeature Matrix Shape :", X.shape)
print("Target Shape         :", y.shape)

# ============================================================
# Train / Validation / Test Split
# ============================================================

print("\nSplitting dataset...")

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

print(f"Train      : {len(X_train):,}")
print(f"Validation : {len(X_validation):,}")
print(f"Test       : {len(X_test):,}")

# ============================================================
# Scale Numerical Features
# ============================================================

print("\nScaling numerical features...")

scaler = StandardScaler()

numeric_features = X_train.select_dtypes(
    include=["int64", "float64"]
).columns

X_train[numeric_features] = scaler.fit_transform(
    X_train[numeric_features]
)

X_validation[numeric_features] = scaler.transform(
    X_validation[numeric_features]
)

X_test[numeric_features] = scaler.transform(
    X_test[numeric_features]
)

print("Scaling completed.")
# ============================================================
# Save Processed Datasets
# ============================================================

print("\n" + "=" * 70)
print("SAVING DATASETS")
print("=" * 70)

X_train.to_csv(
    "data/final/X_train.csv",
    index=False
)

X_validation.to_csv(
    "data/final/X_validation.csv",
    index=False
)

X_test.to_csv(
    "data/final/X_test.csv",
    index=False
)

y_train.to_csv(
    "data/final/y_train.csv",
    index=False
)

y_validation.to_csv(
    "data/final/y_validation.csv",
    index=False
)

y_test.to_csv(
    "data/final/y_test.csv",
    index=False
)

final_dataset = pd.concat(
    [X, y],
    axis=1
)

final_dataset.to_csv(
    "data/final/customer_ml_dataset.csv",
    index=False
)

print("All datasets saved successfully.")

# ============================================================
# Class Distribution
# ============================================================

print("\n" + "=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

class_counts = y.value_counts().sort_index()

print(class_counts)

print()

class_percentage = (
    y.value_counts(normalize=True)
    .sort_index()
    * 100
)

print(class_percentage.round(2))

# ============================================================
# Dataset Information
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATASET SUMMARY")
print("=" * 70)

print(f"Rows    : {len(final_dataset):,}")
print(f"Columns : {len(final_dataset.columns)}")

print("\nFeature Names:\n")

for column in X.columns:
    print("-", column)

# ============================================================
# Preprocessing Report
# ============================================================

report = f"""
==============================================================
VANTARA CUSTOMER INTELLIGENCE PLATFORM

PREPROCESSING REPORT
==============================================================

Original Customers
------------------
{len(df):,}

Training Set
------------
{len(X_train):,}

Validation Set
--------------
{len(X_validation):,}

Testing Set
-----------
{len(X_test):,}

Total Features
--------------
{len(X.columns)}

Target
------
Churn

Class Distribution
------------------
Active Customers : {class_counts[0]}

Churned Customers : {class_counts[1]}

Active Percentage : {class_percentage[0]:.2f}%

Churn Percentage : {class_percentage[1]:.2f}%

Preprocessing Steps
-------------------
1. Removed duplicate feature (Historical_Revenue)

2. Filled missing numeric values using median

3. Filled missing categorical values using mode

4. One-Hot Encoded Country

5. Standardized numerical variables

6. Stratified Train / Validation / Test split

Dataset Ready for Machine Learning
"""

with open(
    "docs/preprocessing_report.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write(report)

print("\nSaved: docs/preprocessing_report.txt")

# ============================================================
# Validation Checks
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION CHECKS")
print("=" * 70)

assert X_train.isnull().sum().sum() == 0
print("PASS : Training set contains no missing values")

assert X_validation.isnull().sum().sum() == 0
print("PASS : Validation set contains no missing values")

assert X_test.isnull().sum().sum() == 0
print("PASS : Test set contains no missing values")

assert len(X_train.columns) == len(X_validation.columns)
assert len(X_train.columns) == len(X_test.columns)

print("PASS : Feature dimensions are consistent")

# ============================================================
# Completion
# ============================================================

print("\n" + "=" * 70)
print("DAY 5 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nGenerated Files")

print("data/final/customer_ml_dataset.csv")
print("data/final/X_train.csv")
print("data/final/X_validation.csv")
print("data/final/X_test.csv")
print("data/final/y_train.csv")
print("data/final/y_validation.csv")
print("data/final/y_test.csv")
print("docs/preprocessing_report.txt")
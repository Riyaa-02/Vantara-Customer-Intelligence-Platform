import os
import numpy as np
import pandas as pd

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler


# ==========================================================
# Configuration
# ==========================================================

INPUT_PATH = "data/processed/customer_features.csv"
OUTPUT_PATH = "outputs/reports/cleanup_vif_analysis.csv"

os.makedirs("outputs/reports", exist_ok=True)


# ==========================================================
# 1. Load Customer Features
# ==========================================================

print("\nLoading customer features...")

df = pd.read_csv(INPUT_PATH)

print("Dataset Shape:", df.shape)


# ==========================================================
# 2. Select Numeric Predictor Features
# ==========================================================

# Exclude:
# Customer ID -> identifier
# Country -> categorical
# Churn -> target variable

excluded_columns = [
    "Customer ID",
    "Country",
    "Churn"
]

X = df.drop(
    columns=excluded_columns,
    errors="ignore"
)

X = X.select_dtypes(
    include=[np.number]
).copy()

print(
    "Numeric Features:",
    X.shape[1]
)


# ==========================================================
# 3. Validate Data
# ==========================================================

print("\nValidating features...")

if X.isnull().any().any():

    print(
        "Missing values detected. "
        "Filling with median values..."
    )

    X = X.fillna(
        X.median()
    )


# Remove constant columns because VIF cannot be calculated
constant_columns = [

    column

    for column in X.columns

    if X[column].nunique() <= 1

]


if constant_columns:

    print(
        "Removing constant columns:",
        constant_columns
    )

    X = X.drop(
        columns=constant_columns
    )


print(
    "Feature validation completed."
)


# ==========================================================
# 4. Scale Features
# ==========================================================

print(
    "\nScaling features for VIF analysis..."
)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_scaled_df = pd.DataFrame(
    X_scaled,
    columns=X.columns
)


# ==========================================================
# 5. Calculate VIF
# ==========================================================

print(
    "\nCalculating Variance Inflation Factors..."
)

vif_results = []


for index, feature in enumerate(
    X_scaled_df.columns
):

    try:

        vif_value = variance_inflation_factor(
            X_scaled_df.values,
            index
        )

    except Exception:

        vif_value = np.inf


    vif_results.append({

        "Feature":
            feature,

        "VIF":
            vif_value

    })


vif_df = pd.DataFrame(
    vif_results
)


vif_df = vif_df.sort_values(
    "VIF",
    ascending=False
).reset_index(
    drop=True
)


# ==========================================================
# 6. Add Interpretation
# ==========================================================

def interpret_vif(value):

    if np.isinf(value):

        return "Perfect Multicollinearity"

    elif value >= 10:

        return "High Multicollinearity"

    elif value >= 5:

        return "Moderate Multicollinearity"

    else:

        return "Acceptable"


vif_df[
    "Interpretation"
] = vif_df[
    "VIF"
].apply(
    interpret_vif
)


# ==========================================================
# 7. Display Results
# ==========================================================

print(
    "\nVIF Analysis Results"
)

print("-" * 80)

print(

    vif_df

    .round({
        "VIF": 4
    })

    .to_string(
        index=False
    )

)


# ==========================================================
# 8. Summary
# ==========================================================

high_vif_features = vif_df[

    vif_df[
        "VIF"
    ] >= 10

]


moderate_vif_features = vif_df[

    (
        vif_df[
            "VIF"
        ] >= 5
    )

    &

    (
        vif_df[
            "VIF"
        ] < 10
    )

]


print(
    "\nVIF Summary"
)

print("-" * 50)

print(
    "Total Features Analyzed:",
    len(vif_df)
)

print(
    "High Multicollinearity Features:",
    len(
        high_vif_features
    )
)

print(
    "Moderate Multicollinearity Features:",
    len(
        moderate_vif_features
    )
)


# ==========================================================
# 9. Save Results
# ==========================================================

vif_df.to_csv(
    OUTPUT_PATH,
    index=False
)


print(
    "\nSaved to:",
    OUTPUT_PATH
)

print(
    "\nVIF cleanup analysis completed successfully."
)
import os
import numpy as np
import pandas as pd


# ==========================================================
# Configuration
# ==========================================================

INPUT_PATH = "data/processed/customer_features.csv"
OUTPUT_PATH = "outputs/reports/cleanup_data_validation.csv"

os.makedirs("outputs/reports", exist_ok=True)


# ==========================================================
# 1. Load Dataset
# ==========================================================

print("\nLoading customer features...")

df = pd.read_csv(INPUT_PATH)

print("Dataset Shape:", df.shape)


# ==========================================================
# 2. Validation Helper
# ==========================================================

validation_results = []


def add_result(check, status, details):

    validation_results.append({
        "Check": check,
        "Status": status,
        "Details": details
    })

    print(
        f"{status:<6} | "
        f"{check:<30} | "
        f"{details}"
    )


# ==========================================================
# 3. Schema Validation
# ==========================================================

print("\nRunning data validation checks...")
print("-" * 100)


required_columns = [
    "Customer ID",
    "Recency",
    "Frequency",
    "Monetary_Total",
    "Monetary_Average",
    "Total_Quantity",
    "Average_Basket_Size",
    "Average_Order_Value",
    "Unique_Products",
    "Average_Purchase_Gap",
    "Customer_Lifespan_Days",
    "Country",
    "Return_Rate",
    "Price_Sensitivity",
    "Historical_Revenue",
    "Predicted_CLV",
    "Engagement_Score",
    "Churn"
]


missing_columns = [

    column

    for column in required_columns

    if column not in df.columns

]


if not missing_columns:

    add_result(
        "Required Schema",
        "PASS",
        "All required columns are present."
    )

else:

    add_result(
        "Required Schema",
        "FAIL",
        f"Missing columns: {missing_columns}"
    )


# ==========================================================
# 4. Empty Dataset Check
# ==========================================================

if len(df) > 0:

    add_result(
        "Dataset Not Empty",
        "PASS",
        f"{len(df)} customer records found."
    )

else:

    add_result(
        "Dataset Not Empty",
        "FAIL",
        "Dataset contains no records."
    )


# ==========================================================
# 5. Missing Value Check
# ==========================================================

missing_values = int(
    df.isnull().sum().sum()
)


if missing_values == 0:

    add_result(
        "Missing Values",
        "PASS",
        "No missing values found."
    )

else:

    add_result(
        "Missing Values",
        "WARN",
        f"{missing_values} missing values found."
    )


# ==========================================================
# 6. Customer ID Validation
# ==========================================================

duplicate_customer_ids = int(
    df["Customer ID"].duplicated().sum()
)


if duplicate_customer_ids == 0:

    add_result(
        "Unique Customer IDs",
        "PASS",
        "All customer IDs are unique."
    )

else:

    add_result(
        "Unique Customer IDs",
        "FAIL",
        f"{duplicate_customer_ids} duplicate customer IDs found."
    )


# ==========================================================
# 7. Churn Target Validation
# ==========================================================

valid_churn_values = set(
    df["Churn"]
    .dropna()
    .unique()
)


if valid_churn_values.issubset(
    {0, 1}
):

    add_result(
        "Churn Target",
        "PASS",
        f"Valid target values: {sorted(valid_churn_values)}"
    )

else:

    add_result(
        "Churn Target",
        "FAIL",
        f"Invalid values found: {valid_churn_values}"
    )


# ==========================================================
# 8. Infinite Value Check
# ==========================================================

numeric_df = df.select_dtypes(
    include=[np.number]
)


infinite_values = int(
    np.isinf(
        numeric_df
    ).sum().sum()
)


if infinite_values == 0:

    add_result(
        "Infinite Values",
        "PASS",
        "No infinite numeric values found."
    )

else:

    add_result(
        "Infinite Values",
        "FAIL",
        f"{infinite_values} infinite values found."
    )


# ==========================================================
# 9. Core Numeric Sanity Checks
# ==========================================================

sanity_checks = {

    "Recency":
        df["Recency"] >= 0,

    "Frequency":
        df["Frequency"] > 0,

    "Customer_Lifespan_Days":
        df["Customer_Lifespan_Days"] >= 0,

    "Unique_Products":
        df["Unique_Products"] > 0

}


for column, condition in sanity_checks.items():

    invalid_count = int(
        (~condition).sum()
    )


    if invalid_count == 0:

        add_result(
            f"{column} Sanity",
            "PASS",
            "All values are within the expected range."
        )

    else:

        add_result(
            f"{column} Sanity",
            "WARN",
            f"{invalid_count} unusual values detected."
        )


# ==========================================================
# 10. Country Validation
# ==========================================================

blank_country_count = int(

    df["Country"]

    .fillna("")

    .astype(str)

    .str.strip()

    .eq("")

    .sum()

)


if blank_country_count == 0:

    add_result(
        "Country Values",
        "PASS",
        "No blank country values found."
    )

else:

    add_result(
        "Country Values",
        "WARN",
        f"{blank_country_count} blank country values found."
    )


# ==========================================================
# 11. Save Validation Report
# ==========================================================

validation_df = pd.DataFrame(
    validation_results
)


validation_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ==========================================================
# 12. Final Summary
# ==========================================================

pass_count = int(
    (
        validation_df["Status"]
        == "PASS"
    ).sum()
)

warning_count = int(
    (
        validation_df["Status"]
        == "WARN"
    ).sum()
)

fail_count = int(
    (
        validation_df["Status"]
        == "FAIL"
    ).sum()
)


print("\nData Validation Summary")
print("-" * 50)

print("Passed Checks :", pass_count)
print("Warnings      :", warning_count)
print("Failed Checks :", fail_count)

print(
    "\nSaved to:",
    OUTPUT_PATH
)


if fail_count == 0:

    print(
        "\nDataset validation completed successfully."
    )

else:

    print(
        "\nDataset validation found issues requiring review."
    )
"""
==============================================================
VANTARA CUSTOMER INTELLIGENCE PLATFORM
DAY 4 - FEATURE ENGINEERING
==============================================================
"""

from pathlib import Path

from src.features.feature_engineering import build_feature_pipeline
from src.features.churn import create_churn_labels
from src.features.validation import validate_feature_pipeline
from src.features.dataset_split import (
    create_train_validation_test_split,
)

import pandas as pd


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

INPUT_FILE = Path("data/interim/cleaned_data.csv")

OUTPUT_DIR = Path("data/processed")

DOCS_DIR = Path("docs")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

CUTOFF_DATE = pd.Timestamp("2011-06-01")
FUTURE_WINDOW = 90


def save_feature_documentation():

    documentation = """
VANTARA CUSTOMER INTELLIGENCE PLATFORM

FEATURE DOCUMENTATION

Customer ID
Unique customer identifier.

Recency
Days since customer's last purchase before cutoff date.

Frequency
Number of unique invoices placed.

Monetary_Total
Total historical revenue.

Monetary_Average
Average transaction value.

Average_Order_Value
Average revenue per invoice.

Total_Quantity
Total purchased quantity.

Average_Basket_Size
Average quantity purchased.

Unique_Products
Distinct purchased products.

Average_Purchase_Gap
Average days between purchases.

Customer_Lifespan_Days
Days between first and last purchase.

Country
Most frequent customer country.

Return_Rate
Returned transaction percentage.

Price_Sensitivity
Variance of purchased prices.

Historical_Revenue
Revenue before cutoff.

Predicted_CLV
Annualized customer value estimate.

Engagement_Score
Composite RFM score.

Churn

1 = No purchase within 90 days.

0 = Purchased again.
"""

    with open(
        DOCS_DIR / "feature_documentation.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(documentation)


def main():

    print("=" * 70)
    print("DAY 4 : FEATURE ENGINEERING")
    print("=" * 70)

    print("\nBuilding customer features...")

    features, historical_df, future_df = (
        build_feature_pipeline(
            INPUT_FILE,
            cutoff_date=CUTOFF_DATE,
        )
    )

    print("Creating churn labels...")

    features = create_churn_labels(
        customer_features=features,
        future_transactions=future_df,
        cutoff_date=CUTOFF_DATE,
        future_window_days=FUTURE_WINDOW,
    )

    print("Running validation...")

    validate_feature_pipeline(
        historical_df=historical_df,
        future_df=future_df,
        features=features,
        cutoff_date=CUTOFF_DATE,
        future_window_days=FUTURE_WINDOW,
    )

    print("Creating train / validation / test split...")

    train, validation, test = (
        create_train_validation_test_split(
            features
        )
    )

    print("Saving datasets...")

    features.to_csv(
        OUTPUT_DIR / "customer_features.csv"
    )

    train.to_csv(
        OUTPUT_DIR / "train.csv"
    )

    validation.to_csv(
        OUTPUT_DIR / "validation.csv"
    )

    test.to_csv(
        OUTPUT_DIR / "test.csv"
    )

    save_feature_documentation()

    print("\nDatasets Saved Successfully")

    print("\nCustomer Feature Shape :", features.shape)

    print("Train :", train.shape)
    print("Validation :", validation.shape)
    print("Test :", test.shape)

    print("\nDAY 4 COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
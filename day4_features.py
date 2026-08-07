"""
Day 4 - Feature Engineering
"""

import logging
from pathlib import Path

import pandas as pd

from src.features.churn import create_churn_labels
from src.features.dataset_split import (
    create_train_validation_test_split,
)
from src.features.feature_engineering import (
    build_feature_pipeline,
)
from src.features.validation import (
    validate_feature_pipeline,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

INPUT_FILE = Path(
    "data/interim/cleaned_data.csv"
)

OUTPUT_DIR = Path(
    "data/processed"
)

DOCS_DIR = Path(
    "docs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DOCS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CUTOFF_DATE = pd.Timestamp(
    "2011-06-01"
)

FUTURE_WINDOW = 90


def save_feature_documentation() -> None:
    """Save documentation describing engineered features."""

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
    ) as file:
        file.write(documentation)


def main() -> None:
    """Run feature engineering, validation, splitting and saving."""

    logger.info("=" * 70)
    logger.info("DAY 4 : FEATURE ENGINEERING")
    logger.info("=" * 70)

    logger.info(
        "Building customer features..."
    )

    features, historical_df, future_df = (
        build_feature_pipeline(
            INPUT_FILE,
            cutoff_date=CUTOFF_DATE,
        )
    )

    logger.info(
        "Creating churn labels..."
    )

    features = create_churn_labels(
        customer_features=features,
        future_transactions=future_df,
        cutoff_date=CUTOFF_DATE,
        future_window_days=FUTURE_WINDOW,
    )

    logger.info(
        "Running validation..."
    )

    validate_feature_pipeline(
        historical_df=historical_df,
        future_df=future_df,
        features=features,
        cutoff_date=CUTOFF_DATE,
        future_window_days=FUTURE_WINDOW,
    )

    logger.info(
        "Creating train / validation / test split..."
    )

    train, validation, test = (
        create_train_validation_test_split(
            features
        )
    )

    logger.info(
        "Saving datasets..."
    )

    features.to_csv(
        OUTPUT_DIR / "customer_features.csv",
        index=False,
    )

    train.to_csv(
        OUTPUT_DIR / "train.csv",
        index=False,
    )

    validation.to_csv(
        OUTPUT_DIR / "validation.csv",
        index=False,
    )

    test.to_csv(
        OUTPUT_DIR / "test.csv",
        index=False,
    )

    save_feature_documentation()

    logger.info(
        "Datasets saved successfully."
    )

    logger.info(
        "Customer Feature Shape: %s",
        features.shape,
    )

    logger.info(
        "Train: %s",
        train.shape,
    )

    logger.info(
        "Validation: %s",
        validation.shape,
    )

    logger.info(
        "Test: %s",
        test.shape,
    )

    logger.info(
        "DAY 4 COMPLETED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()
"""
Day 2 - Data Cleaning
"""

import logging

from src.data.cleaning import clean_retail_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

INPUT_FILE = "data/raw/online_retail_ii_combined.csv"
OUTPUT_FILE = "data/interim/cleaned_data.csv"


def main() -> None:
    """Run the Day 2 retail data-cleaning pipeline."""

    logger.info("=" * 60)
    logger.info("DAY 2 : DATA CLEANING")
    logger.info("=" * 60)

    cleaned_df, stats = clean_retail_file(
        INPUT_FILE,
        OUTPUT_FILE,
    )

    logger.info("=" * 60)
    logger.info("DATA CLEANING SUMMARY")
    logger.info("=" * 60)

    logger.info(
        "Original Rows: %s",
        f"{stats['original_rows']:,}",
    )
    logger.info(
        "Final Rows: %s",
        f"{stats['final_rows']:,}",
    )
    logger.info(
        "Duplicates Removed: %s",
        f"{stats['duplicates_removed']:,}",
    )
    logger.info(
        "Invalid Prices Removed: %s",
        f"{stats['invalid_prices_removed']:,}",
    )
    logger.info(
        "Unique Customers: %s",
        f"{stats['unique_customers']:,}",
    )
    logger.info(
        "Unique Products: %s",
        f"{stats['unique_products']:,}",
    )
    logger.info(
        "Returns Retained: %s",
        f"{stats['returns_retained']:,}",
    )
    logger.info(
        "Quantity Outliers: %s",
        f"{stats['quantity_outliers']:,}",
    )
    logger.info(
        "Price Outliers: %s",
        f"{stats['price_outliers']:,}",
    )

    logger.info("Clean dataset saved successfully.")
    logger.info("Location: %s", OUTPUT_FILE)

    # Keep reference to avoid unused-variable concerns
    _ = cleaned_df


if __name__ == "__main__":
    main()
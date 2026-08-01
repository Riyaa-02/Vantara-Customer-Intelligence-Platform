"""
Day 2 - Data Cleaning
"""

from src.data.cleaning import clean_retail_file

INPUT_FILE = "data/raw/online_retail_ii_combined.csv"
OUTPUT_FILE = "data/interim/cleaned_data.csv"


def main() -> None:
    print("=" * 60)
    print("DAY 2 : DATA CLEANING")
    print("=" * 60)

    cleaned_df, stats = clean_retail_file(
        INPUT_FILE,
        OUTPUT_FILE,
    )

    print("\n" + "=" * 60)
    print("DATA CLEANING SUMMARY")
    print("=" * 60)

    print(f"Original Rows           : {stats['original_rows']:,}")
    print(f"Final Rows              : {stats['final_rows']:,}")
    print(f"Duplicates Removed      : {stats['duplicates_removed']:,}")
    print(f"Invalid Prices Removed  : {stats['invalid_prices_removed']:,}")
    print(f"Unique Customers        : {stats['unique_customers']:,}")
    print(f"Unique Products         : {stats['unique_products']:,}")
    print(f"Returns Retained        : {stats['returns_retained']:,}")
    print(f"Quantity Outliers       : {stats['quantity_outliers']:,}")
    print(f"Price Outliers          : {stats['price_outliers']:,}")

    print("\nClean dataset saved successfully.")
    print(f"Location : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
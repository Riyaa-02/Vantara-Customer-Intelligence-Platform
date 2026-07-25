"""
Day 1 - Download & Combine Online Retail II Dataset

Purpose:
    - Load both sheets from the Online Retail II Excel dataset.
    - Merge them into one DataFrame.
    - Save the combined dataset as a CSV file.

Output:
    data/raw/online_retail_ii_combined.csv
"""

from pathlib import Path
import pandas as pd

# -----------------------------
# Configuration
# -----------------------------
RAW_DATA_PATH = Path("data/raw/online_retail_II.xlsx")
OUTPUT_PATH = Path("data/raw/online_retail_ii_combined.csv")

SHEETS = [
    "Year 2009-2010",
    "Year 2010-2011"
]


def load_sheet(file_path: Path, sheet_name: str) -> pd.DataFrame:
    """
    Load a single sheet from the Excel file.
    """
    return pd.read_excel(file_path, sheet_name=sheet_name)


def combine_datasets() -> pd.DataFrame:
    """
    Load and combine both yearly datasets.
    """
    print("Loading Online Retail II dataset...")

    dataframes = [load_sheet(RAW_DATA_PATH, sheet) for sheet in SHEETS]

    combined_df = pd.concat(
        dataframes,
        ignore_index=True
    )

    return combined_df


def save_dataset(df: pd.DataFrame) -> None:
    """
    Save combined dataset as CSV.
    """
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)


def main() -> None:
    """
    Main execution function.
    """
    df = combine_datasets()

    save_dataset(df)

    print("\nDataset Successfully Combined!")
    print(f"Total Rows      : {len(df):,}")
    print(f"Total Columns   : {len(df.columns)}")
    print(f"Saved File      : {OUTPUT_PATH}")

    print("\nColumns:")
    for col in df.columns:
        print(f" - {col}")


if __name__ == "__main__":
    main()
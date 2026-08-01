"""
Reusable data-cleaning functions for the Online Retail II dataset.
"""

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country",
]


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: list[str] | None = None,
) -> None:
    """
    Confirm that all required columns are present.

    Raises:
        ValueError: If one or more required columns are missing.
    """
    columns = required_columns or REQUIRED_COLUMNS
    missing_columns = [column for column in columns if column not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")


def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert essential numeric and datetime columns safely.

    Invalid values are converted to NaN or NaT.
    """
    cleaned_df = df.copy()

    cleaned_df["Customer ID"] = pd.to_numeric(
        cleaned_df["Customer ID"],
        errors="coerce",
    )
    cleaned_df["Quantity"] = pd.to_numeric(
        cleaned_df["Quantity"],
        errors="coerce",
    )
    cleaned_df["Price"] = pd.to_numeric(
        cleaned_df["Price"],
        errors="coerce",
    )
    cleaned_df["InvoiceDate"] = pd.to_datetime(
        cleaned_df["InvoiceDate"],
        errors="coerce",
    )

    return cleaned_df


def remove_invalid_essential_rows(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """
    Remove rows with invalid essential customer, quantity, price,
    or invoice-date values.
    """
    before = len(df)

    cleaned_df = df.dropna(
        subset=[
            "Customer ID",
            "Quantity",
            "Price",
            "InvoiceDate",
        ]
    ).copy()

    removed_count = before - len(cleaned_df)
    return cleaned_df, removed_count


def validate_customer_ids(
    df: pd.DataFrame,
    maximum_null_rate: float = 0.30,
) -> tuple[pd.DataFrame, float]:
    """
    Validate the Customer ID null rate and convert valid IDs to integers.
    """
    cleaned_df = df.copy()
    null_rate = cleaned_df["Customer ID"].isna().mean()

    if null_rate > maximum_null_rate:
        raise ValueError(
            f"Customer ID null rate too high ({null_rate:.1%})"
        )

    cleaned_df["Customer ID"] = cleaned_df["Customer ID"].astype(int)

    return cleaned_df, null_rate


def remove_exact_duplicates(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """
    Remove exact duplicate rows.
    """
    before = len(df)
    cleaned_df = df.drop_duplicates().copy()
    removed_count = before - len(cleaned_df)

    return cleaned_df, removed_count


def remove_invalid_prices(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """
    Remove rows whose price is zero or negative.
    """
    before = len(df)
    cleaned_df = df[df["Price"] > 0].copy()
    removed_count = before - len(cleaned_df)

    return cleaned_df, removed_count


def add_outlier_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add IQR-based quantity and price outlier flags.

    Outliers are flagged but not removed.
    """
    cleaned_df = df.copy()

    quantity_q1 = cleaned_df["Quantity"].quantile(0.25)
    quantity_q3 = cleaned_df["Quantity"].quantile(0.75)
    quantity_iqr = quantity_q3 - quantity_q1

    quantity_lower = quantity_q1 - 1.5 * quantity_iqr
    quantity_upper = quantity_q3 + 1.5 * quantity_iqr

    price_q1 = cleaned_df["Price"].quantile(0.25)
    price_q3 = cleaned_df["Price"].quantile(0.75)
    price_iqr = price_q3 - price_q1

    price_lower = price_q1 - 1.5 * price_iqr
    price_upper = price_q3 + 1.5 * price_iqr

    cleaned_df["Qty_Outlier"] = (
        (cleaned_df["Quantity"] < quantity_lower)
        | (cleaned_df["Quantity"] > quantity_upper)
    )

    cleaned_df["Price_Outlier"] = (
        (cleaned_df["Price"] < price_lower)
        | (cleaned_df["Price"] > price_upper)
    )

    return cleaned_df


def standardize_product_descriptions(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Standardize each product description using the most common
    description associated with its StockCode.
    """
    cleaned_df = df.copy()

    description_lookup = (
        cleaned_df.dropna(subset=["Description"])
        .groupby("StockCode")["Description"]
        .agg(
            lambda values: (
                values.mode().iloc[0]
                if not values.mode().empty
                else values.iloc[0]
            )
        )
        .to_dict()
    )

    cleaned_df["Description_Clean"] = (
        cleaned_df["StockCode"]
        .map(description_lookup)
        .str.strip()
        .str.upper()
    )

    return cleaned_df, description_lookup


def add_total_sales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the Total column as Quantity multiplied by Price.
    """
    cleaned_df = df.copy()
    cleaned_df["Total"] = cleaned_df["Quantity"] * cleaned_df["Price"]

    return cleaned_df


def clean_retail_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Run the complete Online Retail II cleaning pipeline.

    Returns:
        A tuple containing:
        - cleaned DataFrame
        - dictionary of cleaning statistics
    """
    original_rows = len(df)

    validate_required_columns(df)

    cleaned_df = convert_data_types(df)

    cleaned_df, invalid_rows_removed = remove_invalid_essential_rows(
        cleaned_df
    )

    cleaned_df, customer_id_null_rate = validate_customer_ids(cleaned_df)

    cleaned_df, duplicates_removed = remove_exact_duplicates(cleaned_df)

    cleaned_df, invalid_prices_removed = remove_invalid_prices(cleaned_df)

    cleaned_df = add_outlier_flags(cleaned_df)

    cleaned_df, description_lookup = standardize_product_descriptions(
        cleaned_df
    )

    cleaned_df = add_total_sales(cleaned_df)

    statistics = {
        "original_rows": original_rows,
        "final_rows": len(cleaned_df),
        "invalid_rows_removed": invalid_rows_removed,
        "customer_id_null_rate": customer_id_null_rate,
        "duplicates_removed": duplicates_removed,
        "invalid_prices_removed": invalid_prices_removed,
        "unique_customers": cleaned_df["Customer ID"].nunique(),
        "unique_products": cleaned_df["StockCode"].nunique(),
        "returns_retained": int((cleaned_df["Quantity"] < 0).sum()),
        "quantity_outliers": int(cleaned_df["Qty_Outlier"].sum()),
        "price_outliers": int(cleaned_df["Price_Outlier"].sum()),
        "description_lookup_size": len(description_lookup),
    }

    return cleaned_df, statistics


def clean_retail_file(
    input_file: str | Path,
    output_file: str | Path,
) -> tuple[pd.DataFrame, dict]:
    """
    Load, clean, and save an Online Retail II CSV file.
    """
    input_path = Path(input_file)
    output_path = Path(output_file)

    df = pd.read_csv(input_path)
    cleaned_df, statistics = clean_retail_data(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(output_path, index=False)

    return cleaned_df, statistics
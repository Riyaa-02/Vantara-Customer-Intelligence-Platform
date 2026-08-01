import pandas as pd
import pytest

from src.data.cleaning import (
    add_outlier_flags,
    add_total_sales,
    convert_data_types,
    remove_exact_duplicates,
    remove_invalid_essential_rows,
    remove_invalid_prices,
    standardize_product_descriptions,
    validate_required_columns,
)


def make_sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Invoice": ["1001", "1001", "1002", "1003"],
            "StockCode": ["A", "A", "B", "C"],
            "Description": [
                "red mug",
                "red mug",
                "blue plate",
                "green bowl",
            ],
            "Quantity": [2, 2, 3, 100],
            "InvoiceDate": [
                "2011-01-01",
                "2011-01-01",
                "2011-01-03",
                "2011-01-04",
            ],
            "Price": [5.0, 5.0, -1.0, 50.0],
            "Customer ID": [1, 1, 2, 3],
            "Country": ["UK", "UK", "France", "Germany"],
        }
    )


def test_validate_required_columns_passes():
    df = make_sample_dataframe()

    validate_required_columns(df)


def test_validate_required_columns_raises_for_missing_column():
    df = make_sample_dataframe().drop(columns=["Country"])

    with pytest.raises(ValueError, match="Missing columns"):
        validate_required_columns(df)


def test_convert_data_types():
    df = make_sample_dataframe()

    result = convert_data_types(df)

    assert pd.api.types.is_numeric_dtype(result["Customer ID"])
    assert pd.api.types.is_numeric_dtype(result["Quantity"])
    assert pd.api.types.is_numeric_dtype(result["Price"])
    assert pd.api.types.is_datetime64_any_dtype(result["InvoiceDate"])


def test_remove_invalid_essential_rows():
    df = make_sample_dataframe()
    df.loc[0, "Customer ID"] = None

    converted = convert_data_types(df)
    result, removed = remove_invalid_essential_rows(converted)

    assert removed == 1
    assert result["Customer ID"].isna().sum() == 0


def test_remove_exact_duplicates():
    df = make_sample_dataframe()

    result, removed = remove_exact_duplicates(df)

    assert removed == 1
    assert len(result) == 3


def test_remove_invalid_prices():
    df = make_sample_dataframe()

    result, removed = remove_invalid_prices(df)

    assert removed == 1
    assert (result["Price"] > 0).all()


def test_add_outlier_flags():
    df = make_sample_dataframe()

    result = add_outlier_flags(df)

    assert "Qty_Outlier" in result.columns
    assert "Price_Outlier" in result.columns
    assert result["Qty_Outlier"].dtype == bool
    assert result["Price_Outlier"].dtype == bool


def test_standardize_product_descriptions():
    df = make_sample_dataframe()

    result, lookup = standardize_product_descriptions(df)

    assert "Description_Clean" in result.columns
    assert result.loc[0, "Description_Clean"] == "RED MUG"
    assert lookup["A"] == "red mug"


def test_add_total_sales():
    df = make_sample_dataframe()

    result = add_total_sales(df)

    assert "Total" in result.columns
    assert result.loc[0, "Total"] == 10.0
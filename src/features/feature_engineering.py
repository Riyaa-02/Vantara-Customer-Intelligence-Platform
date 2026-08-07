"""
Reusable feature-engineering functions for the
Vantara Customer Intelligence Platform.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

DEFAULT_CUTOFF_DATE = pd.Timestamp("2011-06-01")


def load_and_prepare_data(
    input_file: str | Path,
) -> pd.DataFrame:
    """
    Load the cleaned transaction dataset and prepare basic columns.

    Parameters
    ----------
    input_file:
        Path to the cleaned transaction CSV.

    Returns
    -------
    pandas.DataFrame
        Prepared transaction dataframe.
    """
    df = pd.read_csv(input_file)

    required_columns = {
        "Invoice",
        "StockCode",
        "Description_Clean",
        "Quantity",
        "InvoiceDate",
        "Price",
        "Customer ID",
        "Country",
    }

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Required columns are missing: {missing_text}"
        )

    df = df.copy()

    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"],
        errors="coerce",
    )

    if df["InvoiceDate"].isna().any():
        invalid_dates = int(df["InvoiceDate"].isna().sum())

        raise ValueError(
            f"Dataset contains {invalid_dates} invalid InvoiceDate values."
        )

    df["Total"] = df["Quantity"] * df["Price"]

    return df


def split_historical_future(
    df: pd.DataFrame,
    cutoff_date: pd.Timestamp = DEFAULT_CUTOFF_DATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split transactions into historical and future datasets.

    Historical data contains transactions before the cutoff date.
    Future data contains transactions on or after the cutoff date.
    """
    cutoff_date = pd.Timestamp(cutoff_date)

    historical_df = df[
        df["InvoiceDate"] < cutoff_date
    ].copy()

    future_df = df[
        df["InvoiceDate"] >= cutoff_date
    ].copy()

    if historical_df.empty:
        raise ValueError(
            "Historical dataset is empty for the selected cutoff date."
        )

    return historical_df, future_df


def create_customer_index(
    historical_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create an empty feature table indexed by Customer ID.
    """
    customer_ids = (
        historical_df.groupby("Customer ID")
        .size()
        .index
    )

    features = pd.DataFrame(index=customer_ids)
    features.index.name = "Customer ID"

    return features


def create_rfm_features(
    historical_df: pd.DataFrame,
    cutoff_date: pd.Timestamp = DEFAULT_CUTOFF_DATE,
) -> pd.DataFrame:
    """
    Create Recency, Frequency, and Monetary features.
    """
    cutoff_date = pd.Timestamp(cutoff_date)

    customers = historical_df.groupby("Customer ID")

    last_purchase = customers["InvoiceDate"].max()

    rfm_features = pd.DataFrame(
        index=last_purchase.index
    )

    rfm_features.index.name = "Customer ID"

    rfm_features["Recency"] = (
        cutoff_date - last_purchase
    ).dt.days

    rfm_features["Frequency"] = (
        customers["Invoice"].nunique()
    )

    rfm_features["Monetary_Total"] = (
        customers["Total"].sum()
    )

    rfm_features["Monetary_Average"] = (
        customers["Total"].mean()
    )

    return rfm_features


def calculate_average_purchase_gap(
    customer_transactions: pd.DataFrame,
) -> float:
    """
    Calculate the average number of days between purchases.

    Returns zero when fewer than two distinct purchase dates exist.
    """
    purchase_dates = (
        customer_transactions["InvoiceDate"]
        .sort_values()
        .drop_duplicates()
    )

    if len(purchase_dates) < 2:
        return 0.0

    gaps = (
        purchase_dates
        .diff()
        .dt.days
        .dropna()
    )

    if gaps.empty:
        return 0.0

    return float(gaps.mean())


def create_purchase_features(
    historical_df: pd.DataFrame,
    rfm_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create customer purchase-behaviour features.
    """
    customers = historical_df.groupby("Customer ID")

    purchase_features = pd.DataFrame(
        index=rfm_features.index
    )

    purchase_features.index.name = "Customer ID"

    purchase_features["Total_Quantity"] = (
        customers["Quantity"].sum()
    )

    purchase_features["Average_Basket_Size"] = (
        customers["Quantity"].mean()
    )

    frequency = rfm_features["Frequency"].replace(
        0,
        np.nan,
    )

    purchase_features["Average_Order_Value"] = (
        rfm_features["Monetary_Total"] / frequency
    ).fillna(0)

    purchase_features["Unique_Products"] = (
        customers["StockCode"].nunique()
    )

    average_gaps = (
        historical_df
        .groupby("Customer ID")
        .apply(
            calculate_average_purchase_gap,
            include_groups=False,
        )
    )

    purchase_features["Average_Purchase_Gap"] = (
        average_gaps.reindex(purchase_features.index)
        .fillna(0)
    )

    return purchase_features


def get_customer_mode_country(
    country_series: pd.Series,
) -> str:
    """
    Return the most frequently occurring country.

    When no mode is available, return 'Unknown'.
    """
    mode_values = country_series.dropna().mode()

    if mode_values.empty:
        return "Unknown"

    return str(mode_values.iloc[0])


def create_customer_profile_features(
    historical_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create customer lifespan and country-profile features.
    """
    customers = historical_df.groupby("Customer ID")

    first_purchase = customers["InvoiceDate"].min()
    last_purchase = customers["InvoiceDate"].max()

    profile_features = pd.DataFrame(
        index=first_purchase.index
    )

    profile_features.index.name = "Customer ID"

    profile_features["Customer_Lifespan_Days"] = (
        last_purchase - first_purchase
    ).dt.days

    profile_features["Country"] = (
        customers["Country"]
        .agg(get_customer_mode_country)
    )

    return profile_features


def create_return_features(
    historical_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create the percentage of returned transactions per customer.
    """
    total_transactions = (
        historical_df
        .groupby("Customer ID")
        .size()
    )

    returned_transactions = (
        historical_df[
            historical_df["Quantity"] < 0
        ]
        .groupby("Customer ID")
        .size()
    )

    return_features = pd.DataFrame(
        index=total_transactions.index
    )

    return_features.index.name = "Customer ID"

    return_features["Return_Rate"] = (
        returned_transactions
        .div(total_transactions)
        .mul(100)
        .fillna(0)
    )

    return return_features


def create_price_sensitivity_feature(
    historical_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create customer price-sensitivity feature using price variance.
    """
    price_variance = (
        historical_df
        .groupby("Customer ID")["Price"]
        .var()
        .fillna(0)
    )

    result = pd.DataFrame(
        index=price_variance.index
    )

    result.index.name = "Customer ID"
    result["Price_Sensitivity"] = price_variance

    return result


def create_clv_features(
    monetary_total: pd.Series,
    customer_lifespan_days: pd.Series,
    clipping_quantile: float = 0.99,
) -> pd.DataFrame:
    """
    Create historical revenue and annualized predicted CLV features.
    """
    if not 0 < clipping_quantile <= 1:
        raise ValueError(
            "clipping_quantile must be greater than 0 and at most 1."
        )

    clv_features = pd.DataFrame(
        index=monetary_total.index
    )

    clv_features.index.name = "Customer ID"

    clv_features["Historical_Revenue"] = (
        monetary_total
    )

    daily_value = (
        clv_features["Historical_Revenue"]
        / (customer_lifespan_days + 1)
    )

    predicted_clv = daily_value * 365

    clip_value = predicted_clv.quantile(
        clipping_quantile
    )

    clv_features["Predicted_CLV"] = (
        predicted_clv.clip(upper=clip_value)
    )

    return clv_features


def create_engagement_score(
    features: pd.DataFrame,
) -> pd.Series:
    """
    Create a 0-100 engagement score from RFM values.

    Higher recency values represent weaker engagement, so Recency is
    reversed before scaling.
    """
    required_columns = {
        "Recency",
        "Frequency",
        "Monetary_Total",
    }

    missing_columns = required_columns.difference(
        features.columns
    )

    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))

        raise ValueError(
            "Cannot calculate Engagement_Score. "
            f"Missing columns: {missing_text}"
        )

    rfm = features[
        [
            "Recency",
            "Frequency",
            "Monetary_Total",
        ]
    ].copy()

    rfm["Recency"] = (
        rfm["Recency"].max() - rfm["Recency"]
    )

    scaler = MinMaxScaler(
        feature_range=(0, 100)
    )

    scaled_rfm = scaler.fit_transform(rfm)

    engagement_score = pd.Series(
        scaled_rfm.mean(axis=1),
        index=features.index,
        name="Engagement_Score",
    )

    return engagement_score


def build_customer_features(
    historical_df: pd.DataFrame,
    cutoff_date: pd.Timestamp = DEFAULT_CUTOFF_DATE,
) -> pd.DataFrame:
    """
    Build the complete customer-level feature table.
    """
    features = create_customer_index(
        historical_df
    )

    rfm_features = create_rfm_features(
        historical_df,
        cutoff_date=cutoff_date,
    )

    purchase_features = create_purchase_features(
        historical_df,
        rfm_features=rfm_features,
    )

    profile_features = (
        create_customer_profile_features(
            historical_df
        )
    )

    return_features = create_return_features(
        historical_df
    )

    price_features = (
        create_price_sensitivity_feature(
            historical_df
        )
    )

    feature_tables = [
        rfm_features,
        purchase_features,
        profile_features,
        return_features,
        price_features,
    ]

    for feature_table in feature_tables:
        features = features.join(
            feature_table,
            how="left",
        )

    clv_features = create_clv_features(
        monetary_total=features["Monetary_Total"],
        customer_lifespan_days=features[
            "Customer_Lifespan_Days"
        ],
    )

    features = features.join(
        clv_features,
        how="left",
    )

    features["Engagement_Score"] = (
        create_engagement_score(features)
    )

    numeric_columns = (
        features
        .select_dtypes(include=[np.number])
        .columns
    )

    features[numeric_columns] = (
        features[numeric_columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    return features


def build_feature_pipeline(
    input_file: str | Path,
    cutoff_date: pd.Timestamp = DEFAULT_CUTOFF_DATE,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Run data loading, temporal splitting, and feature creation.

    Returns
    -------
    tuple
        customer_features, historical_transactions, future_transactions
    """
    df = load_and_prepare_data(input_file)

    historical_df, future_df = (
        split_historical_future(
            df,
            cutoff_date=cutoff_date,
        )
    )

    customer_features = build_customer_features(
        historical_df,
        cutoff_date=cutoff_date,
    )

    return (
        customer_features,
        historical_df,
        future_df,
    )
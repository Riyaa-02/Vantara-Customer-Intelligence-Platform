from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8000",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CUSTOMER_SEGMENTS_FILE = (
    PROJECT_ROOT / "data" / "final" / "customer_segments.csv"
)

TRANSACTIONS_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "online_retail_ii_combined.csv"
)


@st.cache_data(show_spinner=False)
def load_customer_segments() -> pd.DataFrame:
    """Load the customer-level segmentation and analytics dataset."""

    if not CUSTOMER_SEGMENTS_FILE.exists():
        raise FileNotFoundError(
            f"Customer segments file not found: {CUSTOMER_SEGMENTS_FILE}"
        )

    dataframe = pd.read_csv(CUSTOMER_SEGMENTS_FILE)

    numeric_columns = [
        "Recency",
        "Frequency",
        "Monetary_Total",
        "Predicted_CLV",
        "Engagement_Score",
        "Churn",
        "Customer_Segment",
    ]

    for column in numeric_columns:
        if column in dataframe.columns:
            dataframe[column] = pd.to_numeric(
                dataframe[column],
                errors="coerce",
            )

    return dataframe


@st.cache_data(show_spinner=False)
def load_transactions() -> pd.DataFrame:
    """Load and prepare valid retail transactions for analytics."""

    if not TRANSACTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Transaction file not found: {TRANSACTIONS_FILE}"
        )

    dataframe = pd.read_csv(TRANSACTIONS_FILE)

    dataframe["InvoiceDate"] = pd.to_datetime(
        dataframe["InvoiceDate"],
        errors="coerce",
    )

    dataframe["Quantity"] = pd.to_numeric(
        dataframe["Quantity"],
        errors="coerce",
    )

    dataframe["Price"] = pd.to_numeric(
        dataframe["Price"],
        errors="coerce",
    )

    dataframe = dataframe.dropna(
        subset=["InvoiceDate", "Quantity", "Price"]
    )

    dataframe = dataframe[
        (dataframe["Quantity"] > 0)
        & (dataframe["Price"] > 0)
    ].copy()

    dataframe["Revenue"] = (
        dataframe["Quantity"] * dataframe["Price"]
    )

    dataframe["Month"] = (
        dataframe["InvoiceDate"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    return dataframe


@st.cache_data(show_spinner=False)
def load_monthly_revenue() -> pd.DataFrame:
    """Prepare monthly revenue, order and quantity summaries."""

    transactions = load_transactions()

    return (
        transactions.groupby("Month", as_index=False)
        .agg(
            Revenue=("Revenue", "sum"),
            Orders=("Invoice", "nunique"),
            Quantity=("Quantity", "sum"),
        )
        .sort_values("Month")
        .reset_index(drop=True)
    )


@st.cache_data(ttl=15, show_spinner=False)
def check_api_status() -> bool:
    """Return True when the FastAPI health endpoint is available."""

    try:
        response = requests.get(
            f"{API_BASE_URL}/health",
            timeout=3,
        )

        return response.status_code == 200

    except requests.RequestException:
        return False


@st.cache_data(ttl=20, show_spinner=False)
def get_prediction_history(
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Retrieve recent prediction records from FastAPI/PostgreSQL."""

    try:
        response = requests.get(
            f"{API_BASE_URL}/predictions",
            params={"limit": int(limit)},
            timeout=10,
        )

        response.raise_for_status()

        result = response.json()

        return result if isinstance(result, list) else []

    except requests.RequestException:
        return []


def clear_api_cache() -> None:
    """Refresh API-dependent dashboard information."""

    check_api_status.clear()
    get_prediction_history.clear()
"""
Churn label creation utilities.
"""

from __future__ import annotations

import pandas as pd


def create_churn_labels(
    customer_features: pd.DataFrame,
    future_transactions: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    future_window_days: int = 90,
) -> pd.DataFrame:
    """
    Create churn labels.

    Churn = 1
        Customer does NOT purchase within the next N days.

    Active = 0
        Customer purchases again within the future window.
    """

    active_window_end = (
        pd.Timestamp(cutoff_date)
        + pd.Timedelta(days=future_window_days)
    )

    active_customers = (
        future_transactions[
            future_transactions["InvoiceDate"]
            <= active_window_end
        ]["Customer ID"]
        .unique()
    )

    features = customer_features.copy()

    features["Churn"] = (
        ~features.index.isin(active_customers)
    ).astype(int)

    return features
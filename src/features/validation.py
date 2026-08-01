"""
Validation utilities for the feature-engineering pipeline.
"""

from __future__ import annotations

import pandas as pd


def validate_feature_pipeline(
    historical_df: pd.DataFrame,
    future_df: pd.DataFrame,
    features: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    future_window_days: int = 90,
) -> None:
    """
    Validate that feature engineering does not leak future information.
    """

    cutoff_date = pd.Timestamp(cutoff_date)

    assert (
        historical_df["InvoiceDate"].max()
        < cutoff_date
    ), "Historical data contains future records."

    active_window_end = (
        cutoff_date
        + pd.Timedelta(days=future_window_days)
    )

    future_subset = future_df[
        future_df["InvoiceDate"]
        <= active_window_end
    ]

    if not future_subset.empty:
        assert (
            future_subset["InvoiceDate"].min()
            >= cutoff_date
        ), "Future window starts before cutoff."

    assert (
        features["Recency"].min()
        >= 0
    ), "Negative Recency detected."
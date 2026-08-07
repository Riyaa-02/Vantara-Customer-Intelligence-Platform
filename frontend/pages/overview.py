from __future__ import annotations

import pandas as pd
import streamlit as st

from frontend.components import metric_card, page_title
from frontend.data_service import (
    check_api_status,
    get_prediction_history,
    load_customer_segments,
    load_monthly_revenue,
)


def render() -> None:
    """Render the executive dashboard overview."""

    segments_df = load_customer_segments()
    monthly_revenue_df = load_monthly_revenue()
    prediction_history = get_prediction_history(limit=5)
    api_connected = check_api_status()

    total_customers = len(segments_df)

    total_segments = int(
        segments_df["Customer_Segment"].nunique()
    )

    high_risk_customers = int(
        (segments_df["Churn"] == 1).sum()
    )

    average_churn_risk = (
        segments_df["Churn"].mean() * 100
        if "Churn" in segments_df.columns
        else 0
    )

    page_title(
        "Dashboard Overview",
        "Monitor customer risk, value, segments and recent activity.",
    )

    st.markdown(
        f"""
        <div class="status-row">
            <span class="status-dot {'online' if api_connected else 'offline'}"></span>
            <span>
                API Status:
                <strong>
                    {'Connected' if api_connected else 'Offline'}
                </strong>
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card(
            "Total Customers",
            f"{total_customers:,}",
            "Customer profiles available",
        )

    with col2:
        metric_card(
            "Customer Segments",
            f"{total_segments}",
            "Active business segments",
        )

    with col3:
        metric_card(
            "High-Risk Customers",
            f"{high_risk_customers:,}",
            f"{average_churn_risk:.1f}% of customers",
        )

    with col4:
        metric_card(
            "Stored Predictions",
            f"{len(prediction_history):,}",
            "Latest records retrieved",
        )

    st.write("")

    chart_col1, chart_col2 = st.columns([1, 1])

    with chart_col1:
        st.subheader("Customer Segments")

        segment_counts = (
            segments_df["Customer_Segment"]
            .value_counts()
            .sort_index()
            .rename_axis("Segment")
            .reset_index(name="Customers")
        )

        st.bar_chart(
            segment_counts.set_index("Segment"),
            use_container_width=True,
        )

    with chart_col2:
        st.subheader("Revenue Trend")

        revenue_chart = monthly_revenue_df[
            ["Month", "Revenue"]
        ].copy()

        revenue_chart["Month"] = pd.to_datetime(
            revenue_chart["Month"]
        )

        st.line_chart(
            revenue_chart.set_index("Month"),
            use_container_width=True,
        )

    st.write("")

    left, right = st.columns([2, 1])

    with left:
        st.subheader("Recent Predictions")

        if prediction_history:
            history_df = pd.DataFrame(prediction_history)

            display_columns = [
                column
                for column in [
                    "customer_id",
                    "risk_level",
                    "churn_probability",
                    "prediction_type",
                    "created_at",
                ]
                if column in history_df.columns
            ]

            history_df = history_df[display_columns]

            if "churn_probability" in history_df.columns:
                history_df["churn_probability"] = (
                    pd.to_numeric(
                        history_df["churn_probability"],
                        errors="coerce",
                    )
                    * 100
                ).round(2)

            history_df = history_df.rename(
                columns={
                    "customer_id": "Customer ID",
                    "risk_level": "Risk Level",
                    "churn_probability": "Probability (%)",
                    "prediction_type": "Type",
                    "created_at": "Created At",
                }
            )

            st.dataframe(
                history_df,
                use_container_width=True,
                hide_index=True,
            )

        else:
            st.info("No prediction history is available yet.")

    with right:
        st.subheader("Business Insight")

        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-title">
                    Customer Retention Priority
                </div>

                <div class="insight-value">
                    {average_churn_risk:.1f}%
                </div>

                <p>
                    of customers are marked as high risk.
                    Prioritize valuable at-risk customers for
                    targeted retention campaigns.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
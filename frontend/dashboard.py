import os
import numpy as np
import streamlit as st
import requests
import pandas as pd


API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Vantara Customer Intelligence Platform",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Vantara Customer Intelligence Platform")

page = st.sidebar.radio(
    "Dashboard Navigation",
    [
        "Churn Prediction",
        "Customer Explanation",
        "Batch Prediction",
        "Prediction History",
        "Customer Segmentation",
        "Churn Leaderboard",
        "Revenue & CLV Analytics"
    ]
)
st.markdown(
    """
Analyze customer behavior using machine learning to predict customer churn,
estimate Customer Lifetime Value (CLV), and support data-driven business decisions.
"""
)

st.markdown("---")


# ---------------------------------------------------------
# CHURN PREDICTION PAGE
# ---------------------------------------------------------

if page == "Churn Prediction":

    st.header("Customer Churn Prediction")

    st.write(
        "Fill in the customer details below and click "
        "**Predict Customer Churn** to analyze the customer's risk."
    )

    col1, col2 = st.columns(2)

    with col1:
        recency = st.number_input(
            "Recency",
            min_value=0.0,
            value=25.0
        )

        frequency = st.number_input(
            "Frequency",
            min_value=0.0,
            value=12.0
        )

        monetary_total = st.number_input(
            "Monetary Total",
            min_value=0.0,
            value=2500.0
        )

        monetary_average = st.number_input(
            "Monetary Average",
            min_value=0.0,
            value=208.3
        )

        total_quantity = st.number_input(
            "Total Quantity",
            min_value=0.0,
            value=40.0
        )

        average_basket_size = st.number_input(
            "Average Basket Size",
            min_value=0.0,
            value=3.3
        )

        average_order_value = st.number_input(
            "Average Order Value",
            min_value=0.0,
            value=208.3
        )

    with col2:
        unique_products = st.number_input(
            "Unique Products",
            min_value=0.0,
            value=15.0
        )

        average_purchase_gap = st.number_input(
            "Average Purchase Gap",
            min_value=0.0,
            value=18.0
        )

        customer_lifespan_days = st.number_input(
            "Customer Lifespan Days",
            min_value=0.0,
            value=320.0
        )

        return_rate = st.number_input(
            "Return Rate",
            min_value=0.0,
            value=0.02
        )

        price_sensitivity = st.number_input(
            "Price Sensitivity",
            min_value=0.0,
            value=0.35
        )

        predicted_clv = st.number_input(
            "Predicted CLV",
            min_value=0.0,
            value=4200.0
        )

        engagement_score = st.number_input(
            "Engagement Score",
            min_value=0.0,
            value=78.0
        )

    country = st.selectbox(
        "Country",
        [
            "Australia",
            "Austria",
            "Bahrain",
            "Belgium",
            "Brazil",
            "Canada",
            "Channel Islands",
            "Cyprus",
            "Czech Republic",
            "Denmark",
            "EIRE",
            "European Community",
            "Finland",
            "France",
            "Germany",
            "Greece",
            "Iceland",
            "Israel",
            "Italy",
            "Japan",
            "Korea",
            "Lebanon",
            "Lithuania",
            "Malta",
            "Netherlands",
            "Nigeria",
            "Norway",
            "Poland",
            "Portugal",
            "RSA",
            "Saudi Arabia",
            "Singapore",
            "Spain",
            "Sweden",
            "Switzerland",
            "Thailand",
            "USA",
            "United Arab Emirates",
            "United Kingdom",
            "Unspecified",
            "West Indies"
        ],
        index=38
    )

    st.markdown("---")

    if st.button("🔍 Predict Customer Churn"):

        payload = {
            "Recency": recency,
            "Frequency": frequency,
            "Monetary_Total": monetary_total,
            "Monetary_Average": monetary_average,
            "Total_Quantity": total_quantity,
            "Average_Basket_Size": average_basket_size,
            "Average_Order_Value": average_order_value,
            "Unique_Products": unique_products,
            "Average_Purchase_Gap": average_purchase_gap,
            "Customer_Lifespan_Days": customer_lifespan_days,
            "Return_Rate": return_rate,
            "Price_Sensitivity": price_sensitivity,
            "Predicted_CLV": predicted_clv,
            "Engagement_Score": engagement_score,
            "Country": country
        }

        try:
            response = requests.post(
                f"{API_BASE_URL}/predict",
                json=payload,
                timeout=30
            )

            if response.status_code == 200:

                result = response.json()

                st.success(
    				"Prediction completed successfully! "
    				"The prediction has also been saved to the database."
		)

                metric_col1, metric_col2, metric_col3 = st.columns(3)

                with metric_col1:
                    st.metric(
                        "Churn Probability",
                        f"{result['churn_probability'] * 100:.2f}%"
                    )

                with metric_col2:
                    st.metric(
                        "Risk Level",
                        result["risk_level"]
                    )

                with metric_col3:
                    st.metric(
                        "Prediction",
                        (
                            "Likely to Churn"
                            if result["churn_prediction"] == 1
                            else "Not Likely to Churn"
                        )
                    )

            else:
                st.error(
                    f"Prediction failed. API response: {response.text}"
                )

        except requests.exceptions.ConnectionError:
            st.error(
                "Could not connect to the FastAPI service. "
                "Make sure the API is running on port 8000."
            )

        except requests.exceptions.Timeout:
            st.error(
                "The prediction request timed out."
            )

        except Exception as error:
            st.error(
                f"Unexpected error: {error}"
            )
# ---------------------------------------------------------
# CUSTOMER EXPLANATION PAGE
# ---------------------------------------------------------

elif page == "Customer Explanation":

    st.header("Customer Churn Explanation")

    st.write(
        "Enter customer information to predict churn and view "
        "the most influential features affecting the prediction."
    )

    col1, col2 = st.columns(2)

    with col1:
        recency = st.number_input(
            "Recency",
            min_value=0.0,
            value=45.0,
            key="exp_recency",
        )

        frequency = st.number_input(
            "Frequency",
            min_value=0.0,
            value=8.0,
            key="exp_frequency",
        )

        monetary_total = st.number_input(
            "Monetary Total",
            min_value=0.0,
            value=4200.0,
            key="exp_monetary_total",
        )

        monetary_average = st.number_input(
            "Monetary Average",
            min_value=0.0,
            value=525.0,
            key="exp_monetary_average",
        )

        total_quantity = st.number_input(
            "Total Quantity",
            min_value=0.0,
            value=18.0,
            key="exp_total_quantity",
        )

        average_basket_size = st.number_input(
            "Average Basket Size",
            min_value=0.0,
            value=2.25,
            key="exp_average_basket_size",
        )

        average_order_value = st.number_input(
            "Average Order Value",
            min_value=0.0,
            value=525.0,
            key="exp_average_order_value",
        )

    with col2:
        unique_products = st.number_input(
            "Unique Products",
            min_value=0.0,
            value=12.0,
            key="exp_unique_products",
        )

        average_purchase_gap = st.number_input(
            "Average Purchase Gap",
            min_value=0.0,
            value=30.0,
            key="exp_average_purchase_gap",
        )

        customer_lifespan_days = st.number_input(
            "Customer Lifespan Days",
            min_value=0.0,
            value=240.0,
            key="exp_customer_lifespan_days",
        )

        return_rate = st.number_input(
            "Return Rate",
            min_value=0.0,
            max_value=1.0,
            value=0.05,
            step=0.01,
            key="exp_return_rate",
        )

        price_sensitivity = st.number_input(
            "Price Sensitivity",
            min_value=0.0,
            max_value=1.0,
            value=0.35,
            step=0.01,
            key="exp_price_sensitivity",
        )

        predicted_clv = st.number_input(
            "Predicted CLV",
            min_value=0.0,
            value=6500.0,
            key="exp_predicted_clv",
        )

        engagement_score = st.number_input(
            "Engagement Score",
            min_value=0.0,
            value=62.0,
            key="exp_engagement_score",
        )

    country = st.selectbox(
        "Country",
        [
            "Australia",
            "Austria",
            "Bahrain",
            "Belgium",
            "Brazil",
            "Canada",
            "Channel Islands",
            "Cyprus",
            "Czech Republic",
            "Denmark",
            "EIRE",
            "European Community",
            "Finland",
            "France",
            "Germany",
            "Greece",
            "Iceland",
            "Israel",
            "Italy",
            "Japan",
            "Korea",
            "Lebanon",
            "Lithuania",
            "Malta",
            "Netherlands",
            "Nigeria",
            "Norway",
            "Poland",
            "Portugal",
            "RSA",
            "Saudi Arabia",
            "Singapore",
            "Spain",
            "Sweden",
            "Switzerland",
            "Thailand",
            "USA",
            "United Arab Emirates",
            "United Kingdom",
            "Unspecified",
            "West Indies",
        ],
        index=38,
        key="exp_country",
    )

    st.markdown("---")

    if st.button(
        "🔎 Explain Customer Churn",
        key="explain_customer_button",
    ):

        payload = {
            "Recency": recency,
            "Frequency": frequency,
            "Monetary_Total": monetary_total,
            "Monetary_Average": monetary_average,
            "Total_Quantity": total_quantity,
            "Average_Basket_Size": average_basket_size,
            "Average_Order_Value": average_order_value,
            "Unique_Products": unique_products,
            "Average_Purchase_Gap": average_purchase_gap,
            "Customer_Lifespan_Days": customer_lifespan_days,
            "Return_Rate": return_rate,
            "Price_Sensitivity": price_sensitivity,
            "Predicted_CLV": predicted_clv,
            "Engagement_Score": engagement_score,
            "Country": country,
        }

        try:
            with st.spinner(
                "Generating prediction and SHAP explanation..."
            ):
                response = requests.post(
                    f"{API_BASE_URL}/predict/explain",
                    json=payload,
                    timeout=60,
                )

            if response.status_code == 200:
                result = response.json()

                st.success(
                    "Prediction explanation generated successfully."
                )

                metric_col1, metric_col2, metric_col3 = st.columns(3)

                with metric_col1:
                    st.metric(
                        "Churn Probability",
                        f"{result['churn_probability'] * 100:.2f}%",
                    )

                with metric_col2:
                    st.metric(
                        "Risk Level",
                        result["risk_level"],
                    )

                with metric_col3:
                    prediction_label = (
                        "Likely to Churn"
                        if result["churn_prediction"] == 1
                        else "Not Likely to Churn"
                    )

                    st.metric(
                        "Prediction",
                        prediction_label,
                    )

                st.markdown("---")
                st.subheader("Top Factors Influencing Churn Prediction")

                top_features = result.get(
                    "top_features",
                    [],
                )

                feature_name_mapping = {
                    "Engagement_Score": "Engagement Score",
                    "Recency": "Recency",
                    "Frequency": "Purchase Frequency",
                    "Monetary_Total": "Total Spending",
                    "Monetary_Average": "Average Spending",
                    "Total_Quantity": "Total Quantity Purchased",
                    "Average_Basket_Size": "Average Basket Size",
                    "Average_Order_Value": "Average Order Value",
                    "Unique_Products": "Unique Products Purchased",
                    "Average_Purchase_Gap": "Average Purchase Gap",
                    "Customer_Lifespan_Days": "Customer Lifespan",
                    "Return_Rate": "Return Rate",
                    "Price_Sensitivity": "Price Sensitivity",
                    "Predicted_CLV": (
                        "Predicted Customer Lifetime Value"
                    ),
                }

                if top_features:
                    explanation_df = pd.DataFrame(
                        top_features
                    )

                    display_columns = [
                        column
                        for column in [
                            "feature",
                            "feature_value",
                            "shap_value",
                            "impact",
                        ]
                        if column in explanation_df.columns
                    ]

                    if display_columns:
                        explanation_df = explanation_df[
                            display_columns
                        ]

                    explanation_df = explanation_df.rename(
                        columns={
                            "feature": "Feature",
                            "feature_value": "Customer Value",
                            "shap_value": "SHAP Value",
                            "impact": "Impact on Churn Risk",
                        }
                    )

                    if "Feature" in explanation_df.columns:
                        explanation_df["Feature"] = (
                            explanation_df["Feature"]
                            .replace(feature_name_mapping)
                            .str.replace(
                                "Country_",
                                "Country: ",
                                regex=False,
                            )
                        )

                    if "SHAP Value" in explanation_df.columns:
                        explanation_df["SHAP Value"] = (
                            pd.to_numeric(
                                explanation_df["SHAP Value"],
                                errors="coerce",
                            ).round(4)
                        )

                    st.dataframe(
                        explanation_df,
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.subheader("Plain-Language Explanation")

                    for index, feature in enumerate(top_features, start=1):

                        original_feature_name = feature.get(
                            "feature",
                            "Unknown feature",
                    )

                        feature_name = feature_name_mapping.get(
                        original_feature_name,
                        original_feature_name.replace(
                                "Country_",
                                "Country: ",
                        ),
                    )
                        impact = feature.get(
                            "impact",
                            "Impact unavailable",
                        )

                        st.markdown(
                            f"**{index}. {feature_name}** — {impact}"
                    )

                else:
                    st.info(
                        "The prediction was generated, but no "
                        "feature explanations were returned."
                    )

            else:
                try:
                    error_detail = response.json().get(
                        "detail",
                        response.text,
                    )

                except ValueError:
                    error_detail = response.text

                st.error(
                    "Explanation failed. "
                    f"API response: {error_detail}"
                )

        except requests.exceptions.ConnectionError:
            st.error(
                "Could not connect to the FastAPI service. "
                "Make sure the API is running on port 8000."
            )

        except requests.exceptions.Timeout:
            st.error(
                "The explanation request timed out."
            )

        except requests.exceptions.RequestException as error:
            st.error(
                f"API request failed: {error}"
            )

        except Exception as error:
            st.error(
                f"Unexpected error: {error}"
            )
# ---------------------------------------------------------
# BATCH PREDICTION PAGE
# ---------------------------------------------------------

elif page == "Batch Prediction":

    st.header("Batch Customer Churn Prediction")

    st.write(
        "Upload a CSV file containing multiple customers. "
        "The model will predict churn for each customer."
    )

    uploaded_file = st.file_uploader(
        "Upload Customer CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        st.success("CSV uploaded successfully!")

        preview_df = pd.read_csv(uploaded_file)

        st.subheader("Dataset Preview")

        st.dataframe(
            preview_df.head(),
            use_container_width=True
        )

        uploaded_file.seek(0)

        if st.button("🚀 Run Batch Prediction"):

            try:

                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file,
                        "text/csv"
                    )
                }

                response = requests.post(
                    f"{API_BASE_URL}/predict/batch",
                    files=files,
                    timeout=120
                )

                if response.status_code == 200:

                    st.success(
                        "Batch prediction completed successfully!"
                    )

                    st.download_button(
                        label="⬇️ Download Prediction Results",
                        data=response.content,
                        file_name="batch_churn_predictions.csv",
                        mime="text/csv"
                    )

                else:

                    st.error(
                        f"API Error: {response.text}"
                    )

            except requests.exceptions.ConnectionError:

                st.error(
                    "Could not connect to FastAPI."
                )

            except requests.exceptions.Timeout:

                st.error(
                    "Batch prediction timed out."
                )

            except Exception as error:

                st.error(
                    f"Unexpected error: {error}"
                )

# ---------------------------------------------------------
# PREDICTION HISTORY PAGE
# ---------------------------------------------------------

elif page == "Prediction History":

    st.header("Prediction History")

    st.write(
        "View the latest churn predictions saved in the PostgreSQL database."
    )

    limit = st.number_input(
        "Number of records to display",
        min_value=1,
        max_value=1000,
        value=100,
        step=10
    )

    if st.button("🔄 Load Prediction History"):

        try:
            response = requests.get(
                f"{API_BASE_URL}/predictions",
                params={"limit": int(limit)},
                timeout=30
            )

            if response.status_code == 200:

                history_data = response.json()

                if history_data:

                    history_df = pd.DataFrame(history_data)

                    history_df["churn_probability"] = (
                        history_df["churn_probability"] * 100
                    ).round(2)

                    history_df["churn_prediction"] = (
                        history_df["churn_prediction"]
                        .map({
                            1: "Likely to Churn",
                            0: "Not Likely to Churn"
                        })
                    )

                    history_df["created_at"] = pd.to_datetime(
                        history_df["created_at"],
                        errors="coerce"
                    )

                    history_df = history_df.rename(
                        columns={
                            "id": "ID",
                            "customer_id": "Customer ID",
                            "churn_prediction": "Prediction",
                            "churn_probability": "Churn Probability (%)",
                            "risk_level": "Risk Level",
                            "prediction_type": "Prediction Type",
                            "created_at": "Created At"
                        }
                    )

                    st.success(
                        f"{len(history_df)} prediction record(s) loaded."
                    )

                    st.dataframe(
                        history_df,
                        use_container_width=True,
                        hide_index=True
                    )

                    csv_data = history_df.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        label="⬇️ Download Prediction History",
                        data=csv_data,
                        file_name="prediction_history.csv",
                        mime="text/csv"
                    )

                else:
                    st.info(
                        "No prediction records are available yet."
                    )

            else:
                st.error(
                    f"Could not load prediction history. "
                    f"API response: {response.text}"
                )

        except requests.exceptions.ConnectionError:
            st.error(
                "Could not connect to FastAPI. "
                "Make sure the API is running on port 8000."
            )

        except requests.exceptions.Timeout:
            st.error(
                "The prediction history request timed out."
            )

        except Exception as error:
            st.error(
                f"Unexpected error: {error}"
            )

# ---------------------------------------------------------
# LOAD CUSTOMER SEGMENT DATA
# ---------------------------------------------------------

# ============================
# Load Data
# ============================

# Customer Segmentation Dataset
segments_df = pd.read_csv("data/final/customer_segments.csv")

# Raw Transaction Dataset (Used for Revenue Analytics)
transactions_df = pd.read_csv("data/raw/online_retail_ii_combined.csv")


# ============================
# Prepare Revenue Analytics Data
# ============================

transactions_df["InvoiceDate"] = pd.to_datetime(
    transactions_df["InvoiceDate"],
    errors="coerce"
)

transactions_df["Quantity"] = pd.to_numeric(
    transactions_df["Quantity"],
    errors="coerce"
)

transactions_df["Price"] = pd.to_numeric(
    transactions_df["Price"],
    errors="coerce"
)

# Remove invalid records
transactions_df = transactions_df.dropna(
    subset=["InvoiceDate", "Quantity", "Price"]
)

# Remove returns/cancelled orders
transactions_df = transactions_df[
    (transactions_df["Quantity"] > 0)
    & (transactions_df["Price"] > 0)
]

# Revenue per transaction
transactions_df["Revenue"] = (
    transactions_df["Quantity"]
    * transactions_df["Price"]
)

# Monthly revenue
transactions_df["Month"] = (
    transactions_df["InvoiceDate"]
        .dt.to_period("M")
        .dt.to_timestamp()
)

monthly_revenue_df = (
    transactions_df
    .groupby("Month", as_index=False)
    .agg(
        Revenue=("Revenue", "sum"),
        Orders=("Invoice", "nunique"),
        Quantity=("Quantity", "sum")
    )
    .sort_values("Month")
)
# ---------------------------------------------------------
# CUSTOMER SEGMENTATION PAGE
# ---------------------------------------------------------

if page == "Customer Segmentation":

    st.header("Customer Segmentation View")
        # Create customer value tiers from predicted CLV
    segments_df["Value_Tier"] = pd.qcut(
        segments_df["Predicted_CLV"],
        q=3,
        labels=["Low Value", "Medium Value", "High Value"],
        duplicates="drop"
    )

    filter_col1, filter_col2, filter_col3 = st.columns(3)

    with filter_col1:
        selected_segments = st.multiselect(
            "Filter by Segment",
            options=sorted(
                segments_df["Customer_Segment"].unique().tolist()
            ),
            default=sorted(
                segments_df["Customer_Segment"].unique().tolist()
            )
        )

    with filter_col2:
        selected_countries = st.multiselect(
            "Filter by Country",
            options=sorted(
                segments_df["Country"].dropna().unique().tolist()
            ),
            default=sorted(
                segments_df["Country"].dropna().unique().tolist()
            )
        )

    with filter_col3:
        selected_value_tiers = st.multiselect(
            "Filter by Value Tier",
            options=["Low Value", "Medium Value", "High Value"],
            default=["Low Value", "Medium Value", "High Value"]
        )

    filtered_segments_df = segments_df[
        segments_df["Customer_Segment"].isin(selected_segments)
        & segments_df["Country"].isin(selected_countries)
        & segments_df["Value_Tier"].isin(selected_value_tiers)
    ]

    segment_summary = (
        filtered_segments_df.groupby("Customer_Segment")
        .agg(
            Customer_Count=("Customer ID", "count"),
            Average_Recency=("Recency", "mean"),
            Average_Frequency=("Frequency", "mean"),
            Average_Revenue=("Monetary_Total", "mean"),
            Average_CLV=("Predicted_CLV", "mean"),
            Average_Engagement=("Engagement_Score", "mean"),
            Churn_Rate=("Churn", "mean")
        )
        .reset_index()
    )

    segment_summary["Average_Recency"] = (
        segment_summary["Average_Recency"].round(2)
    )

    segment_summary["Average_Frequency"] = (
        segment_summary["Average_Frequency"].round(2)
    )

    segment_summary["Average_Revenue"] = (
        segment_summary["Average_Revenue"].round(2)
    )

    segment_summary["Average_CLV"] = (
        segment_summary["Average_CLV"].round(2)
    )

    segment_summary["Average_Engagement"] = (
        segment_summary["Average_Engagement"].round(2)
    )

    segment_summary["Churn_Rate"] = (
        segment_summary["Churn_Rate"] * 100
    ).round(2)

    st.dataframe(
        segment_summary,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Customer Count by Segment")

    st.bar_chart(
        segment_summary.set_index(
            "Customer_Segment"
        )["Customer_Count"]
    )


# ---------------------------------------------------------
# CHURN LEADERBOARD PAGE
# ---------------------------------------------------------

elif page == "Churn Leaderboard":

    st.header("Churn Risk Leaderboard")

    top_risk = (
        segments_df.sort_values(
            by=["Churn", "Predicted_CLV"],
            ascending=[False, False]
        )
        .head(20)
    )

    st.write(
        "Customers with the highest churn risk and predicted "
        "lifetime value are displayed first."
    )

    st.dataframe(
        top_risk[
            [
                "Customer ID",
                "Customer_Segment",
                "Predicted_CLV",
                "Recency",
                "Frequency",
                "Monetary_Total",
                "Churn"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )
# ---------------------------------------------------------
# REVENUE & CLV ANALYTICS PAGE
# ---------------------------------------------------------

elif page == "Revenue & CLV Analytics":

    st.header("Revenue & Customer Lifetime Value Analytics")

    st.write(
        "Analyze monthly sales performance, revenue trends, "
        "order volume, and customer lifetime value."
    )

    # -----------------------------------------------------
    # Prepare and validate revenue data
    # -----------------------------------------------------

    revenue_data = monthly_revenue_df[
        ["Month", "Revenue", "Orders", "Quantity"]
    ].copy()

    revenue_data["Month"] = pd.to_datetime(
        revenue_data["Month"],
        errors="coerce",
    )

    revenue_data["Revenue"] = pd.to_numeric(
        revenue_data["Revenue"],
        errors="coerce",
    )

    revenue_data["Orders"] = pd.to_numeric(
        revenue_data["Orders"],
        errors="coerce",
    )

    revenue_data["Quantity"] = pd.to_numeric(
        revenue_data["Quantity"],
        errors="coerce",
    )

    revenue_data = (
        revenue_data.dropna(
            subset=[
                "Month",
                "Revenue",
                "Orders",
                "Quantity",
            ]
        )
        .sort_values("Month")
        .reset_index(drop=True)
    )

    if revenue_data.empty:
        st.error("Revenue analytics data is unavailable.")
        st.stop()

    # -----------------------------------------------------
    # Overall KPIs
    # -----------------------------------------------------

    total_revenue = revenue_data["Revenue"].sum()
    total_orders = revenue_data["Orders"].sum()
    total_quantity = revenue_data["Quantity"].sum()

    metric_col1, metric_col2, metric_col3 = st.columns(3)

    with metric_col1:
        st.metric(
            "Total Revenue",
            f"£{total_revenue:,.2f}",
        )

    with metric_col2:
        st.metric(
            "Total Orders",
            f"{total_orders:,.0f}",
        )

    with metric_col3:
        st.metric(
            "Total Quantity Sold",
            f"{total_quantity:,.0f}",
        )

    st.markdown("---")

    # -----------------------------------------------------
    # Monthly Revenue Trend with Forecast
    # -----------------------------------------------------

    st.subheader("Monthly Revenue Trend with 3-Month Forecast")

    historical_revenue = revenue_data[
        ["Month", "Revenue"]
    ].copy()

    historical_revenue = historical_revenue.rename(
        columns={
            "Revenue": "Actual Revenue",
        }
    )

    # Create a simple three-month linear trend forecast.
    if len(historical_revenue) >= 2:

        historical_x = np.arange(
            len(historical_revenue),
            dtype=float,
        )

        actual_revenue_values = historical_revenue[
            "Actual Revenue"
        ].astype(float).to_numpy()

        trend_slope, trend_intercept = np.polyfit(
            historical_x,
            actual_revenue_values,
            1,
        )

        forecast_months = pd.date_range(
            start=(
                historical_revenue["Month"].max()
                + pd.offsets.MonthBegin(1)
            ),
            periods=3,
            freq="MS",
        )

        forecast_x = np.arange(
            len(historical_revenue),
            len(historical_revenue) + 3,
            dtype=float,
        )

        forecast_values = (
            trend_slope * forecast_x
            + trend_intercept
        )

        forecast_df = pd.DataFrame(
            {
                "Month": forecast_months,
                "Forecast Revenue": forecast_values,
            }
        )

        revenue_chart_df = (
            historical_revenue.merge(
                forecast_df,
                on="Month",
                how="outer",
            )
            .sort_values("Month")
            .set_index("Month")
        )

        st.line_chart(
            revenue_chart_df,
            use_container_width=True,
        )

        st.caption(
            "The next three months are estimated using "
            "a simple linear trend fitted to historical revenue."
        )

    else:
        st.line_chart(
            historical_revenue.set_index("Month"),
            use_container_width=True,
        )

        st.warning(
            "At least two months of historical data are required "
            "to generate a revenue forecast."
        )

    # -----------------------------------------------------
    # Monthly Orders and Quantity
    # -----------------------------------------------------

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Monthly Order Volume")

        st.bar_chart(
            revenue_data.set_index("Month")["Orders"],
            use_container_width=True,
        )

    with chart_col2:
        st.subheader("Monthly Quantity Sold")

        st.bar_chart(
            revenue_data.set_index("Month")["Quantity"],
            use_container_width=True,
        )

    st.markdown("---")

    # -----------------------------------------------------
    # Revenue and CLV by Customer Segment
    # -----------------------------------------------------

    st.subheader("Revenue and CLV by Customer Segment")

    segment_revenue_df = (
        segments_df.groupby(
            "Customer_Segment",
            as_index=False,
        )
        .agg(
            Total_Revenue=(
                "Monetary_Total",
                "sum",
            ),
            Average_CLV=(
                "Predicted_CLV",
                "mean",
            ),
            Customer_Count=(
                "Customer ID",
                "count",
            ),
        )
    )

    segment_revenue_df["Total_Revenue"] = (
        pd.to_numeric(
            segment_revenue_df["Total_Revenue"],
            errors="coerce",
        )
        .fillna(0)
        .round(2)
    )

    segment_revenue_df["Average_CLV"] = (
        pd.to_numeric(
            segment_revenue_df["Average_CLV"],
            errors="coerce",
        )
        .fillna(0)
        .round(2)
    )

    st.dataframe(
        segment_revenue_df,
        use_container_width=True,
        hide_index=True,
    )

    segment_col1, segment_col2 = st.columns(2)

    with segment_col1:
        st.subheader("Total Revenue by Segment")

        st.bar_chart(
            segment_revenue_df.set_index(
                "Customer_Segment"
            )["Total_Revenue"],
            use_container_width=True,
        )

    with segment_col2:
        st.subheader("Average Predicted CLV by Segment")

        st.bar_chart(
            segment_revenue_df.set_index(
                "Customer_Segment"
            )["Average_CLV"],
            use_container_width=True,
        )
import os

import numpy as np
import pandas as pd
import requests
import streamlit as st
from components import metric_card, page_title
from styles import load_css

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Vantara Customer Intelligence Platform", page_icon="📊", layout="wide"
)
load_css()


@st.cache_data(show_spinner=False)
def load_dashboard_data():
    """Load and prepare shared analytics datasets once per session."""
    segments = pd.read_csv("data/final/customer_segments.csv")
    transactions = pd.read_csv("data/raw/online_retail_ii_combined.csv")

    transactions["InvoiceDate"] = pd.to_datetime(
        transactions["InvoiceDate"],
        errors="coerce",
    )
    transactions["Quantity"] = pd.to_numeric(
        transactions["Quantity"],
        errors="coerce",
    )
    transactions["Price"] = pd.to_numeric(
        transactions["Price"],
        errors="coerce",
    )

    transactions = transactions.dropna(subset=["InvoiceDate", "Quantity", "Price"])
    transactions = transactions[
        (transactions["Quantity"] > 0) & (transactions["Price"] > 0)
    ].copy()

    transactions["Revenue"] = transactions["Quantity"] * transactions["Price"]
    transactions["Month"] = (
        transactions["InvoiceDate"].dt.to_period("M").dt.to_timestamp()
    )

    monthly_revenue = (
        transactions.groupby("Month", as_index=False)
        .agg(
            Revenue=("Revenue", "sum"),
            Orders=("Invoice", "nunique"),
            Quantity=("Quantity", "sum"),
        )
        .sort_values("Month")
    )

    return segments, transactions, monthly_revenue


@st.cache_data(ttl=15, show_spinner=False)
def get_api_status():
    """Return the live FastAPI health state."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/health",
            timeout=3,
        )
        return response.status_code == 200
    except requests.RequestException:
        return False


@st.cache_data(ttl=30, show_spinner=False)
def get_prediction_history(limit=1000):
    """Fetch stored predictions for overview metrics."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/predictions",
            params={"limit": int(limit)},
            timeout=10,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return []


segments_df, transactions_df, monthly_revenue_df = load_dashboard_data()
api_connected = get_api_status()
prediction_history = get_prediction_history()


st.sidebar.markdown(
    """
    <div style="padding: 0.5rem 0 1rem 0;">
        <div style="font-size: 1.45rem; font-weight: 800;">VANTARA</div>
        <div style="font-size: 0.85rem; opacity: 0.78;">
            Customer Intelligence Platform
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "🎯 Churn Prediction",
        "🧠 Customer Explanation",
        "📂 Batch Prediction",
        "🕒 Prediction History",
        "👥 Customer Segmentation",
        "⚠️ Churn Leaderboard",
        "📈 Revenue & CLV Analytics",
        "🖥️ System Status",
    ],
)

st.sidebar.markdown("---")
if api_connected:
    st.sidebar.success("API Connected")
else:
    st.sidebar.error("API Offline")
st.sidebar.caption("Version 2.0")

# ---------------------------------------------------------
# CHURN PREDICTION PAGE
# ---------------------------------------------------------
if page == "🏠 Dashboard":

    page_title(
        "🏠 Executive Dashboard",
        "Live customer intelligence, churn risk, revenue and model activity",
    )

    total_customers = len(segments_df)
    total_segments = int(segments_df["Customer_Segment"].nunique())
    high_risk_customers = int((segments_df["Churn"] == 1).sum())
    total_predictions = len(prediction_history)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("Total Customers", f"{total_customers:,}")

    with c2:
        metric_card("Customer Segments", f"{total_segments:,}")

    with c3:
        metric_card("Stored Predictions", f"{total_predictions:,}")

    with c4:
        metric_card("High-Risk Customers", f"{high_risk_customers:,}")

    st.write("")
    left, right = st.columns([2, 1])

    with left:
        st.subheader("Monthly Revenue Trend")
        st.line_chart(
            monthly_revenue_df.set_index("Month")[["Revenue"]],
            use_container_width=True,
        )

    with right:
        st.subheader("Customer Distribution")
        segment_counts = (
            segments_df["Customer_Segment"]
            .value_counts()
            .sort_index()
            .rename("Customers")
        )
        st.bar_chart(segment_counts, use_container_width=True)

    st.write("")
    insight_col, status_col = st.columns([2, 1])

    with insight_col:
        high_risk_rate = (
            high_risk_customers / total_customers * 100 if total_customers else 0
        )
        st.markdown(
            f"""
            <div class="warning-card">
                <strong>Business Insight</strong><br>
                {high_risk_rate:.1f}% of analysed customers are marked as
                high churn risk. Prioritise high-value customers in the
                churn leaderboard for targeted retention campaigns.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with status_col:
        if api_connected:
            st.markdown(
                """
                <div class="success-card">
                    <strong>System Status</strong><br>
                    FastAPI is healthy and ready to serve predictions.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="danger-card">
                    <strong>System Status</strong><br>
                    FastAPI is offline. Start the API or Docker stack.
                </div>
                """,
                unsafe_allow_html=True,
            )

    if prediction_history:
        st.write("")
        st.subheader("Recent Predictions")
        recent_df = pd.DataFrame(prediction_history[:8]).copy()
        if not recent_df.empty:
            if "churn_probability" in recent_df.columns:
                recent_df["churn_probability"] = (
                    pd.to_numeric(
                        recent_df["churn_probability"],
                        errors="coerce",
                    )
                    * 100
                ).round(2)
            recent_df = recent_df.rename(
                columns={
                    "customer_id": "Customer ID",
                    "churn_probability": "Churn Probability (%)",
                    "risk_level": "Risk Level",
                    "prediction_type": "Prediction Type",
                    "created_at": "Created At",
                }
            )
            display_cols = [
                column
                for column in [
                    "Customer ID",
                    "Churn Probability (%)",
                    "Risk Level",
                    "Prediction Type",
                    "Created At",
                ]
                if column in recent_df.columns
            ]
            st.dataframe(
                recent_df[display_cols],
                use_container_width=True,
                hide_index=True,
            )


elif page == "🎯 Churn Prediction":

    st.header("🎯 Customer Churn Prediction")

    st.write(
        "Fill in the customer details below and click "
        "**Predict Customer Churn** to analyze the customer's risk."
    )

    col1, col2 = st.columns(2)

    with col1:
        recency = st.number_input("Recency", min_value=0.0, value=25.0)

        frequency = st.number_input("Frequency", min_value=0.0, value=12.0)

        monetary_total = st.number_input("Monetary Total", min_value=0.0, value=2500.0)

        monetary_average = st.number_input(
            "Monetary Average", min_value=0.0, value=208.3
        )

        total_quantity = st.number_input("Total Quantity", min_value=0.0, value=40.0)

        average_basket_size = st.number_input(
            "Average Basket Size", min_value=0.0, value=3.3
        )

        average_order_value = st.number_input(
            "Average Order Value", min_value=0.0, value=208.3
        )

    with col2:
        unique_products = st.number_input("Unique Products", min_value=0.0, value=15.0)

        average_purchase_gap = st.number_input(
            "Average Purchase Gap", min_value=0.0, value=18.0
        )

        customer_lifespan_days = st.number_input(
            "Customer Lifespan Days", min_value=0.0, value=320.0
        )

        return_rate = st.number_input("Return Rate", min_value=0.0, value=0.02)

        price_sensitivity = st.number_input(
            "Price Sensitivity", min_value=0.0, value=0.35
        )

        predicted_clv = st.number_input("Predicted CLV", min_value=0.0, value=4200.0)

        engagement_score = st.number_input(
            "Engagement Score", min_value=0.0, value=78.0
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
            "Country": country,
        }

        try:
            response = requests.post(
                f"{API_BASE_URL}/predict", json=payload, timeout=30
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
                        "Churn Probability", f"{result['churn_probability'] * 100:.2f}%"
                    )

                with metric_col2:
                    st.metric("Risk Level", result["risk_level"])

                with metric_col3:
                    st.metric(
                        "Prediction",
                        (
                            "Likely to Churn"
                            if result["churn_prediction"] == 1
                            else "Not Likely to Churn"
                        ),
                    )

            else:
                st.error(f"Prediction failed. API response: {response.text}")

        except requests.exceptions.ConnectionError:
            st.error(
                "Could not connect to the FastAPI service. "
                "Make sure the API is running on port 8000."
            )

        except requests.exceptions.Timeout:
            st.error("The prediction request timed out.")

        except Exception as error:
            st.error(f"Unexpected error: {error}")
# ---------------------------------------------------------
# CUSTOMER EXPLANATION PAGE
# ---------------------------------------------------------

elif page == "🧠 Customer Explanation":

    st.header("🧠 Customer Churn Explanation")

    st.write(
        "Enter customer information to predict churn and view "
        "the most influential features affecting the prediction."
    )
    st.subheader("Search Existing Customer")

    customer_id_lookup = st.text_input(
        "Customer ID",
        placeholder="Example: 12354",
        key="exp_customer_id_lookup",
    )

    if st.button(
        "🔎 Load Customer",
        key="load_customer_button",
    ):
        if not customer_id_lookup.strip():
            st.warning("Enter a Customer ID first.")
        else:
            try:
                response = requests.get(
                    f"{API_BASE_URL}/dashboard/customers/"
                    f"{customer_id_lookup.strip()}",
                    timeout=10,
                )

                if response.status_code == 200:
                    customer_data = response.json()

                    st.session_state["exp_recency"] = float(
                        customer_data.get("recency") or 0
                    )

                    st.session_state["exp_frequency"] = float(
                        customer_data.get("frequency") or 0
                    )

                    st.session_state["exp_monetary_total"] = float(
                        customer_data.get("monetary_total") or 0
                    )

                    st.session_state["exp_predicted_clv"] = float(
                        customer_data.get("predicted_clv") or 0
                    )

                    st.session_state["exp_engagement_score"] = float(
                        customer_data.get("engagement_score") or 0
                    )

                    country_value = customer_data.get("country")

                    if country_value:
                        st.session_state["exp_country"] = country_value

                    st.success(
                        f"Customer {customer_data['customer_id']} loaded. "
                        f"Segment: {customer_data.get('segment', 'Unknown')}"
                    )

                    st.rerun()

                elif response.status_code == 404:
                    st.error("Customer ID was not found.")

                else:
                    st.error(
                        f"Customer lookup failed: {response.text}"
                    )

            except requests.exceptions.ConnectionError:
                st.error(
                    "Could not connect to FastAPI."
                )

            except requests.exceptions.RequestException as error:
                st.error(
                    f"Customer lookup failed: {error}"
                )

    st.markdown("---")
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
            with st.spinner("Generating prediction and SHAP explanation..."):
                response = requests.post(
                    f"{API_BASE_URL}/predict/explain",
                    json=payload,
                    timeout=60,
                )

            if response.status_code == 200:
                result = response.json()

                st.success("Prediction explanation generated successfully.")

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
                    "Predicted_CLV": ("Predicted Customer Lifetime Value"),
                }

                if top_features:
                    explanation_df = pd.DataFrame(top_features)

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
                        explanation_df = explanation_df[display_columns]

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
                        explanation_df["SHAP Value"] = pd.to_numeric(
                            explanation_df["SHAP Value"],
                            errors="coerce",
                        ).round(4)

                    st.dataframe(
                        explanation_df,
                        use_container_width=True,
                        hide_index=True,
                    )

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

                        st.markdown(f"**{index}. {feature_name}** — {impact}")

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

                st.error("Explanation failed. " f"API response: {error_detail}")

        except requests.exceptions.ConnectionError:
            st.error(
                "Could not connect to the FastAPI service. "
                "Make sure the API is running on port 8000."
            )

        except requests.exceptions.Timeout:
            st.error("The explanation request timed out.")

        except requests.exceptions.RequestException as error:
            st.error(f"API request failed: {error}")

        except Exception as error:
            st.error(f"Unexpected error: {error}")
# ---------------------------------------------------------
# BATCH PREDICTION PAGE
# ---------------------------------------------------------

elif page == "📂 Batch Prediction":

    st.header("📂 Batch Customer Churn Prediction")

    st.write(
        "Upload a CSV file containing multiple customers. "
        "The model will predict churn for each customer."
    )

    uploaded_file = st.file_uploader("Upload Customer CSV", type=["csv"])

    if uploaded_file is not None:

        st.success("CSV uploaded successfully!")

        preview_df = pd.read_csv(uploaded_file)

        st.subheader("Dataset Preview")

        st.dataframe(preview_df.head(), use_container_width=True)

        uploaded_file.seek(0)

        if st.button("🚀 Run Batch Prediction"):

            try:

                files = {"file": (uploaded_file.name, uploaded_file, "text/csv")}

                response = requests.post(
                    f"{API_BASE_URL}/predict/batch", files=files, timeout=120
                )

                if response.status_code == 200:

                    st.success("Batch prediction completed successfully!")

                    st.download_button(
                        label="⬇️ Download Prediction Results",
                        data=response.content,
                        file_name="batch_churn_predictions.csv",
                        mime="text/csv",
                    )

                else:

                    st.error(f"API Error: {response.text}")

            except requests.exceptions.ConnectionError:

                st.error("Could not connect to FastAPI.")

            except requests.exceptions.Timeout:

                st.error("Batch prediction timed out.")

            except Exception as error:

                st.error(f"Unexpected error: {error}")

# ---------------------------------------------------------
# PREDICTION HISTORY PAGE
# ---------------------------------------------------------

elif page == "🕒 Prediction History":

    st.header("🕒 Prediction History")

    st.write("View the latest churn predictions saved in the PostgreSQL database.")

    limit = st.number_input(
        "Number of records to display", min_value=1, max_value=1000, value=100, step=10
    )

    if st.button("🔄 Load Prediction History"):

        try:
            response = requests.get(
                f"{API_BASE_URL}/predictions", params={"limit": int(limit)}, timeout=30
            )

            if response.status_code == 200:

                history_data = response.json()

                if history_data:

                    history_df = pd.DataFrame(history_data)

                    history_df["churn_probability"] = (
                        history_df["churn_probability"] * 100
                    ).round(2)

                    history_df["churn_prediction"] = history_df["churn_prediction"].map(
                        {1: "Likely to Churn", 0: "Not Likely to Churn"}
                    )

                    history_df["created_at"] = pd.to_datetime(
                        history_df["created_at"], errors="coerce"
                    )

                    history_df = history_df.rename(
                        columns={
                            "id": "ID",
                            "customer_id": "Customer ID",
                            "churn_prediction": "Prediction",
                            "churn_probability": "Churn Probability (%)",
                            "risk_level": "Risk Level",
                            "prediction_type": "Prediction Type",
                            "created_at": "Created At",
                        }
                    )

                    st.success(f"{len(history_df)} prediction record(s) loaded.")

                    st.dataframe(history_df, use_container_width=True, hide_index=True)

                    csv_data = history_df.to_csv(index=False).encode("utf-8")

                    st.download_button(
                        label="⬇️ Download Prediction History",
                        data=csv_data,
                        file_name="prediction_history.csv",
                        mime="text/csv",
                    )

                else:
                    st.info("No prediction records are available yet.")

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
            st.error("The prediction history request timed out.")

        except Exception as error:
            st.error(f"Unexpected error: {error}")

# ---------------------------------------------------------
# CUSTOMER SEGMENTATION PAGE
# ---------------------------------------------------------

elif page == "👥 Customer Segmentation":

    st.header("👥 Customer Segmentation View")
    segment_name_map = {
        0: "Active Customers",
        1: "Champions",
        2: "Dormant Customers",
        3: "At-Risk Customers",
    }

    segments_df["Segment_Name"] = (
        segments_df["Customer_Segment"]
        .map(segment_name_map)
    )
    # Create customer value tiers from predicted CLV
    segments_df["Value_Tier"] = pd.qcut(
        segments_df["Predicted_CLV"],
        q=3,
        labels=["Low Value", "Medium Value", "High Value"],
        duplicates="drop",
    )

    filter_col1, filter_col2, filter_col3 = st.columns(3)

    with filter_col1:
        selected_segments = st.multiselect(
            "Filter by Segment",
    options=sorted(
        segments_df["Segment_Name"].dropna().unique().tolist()
    ),
    default=sorted(
        segments_df["Segment_Name"].dropna().unique().tolist()
    )
)

    with filter_col2:
        selected_countries = st.multiselect(
            "Filter by Country",
            options=sorted(segments_df["Country"].dropna().unique().tolist()),
            default=sorted(segments_df["Country"].dropna().unique().tolist()),
        )

    with filter_col3:
        selected_value_tiers = st.multiselect(
            "Filter by Value Tier",
            options=["Low Value", "Medium Value", "High Value"],
            default=["Low Value", "Medium Value", "High Value"],
        )

    filtered_segments_df = segments_df[
        segments_df["Segment_Name"].isin(selected_segments)
        & segments_df["Country"].isin(selected_countries)
        & segments_df["Value_Tier"].isin(selected_value_tiers)
    ]

    segment_summary = (
        filtered_segments_df.groupby("Segment_Name")
        .agg(
            Customer_Count=("Customer ID", "count"),
            Average_Recency=("Recency", "mean"),
            Average_Frequency=("Frequency", "mean"),
            Average_Revenue=("Monetary_Total", "mean"),
            Average_CLV=("Predicted_CLV", "mean"),
            Average_Engagement=("Engagement_Score", "mean"),
            Churn_Rate=("Churn", "mean"),
        )
        .reset_index()
    )

    segment_summary["Average_Recency"] = segment_summary["Average_Recency"].round(2)

    segment_summary["Average_Frequency"] = segment_summary["Average_Frequency"].round(2)

    segment_summary["Average_Revenue"] = segment_summary["Average_Revenue"].round(2)

    segment_summary["Average_CLV"] = segment_summary["Average_CLV"].round(2)

    segment_summary["Average_Engagement"] = segment_summary["Average_Engagement"].round(
        2
    )

    segment_summary["Churn_Rate"] = (segment_summary["Churn_Rate"] * 100).round(2)

    st.dataframe(segment_summary, use_container_width=True, hide_index=True)

    st.subheader("Customer Count by Segment")

    st.bar_chart(segment_summary.set_index("Segment_Name")["Customer_Count"])


# ---------------------------------------------------------
# CHURN LEADERBOARD PAGE
# ---------------------------------------------------------

elif page == "⚠️ Churn Leaderboard":

    st.header("⚠️ Churn Risk Leaderboard")

    top_risk = segments_df.sort_values(
        by=["Churn", "Predicted_CLV"], ascending=[False, False]
    ).head(20)

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
                "Churn",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )
# ---------------------------------------------------------
# REVENUE & CLV ANALYTICS PAGE
# ---------------------------------------------------------

elif page == "📈 Revenue & CLV Analytics":

    st.header("📈 Revenue & Customer Lifetime Value Analytics")

    st.write(
        "Analyze monthly sales performance, revenue trends, "
        "order volume, and customer lifetime value."
    )

    # -----------------------------------------------------
    # Prepare and validate revenue data
    # -----------------------------------------------------

    revenue_data = monthly_revenue_df[["Month", "Revenue", "Orders", "Quantity"]].copy()

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

    historical_revenue = revenue_data[["Month", "Revenue"]].copy()

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

        actual_revenue_values = (
            historical_revenue["Actual Revenue"].astype(float).to_numpy()
        )

        trend_slope, trend_intercept = np.polyfit(
            historical_x,
            actual_revenue_values,
            1,
        )

        forecast_months = pd.date_range(
            start=(historical_revenue["Month"].max() + pd.offsets.MonthBegin(1)),
            periods=3,
            freq="MS",
        )

        forecast_x = np.arange(
            len(historical_revenue),
            len(historical_revenue) + 3,
            dtype=float,
        )

        forecast_values = trend_slope * forecast_x + trend_intercept

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

    segment_revenue_df = segments_df.groupby(
        "Customer_Segment",
        as_index=False,
    ).agg(
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
            segment_revenue_df.set_index("Customer_Segment")["Total_Revenue"],
            use_container_width=True,
        )

    with segment_col2:
        st.subheader("Average Predicted CLV by Segment")

        st.bar_chart(
            segment_revenue_df.set_index("Customer_Segment")["Average_CLV"],
            use_container_width=True,
        )
    # ---------------------------------------------------------
    # SYSTEM STATUS PAGE
    # ---------------------------------------------------------

elif page == "🖥️ System Status":

    st.header("🖥️ System Status")

    st.write(
        "Check the availability of the API, datasets, models, "
        "database activity, and analytics components."
    )

    api_col, data_col, prediction_col = st.columns(3)

    with api_col:
        if api_connected:
            st.success("FastAPI: Connected")
        else:
            st.error("FastAPI: Offline")

    with data_col:
        st.success(f"Customer Dataset: {len(segments_df):,} records")

    with prediction_col:
        st.info(f"Stored Predictions: {len(prediction_history):,}")

    st.markdown("---")

    st.subheader("Data Status")

    data_status_col1, data_status_col2, data_status_col3 = st.columns(3)

    with data_status_col1:
        st.metric(
            "Customers",
            f"{len(segments_df):,}",
        )

    with data_status_col2:
        st.metric(
            "Customer Segments",
            int(segments_df["Customer_Segment"].nunique()),
        )

    with data_status_col3:
        st.metric(
            "Revenue Months",
            f"{len(monthly_revenue_df):,}",
        )

    st.markdown("---")

    st.subheader("Model and Feature Status")

    st.success("Churn Prediction Model: Ready")
    st.success("Customer Segmentation Model: Ready")
    st.success("Customer Lifetime Value Analytics: Ready")

    if api_connected:
        st.success("SHAP Explanation Service: Ready")
        st.success("Batch Prediction Service: Ready")
        st.success("PostgreSQL Prediction History: Available")
    else:
        st.warning("API-dependent services cannot be checked while FastAPI is offline.")

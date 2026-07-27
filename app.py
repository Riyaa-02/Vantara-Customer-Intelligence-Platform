import streamlit as st
import requests
import pandas as pd

st.set_page_config(
    page_title="Vantara Customer Intelligence Platform",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Vantara Customer Intelligence Platform")

st.markdown(
    """
Predict customer churn using an AI-powered neural network model.

Fill in the customer details below and click **Predict Customer Churn** to analyze the customer's churn probability and risk level.
"""
)
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    recency = st.number_input("Recency", min_value=0.0, value=25.0)
    frequency = st.number_input("Frequency", min_value=0.0, value=12.0)
    monetary_total = st.number_input("Monetary Total", min_value=0.0, value=2500.0)
    monetary_average = st.number_input("Monetary Average", min_value=0.0, value=208.3)
    total_quantity = st.number_input("Total Quantity", min_value=0.0, value=40.0)
    average_basket_size = st.number_input("Average Basket Size", min_value=0.0, value=3.3)
    average_order_value = st.number_input("Average Order Value", min_value=0.0, value=208.3)

with col2:
    unique_products = st.number_input("Unique Products", min_value=0.0, value=15.0)
    average_purchase_gap = st.number_input("Average Purchase Gap", min_value=0.0, value=18.0)
    customer_lifespan_days = st.number_input("Customer Lifespan Days", min_value=0.0, value=320.0)
    return_rate = st.number_input("Return Rate", min_value=0.0, value=0.02)
    price_sensitivity = st.number_input("Price Sensitivity", min_value=0.0, value=0.35)
    predicted_clv = st.number_input("Predicted CLV", min_value=0.0, value=4200.0)
    engagement_score = st.number_input("Engagement Score", min_value=0.0, value=78.0)

country = st.selectbox(
    "Country",
    [
        "Australia","Austria","Bahrain","Belgium","Brazil","Canada",
        "Channel Islands","Cyprus","Czech Republic","Denmark","EIRE",
        "European Community","Finland","France","Germany","Greece",
        "Iceland","Israel","Italy","Japan","Korea","Lebanon",
        "Lithuania","Malta","Netherlands","Nigeria","Norway","Poland",
        "Portugal","RSA","Saudi Arabia","Singapore","Spain","Sweden",
        "Switzerland","Thailand","USA","United Arab Emirates",
        "United Kingdom","Unspecified","West Indies"
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
            "http://127.0.0.1:8000/predict",
            json=payload
        )

        if response.status_code == 200:

            result = response.json()

            st.success("Prediction Completed Successfully!")

            st.metric(
                "Churn Probability",
                f"{result['churn_probability']*100:.2f}%"
            )

            st.metric(
                "Risk Level",
                result["risk_level"]
            )

            st.metric(
                "Prediction",
                "Likely to Churn"
                if result["churn_prediction"] == 1
                else "Not Likely to Churn"
            )

        else:
            st.error(response.text)

    except Exception as e:
        st.error(f"Connection Error: {e}")
st.markdown("---")
st.header("Customer Segmentation View")

segments_df = pd.read_csv("data/final/customer_segments.csv")

segment_summary = (
    segments_df.groupby("Customer_Segment")
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

segment_summary["Average_Recency"] = segment_summary["Average_Recency"].round(2)
segment_summary["Average_Frequency"] = segment_summary["Average_Frequency"].round(2)
segment_summary["Average_Revenue"] = segment_summary["Average_Revenue"].round(2)
segment_summary["Average_CLV"] = segment_summary["Average_CLV"].round(2)
segment_summary["Average_Engagement"] = segment_summary["Average_Engagement"].round(2)
segment_summary["Churn_Rate"] = (
    segment_summary["Churn_Rate"] * 100
).round(2)

st.dataframe(
    segment_summary,
    use_container_width=True,
    hide_index=True
)

st.bar_chart(
    segment_summary.set_index("Customer_Segment")["Customer_Count"]
)
st.markdown("---")
st.header("Churn Leaderboard")

top_risk = (
    segments_df.sort_values(
        by=["Churn", "Predicted_CLV"],
        ascending=[False, False]
    )
    .head(20)
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
st.markdown("---")
st.header("Revenue Trends")

revenue_df = (
    segments_df.groupby("Customer_Segment")
    .agg(
        Total_Revenue=("Monetary_Total", "sum"),
        Average_CLV=("Predicted_CLV", "mean")
    )
    .reset_index()
)

st.subheader("Total Revenue by Customer Segment")
st.bar_chart(
    revenue_df.set_index("Customer_Segment")["Total_Revenue"]
)

st.subheader("Average Predicted CLV by Customer Segment")
st.line_chart(
    revenue_df.set_index("Customer_Segment")["Average_CLV"]
)
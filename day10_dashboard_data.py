import os
import json
import pandas as pd


# --------------------------------------------------
# File Paths
# --------------------------------------------------

INPUT_PATH = "data/final/customer_recommendations.csv"

KPI_OUTPUT_PATH = "data/final/dashboard_kpis.json"
SEGMENT_OUTPUT_PATH = "data/final/dashboard_segments.csv"
RISK_OUTPUT_PATH = "data/final/dashboard_churn_risk.csv"
VALUE_OUTPUT_PATH = "data/final/dashboard_customer_value.csv"
PRIORITY_OUTPUT_PATH = "data/final/dashboard_priority.csv"
STRATEGY_OUTPUT_PATH = "data/final/dashboard_strategy.csv"


# --------------------------------------------------
# 1. Load Final Customer Dataset
# --------------------------------------------------

print("Loading customer recommendations dataset...")

df = pd.read_csv(INPUT_PATH)

print(f"Dataset Shape: {df.shape}")


# --------------------------------------------------
# 2. Validate Important Columns
# --------------------------------------------------

print("\nValidating dashboard columns...")

required_columns = [
    "Predicted_CLV",
    "Churn_Probability",
    "Customer_Segment",
    "Churn_Risk",
    "Customer_Value",
    "Customer_Priority",
    "Customer_Strategy"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print("Dashboard data validation completed successfully.")


# --------------------------------------------------
# 3. Generate Main Dashboard KPIs
# --------------------------------------------------

print("\nGenerating dashboard KPIs...")

total_customers = len(df)

average_clv = df["Predicted_CLV"].mean()

average_churn_probability = (
    df["Churn_Probability"].mean()
)

high_risk_customers = (
    df["Churn_Risk"] == "High Risk"
).sum()

medium_risk_customers = (
    df["Churn_Risk"] == "Medium Risk"
).sum()

low_risk_customers = (
    df["Churn_Risk"] == "Low Risk"
).sum()

high_value_customers = (
    df["Customer_Value"] == "High Value"
).sum()

critical_customers = (
    df["Customer_Priority"] == "Critical"
).sum()


dashboard_kpis = {
    "Total_Customers": int(total_customers),
    "Average_Predicted_CLV": round(
        float(average_clv),
        2
    ),
    "Average_Churn_Probability": round(
        float(average_churn_probability),
        4
    ),
    "High_Risk_Customers": int(
        high_risk_customers
    ),
    "Medium_Risk_Customers": int(
        medium_risk_customers
    ),
    "Low_Risk_Customers": int(
        low_risk_customers
    ),
    "High_Value_Customers": int(
        high_value_customers
    ),
    "Critical_Priority_Customers": int(
        critical_customers
    )
}


# --------------------------------------------------
# 4. Save KPI Data
# --------------------------------------------------

os.makedirs(
    os.path.dirname(KPI_OUTPUT_PATH),
    exist_ok=True
)

with open(
    KPI_OUTPUT_PATH,
    "w"
) as file:

    json.dump(
        dashboard_kpis,
        file,
        indent=4
    )

print("Dashboard KPIs saved successfully.")


# --------------------------------------------------
# 5. Prepare Segment Distribution
# --------------------------------------------------

segment_data = (
    df["Customer_Segment"]
    .value_counts()
    .sort_index()
    .reset_index()
)

segment_data.columns = [
    "Customer_Segment",
    "Customer_Count"
]

segment_data.to_csv(
    SEGMENT_OUTPUT_PATH,
    index=False
)


# --------------------------------------------------
# 6. Prepare Churn Risk Distribution
# --------------------------------------------------

risk_data = (
    df["Churn_Risk"]
    .value_counts()
    .reset_index()
)

risk_data.columns = [
    "Churn_Risk",
    "Customer_Count"
]

risk_data.to_csv(
    RISK_OUTPUT_PATH,
    index=False
)


# --------------------------------------------------
# 7. Prepare Customer Value Distribution
# --------------------------------------------------

value_data = (
    df["Customer_Value"]
    .value_counts()
    .reset_index()
)

value_data.columns = [
    "Customer_Value",
    "Customer_Count"
]

value_data.to_csv(
    VALUE_OUTPUT_PATH,
    index=False
)


# --------------------------------------------------
# 8. Prepare Customer Priority Distribution
# --------------------------------------------------

priority_data = (
    df["Customer_Priority"]
    .value_counts()
    .reset_index()
)

priority_data.columns = [
    "Customer_Priority",
    "Customer_Count"
]

priority_data.to_csv(
    PRIORITY_OUTPUT_PATH,
    index=False
)


# --------------------------------------------------
# 9. Prepare Strategy Distribution
# --------------------------------------------------

strategy_data = (
    df["Customer_Strategy"]
    .value_counts()
    .reset_index()
)

strategy_data.columns = [
    "Customer_Strategy",
    "Customer_Count"
]

strategy_data.to_csv(
    STRATEGY_OUTPUT_PATH,
    index=False
)


# --------------------------------------------------
# 10. Display Dashboard Summary
# --------------------------------------------------

print("\nDashboard KPI Summary")
print("-" * 45)

for key, value in dashboard_kpis.items():
    print(f"{key}: {value}")


print("\nDashboard files created:")

print(KPI_OUTPUT_PATH)
print(SEGMENT_OUTPUT_PATH)
print(RISK_OUTPUT_PATH)
print(VALUE_OUTPUT_PATH)
print(PRIORITY_OUTPUT_PATH)
print(STRATEGY_OUTPUT_PATH)


print("\nDay 10 completed successfully.")
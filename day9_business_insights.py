import os
import pandas as pd


# --------------------------------------------------
# File Paths
# --------------------------------------------------

INPUT_PATH = "data/final/customer_intelligence.csv"
OUTPUT_PATH = "data/final/customer_recommendations.csv"


# --------------------------------------------------
# 1. Load Customer Intelligence Data
# --------------------------------------------------

print("Loading customer intelligence dataset...")

df = pd.read_csv(INPUT_PATH)

print(f"Dataset Shape: {df.shape}")


# --------------------------------------------------
# 2. Validate Required Columns
# --------------------------------------------------

print("\nValidating required columns...")

required_columns = [
    "Recency",
    "Frequency",
    "Monetary_Total",
    "Predicted_CLV",
    "Engagement_Score",
    "Customer_Segment",
    "Churn_Risk",
    "Customer_Value"
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

print("Dataset validation completed successfully.")


# --------------------------------------------------
# 3. Generate Recommended Actions
# --------------------------------------------------

print("\nGenerating customer recommendations...")


def generate_recommendation(row):

    churn_risk = row["Churn_Risk"]
    customer_value = row["Customer_Value"]

    # High-value customers needing attention
    if (
        customer_value == "High Value"
        and churn_risk in ["Medium Risk", "High Risk"]
    ):
        return (
            "Priority Retention: Provide personalized offers, "
            "loyalty rewards, and proactive engagement."
        )

    # High-value loyal customers
    elif (
        customer_value == "High Value"
        and churn_risk == "Low Risk"
    ):
        return (
            "VIP Growth: Offer premium benefits, "
            "exclusive products, and loyalty rewards."
        )

    # Medium-value customers at risk
    elif (
        customer_value == "Medium Value"
        and churn_risk in ["Medium Risk", "High Risk"]
    ):
        return (
            "Re-engagement: Send targeted discounts "
            "and personalized product recommendations."
        )

    # Low-value customers at risk
    elif (
        customer_value == "Low Value"
        and churn_risk in ["Medium Risk", "High Risk"]
    ):
        return (
            "Automated Retention: Use low-cost email campaigns "
            "and promotional offers."
        )

    # Other customers
    else:
        return (
            "Maintain Engagement: Continue regular communication "
            "and monitor customer activity."
        )


df["Recommended_Action"] = df.apply(
    generate_recommendation,
    axis=1
)


# --------------------------------------------------
# 4. Create Customer Priority Levels
# --------------------------------------------------

print("Creating customer priority levels...")


def assign_priority(row):

    if (
        row["Customer_Value"] == "High Value"
        and row["Churn_Risk"] in ["Medium Risk", "High Risk"]
    ):
        return "Critical"

    elif (
        row["Customer_Value"] == "Medium Value"
        and row["Churn_Risk"] in ["Medium Risk", "High Risk"]
    ):
        return "High"

    elif row["Churn_Risk"] in ["Medium Risk", "High Risk"]:
        return "Medium"

    else:
        return "Low"


df["Customer_Priority"] = df.apply(
    assign_priority,
    axis=1
)


# --------------------------------------------------
# 5. Create Customer Strategy Labels
# --------------------------------------------------

print("Creating customer strategy labels...")


def assign_strategy(row):

    if row["Customer_Priority"] == "Critical":
        return "Retain Immediately"

    elif row["Customer_Priority"] == "High":
        return "Re-engage Customer"

    elif row["Customer_Value"] == "High Value":
        return "Grow Customer Value"

    elif row["Customer_Priority"] == "Medium":
        return "Automated Retention"

    else:
        return "Maintain Relationship"


df["Customer_Strategy"] = df.apply(
    assign_strategy,
    axis=1
)


# --------------------------------------------------
# 6. Save Recommendations Dataset
# --------------------------------------------------

print("\nSaving customer recommendations...")

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print(f"Final Dataset Shape: {df.shape}")
print(f"Saved to: {OUTPUT_PATH}")


# --------------------------------------------------
# 7. Display Business Insights
# --------------------------------------------------

print("\nBusiness Insights Summary")
print("-" * 45)

print("\nCustomer Priority Distribution:")
print(df["Customer_Priority"].value_counts())

print("\nCustomer Strategy Distribution:")
print(df["Customer_Strategy"].value_counts())

print("\nRecommended Action Distribution:")
print(df["Recommended_Action"].value_counts())


# --------------------------------------------------
# 8. Display Key Business Metrics
# --------------------------------------------------

print("\nKey Business Metrics")
print("-" * 45)

print(f"Total Customers: {len(df)}")

critical_customers = (
    df["Customer_Priority"] == "Critical"
).sum()

print(
    f"Critical Priority Customers: "
    f"{critical_customers}"
)

high_value_customers = (
    df["Customer_Value"] == "High Value"
).sum()

print(
    f"High Value Customers: "
    f"{high_value_customers}"
)

at_risk_customers = (
    df["Churn_Risk"].isin(
        ["Medium Risk", "High Risk"]
    )
).sum()

print(
    f"Customers Requiring Retention Attention: "
    f"{at_risk_customers}"
)


print("\nDay 9 completed successfully.")
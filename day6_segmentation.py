"""
Day 6: Customer Segmentation
Vantara Customer Intelligence Platform
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib


# -------------------------------
# 1. Load Customer Features
# -------------------------------

print("Loading customer features...")

df = pd.read_csv(
    "data/processed/customer_features.csv"
)

print("Dataset shape:", df.shape)


# -------------------------------
# 2. Select Segmentation Features
# -------------------------------

features = [
    "Recency",
    "Frequency",
    "Monetary_Total",
    "Average_Order_Value",
    "Unique_Products",
    "Engagement_Score",
    "Return_Rate"
]

X = df[features]


# -------------------------------
# 3. Feature Scaling
# -------------------------------

print("\nScaling features...")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# -------------------------------
# 4. Train K-Means Model
# -------------------------------

print("\nTraining K-Means model...")

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)


df["Customer_Segment"] = kmeans.fit_predict(X_scaled)


# -------------------------------
# 5. Save Model Artifacts
# -------------------------------

print("\nSaving models...")

joblib.dump(
    kmeans,
    "models/customer_segmentation_model.pkl"
)

joblib.dump(
    scaler,
    "models/customer_scaler.pkl"
)


# -------------------------------
# 6. Save Segmented Customers
# -------------------------------

df.to_csv(
    "data/final/customer_segments.csv",
    index=False
)


# -------------------------------
# 7. Cluster Profiling
# -------------------------------

print("\nGenerating cluster profile...")

profile = df.groupby("Customer_Segment").agg({

    "Customer ID": "count",
    "Recency": "mean",
    "Frequency": "mean",
    "Monetary_Total": "mean",
    "Predicted_CLV": "mean",
    "Engagement_Score": "mean",
    "Churn": "mean"

})


profile.columns = [

    "Customer Count",
    "Avg Recency",
    "Avg Frequency",
    "Avg Spending",
    "Avg CLV",
    "Avg Engagement",
    "Churn Rate"

]


profile = profile.round(2)


print("\nCustomer Segment Profile:")
print(profile)


print("\nDay 6 Customer Segmentation Completed Successfully!")
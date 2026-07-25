import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import os


# Paths
DATA_PATH = "data/processed/customer_features.csv"
MODEL_PATH = "models/customer_segmentation_model.pkl"
SCALER_PATH = "models/customer_scaler.pkl"
OUTPUT_PATH = "data/final/customer_segments.csv"
# Load customer data
df = pd.read_csv(DATA_PATH)

print("Dataset loaded:", df.shape)


# Features for segmentation
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


# Scaling
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# K-Means model
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)


# Train model
df["Customer_Segment"] = kmeans.fit_predict(X_scaled)


# Save model files
joblib.dump(
    kmeans,
    MODEL_PATH
)

joblib.dump(
    scaler,
    SCALER_PATH
)


# Save segmented data
df.to_csv(
    OUTPUT_PATH,
    index=False
)


print("\nCustomer segmentation completed!")

print("\nCluster distribution:")
print(df["Customer_Segment"].value_counts())
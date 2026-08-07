import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)
import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Paths
DATA_PATH = "data/processed/customer_features.csv"
MODEL_PATH = "models/customer_segmentation_model.pkl"
SCALER_PATH = "models/customer_scaler.pkl"
OUTPUT_PATH = "data/final/customer_segments.csv"
# Load customer data
df = pd.read_csv(DATA_PATH)

logger.info("Dataset loaded: %s", df.shape)


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


logger.info("\nCustomer segmentation completed!")

logger.info("Cluster distribution:")
logger.info(df["Customer_Segment"].value_counts())
# ==========================================================
# VANTARA CUSTOMER INTELLIGENCE PLATFORM
# DAY 9 - CUSTOMER SEGMENTATION ANALYSIS
# K-Means + Gaussian Mixture Model
# ==========================================================

import os
import joblib
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score
)


# ==========================================================
# Configuration
# ==========================================================

RANDOM_SEED = 42

INPUT_PATH = (
    "data/processed/customer_features.csv"
)

K_ANALYSIS_PATH = (
    "outputs/reports/"
    "day9_kmeans_k_analysis.csv"
)

COMPARISON_PATH = (
    "outputs/reports/"
    "day9_clustering_comparison.csv"
)

PROFILE_PATH = (
    "outputs/reports/"
    "day9_segment_profiles.csv"
)

SEGMENT_OUTPUT_PATH = (
    "data/final/"
    "customer_segments_final.csv"
)

MODEL_PATH = (
    "models/"
    "final_segmentation_model.pkl"
)

SCALER_PATH = (
    "models/"
    "final_segmentation_scaler.pkl"
)


os.makedirs(
    "outputs/reports",
    exist_ok=True
)

os.makedirs(
    "data/final",
    exist_ok=True
)

os.makedirs(
    "models",
    exist_ok=True
)


# ==========================================================
# 1. Load Customer Features
# ==========================================================

print("\nLoading customer features...")

df = pd.read_csv(
    INPUT_PATH
)

print(
    "Dataset Shape:",
    df.shape
)


# ==========================================================
# 2. Select Clustering Features
# ==========================================================

print(
    "\nPreparing clustering features..."
)

clustering_features = [

    "Recency",

    "Frequency",

    "Monetary_Total",

    "Average_Order_Value",

    "Unique_Products",

    "Average_Purchase_Gap",

    "Customer_Lifespan_Days",

    "Return_Rate",

    "Engagement_Score"

]


missing_features = [

    feature

    for feature in clustering_features

    if feature not in df.columns

]


if missing_features:

    raise ValueError(

        "Missing clustering features: "

        f"{missing_features}"

    )


X = df[
    clustering_features
].copy()


# ==========================================================
# 3. Check Missing Values
# ==========================================================

print(
    "\nChecking missing values..."
)

missing_values = (
    X.isnull().sum()
)

if missing_values.sum() > 0:

    print(
        "Missing values found."
    )

    X = X.fillna(
        X.median(
            numeric_only=True
        )
    )

else:

    print(
        "No missing values found."
    )


# ==========================================================
# 4. Cap Extreme Values
# ==========================================================

print(
    "\nCapping extreme values "
    "using 1st and 99th percentiles..."
)


for column in clustering_features:

    lower_limit = (
        X[column]
        .quantile(0.01)
    )

    upper_limit = (
        X[column]
        .quantile(0.99)
    )

    X[column] = (
        X[column]
        .clip(
            lower=lower_limit,
            upper=upper_limit
        )
    )


print(
    "Extreme value treatment completed."
)


# ==========================================================
# 5. Scale Features
# ==========================================================

print(
    "\nScaling clustering features..."
)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X
)


# ==========================================================
# 6. Evaluate Different K Values
# ==========================================================

print(
    "\nEvaluating K-Means "
    "cluster values..."
)


k_results = []


for k in range(2, 9):

    model = KMeans(

        n_clusters=k,

        random_state=RANDOM_SEED,

        n_init=20

    )


    labels = model.fit_predict(
        X_scaled
    )


    silhouette = (
        silhouette_score(
            X_scaled,
            labels
        )
    )


    davies_bouldin = (
        davies_bouldin_score(
            X_scaled,
            labels
        )
    )


    cluster_sizes = (
        pd.Series(labels)
        .value_counts()
    )


    smallest_cluster = int(
        cluster_sizes.min()
    )


    k_results.append(

        {

            "K":
                k,

            "Inertia":
                model.inertia_,

            "Silhouette_Score":
                silhouette,

            "Davies_Bouldin_Score":
                davies_bouldin,

            "Smallest_Cluster_Size":
                smallest_cluster

        }

    )


    print(

        f"K = {k} | "

        f"Silhouette = "
        f"{silhouette:.4f} | "

        f"Davies-Bouldin = "
        f"{davies_bouldin:.4f} | "

        f"Smallest Cluster = "
        f"{smallest_cluster}"

    )


k_results_df = pd.DataFrame(
    k_results
)


k_results_df.to_csv(

    K_ANALYSIS_PATH,

    index=False

)


# ==========================================================
# 7. Select Best Practical K
# ==========================================================

print(
    "\nSelecting best practical K..."
)


minimum_cluster_size = max(
    25,
    int(
        len(df) * 0.01
    )
)


valid_k_results = (

    k_results_df[

        k_results_df[
            "Smallest_Cluster_Size"
        ]
        >= minimum_cluster_size

    ]

)


if valid_k_results.empty:

    print(

        "Warning: No K value met the "
        "minimum cluster-size rule."

    )

    valid_k_results = (
        k_results_df.copy()
    )


best_k_row = (

    valid_k_results

    .sort_values(

        by=[

            "Silhouette_Score",

            "Davies_Bouldin_Score"

        ],

        ascending=[
            False,
            True
        ]

    )

    .iloc[0]

)


best_k = int(
    best_k_row["K"]
)


print(
    "Minimum acceptable "
    "cluster size:",
    minimum_cluster_size
)

print(
    "Selected K:",
    best_k
)


# ==========================================================
# 8. Train Final K-Means Model
# ==========================================================

print(
    "\nTraining final K-Means model..."
)


kmeans_model = KMeans(

    n_clusters=best_k,

    random_state=RANDOM_SEED,

    n_init=20

)


kmeans_labels = (
    kmeans_model
    .fit_predict(
        X_scaled
    )
)


kmeans_silhouette = (
    silhouette_score(
        X_scaled,
        kmeans_labels
    )
)


kmeans_db = (
    davies_bouldin_score(
        X_scaled,
        kmeans_labels
    )
)


# ==========================================================
# 9. Train Gaussian Mixture Model
# ==========================================================

print(
    "\nTraining Gaussian Mixture Model..."
)


gmm_model = GaussianMixture(

    n_components=best_k,

    covariance_type="full",

    random_state=RANDOM_SEED,

    n_init=5

)


gmm_labels = (
    gmm_model
    .fit_predict(
        X_scaled
    )
)


gmm_silhouette = (
    silhouette_score(
        X_scaled,
        gmm_labels
    )
)


gmm_db = (
    davies_bouldin_score(
        X_scaled,
        gmm_labels
    )
)


# ==========================================================
# 10. Compare Clustering Models
# ==========================================================

comparison = pd.DataFrame(

    [

        {

            "Model":
                "K-Means",

            "Clusters":
                best_k,

            "Silhouette_Score":
                kmeans_silhouette,

            "Davies_Bouldin_Score":
                kmeans_db

        },

        {

            "Model":
                "Gaussian Mixture Model",

            "Clusters":
                best_k,

            "Silhouette_Score":
                gmm_silhouette,

            "Davies_Bouldin_Score":
                gmm_db

        }

    ]

)


print(
    "\nClustering Model Comparison"
)

print("-" * 75)


print(

    comparison

    .round(4)

    .to_string(
        index=False
    )

)


comparison.to_csv(

    COMPARISON_PATH,

    index=False

)


# ==========================================================
# 11. Select Final Clustering Model
# ==========================================================

best_model_row = (

    comparison

    .sort_values(

        by=[

            "Silhouette_Score",

            "Davies_Bouldin_Score"

        ],

        ascending=[
            False,
            True
        ]

    )

    .iloc[0]

)


best_model_name = (
    best_model_row[
        "Model"
    ]
)


if best_model_name == "K-Means":

    final_labels = (
        kmeans_labels
    )

    final_model = (
        kmeans_model
    )

else:

    final_labels = (
        gmm_labels
    )

    final_model = (
        gmm_model
    )


print(

    "\nSelected Final "
    "Clustering Model:",

    best_model_name

)


# ==========================================================
# 12. Add Final Segment Labels
# ==========================================================

df[
    "Customer_Segment"
] = final_labels


# ==========================================================
# 13. Generate Detailed Segment Profiles
# ==========================================================

segment_profile = (

    df

    .groupby(
        "Customer_Segment"
    )

    .agg(

        Customer_Count=(
            "Customer_Segment",
            "size"
        ),

        Avg_Recency=(
            "Recency",
            "mean"
        ),

        Avg_Frequency=(
            "Frequency",
            "mean"
        ),

        Avg_Monetary=(
            "Monetary_Total",
            "mean"
        ),

        Avg_Order_Value=(
            "Average_Order_Value",
            "mean"
        ),

        Avg_Unique_Products=(
            "Unique_Products",
            "mean"
        ),

        Avg_Purchase_Gap=(
            "Average_Purchase_Gap",
            "mean"
        ),

        Avg_Return_Rate=(
            "Return_Rate",
            "mean"
        ),

        Avg_Engagement=(
            "Engagement_Score",
            "mean"
        ),

        Churn_Rate=(
            "Churn",
            "mean"
        )

    )

    .round(2)

)


print(
    "\nCustomer Segment Profile"
)

print("-" * 100)


print(
    segment_profile
    .to_string()
)


# ==========================================================
# 14. Create Business-Friendly Segment Names
# ==========================================================

print(
    "\nCreating business-friendly "
    "segment names..."
)


segment_names = {
    0: "High-Spend Occasional Customers",
    1: "Developing Customers",
    2: "Loyal Regular Customers",
    3: "Irregular At-Risk Customers",
    4: "High-Return Lost Customers",
    5: "Champions",
    6: "Dormant Customers"
}


df[
    "Segment_Name"
] = (

    df[
        "Customer_Segment"
    ]

    .map(
        segment_names
    )

)


print(
    "\nSegment Names:"
)


for segment_id, name in (
    segment_names.items()
):

    print(

        f"Segment {segment_id}: "
        f"{name}"

    )
# ==========================================================
# 15. Add Names to Segment Profile
# ==========================================================

segment_profile[
    "Segment_Name"
] = (

    segment_profile.index

    .map(
        segment_names
    )

)


segment_profile.to_csv(

    PROFILE_PATH

)


# ==========================================================
# 16. Save Final Dataset
# ==========================================================

df.to_csv(

    SEGMENT_OUTPUT_PATH,

    index=False

)


# ==========================================================
# 17. Save Final Model and Scaler
# ==========================================================

joblib.dump(

    final_model,

    MODEL_PATH

)


joblib.dump(

    scaler,

    SCALER_PATH

)


# ==========================================================
# 18. Final Summary
# ==========================================================

print(
    "\nSaved clustering outputs:"
)

print(
    K_ANALYSIS_PATH
)

print(
    COMPARISON_PATH
)

print(
    PROFILE_PATH
)

print(
    SEGMENT_OUTPUT_PATH
)

print(
    MODEL_PATH
)

print(
    SCALER_PATH
)


print(

    "\nDay 9 Customer Segmentation "
    "Analysis completed successfully."

)
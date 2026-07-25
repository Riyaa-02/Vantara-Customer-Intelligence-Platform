import os
import joblib
import pandas as pd


# --------------------------------------------------
# File Paths
# --------------------------------------------------

ML_DATA_PATH = "data/final/customer_ml_dataset.csv"
SEGMENTS_PATH = "data/final/customer_segments.csv"
CHURN_MODEL_PATH = "models/customer_churn_model.pkl"

OUTPUT_PATH = "data/final/customer_intelligence.csv"


# --------------------------------------------------
# 1. Load Data
# --------------------------------------------------

print("Loading customer datasets...")

ml_data = pd.read_csv(ML_DATA_PATH)
segments = pd.read_csv(SEGMENTS_PATH)

print(f"ML Dataset Shape      : {ml_data.shape}")
print(f"Segments Dataset Shape: {segments.shape}")


# --------------------------------------------------
# 2. Load Churn Model
# --------------------------------------------------

print("\nLoading churn prediction model...")

churn_model = joblib.load(CHURN_MODEL_PATH)

print("Model loaded successfully.")


# --------------------------------------------------
# 3. Prepare Features for Churn Prediction
# --------------------------------------------------

print("\nPreparing customer features...")

# Use the exact features that were used during model training
if hasattr(churn_model, "feature_names_in_"):

    model_features = list(churn_model.feature_names_in_)

    missing_features = [
        feature
        for feature in model_features
        if feature not in ml_data.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing features required by the churn model: {missing_features}"
        )

    X = ml_data[model_features]

else:

    # Fallback if the saved model does not contain feature names
    X = ml_data.drop(columns=["Churn"], errors="ignore")


# --------------------------------------------------
# 4. Generate Churn Predictions
# --------------------------------------------------

print("\nGenerating churn predictions...")

ml_data["Predicted_Churn"] = churn_model.predict(X)

if hasattr(churn_model, "predict_proba"):
    ml_data["Churn_Probability"] = (
        churn_model.predict_proba(X)[:, 1]
    )

print("Churn predictions generated successfully.")


# --------------------------------------------------
# 5. Add Customer Segments
# --------------------------------------------------

print("\nCombining customer segmentation results...")

# Find common customer identifier if available
possible_ids = [
    "CustomerID",
    "Customer_ID",
    "customer_id"
]

customer_id = None

for column in possible_ids:
    if column in ml_data.columns and column in segments.columns:
        customer_id = column
        break


if customer_id:

    segment_columns = [
        customer_id,
        "Customer_Segment"
    ]

    final_data = ml_data.merge(
        segments[segment_columns],
        on=customer_id,
        how="left"
    )

else:

    # If both datasets contain the same customers in the same order
    if len(ml_data) != len(segments):
        raise ValueError(
            "Datasets have different numbers of rows and "
            "no common customer ID was found."
        )

    final_data = ml_data.copy()

    if "Customer_Segment" in segments.columns:
        final_data["Customer_Segment"] = (
            segments["Customer_Segment"].values
        )
    else:
        raise ValueError(
            "Customer_Segment column not found "
            "in customer_segments.csv."
        )


# --------------------------------------------------
# 6. Create Churn Risk Levels
# --------------------------------------------------

print("\nCreating customer risk categories...")

if "Churn_Probability" in final_data.columns:

    final_data["Churn_Risk"] = pd.cut(
        final_data["Churn_Probability"],
        bins=[0, 0.40, 0.60, 1.0],
        labels=[
            "Low Risk",
            "Medium Risk",
            "High Risk"
        ],
        include_lowest=True
    )


# --------------------------------------------------
# 7. Create Customer Value Categories
# --------------------------------------------------

if "Predicted_CLV" in final_data.columns:

    final_data["Customer_Value"] = pd.qcut(
        final_data["Predicted_CLV"],
        q=3,
        labels=[
            "Low Value",
            "Medium Value",
            "High Value"
        ],
        duplicates="drop"
    )


# --------------------------------------------------
# 8. Save Final Customer Intelligence Dataset
# --------------------------------------------------

print("\nSaving final customer intelligence dataset...")

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)

final_data.to_csv(
    OUTPUT_PATH,
    index=False
)

print(f"Final Dataset Shape: {final_data.shape}")
print(f"Saved to: {OUTPUT_PATH}")


# --------------------------------------------------
# 9. Display Summary
# --------------------------------------------------

print("\nCustomer Intelligence Summary")
print("-" * 45)

if "Customer_Segment" in final_data.columns:
    print("\nCustomer Segments:")
    print(final_data["Customer_Segment"].value_counts().sort_index())

if "Churn_Risk" in final_data.columns:
    print("\nChurn Risk Distribution:")
    print(final_data["Churn_Risk"].value_counts())

if "Customer_Value" in final_data.columns:
    print("\nCustomer Value Distribution:")
    print(final_data["Customer_Value"].value_counts())


print("\nDay 8 completed successfully.")
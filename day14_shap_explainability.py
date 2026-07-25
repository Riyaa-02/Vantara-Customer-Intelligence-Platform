# ==========================================================
# VANTARA CUSTOMER INTELLIGENCE PLATFORM
# DAY 14 - SHAP EXPLAINABILITY FOR ANN
# ==========================================================

import os
import joblib
import numpy as np
import pandas as pd
import torch
import shap
import matplotlib.pyplot as plt

from torch import nn


# ==========================================================
# Configuration
# ==========================================================

SEED = 42

os.makedirs(
    "outputs/explainability",
    exist_ok=True
)


# ==========================================================
# 1. Load Data
# ==========================================================

print("\nLoading datasets...")

X_train = pd.read_csv(
    "data/final/X_train.csv"
).astype("float32")

X_test = pd.read_csv(
    "data/final/X_test.csv"
).astype("float32")


feature_names = joblib.load(
    "models/ann_feature_names.pkl"
)


print(
    "Training Set:",
    X_train.shape
)

print(
    "Testing Set:",
    X_test.shape
)


# ==========================================================
# 2. Rebuild ANN Architecture
# ==========================================================

class ANNModel(nn.Module):

    def __init__(
        self,
        input_size
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_size,
                128
            ),

            nn.BatchNorm1d(
                128
            ),

            nn.ReLU(),

            nn.Dropout(
                0.30
            ),

            nn.Linear(
                128,
                64
            ),

            nn.BatchNorm1d(
                64
            ),

            nn.ReLU(),

            nn.Dropout(
                0.30
            ),

            nn.Linear(
                64,
                32
            ),

            nn.ReLU(),

            nn.Dropout(
                0.20
            ),

            nn.Linear(
                32,
                1
            )

        )


    def forward(
        self,
        x
    ):

        return self.network(
            x
        )


model = ANNModel(
    len(feature_names)
)


model.load_state_dict(

    torch.load(

        "models/"
        "ann_churn_model.pth",

        map_location="cpu"

    )

)


model.eval()


print(
    "ANN model loaded successfully."
)


# ==========================================================
# 3. Prepare SHAP Data
# ==========================================================

print(
    "\nPreparing SHAP background data..."
)


background_sample = (

    X_train.sample(

        n=min(
            200,
            len(X_train)
        ),

        random_state=SEED

    )

)


explanation_sample = (

    X_test.sample(

        n=min(
            300,
            len(X_test)
        ),

        random_state=SEED

    )

)


background_tensor = torch.tensor(

    background_sample.values,

    dtype=torch.float32

)


explanation_tensor = torch.tensor(

    explanation_sample.values,

    dtype=torch.float32

)


# ==========================================================
# 4. Generate SHAP Values
# ==========================================================

print(
    "\nGenerating SHAP explanations..."
)


explainer = shap.GradientExplainer(

    model,

    background_tensor

)


shap_values = explainer.shap_values(

    explanation_tensor

)


# Handle SHAP output format
if isinstance(
    shap_values,
    list
):

    shap_values = shap_values[0]


shap_values = np.array(
    shap_values
)


if shap_values.ndim == 3:

    shap_values = (
        shap_values[:, :, 0]
    )


print(
    "SHAP values generated successfully."
)


# ==========================================================
# 5. Global Feature Importance
# ==========================================================

print(
    "\nGenerating global feature importance..."
)


mean_absolute_shap = (

    np.abs(
        shap_values
    )

    .mean(
        axis=0
    )

)


importance_df = pd.DataFrame({

    "Feature":
        feature_names,

    "Mean_Absolute_SHAP":
        mean_absolute_shap

})


importance_df = (

    importance_df

    .sort_values(

        "Mean_Absolute_SHAP",

        ascending=False

    )

)


importance_df.to_csv(

    "outputs/explainability/"
    "day14_shap_global_importance.csv",

    index=False

)


print(
    "\nTop 10 SHAP Features"
)

print("-" * 55)


print(

    importance_df

    .head(10)

    .round(6)

    .to_string(
        index=False
    )

)


# ==========================================================
# 6. Save SHAP Summary Plot
# ==========================================================

print(
    "\nSaving SHAP summary plot..."
)


shap.summary_plot(

    shap_values,

    explanation_sample.values,

    feature_names=feature_names,

    show=False

)


plt.tight_layout()


plt.savefig(

    "outputs/explainability/"
    "day14_shap_summary.png",

    dpi=300,

    bbox_inches="tight"

)


plt.close()


# ==========================================================
# 7. Find Representative Customers
# ==========================================================

print(
    "\nSelecting representative customers..."
)


X_test_tensor = torch.tensor(

    X_test.values,

    dtype=torch.float32

)


with torch.no_grad():

    probabilities = (

        torch.sigmoid(

            model(
                X_test_tensor
            )

        )

        .numpy()

        .ravel()

    )


low_index = int(
    np.argmin(
        probabilities
    )
)


high_index = int(
    np.argmax(
        probabilities
    )
)


borderline_index = int(

    np.argmin(

        np.abs(

            probabilities
            - 0.50

        )

    )

)


representative_customers = {

    "Low_Risk":
        low_index,

    "High_Risk":
        high_index,

    "Borderline":
        borderline_index

}


# ==========================================================
# 8. Generate Individual SHAP Explanations
# ==========================================================

print(
    "\nGenerating individual explanations..."
)


for customer_type, index in (

    representative_customers.items()

):


    customer_data = (

        X_test

        .iloc[
            [
                index
            ]
        ]

    )


    customer_tensor = torch.tensor(

        customer_data.values,

        dtype=torch.float32

    )


    customer_shap = (

        explainer.shap_values(

            customer_tensor

        )

    )


    if isinstance(
        customer_shap,
        list
    ):

        customer_shap = (
            customer_shap[0]
        )


    customer_shap = np.array(
        customer_shap
    )


    if customer_shap.ndim == 3:

        customer_shap = (

            customer_shap[
                :,
                :,
                0
            ]

        )


    customer_shap = (

        customer_shap[0]

    )


    explanation_df = pd.DataFrame({

        "Feature":
            feature_names,

        "Feature_Value":
            customer_data
            .iloc[0]
            .values,

        "SHAP_Value":
            customer_shap

    })


    explanation_df[
        "Absolute_SHAP"
    ] = (

        explanation_df[
            "SHAP_Value"
        ]

        .abs()

    )


    explanation_df = (

        explanation_df

        .sort_values(

            "Absolute_SHAP",

            ascending=False

        )

    )


    explanation_df.to_csv(

        "outputs/explainability/"

        f"day14_"

        f"{customer_type.lower()}"

        "_explanation.csv",

        index=False

    )


    print(

        f"{customer_type} customer | "

        f"Churn Probability: "

        f"{probabilities[index]:.4f}"

    )


# ==========================================================
# 9. Save Representative Customer Summary
# ==========================================================

representative_df = pd.DataFrame([

    {

        "Customer_Type":
            customer_type,

        "Test_Row_Index":
            index,

        "Churn_Probability":
            probabilities[index]

    }

    for customer_type, index

    in representative_customers.items()

])


representative_df.to_csv(

    "outputs/explainability/"
    "day14_representative_customers.csv",

    index=False

)


# ==========================================================
# Completion
# ==========================================================

print(
    "\nSaved SHAP outputs:"
)

print(
    "outputs/explainability/"
    "day14_shap_global_importance.csv"
)

print(
    "outputs/explainability/"
    "day14_shap_summary.png"
)

print(
    "outputs/explainability/"
    "day14_low_risk_explanation.csv"
)

print(
    "outputs/explainability/"
    "day14_high_risk_explanation.csv"
)

print(
    "outputs/explainability/"
    "day14_borderline_explanation.csv"
)


print(
    "\nDay 14 SHAP Explainability "
    "completed successfully."
)
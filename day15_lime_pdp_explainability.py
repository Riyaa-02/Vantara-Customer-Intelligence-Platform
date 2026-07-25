# ==========================================================
# VANTARA CUSTOMER INTELLIGENCE PLATFORM
# DAY 15 - LIME, PDP AND PLAIN-ENGLISH EXPLANATIONS
# ==========================================================

import os
import joblib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from torch import nn
from lime.lime_tabular import LimeTabularExplainer


# ==========================================================
# Configuration
# ==========================================================

SEED = 42

OUTPUT_DIR = "outputs/explainability"

os.makedirs(
    OUTPUT_DIR,
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
# 2. Rebuild ANN Model
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
    len(
        feature_names
    )
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
# 3. Prediction Functions
# ==========================================================

def predict_probability(
    X
):

    X = np.asarray(
        X,
        dtype=np.float32
    )


    tensor = torch.tensor(
        X,
        dtype=torch.float32
    )


    model.eval()


    with torch.no_grad():

        probabilities = (

            torch.sigmoid(

                model(
                    tensor
                )

            )

            .numpy()

            .ravel()

        )


    return probabilities


def predict_proba(
    X
):

    churn_probability = (
        predict_probability(
            X
        )
    )


    return np.column_stack(

        [

            1
            - churn_probability,

            churn_probability

        ]

    )


# ==========================================================
# 4. Select High-Risk Customer
# ==========================================================

print(
    "\nSelecting high-risk customer..."
)


test_probabilities = (

    predict_probability(

        X_test.values

    )

)


high_risk_index = int(

    np.argmax(

        test_probabilities

    )

)


high_risk_customer = (

    X_test

    .iloc[
        high_risk_index
    ]

)


high_risk_probability = float(

    test_probabilities[
        high_risk_index
    ]

)


print(
    "Selected Test Row:",
    high_risk_index
)


print(
    "Churn Probability:",
    round(
        high_risk_probability,
        4
    )
)


# ==========================================================
# 5. Generate LIME Explanation
# ==========================================================

print(
    "\nGenerating LIME explanation..."
)


lime_explainer = LimeTabularExplainer(

    training_data=
        X_train.values,

    feature_names=
        feature_names,

    class_names=[
        "No Churn",
        "Churn"
    ],

    mode=
        "classification",

    random_state=
        SEED

)


lime_explanation = (

    lime_explainer

    .explain_instance(

        high_risk_customer.values,

        predict_proba,

        num_features=10

    )

)


lime_results = pd.DataFrame(

    lime_explanation.as_list(),

    columns=[

        "Feature_Condition",

        "LIME_Weight"

    ]

)


lime_output_path = (

    f"{OUTPUT_DIR}/"
    "day15_lime_high_risk_explanation.csv"

)


lime_results.to_csv(

    lime_output_path,

    index=False

)


print(
    "\nTop LIME Explanation Factors"
)

print("-" * 75)


print(

    lime_results

    .round(6)

    .to_string(
        index=False
    )

)


# ==========================================================
# 6. Generate Partial Dependence Plots Manually
# ==========================================================

print(
    "\nGenerating Partial Dependence Plots..."
)


pdp_features = [

    "Engagement_Score",

    "Recency",

    "Average_Purchase_Gap"

]


for feature in pdp_features:


    if feature not in X_train.columns:

        print(

            f"Skipping missing feature: "
            f"{feature}"

        )

        continue


    feature_index = (

        X_train.columns

        .get_loc(
            feature
        )

    )


    feature_values = np.linspace(

        float(
            X_train[
                feature
            ]
            .quantile(
                0.05
            )
        ),

        float(
            X_train[
                feature
            ]
            .quantile(
                0.95
            )
        ),

        30,

        dtype=np.float32

    )


    average_predictions = []


    for value in feature_values:


        modified_data = (

            X_train

            .copy()

        )


        modified_data.iloc[

            :,

            feature_index

        ] = np.float32(
            value
        )


        probabilities = (

            predict_probability(

                modified_data.values

            )

        )


        average_predictions.append(

            float(

                probabilities.mean()

            )

        )


    plt.figure(

        figsize=(
            8,
            5
        )

    )


    plt.plot(

        feature_values,

        average_predictions

    )


    plt.xlabel(
        feature
    )


    plt.ylabel(

        "Average Predicted "
        "Churn Probability"

    )


    plt.title(

        f"Partial Dependence - "
        f"{feature}"

    )


    plt.tight_layout()


    output_path = (

        f"{OUTPUT_DIR}/"

        f"day15_pdp_"

        f"{feature.lower()}.png"

    )


    plt.savefig(

        output_path,

        dpi=300,

        bbox_inches="tight"

    )


    plt.close()


    print(

        "Saved:",

        output_path

    )


# ==========================================================
# 7. Create Plain-English Explanation
# ==========================================================

print(
    "\nCreating plain-English explanation..."
)


positive_factors = (

    lime_results[

        lime_results[
            "LIME_Weight"
        ]
        > 0

    ]

    .sort_values(

        "LIME_Weight",

        ascending=False

    )

    .head(3)

)


factor_lines = []


for factor in (

    positive_factors[
        "Feature_Condition"
    ]

):

    factor_lines.append(

        f"- {factor}"

    )


factor_text = "\n".join(

    factor_lines

)


plain_explanation = f"""
CUSTOMER CHURN EXPLANATION
==========================

Predicted Churn Risk:
{high_risk_probability * 100:.2f}%

Why this customer is flagged:

The model identified this customer as having a high
probability of churn based on their recent behaviour
and customer profile.

The strongest factors increasing the churn prediction are:

{factor_text}

Recommended Business Action:

Prioritize this customer for a retention campaign.
Possible actions include personalized offers,
targeted communication, loyalty rewards, or
proactive customer support.

Important Note:

This explanation describes factors influencing the
model's prediction. It does not prove that these
factors directly cause customer churn.
"""


print(
    plain_explanation
)


plain_explanation_path = (

    f"{OUTPUT_DIR}/"
    "day15_plain_english_explanation.txt"

)


with open(

    plain_explanation_path,

    "w",

    encoding="utf-8"

) as file:

    file.write(

        plain_explanation

    )


# ==========================================================
# 8. Save Customer Summary
# ==========================================================

customer_summary = pd.DataFrame(

    [

        {

            "Test_Row_Index":
                high_risk_index,

            "Churn_Probability":
                high_risk_probability,

            "Risk_Level":
                "High Risk"

        }

    ]

)


customer_summary.to_csv(

    f"{OUTPUT_DIR}/"
    "day15_explained_customer.csv",

    index=False

)


# ==========================================================
# 9. Completion
# ==========================================================

print(
    "\nSaved Day 15 outputs:"
)


print(
    lime_output_path
)


print(

    f"{OUTPUT_DIR}/"
    "day15_pdp_engagement_score.png"

)


print(

    f"{OUTPUT_DIR}/"
    "day15_pdp_recency.png"

)


print(

    f"{OUTPUT_DIR}/"
    "day15_pdp_average_purchase_gap.png"

)


print(
    plain_explanation_path
)


print(

    "\nDay 15 LIME, PDP and "
    "Plain-English Explainability "
    "completed successfully."

)
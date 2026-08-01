from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn


# ------------------------------------------------------------------
# Project paths
# ------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"


# ------------------------------------------------------------------
# ANN architecture
# ------------------------------------------------------------------

class ANNChurnModel(nn.Module):
    """Artificial neural network used for customer churn prediction."""

    def __init__(self, input_size: int):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the network."""
        return self.network(x)


# ------------------------------------------------------------------
# Load model artifacts
# ------------------------------------------------------------------

feature_names = joblib.load(
    MODELS_DIR / "ann_feature_names.pkl"
)

numeric_features = joblib.load(
    MODELS_DIR / "ann_numeric_features.pkl"
)

country_values = joblib.load(
    MODELS_DIR / "ann_country_values.pkl"
)

scaler = joblib.load(
    MODELS_DIR / "ann_input_scaler.pkl"
)


model = ANNChurnModel(
    input_size=len(feature_names)
)

model.load_state_dict(
    torch.load(
        MODELS_DIR / "ann_churn_model.pth",
        map_location=torch.device("cpu"),
    )
)

model.eval()


# ------------------------------------------------------------------
# Customer preprocessing
# ------------------------------------------------------------------

def preprocess_customer(customer_data: dict) -> np.ndarray:
    """
    Convert one raw customer record into the exact input format
    required by the trained ANN model.
    """

    customer_values = customer_data.copy()

    if "Country" not in customer_values:
        raise ValueError("Country is required.")

    country = customer_values.pop("Country")

    if country not in country_values:
        raise ValueError(
            f"Unsupported country: {country}"
        )

    missing_features = [
        feature
        for feature in numeric_features
        if feature not in customer_values
    ]

    if missing_features:
        raise ValueError(
            "Missing required numeric features: "
            + ", ".join(missing_features)
        )

    numeric_dataframe = pd.DataFrame(
        [
            {
                feature: customer_values[feature]
                for feature in numeric_features
            }
        ],
        columns=numeric_features,
    )

    if numeric_dataframe.isna().any().any():
        raise ValueError(
            "Numeric features cannot contain missing values."
        )

    try:
        numeric_dataframe = numeric_dataframe.astype(
            np.float32
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "All numeric features must contain valid numbers."
        ) from error

    scaled_numeric_values = scaler.transform(
        numeric_dataframe
    )

    final_input = np.zeros(
        (1, len(feature_names)),
        dtype=np.float32,
    )

    for numeric_index, feature in enumerate(
        numeric_features
    ):
        feature_position = feature_names.index(feature)

        final_input[0, feature_position] = (
            scaled_numeric_values[0, numeric_index]
        )

    country_feature = f"Country_{country}"

    if country_feature in feature_names:
        country_position = feature_names.index(
            country_feature
        )

        final_input[0, country_position] = 1.0

    return final_input


# ------------------------------------------------------------------
# ANN prediction
# ------------------------------------------------------------------

def score_customer(customer_data: dict) -> dict:
    """
    Generate a churn prediction for one customer.
    """

    processed_input = preprocess_customer(
        customer_data
    )

    input_tensor = torch.tensor(
        processed_input,
        dtype=torch.float32,
    )

    with torch.no_grad():
        logit = model(input_tensor)
        probability = torch.sigmoid(logit).item()

    prediction = int(probability >= 0.5)

    if probability >= 0.7:
        risk_level = "High"
    elif probability >= 0.4:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return {
        "churn_prediction": prediction,
        "churn_probability": round(
            probability,
            4,
        ),
        "risk_level": risk_level,
    }
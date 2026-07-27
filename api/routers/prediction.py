import joblib
import numpy as np
import torch
from fastapi import APIRouter, HTTPException
from torch import nn

from api.schemas.prediction_schema import CustomerInput
from api.schemas.response_schema import PredictionResponse

router = APIRouter(tags=["Prediction"])


class ANNChurnModel(nn.Module):
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

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)


feature_names = joblib.load("models/ann_feature_names.pkl")
numeric_features = joblib.load("models/ann_numeric_features.pkl")
country_values = joblib.load("models/ann_country_values.pkl")
scaler = joblib.load("models/ann_input_scaler.pkl")

model = ANNChurnModel(input_size=len(feature_names))
model.load_state_dict(
    torch.load(
        "models/ann_churn_model.pth",
        map_location=torch.device("cpu")
    )
)
model.eval()


@router.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerInput):
    try:
        customer_data = customer.model_dump()

        country = customer_data.pop("Country")

        if country not in country_values:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported country: {country}"
            )

        numeric_values = np.array(
            [[customer_data[feature] for feature in numeric_features]],
            dtype=np.float32
        )

        scaled_numeric_values = scaler.transform(numeric_values)

        final_input = np.zeros((1, len(feature_names)), dtype=np.float32)

        for index, feature in enumerate(numeric_features):
            feature_position = feature_names.index(feature)
            final_input[0, feature_position] = scaled_numeric_values[0, index]

        country_feature = f"Country_{country}"

        if country_feature in feature_names:
            country_position = feature_names.index(country_feature)
            final_input[0, country_position] = 1.0

        input_tensor = torch.tensor(final_input, dtype=torch.float32)

        with torch.no_grad():
            logit = model(input_tensor)
            probability = torch.sigmoid(logit).item()

        prediction = int(probability >= 0.5)

        return PredictionResponse(
            churn_prediction=prediction,
            churn_probability=round(probability, 4),
            risk_level=(
                "High"
                if probability >= 0.7
                else "Medium"
                if probability >= 0.4
                else "Low"
            )
        )

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
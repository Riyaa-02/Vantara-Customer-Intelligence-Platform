from datetime import datetime

from pydantic import BaseModel, ConfigDict


class PredictionResponse(BaseModel):
    """Response returned after generating a churn prediction."""

    churn_prediction: int
    churn_probability: float
    risk_level: str


class PredictionHistoryResponse(BaseModel):
    """One prediction record retrieved from PostgreSQL."""

    id: int
    customer_id: str | None
    churn_prediction: int
    churn_probability: float
    risk_level: str
    prediction_type: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
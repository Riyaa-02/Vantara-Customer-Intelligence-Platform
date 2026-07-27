from pydantic import BaseModel


class PredictionResponse(BaseModel):
    churn_prediction: int
    churn_probability: float
    risk_level: str
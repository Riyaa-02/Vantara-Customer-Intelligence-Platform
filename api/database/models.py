from sqlalchemy import Column, Integer, Float, String, DateTime
from datetime import datetime

from .database import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)

    customer_id = Column(String(100), nullable=True)

    churn_prediction = Column(Integer, nullable=False)
    churn_probability = Column(Float, nullable=False)
    risk_level = Column(String(50), nullable=False)

    prediction_type = Column(String(20), default="single")

    created_at = Column(DateTime, default=datetime.utcnow)
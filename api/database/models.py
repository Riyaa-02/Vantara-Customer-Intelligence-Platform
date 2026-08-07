from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
)
from sqlalchemy.orm import relationship

from .database import Base


class Segment(Base):
    """Store business-readable customer segment information."""

    __tablename__ = "segments"

    id = Column(Integer, primary_key=True, index=True)
    segment_number = Column(Integer, unique=True, nullable=False)
    segment_name = Column(String(100), nullable=False)
    description = Column(String(255), nullable=True)

    customers = relationship(
        "Customer",
        back_populates="segment",
    )


class Customer(Base):
    """Store customer-level features and segment assignments."""

    __tablename__ = "customers"

    customer_id = Column(
        String(100),
        primary_key=True,
        index=True,
    )

    country = Column(String(100), nullable=True)
    recency = Column(Float, nullable=True)
    frequency = Column(Float, nullable=True)
    monetary_total = Column(Float, nullable=True)
    predicted_clv = Column(Float, nullable=True)
    engagement_score = Column(Float, nullable=True)
    churn = Column(Integer, nullable=True)

    segment_id = Column(
        Integer,
        ForeignKey("segments.id"),
        nullable=True,
    )

    segment = relationship(
        "Segment",
        back_populates="customers",
    )

    transactions = relationship(
        "Transaction",
        back_populates="customer",
    )

    predictions = relationship(
        "Prediction",
        back_populates="customer",
    )


class Transaction(Base):
    """Store transaction-level retail records."""

    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)

    invoice = Column(String(100), nullable=False)
    stock_code = Column(String(100), nullable=True)
    description = Column(String(255), nullable=True)
    quantity = Column(Float, nullable=True)
    invoice_date = Column(DateTime, nullable=True)
    price = Column(Float, nullable=True)
    country = Column(String(100), nullable=True)

    customer_id = Column(
        String(100),
        ForeignKey("customers.customer_id"),
        nullable=True,
        index=True,
    )

    customer = relationship(
        "Customer",
        back_populates="transactions",
    )


class Prediction(Base):
    """Store single and batch churn prediction results."""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)

    customer_id = Column(
        String(100),
        ForeignKey("customers.customer_id"),
        nullable=True,
        index=True,
    )

    churn_prediction = Column(Integer, nullable=False)
    churn_probability = Column(Float, nullable=False)
    risk_level = Column(String(50), nullable=False)

    prediction_type = Column(
        String(20),
        default="single",
        nullable=False,
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    customer = relationship(
        "Customer",
        back_populates="predictions",
    )
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from api.database.database import get_db
from api.database.models import Customer, Prediction

router = APIRouter(
    prefix="/dashboard",
    tags=["Dashboard"],
)

BASE_DIR = Path(__file__).resolve().parents[2]

CUSTOMER_DATASET_PATH = (
    BASE_DIR
    / "data"
    / "final"
    / "customer_segments.csv"
)

TRANSACTION_DATASET_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "online_retail_ii_combined.csv"
)


@lru_cache(maxsize=1)
def load_dashboard_files() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and prepare customer and transaction dashboard datasets."""

    if not CUSTOMER_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Customer dataset not found: {CUSTOMER_DATASET_PATH}"
        )

    if not TRANSACTION_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Transaction dataset not found: {TRANSACTION_DATASET_PATH}"
        )

    customers = pd.read_csv(CUSTOMER_DATASET_PATH)
    transactions = pd.read_csv(TRANSACTION_DATASET_PATH)

    required_customer_columns = {
        "Customer ID",
        "Churn",
        "Customer_Segment",
        "Predicted_CLV",
        "Monetary_Total",
    }

    missing_customer_columns = (
        required_customer_columns.difference(customers.columns)
    )

    if missing_customer_columns:
        raise ValueError(
            "Customer dataset is missing columns: "
            + ", ".join(sorted(missing_customer_columns))
        )

    required_transaction_columns = {
        "Invoice",
        "InvoiceDate",
        "Quantity",
        "Price",
    }

    missing_transaction_columns = (
        required_transaction_columns.difference(transactions.columns)
    )

    if missing_transaction_columns:
        raise ValueError(
            "Transaction dataset is missing columns: "
            + ", ".join(sorted(missing_transaction_columns))
        )

    for column in [
        "Churn",
        "Customer_Segment",
        "Predicted_CLV",
        "Monetary_Total",
    ]:
        customers[column] = pd.to_numeric(
            customers[column],
            errors="coerce",
        )

    transactions["InvoiceDate"] = pd.to_datetime(
        transactions["InvoiceDate"],
        errors="coerce",
    )

    transactions["Quantity"] = pd.to_numeric(
        transactions["Quantity"],
        errors="coerce",
    )

    transactions["Price"] = pd.to_numeric(
        transactions["Price"],
        errors="coerce",
    )

    transactions = transactions.dropna(
        subset=[
            "InvoiceDate",
            "Quantity",
            "Price",
        ]
    )

    # Revenue charts exclude returns, cancellations and invalid prices.
    transactions = transactions[
        (transactions["Quantity"] > 0)
        & (transactions["Price"] > 0)
    ].copy()

    transactions["Revenue"] = (
        transactions["Quantity"]
        * transactions["Price"]
    )

    transactions["Month"] = (
        transactions["InvoiceDate"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    return customers, transactions


@router.get("/")
def get_dashboard_summary(
    db: Session = Depends(get_db),
) -> dict:
    """Return real dashboard KPIs, revenue trend and segment data."""

    try:
        customers, transactions = load_dashboard_files()
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Dashboard data loading failed: {error}",
        ) from error

    total_customers = int(
        customers["Customer ID"].nunique()
    )

    high_risk_customers = int(
        (customers["Churn"] == 1).sum()
    )

    total_segments = int(
        customers["Customer_Segment"].nunique()
    )

    average_clv = float(
        customers["Predicted_CLV"].mean()
    )

    total_revenue = float(
        transactions["Revenue"].sum()
    )

    prediction_count = int(
        db.query(func.count(Prediction.id)).scalar() or 0
    )

    segment_distribution = (
        customers["Customer_Segment"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    segment_distribution = {
        str(int(segment)): int(count)
        for segment, count in segment_distribution.items()
    }

    monthly_revenue_df = (
        transactions
        .groupby("Month", as_index=False)
        .agg(
            Revenue=("Revenue", "sum"),
            Orders=("Invoice", "nunique"),
        )
        .sort_values("Month")
    )

    monthly_revenue = [
        {
            "month": row.Month.strftime("%b %Y"),
            "revenue": round(float(row.Revenue), 2),
            "orders": int(row.Orders),
        }
        for row in monthly_revenue_df.itertuples(index=False)
    ]

    return {
        "total_customers": total_customers,
        "high_risk_customers": high_risk_customers,
        "segments": total_segments,
        "average_clv": round(average_clv, 2),
        "total_revenue": round(total_revenue, 2),
        "prediction_count": prediction_count,
        "segment_distribution": segment_distribution,
        "monthly_revenue": monthly_revenue,
    }
@router.get("/customers/{customer_id}")
def get_customer_by_id(
    customer_id: str,
    db: Session = Depends(get_db),
) -> dict:
    """Return stored customer information for dashboard lookup."""

    customer = (
        db.query(Customer)
        .filter(Customer.customer_id == customer_id)
        .first()
    )

    if customer is None:
        raise HTTPException(
            status_code=404,
            detail=f"Customer {customer_id} was not found.",
        )

    return {
        "customer_id": customer.customer_id,
        "country": customer.country,
        "recency": customer.recency,
        "frequency": customer.frequency,
        "monetary_total": customer.monetary_total,
        "predicted_clv": customer.predicted_clv,
        "engagement_score": customer.engagement_score,
        "churn": customer.churn,
        "segment": (
            customer.segment.segment_name
            if customer.segment is not None
            else None
        ),
    }
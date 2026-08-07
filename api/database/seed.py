"""Seed PostgreSQL with customer profiles and business-readable segments."""

from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

from api.database.database import Base, SessionLocal, engine
from api.database.models import Customer, Segment

DATA_FILE = Path("data/final/customer_segments.csv")

SEGMENT_PROFILES = {
    0: {
        "name": "Active Customers",
        "description": "Recently active customers with moderate engagement and churn risk.",
    },
    1: {
        "name": "Champions",
        "description": "Highly engaged, recent and valuable customers with low churn risk.",
    },
    2: {
        "name": "Dormant Customers",
        "description": "Long-inactive customers with very high churn risk.",
    },
    3: {
        "name": "At-Risk Customers",
        "description": "Lapsed customers showing low engagement and high churn risk.",
    },
}


def normalize_customer_id(value: object) -> str:
    """Convert values such as 12346.0 into stable customer IDs such as 12346."""
    if pd.isna(value):
        raise ValueError("Customer ID cannot be missing.")

    text = str(value).strip()

    text = text.removesuffix(".0")

    return text


def optional_float(value: object) -> float | None:
    """Return a float value or None for missing data."""
    if pd.isna(value):
        return None

    return float(value)


def seed_segments(database: Session) -> dict[int, Segment]:
    """Create or update the four business-readable customer segments."""
    segment_records: dict[int, Segment] = {}

    for segment_number, profile in SEGMENT_PROFILES.items():
        segment = (
            database.query(Segment)
            .filter(Segment.segment_number == segment_number)
            .first()
        )

        if segment is None:
            segment = Segment(
                segment_number=segment_number,
                segment_name=profile["name"],
                description=profile["description"],
            )
            database.add(segment)
        else:
            segment.segment_name = profile["name"]
            segment.description = profile["description"]

        database.flush()
        segment_records[segment_number] = segment

    return segment_records


def seed_customers(
    database: Session,
    dataframe: pd.DataFrame,
    segments: dict[int, Segment],
) -> int:
    """Create or update customer profiles from the segmentation output."""
    processed = 0

    for row in dataframe.to_dict(orient="records"):
        customer_id = normalize_customer_id(row["Customer ID"])
        segment_number = int(row["Customer_Segment"])

        customer = database.get(Customer, customer_id)

        values = {
            "country": (
                None
                if pd.isna(row["Country"])
                else str(row["Country"]).strip()
            ),
            "recency": optional_float(row["Recency"]),
            "frequency": optional_float(row["Frequency"]),
            "monetary_total": optional_float(row["Monetary_Total"]),
            "predicted_clv": optional_float(row["Predicted_CLV"]),
            "engagement_score": optional_float(row["Engagement_Score"]),
            "churn": (
                None
                if pd.isna(row["Churn"])
                else int(row["Churn"])
            ),
            "segment_id": segments[segment_number].id,
        }

        if customer is None:
            customer = Customer(
                customer_id=customer_id,
                **values,
            )
            database.add(customer)
        else:
            for field, value in values.items():
                setattr(customer, field, value)

        processed += 1

    return processed


def main() -> None:
    """Create tables and populate segments and customers."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Customer segmentation file not found: {DATA_FILE}"
        )

    Base.metadata.create_all(bind=engine)

    dataframe = pd.read_csv(DATA_FILE)

    required_columns = {
        "Customer ID",
        "Country",
        "Recency",
        "Frequency",
        "Monetary_Total",
        "Predicted_CLV",
        "Engagement_Score",
        "Churn",
        "Customer_Segment",
    }

    missing_columns = required_columns.difference(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    database = SessionLocal()

    try:
        segments = seed_segments(database)
        customer_count = seed_customers(
            database,
            dataframe,
            segments,
        )

        database.commit()

        print("Database seeding completed successfully.")
        print(f"Segments stored: {len(segments)}")
        print(f"Customers stored: {customer_count}")

    except Exception:
        database.rollback()
        raise

    finally:
        database.close()


if __name__ == "__main__":
    main()
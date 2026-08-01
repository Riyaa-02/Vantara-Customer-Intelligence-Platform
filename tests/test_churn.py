import pandas as pd

from src.features.churn import create_churn_labels


def test_create_churn_labels():

    customer_features = pd.DataFrame(
        {
            "Recency": [10, 20, 30]
        },
        index=[101, 102, 103],
    )

    future_transactions = pd.DataFrame(
        {
            "Customer ID": [101, 103],
            "InvoiceDate": [
                pd.Timestamp("2025-04-10"),
                pd.Timestamp("2025-04-20"),
            ],
        }
    )

    result = create_churn_labels(
        customer_features=customer_features,
        future_transactions=future_transactions,
        cutoff_date=pd.Timestamp("2025-04-01"),
        future_window_days=90,
    )

    assert result.loc[101, "Churn"] == 0
    assert result.loc[102, "Churn"] == 1
    assert result.loc[103, "Churn"] == 0
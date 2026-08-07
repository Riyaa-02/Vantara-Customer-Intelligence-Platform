import pandas as pd

from src.features.feature_engineering import (
    create_return_features,
    create_rfm_features,
)


def test_create_rfm_features():
    """
    Verify that RFM features are calculated correctly.
    """

    df = pd.DataFrame(
        {
            "Customer ID": [1, 1, 2],
            "Invoice": [1001, 1002, 1003],
            "InvoiceDate": pd.to_datetime(
                [
                    "2021-01-01",
                    "2021-01-10",
                    "2021-01-05",
                ]
            ),
            "Total": [100, 200, 300],
        }
    )

    cutoff = pd.Timestamp("2021-02-01")

    rfm = create_rfm_features(
        df,
        cutoff_date=cutoff,
    )

    assert len(rfm) == 2

    assert "Recency" in rfm.columns
    assert "Frequency" in rfm.columns
    assert "Monetary_Total" in rfm.columns

    assert rfm.loc[1, "Frequency"] == 2
    assert rfm.loc[2, "Frequency"] == 1

    assert rfm.loc[1, "Monetary_Total"] == 300
    assert rfm.loc[2, "Monetary_Total"] == 300


def test_return_rate():
    """
    Verify return-rate calculation.
    """

    df = pd.DataFrame(
        {
            "Customer ID": [1, 1, 1, 2],
            "Quantity": [1, -1, 2, -3],
        }
    )

    result = create_return_features(df)

    assert "Return_Rate" in result.columns

    assert round(
        result.loc[1, "Return_Rate"],
        2,
    ) == round(100 / 3, 2)

    assert result.loc[2, "Return_Rate"] == 100
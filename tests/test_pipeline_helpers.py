from pathlib import Path

import joblib
import pandas as pd
import pytest

from src.features.dataset_split import (
    create_train_validation_test_split,
)
from src.features.validation import validate_feature_pipeline
from src.preprocessing.saving import (
    generate_preprocessing_report,
    save_processed_datasets,
)
from src.preprocessing.scaling import (
    save_scaler,
    scale_datasets,
)


def make_feature_dataset() -> pd.DataFrame:
    rows = 40

    return pd.DataFrame(
        {
            "Recency": list(range(1, rows + 1)),
            "Frequency": [value % 5 + 1 for value in range(rows)],
            "Monetary_Total": [
                float(value * 100)
                for value in range(1, rows + 1)
            ],
            "Churn": [0, 1] * 20,
        }
    )


def test_train_validation_test_split():
    features = make_feature_dataset()

    train, validation, test = (
        create_train_validation_test_split(features)
    )

    assert len(train) == 28
    assert len(validation) == 6
    assert len(test) == 6

    assert "Churn" in train.columns
    assert "Churn" in validation.columns
    assert "Churn" in test.columns

    assert set(train["Churn"].unique()) == {0, 1}
    assert set(validation["Churn"].unique()) == {0, 1}
    assert set(test["Churn"].unique()) == {0, 1}


def test_feature_validation_passes():
    cutoff = pd.Timestamp("2021-06-01")

    historical = pd.DataFrame(
        {
            "InvoiceDate": pd.to_datetime(
                [
                    "2021-05-01",
                    "2021-05-20",
                ]
            )
        }
    )

    future = pd.DataFrame(
        {
            "InvoiceDate": pd.to_datetime(
                [
                    "2021-06-05",
                    "2021-06-20",
                ]
            )
        }
    )

    features = pd.DataFrame(
        {
            "Recency": [10, 20],
        }
    )

    validate_feature_pipeline(
        historical_df=historical,
        future_df=future,
        features=features,
        cutoff_date=cutoff,
        future_window_days=90,
    )


def test_feature_validation_rejects_future_history():
    cutoff = pd.Timestamp("2021-06-01")

    historical = pd.DataFrame(
        {
            "InvoiceDate": pd.to_datetime(
                [
                    "2021-05-01",
                    "2021-06-10",
                ]
            )
        }
    )

    future = pd.DataFrame(
        {
            "InvoiceDate": pd.to_datetime(
                ["2021-06-20"]
            )
        }
    )

    features = pd.DataFrame(
        {
            "Recency": [10],
        }
    )

    with pytest.raises(
        AssertionError,
        match="Historical data contains future records",
    ):
        validate_feature_pipeline(
            historical_df=historical,
            future_df=future,
            features=features,
            cutoff_date=cutoff,
        )


def test_feature_validation_rejects_negative_recency():
    cutoff = pd.Timestamp("2021-06-01")

    historical = pd.DataFrame(
        {
            "InvoiceDate": pd.to_datetime(
                ["2021-05-01"]
            )
        }
    )

    future = pd.DataFrame(
        {
            "InvoiceDate": pd.to_datetime(
                ["2021-06-10"]
            )
        }
    )

    features = pd.DataFrame(
        {
            "Recency": [-1],
        }
    )

    with pytest.raises(
        AssertionError,
        match="Negative Recency detected",
    ):
        validate_feature_pipeline(
            historical_df=historical,
            future_df=future,
            features=features,
            cutoff_date=cutoff,
        )


def test_scale_datasets():
    X_train = pd.DataFrame(
        {
            "Recency": [10.0, 20.0, 30.0],
            "Frequency": [1.0, 2.0, 3.0],
            "Country_UK": [True, False, True],
        }
    )

    X_validation = pd.DataFrame(
        {
            "Recency": [15.0],
            "Frequency": [2.0],
            "Country_UK": [False],
        }
    )

    X_test = pd.DataFrame(
        {
            "Recency": [25.0],
            "Frequency": [3.0],
            "Country_UK": [True],
        }
    )

    (
        scaled_train,
        scaled_validation,
        scaled_test,
        scaler,
    ) = scale_datasets(
        X_train,
        X_validation,
        X_test,
    )

    assert scaled_train.shape == X_train.shape
    assert scaled_validation.shape == X_validation.shape
    assert scaled_test.shape == X_test.shape

    assert round(
        scaled_train["Recency"].mean(),
        7,
    ) == 0

    assert hasattr(scaler, "mean_")


def test_save_scaler(tmp_path: Path):
    X_train = pd.DataFrame(
        {
            "Recency": [10.0, 20.0, 30.0],
        }
    )

    _, _, _, scaler = scale_datasets(
        X_train,
        X_train.copy(),
        X_train.copy(),
    )

    output_file = tmp_path / "scaler.pkl"

    save_scaler(
        scaler,
        output_path=output_file,
    )

    assert output_file.exists()

    loaded_scaler = joblib.load(output_file)

    assert hasattr(loaded_scaler, "mean_")


def test_save_processed_datasets(tmp_path: Path):
    X_train = pd.DataFrame(
        {
            "Recency": [1.0, 2.0],
        }
    )
    y_train = pd.Series(
        [0, 1],
        name="Churn",
    )

    X_validation = pd.DataFrame(
        {
            "Recency": [3.0],
        }
    )
    y_validation = pd.Series(
        [0],
        name="Churn",
    )

    X_test = pd.DataFrame(
        {
            "Recency": [4.0],
        }
    )
    y_test = pd.Series(
        [1],
        name="Churn",
    )

    final_dataset = save_processed_datasets(
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        X_test=X_test,
        y_test=y_test,
        output_dir=tmp_path,
    )

    expected_files = [
        "X_train.csv",
        "X_validation.csv",
        "X_test.csv",
        "y_train.csv",
        "y_validation.csv",
        "y_test.csv",
        "customer_ml_dataset.csv",
    ]

    for filename in expected_files:
        assert (tmp_path / filename).exists()

    assert final_dataset.shape == (4, 2)


def test_generate_preprocessing_report(tmp_path: Path):
    report_file = tmp_path / "preprocessing_report.txt"

    generate_preprocessing_report(
        output_file=report_file,
        train_size=70,
        validation_size=15,
        test_size=15,
        feature_count=54,
    )

    assert report_file.exists()

    report_text = report_file.read_text(
        encoding="utf-8"
    )

    assert "Training Samples      : 70" in report_text
    assert "Validation Samples    : 15" in report_text
    assert "Testing Samples       : 15" in report_text
    assert "Total Features        : 54" in report_text
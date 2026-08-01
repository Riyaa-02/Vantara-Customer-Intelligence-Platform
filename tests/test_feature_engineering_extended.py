from pathlib import Path

import pandas as pd
import pytest

from src.features.feature_engineering import (
    build_customer_features,
    build_feature_pipeline,
    calculate_average_purchase_gap,
    create_clv_features,
    create_customer_index,
    create_customer_profile_features,
    create_engagement_score,
    create_price_sensitivity_feature,
    create_purchase_features,
    get_customer_mode_country,
    load_and_prepare_data,
    split_historical_future,
)
from src.preprocessing.cleaning import (
    clean_datasets,
    load_datasets,
    remove_duplicate_features,
)
from src.preprocessing.encoding import split_features_target


def make_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Invoice": [
                "1001",
                "1002",
                "1003",
                "1004",
                "1005",
                "1006",
            ],
            "StockCode": [
                "A",
                "B",
                "A",
                "C",
                "D",
                "E",
            ],
            "Description_Clean": [
                "RED MUG",
                "BLUE PLATE",
                "RED MUG",
                "GREEN BOWL",
                "WHITE CUP",
                "BLACK TRAY",
            ],
            "Quantity": [
                2,
                3,
                -1,
                4,
                5,
                2,
            ],
            "InvoiceDate": pd.to_datetime(
                [
                    "2021-01-01",
                    "2021-01-11",
                    "2021-01-21",
                    "2021-02-01",
                    "2021-06-10",
                    "2021-06-20",
                ]
            ),
            "Price": [
                5.0,
                10.0,
                5.0,
                8.0,
                4.0,
                12.0,
            ],
            "Customer ID": [
                1,
                1,
                1,
                2,
                1,
                2,
            ],
            "Country": [
                "UK",
                "UK",
                "UK",
                "France",
                "UK",
                "France",
            ],
        }
    )


def test_load_and_prepare_data(tmp_path: Path):
    input_file = tmp_path / "transactions.csv"

    dataframe = make_transactions()
    dataframe["InvoiceDate"] = dataframe[
        "InvoiceDate"
    ].astype(str)

    dataframe.to_csv(
        input_file,
        index=False,
    )

    result = load_and_prepare_data(input_file)

    assert pd.api.types.is_datetime64_any_dtype(
        result["InvoiceDate"]
    )

    assert "Total" in result.columns
    assert result.loc[0, "Total"] == 10.0


def test_load_and_prepare_data_rejects_missing_columns(
    tmp_path: Path,
):
    input_file = tmp_path / "invalid.csv"

    dataframe = make_transactions().drop(
        columns=["Country"]
    )

    dataframe.to_csv(
        input_file,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="Required columns are missing",
    ):
        load_and_prepare_data(input_file)


def test_split_historical_future():
    dataframe = make_transactions()

    historical, future = split_historical_future(
        dataframe,
        cutoff_date=pd.Timestamp("2021-06-01"),
    )

    assert historical["InvoiceDate"].max() < pd.Timestamp(
        "2021-06-01"
    )

    assert future["InvoiceDate"].min() >= pd.Timestamp(
        "2021-06-01"
    )

    assert len(historical) == 4
    assert len(future) == 2


def test_split_historical_future_rejects_empty_history():
    dataframe = make_transactions()

    with pytest.raises(
        ValueError,
        match="Historical dataset is empty",
    ):
        split_historical_future(
            dataframe,
            cutoff_date=pd.Timestamp("2020-01-01"),
        )


def test_create_customer_index():
    historical = make_transactions().iloc[:4]

    result = create_customer_index(historical)

    assert list(result.index) == [1, 2]
    assert result.index.name == "Customer ID"


def test_calculate_average_purchase_gap():
    customer_transactions = pd.DataFrame(
        {
            "InvoiceDate": pd.to_datetime(
                [
                    "2021-01-01",
                    "2021-01-11",
                    "2021-01-21",
                ]
            )
        }
    )

    result = calculate_average_purchase_gap(
        customer_transactions
    )

    assert result == 10.0


def test_average_purchase_gap_for_one_date_is_zero():
    customer_transactions = pd.DataFrame(
        {
            "InvoiceDate": pd.to_datetime(
                ["2021-01-01"]
            )
        }
    )

    assert (
        calculate_average_purchase_gap(
            customer_transactions
        )
        == 0.0
    )


def test_create_purchase_features():
    historical = make_transactions().iloc[:4].copy()
    historical["Total"] = (
        historical["Quantity"]
        * historical["Price"]
    )

    from src.features.feature_engineering import (
        create_rfm_features,
    )

    rfm = create_rfm_features(
        historical,
        cutoff_date=pd.Timestamp("2021-06-01"),
    )

    result = create_purchase_features(
        historical,
        rfm_features=rfm,
    )

    assert "Total_Quantity" in result.columns
    assert "Average_Basket_Size" in result.columns
    assert "Average_Order_Value" in result.columns
    assert "Unique_Products" in result.columns
    assert "Average_Purchase_Gap" in result.columns

    assert result.loc[1, "Unique_Products"] == 2
    assert result.loc[2, "Unique_Products"] == 1


def test_get_customer_mode_country():
    countries = pd.Series(
        ["UK", "UK", "France"]
    )

    assert get_customer_mode_country(countries) == "UK"


def test_get_customer_mode_country_unknown():
    countries = pd.Series(
        [None, None],
        dtype="object",
    )

    assert (
        get_customer_mode_country(countries)
        == "Unknown"
    )


def test_customer_profile_features():
    historical = make_transactions().iloc[:4]

    result = create_customer_profile_features(
        historical
    )

    assert "Customer_Lifespan_Days" in result.columns
    assert "Country" in result.columns

    assert result.loc[1, "Customer_Lifespan_Days"] == 20
    assert result.loc[1, "Country"] == "UK"


def test_price_sensitivity_feature():
    historical = make_transactions().iloc[:4]

    result = create_price_sensitivity_feature(
        historical
    )

    assert "Price_Sensitivity" in result.columns
    assert result.loc[2, "Price_Sensitivity"] == 0


def test_create_clv_features():
    monetary = pd.Series(
        [100.0, 200.0],
        index=[1, 2],
    )

    lifespan = pd.Series(
        [99, 199],
        index=[1, 2],
    )

    result = create_clv_features(
        monetary_total=monetary,
        customer_lifespan_days=lifespan,
    )

    assert "Historical_Revenue" in result.columns
    assert "Predicted_CLV" in result.columns
    assert result["Predicted_CLV"].ge(0).all()


def test_create_clv_rejects_invalid_quantile():
    monetary = pd.Series([100.0])
    lifespan = pd.Series([10])

    with pytest.raises(
        ValueError,
        match="clipping_quantile",
    ):
        create_clv_features(
            monetary_total=monetary,
            customer_lifespan_days=lifespan,
            clipping_quantile=0,
        )


def test_create_engagement_score():
    features = pd.DataFrame(
        {
            "Recency": [10, 30, 50],
            "Frequency": [10, 5, 1],
            "Monetary_Total": [1000, 500, 100],
        },
        index=[1, 2, 3],
    )

    result = create_engagement_score(features)

    assert result.name == "Engagement_Score"
    assert result.between(0, 100).all()
    assert result.loc[1] > result.loc[3]


def test_engagement_score_rejects_missing_columns():
    features = pd.DataFrame(
        {
            "Recency": [10],
        }
    )

    with pytest.raises(
        ValueError,
        match="Missing columns",
    ):
        create_engagement_score(features)


def test_build_customer_features():
    historical = make_transactions().iloc[:4].copy()

    historical["Total"] = (
        historical["Quantity"]
        * historical["Price"]
    )

    result = build_customer_features(
        historical,
        cutoff_date=pd.Timestamp("2021-06-01"),
    )

    expected_columns = {
        "Recency",
        "Frequency",
        "Monetary_Total",
        "Average_Order_Value",
        "Customer_Lifespan_Days",
        "Return_Rate",
        "Price_Sensitivity",
        "Predicted_CLV",
        "Engagement_Score",
    }

    assert expected_columns.issubset(result.columns)
    assert len(result) == 2
    assert result.isna().sum().sum() == 0


def test_build_feature_pipeline(tmp_path: Path):
    input_file = tmp_path / "transactions.csv"

    dataframe = make_transactions()
    dataframe["InvoiceDate"] = dataframe[
        "InvoiceDate"
    ].astype(str)

    dataframe.to_csv(
        input_file,
        index=False,
    )

    features, historical, future = (
        build_feature_pipeline(
            input_file,
            cutoff_date=pd.Timestamp("2021-06-01"),
        )
    )

    assert len(features) == 2
    assert len(historical) == 4
    assert len(future) == 2
    assert "Engagement_Score" in features.columns


def test_remove_duplicate_features():
    dataframe = pd.DataFrame(
        {
            "Historical_Revenue": [100, 200],
            "Recency": [10, 20],
        }
    )

    result = remove_duplicate_features(dataframe)

    assert "Historical_Revenue" not in result.columns
    assert "Recency" in result.columns


def test_load_and_clean_datasets(tmp_path: Path):
    train = pd.DataFrame(
        {
            "Customer ID": [1, 2],
            "Historical_Revenue": [100, 200],
            "Recency": [10.0, None],
            "Country": ["UK", None],
            "Churn": [0, 1],
        }
    )

    validation = pd.DataFrame(
        {
            "Customer ID": [3],
            "Historical_Revenue": [300],
            "Recency": [None],
            "Country": [None],
            "Churn": [0],
        }
    )

    test = pd.DataFrame(
        {
            "Customer ID": [4],
            "Historical_Revenue": [400],
            "Recency": [None],
            "Country": [None],
            "Churn": [1],
        }
    )

    train.to_csv(
        tmp_path / "train.csv",
        index=False,
    )
    validation.to_csv(
        tmp_path / "validation.csv",
        index=False,
    )
    test.to_csv(
        tmp_path / "test.csv",
        index=False,
    )

    loaded_train, loaded_validation, loaded_test = (
        load_datasets(tmp_path)
    )

    assert len(loaded_train) == 2
    assert len(loaded_validation) == 1
    assert len(loaded_test) == 1

    clean_train, clean_validation, clean_test = (
        clean_datasets(tmp_path)
    )

    assert "Historical_Revenue" not in clean_train.columns
    assert clean_train.isna().sum().sum() == 0
    assert clean_validation.isna().sum().sum() == 0
    assert clean_test.isna().sum().sum() == 0


def test_split_features_target():
    train = pd.DataFrame(
        {
            "Customer ID": [1, 2],
            "Recency": [10, 20],
            "Churn": [0, 1],
        }
    )

    validation = pd.DataFrame(
        {
            "Customer ID": [3],
            "Recency": [30],
            "Churn": [0],
        }
    )

    test = pd.DataFrame(
        {
            "Customer ID": [4],
            "Recency": [40],
            "Churn": [1],
        }
    )

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
    ) = split_features_target(
        train,
        validation,
        test,
    )

    assert "Customer ID" not in X_train.columns
    assert "Churn" not in X_train.columns
    assert list(y_train) == [0, 1]

    assert len(X_validation) == 1
    assert len(y_validation) == 1
    assert len(X_test) == 1
    assert len(y_test) == 1
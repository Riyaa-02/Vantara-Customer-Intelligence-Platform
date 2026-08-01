import pandas as pd

from src.preprocessing.cleaning import fill_missing_values
from src.preprocessing.encoding import encode_country


def test_fill_missing_values():

    train = pd.DataFrame(
        {
            "Country": ["UK", None],
            "Recency": [10, None],
        }
    )

    validation = pd.DataFrame(
        {
            "Country": [None],
            "Recency": [None],
        }
    )

    test = pd.DataFrame(
        {
            "Country": [None],
            "Recency": [None],
        }
    )

    train, validation, test = fill_missing_values(
        train,
        validation,
        test,
    )

    assert train.isnull().sum().sum() == 0
    assert validation.isnull().sum().sum() == 0
    assert test.isnull().sum().sum() == 0


def test_encode_country():

    train = pd.DataFrame(
        {
            "Country": ["UK", "France", "Germany"],
        }
    )

    validation = pd.DataFrame(
        {
            "Country": ["UK", "France"],
        }
    )

    test = pd.DataFrame(
        {
            "Country": ["Germany"],
        }
    )

    train_enc, val_enc, test_enc = encode_country(
        train,
        validation,
        test,
    )

    # Original column should be removed
    assert "Country" not in train_enc.columns

    # One-hot columns should exist
    assert "Country_Germany" in train_enc.columns
    assert "Country_UK" in train_enc.columns

    # All datasets should have identical columns
    assert list(train_enc.columns) == list(val_enc.columns)
    assert list(train_enc.columns) == list(test_enc.columns)
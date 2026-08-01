"""
Encoding utilities for the preprocessing pipeline.
"""

from __future__ import annotations

import pandas as pd


def encode_country(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    One-hot encode the Country column while ensuring
    train, validation and test have identical columns.
    """

    train = train.copy()
    validation = validation.copy()
    test = test.copy()

    combined = pd.concat(
        [train, validation, test],
        keys=["train", "validation", "test"],
    )

    if "Country" in combined.columns:
        combined = pd.get_dummies(
            combined,
            columns=["Country"],
            drop_first=True,
        )

    train = combined.xs("train")
    validation = combined.xs("validation")
    test = combined.xs("test")

    train, validation = train.align(
        validation,
        axis=1,
        fill_value=0,
    )

    train, test = train.align(
        test,
        axis=1,
        fill_value=0,
    )

    validation, test = validation.align(
        test,
        axis=1,
        fill_value=0,
    )

    return train, validation, test


def split_features_target(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    target_column: str = "Churn",
):
    """
    Separate features and target.
    """

    X_train = train.drop(columns=[target_column])
    y_train = train[target_column]

    X_validation = validation.drop(columns=[target_column])
    y_validation = validation[target_column]

    X_test = test.drop(columns=[target_column])
    y_test = test[target_column]

    identifier = "Customer ID"

    for dataset in (
        X_train,
        X_validation,
        X_test,
    ):
        if identifier in dataset.columns:
            dataset.drop(
                columns=[identifier],
                inplace=True,
            )

    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
    )
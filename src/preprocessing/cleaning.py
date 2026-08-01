"""
Cleaning utilities for the preprocessing pipeline.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_datasets(
    processed_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the train, validation and test datasets produced by Day 4.
    """

    processed_dir = Path(processed_dir)

    train = pd.read_csv(processed_dir / "train.csv")
    validation = pd.read_csv(processed_dir / "validation.csv")
    test = pd.read_csv(processed_dir / "test.csv")

    return train, validation, test


def remove_duplicate_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove duplicate or unnecessary features.
    """

    df = df.copy()

    duplicate_columns = [
        "Historical_Revenue",
    ]

    existing = [
        col
        for col in duplicate_columns
        if col in df.columns
    ]

    if existing:
        df = df.drop(columns=existing)

    return df


def fill_missing_values(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fill missing values using statistics calculated only
    from the training dataset.
    """

    train = train.copy()
    validation = validation.copy()
    test = test.copy()

    numeric_columns = train.select_dtypes(
        include=["number"]
    ).columns

    categorical_columns = train.select_dtypes(
        include=["object"]
    ).columns

    for column in numeric_columns:

        median = train[column].median()

        train[column] = train[column].fillna(median)
        validation[column] = validation[column].fillna(median)
        test[column] = test[column].fillna(median)

    for column in categorical_columns:

        mode = train[column].mode()

        if mode.empty:
            continue

        value = mode.iloc[0]

        train[column] = train[column].fillna(value)
        validation[column] = validation[column].fillna(value)
        test[column] = test[column].fillna(value)

    return train, validation, test


def clean_datasets(
    processed_dir: str | Path,
):
    """
    Complete cleaning pipeline.
    """

    train, validation, test = load_datasets(
        processed_dir
    )

    train = remove_duplicate_features(train)
    validation = remove_duplicate_features(validation)
    test = remove_duplicate_features(test)

    train, validation, test = fill_missing_values(
        train,
        validation,
        test,
    )

    return train, validation, test
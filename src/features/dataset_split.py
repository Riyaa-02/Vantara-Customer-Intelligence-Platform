"""
Dataset splitting utilities.
"""

from __future__ import annotations

import pandas as pd

from sklearn.model_selection import train_test_split


def create_train_validation_test_split(
    features: pd.DataFrame,
    target_column: str = "Churn",
    random_state: int = 42,
):
    """
    Create train, validation and test datasets.

    Split ratio:

    Train      70%
    Validation 15%
    Test       15%
    """

    X = features.drop(columns=[target_column])
    y = features[target_column]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=random_state,
        stratify=y,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=random_state,
        stratify=y_temp,
    )

    train = pd.concat(
        [X_train, y_train],
        axis=1,
    )

    validation = pd.concat(
        [X_val, y_val],
        axis=1,
    )

    test = pd.concat(
        [X_test, y_test],
        axis=1,
    )

    return train, validation, test
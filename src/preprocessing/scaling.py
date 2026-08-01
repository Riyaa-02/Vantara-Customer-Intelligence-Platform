"""
Scaling utilities for the preprocessing pipeline.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler


def scale_datasets(
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    StandardScaler,
]:
    """
    Scale numeric features using statistics learned
    only from the training dataset.
    """

    X_train = X_train.copy()
    X_validation = X_validation.copy()
    X_test = X_test.copy()

    scaler = StandardScaler()

    numeric_columns = X_train.select_dtypes(
        include=["number"]
    ).columns

    X_train[numeric_columns] = scaler.fit_transform(
        X_train[numeric_columns]
    )

    X_validation[numeric_columns] = scaler.transform(
        X_validation[numeric_columns]
    )

    X_test[numeric_columns] = scaler.transform(
        X_test[numeric_columns]
    )

    return (
        X_train,
        X_validation,
        X_test,
        scaler,
    )


def save_scaler(
    scaler: StandardScaler,
    output_path: str | Path = "models/customer_scaler.pkl",
) -> None:
    """
    Save the fitted scaler.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        scaler,
        output_path,
    )
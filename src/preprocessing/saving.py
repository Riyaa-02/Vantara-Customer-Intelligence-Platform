"""
Saving utilities for the preprocessing pipeline.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_processed_datasets(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: str | Path = "data/final",
):
    """
    Save all processed datasets.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(output_dir / "X_train.csv", index=False)
    X_validation.to_csv(output_dir / "X_validation.csv", index=False)
    X_test.to_csv(output_dir / "X_test.csv", index=False)

    y_train.to_csv(output_dir / "y_train.csv", index=False)
    y_validation.to_csv(output_dir / "y_validation.csv", index=False)
    y_test.to_csv(output_dir / "y_test.csv", index=False)

    final_dataset = pd.concat(
        [
            pd.concat([X_train, y_train], axis=1),
            pd.concat([X_validation, y_validation], axis=1),
            pd.concat([X_test, y_test], axis=1),
        ],
        ignore_index=True,
    )

    final_dataset.to_csv(
        output_dir / "customer_ml_dataset.csv",
        index=False,
    )

    return final_dataset


def generate_preprocessing_report(
    output_file: str | Path,
    train_size: int,
    validation_size: int,
    test_size: int,
    feature_count: int,
):
    """
    Generate a preprocessing summary report.
    """

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    report = f"""
==============================================================
VANTARA CUSTOMER INTELLIGENCE PLATFORM

PREPROCESSING REPORT
==============================================================

Training Samples      : {train_size}
Validation Samples    : {validation_size}
Testing Samples       : {test_size}

Total Features        : {feature_count}

Completed Steps
---------------
✓ Removed duplicate features
✓ Filled missing values
✓ Encoded categorical variables
✓ Standardized numerical features
✓ Saved ML-ready datasets
"""

    output_file.write_text(
        report,
        encoding="utf-8",
    )
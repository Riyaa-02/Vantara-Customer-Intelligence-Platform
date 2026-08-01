"""
==============================================================
VANTARA CUSTOMER INTELLIGENCE PLATFORM
DAY 5 : DATA PREPROCESSING
==============================================================
"""

from pathlib import Path

from src.preprocessing.cleaning import clean_datasets
from src.preprocessing.encoding import (
    encode_country,
    split_features_target,
)
from src.preprocessing.scaling import (
    scale_datasets,
    save_scaler,
)
from src.preprocessing.saving import (
    save_processed_datasets,
    generate_preprocessing_report,
)

PROCESSED_DIR = Path("data/processed")
FINAL_DIR = Path("data/final")
REPORT_FILE = Path("docs/preprocessing_report.txt")


def main():

    print("=" * 70)
    print("DAY 5 : DATA PREPROCESSING")
    print("=" * 70)

    print("\nLoading Day 4 datasets...")

    train, validation, test = clean_datasets(
        PROCESSED_DIR
    )

    print("Encoding categorical features...")

    train, validation, test = encode_country(
        train,
        validation,
        test,
    )

    print("Preparing ML datasets...")

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

    print("Scaling features...")

    (
        X_train,
        X_validation,
        X_test,
        scaler,
    ) = scale_datasets(
        X_train,
        X_validation,
        X_test,
    )

    save_scaler(scaler)

    print("Saving processed datasets...")

    final_dataset = save_processed_datasets(
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
        output_dir=FINAL_DIR,
    )

    generate_preprocessing_report(
        output_file=REPORT_FILE,
        train_size=len(X_train),
        validation_size=len(X_validation),
        test_size=len(X_test),
        feature_count=len(X_train.columns),
    )

    print("\nSummary")
    print("-" * 50)
    print(f"Training Samples   : {len(X_train):,}")
    print(f"Validation Samples : {len(X_validation):,}")
    print(f"Testing Samples    : {len(X_test):,}")
    print(f"Total Features     : {len(X_train.columns)}")
    print(f"Final Dataset      : {final_dataset.shape}")

    print("\nDAY 5 COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
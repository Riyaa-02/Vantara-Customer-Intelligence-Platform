"""Run the Vantara production data pipeline."""

import subprocess
import sys
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

PIPELINE_STEPS = [
    "day1_download.py",
    "day2_clean.py",
    "day4_features.py",
]


def main() -> None:
    """Run raw-data processing through customer feature generation."""

    for script in PIPELINE_STEPS:
        logger.info("Running %s...", script)

        subprocess.run(
            [sys.executable, script],
            check=True,
        )

    logger.info("Vantara data pipeline completed successfully.")


if __name__ == "__main__":
    main()
"""
Suggested path: src/prediction/load_features.py

SINGLE RESPONSIBILITY: Fetch, filter, and load live/latest feature records 
for batch inference and forecasting pipelines.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.utils.constants import PROCESSED_DATA_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureLoader:
    """
    Feature Loader engine responsible for preparing feature vectors for inference.
    """

    def __init__(self, data_dir: Path | str = PROCESSED_DATA_DIR) -> None:
        self.data_dir = Path(data_dir)

    def load_latest_features(
        self, filename: str = "features_test.parquet", num_rows: int | None = None
    ) -> pd.DataFrame:
        """Loads processed feature set from storage for batch inference."""
        file_path = self.data_dir / filename
        if not file_path.exists():
            error_msg = f"Processed feature file not found at: {file_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        logger.info("Loading latest inference features from: %s", file_path)
        df = pd.read_parquet(file_path)

        if num_rows and num_rows > 0:
            df = df.head(num_rows)

        logger.info("Successfully loaded %d feature records.", len(df))
        return df
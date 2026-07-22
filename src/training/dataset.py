"""
Suggested path: src/training/dataset.py

SINGLE RESPONSIBILITY: Load pre-split feature datasets (Train/Val/Test), 
validate schema and data integrity, isolate features and target, and serve 
DatasetSplits for model training and evaluation routines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.utils.constants import (
    EXPECTED_FEATURE_COUNT,
    METADATA_COLUMNS,
    PROCESSED_DATA_DIR,
    TARGET_COL,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class DatasetSplits:
    """
    Immutable Container holding Pre-Split Training, Validation, 
    and Test Datasets along with Metadata.
    """

    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    feature_names: List[str] = field(default_factory=list)

    @property
    def train_shape(self) -> Tuple[Tuple[int, int], int]:
        return self.X_train.shape, len(self.y_train)

    @property
    def val_shape(self) -> Tuple[Tuple[int, int], int]:
        return self.X_val.shape, len(self.y_val)

    @property
    def test_shape(self) -> Tuple[Tuple[int, int], int]:
        return self.X_test.shape, len(self.y_test)

    def to_numpy(self) -> Dict[str, np.ndarray]:
        """Convert DataFrames/Series into raw NumPy arrays for deep learning / numeric frameworks."""
        return {
            "X_train": self.X_train.to_numpy(dtype=np.float32),
            "y_train": self.y_train.to_numpy(dtype=np.float32),
            "X_val": self.X_val.to_numpy(dtype=np.float32),
            "y_val": self.y_val.to_numpy(dtype=np.float32),
            "X_test": self.X_test.to_numpy(dtype=np.float32),
            "y_test": self.y_test.to_numpy(dtype=np.float32),
        }

    def summary(self) -> Dict[str, Any]:
        """Return a summary of split sizes and feature dimension."""
        return {
            "train_samples": len(self.X_train),
            "val_samples": len(self.X_val),
            "test_samples": len(self.X_test),
            "num_features": len(self.feature_names),
        }


class AQIDatasetLoader:
    """
    Enterprise Dataset Loader for AQI Forecasting Pipeline.
    Manages loading of pre-split Parquet artifacts, strict schema validation,
    data integrity checks, and feature-target separation.
    """

    def __init__(
        self,
        processed_dir: Path | str = PROCESSED_DATA_DIR,
        target_col: str = TARGET_COL,
        expected_feature_count: int = EXPECTED_FEATURE_COUNT,
    ) -> None:
        self.processed_dir = Path(processed_dir)
        self.target_col = target_col
        self.expected_feature_count = expected_feature_count

        self.train_path = self.processed_dir / "features_train.parquet"
        self.val_path = self.processed_dir / "features_val.parquet"
        self.test_path = self.processed_dir / "features_test.parquet"

    def _validate_file_existence(self, file_path: Path) -> None:
        """Validate that target Parquet file exists on disk."""
        if not file_path.exists():
            error_msg = f"Required dataset artifact missing at: {file_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

    def _load_and_validate_single_split(self, file_path: Path, split_name: str) -> pd.DataFrame:
        """
        Load a single Parquet file and run integrity checks:
        - Non-empty validation
        - Duplicate column validation
        - Target column existence
        - Target column NaN validation
        """
        self._validate_file_existence(file_path)
        logger.info("Loading '%s' split from %s", split_name, file_path)

        df = pd.read_parquet(file_path)

        # Validate non-empty DataFrame
        if df.empty:
            error_msg = f"Loaded DataFrame for split '{split_name}' is empty."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Validate duplicate columns
        if df.columns.has_duplicates:
            duplicates = df.columns[df.columns.duplicated()].unique().tolist()
            error_msg = f"Duplicate columns detected in '{split_name}' split: {duplicates}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Validate target presence
        if self.target_col not in df.columns:
            error_msg = f"Target column '{self.target_col}' missing from '{split_name}' split."
            logger.error(error_msg)
            raise KeyError(error_msg)

        # Validate NaN values in target
        null_target_count = df[self.target_col].isnull().sum()
        if null_target_count > 0:
            error_msg = (
                f"Target column '{self.target_col}' in '{split_name}' split contains "
                f"{null_target_count} missing/NaN values."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info("Split '%s' loaded successfully with shape: %s", split_name, df.shape)
        return df

    def _validate_schema_consistency(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> None:
        """Ensure feature and schema consistency across Train, Val, and Test splits."""
        train_cols = list(train_df.columns)
        val_cols = list(val_df.columns)
        test_cols = list(test_df.columns)

        if train_cols != val_cols:
            error_msg = "Schema mismatch detected between Train and Validation splits."
            logger.error(error_msg)
            raise ValueError(error_msg)

        if train_cols != test_cols:
            error_msg = "Schema mismatch detected between Train and Test splits."
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _separate_features_and_target(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """Separate target variable and drop metadata columns from input features."""
        y = df[self.target_col].copy()

        cols_to_drop = [c for c in METADATA_COLUMNS if c in df.columns]
        cols_to_drop.append(self.target_col)

        X = df.drop(columns=cols_to_drop, errors="ignore")
        feature_names = list(X.columns)

        return X, y, feature_names

    def get_splits(self) -> DatasetSplits:
        """
        Loads pre-split feature datasets, runs schema and data validation,
        isolates features and target, and returns DatasetSplits dataclass.
        """
        train_df = self._load_and_validate_single_split(self.train_path, "Train")
        val_df = self._load_and_validate_single_split(self.val_path, "Validation")
        test_df = self._load_and_validate_single_split(self.test_path, "Test")

        self._validate_schema_consistency(train_df, val_df, test_df)

        X_train, y_train, train_features = self._separate_features_and_target(train_df)
        X_val, y_val, _ = self._separate_features_and_target(val_df)
        X_test, y_test, _ = self._separate_features_and_target(test_df)

        extracted_feature_count = len(train_features)
        if extracted_feature_count != self.expected_feature_count:
            error_msg = (
                f"Extracted feature count mismatch: Expected {self.expected_feature_count} features, "
                f"but extracted {extracted_feature_count} features."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(
            "Feature extraction validated successfully. Total feature count: %d",
            extracted_feature_count,
        )

        splits = DatasetSplits(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            feature_names=train_features,
        )

        logger.info(
            "Dataset Loading Completed. Train shape: %s | Val shape: %s | Test shape: %s",
            splits.train_shape[0],
            splits.val_shape[0],
            splits.test_shape[0],
        )

        return splits


def load_prepared_splits(processed_dir: Path | str = PROCESSED_DATA_DIR) -> DatasetSplits:
    """
    Public utility function to fetch pre-split train/val/test data.

    Usage:
        from src.training.dataset import load_prepared_splits
        splits = load_prepared_splits()
        X_train, y_train = splits.X_train, splits.y_train
    """
    loader = AQIDatasetLoader(processed_dir=processed_dir)
    return loader.get_splits()


if __name__ == "__main__":
    try:
        dataset_splits = load_prepared_splits()
        print("\n=== Dataset Load Successful ===")
        print("Summary:", dataset_splits.summary())
    except Exception as err:
        logger.exception("Failed to run dataset.py verification test.")
        raise
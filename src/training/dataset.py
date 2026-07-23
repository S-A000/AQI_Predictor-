"""
Suggested path: src/training/dataset.py

SINGLE RESPONSIBILITY: Load pre-split or cloud-stored feature datasets (Train/Val/Test),
validate schema and data integrity, isolate features and target, and serve
DatasetSplits for model training and evaluation routines.

UPDATED for Hopsworks Feature Store integration & multi-horizon direct forecasting:
    The feature store contains target columns (target_aqi_t+24, target_aqi_t+48, target_aqi_t+72).
    When loading splits for a SPECIFIC horizon, the other two horizon columns
    must be excluded from X.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hopsworks
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.training.forecast_targets import ForecastTargetBuilder
from src.utils.constants import (
    EXPECTED_FEATURE_COUNT,
    METADATA_COLUMNS,
    PROCESSED_DATA_DIR,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_HORIZON_HOURS = 24


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
    horizon_hours: int = DEFAULT_HORIZON_HOURS

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
            "horizon_hours": self.horizon_hours,
            "train_samples": len(self.X_train),
            "val_samples": len(self.X_val),
            "test_samples": len(self.X_test),
            "num_features": len(self.feature_names),
        }


class AQIDatasetLoader:
    """
    Enterprise Dataset Loader for AQI Forecasting Pipeline.
    Manages loading of features from Hopsworks Feature Store, strict schema validation,
    data integrity checks, and feature-target separation — for ONE
    forecast horizon at a time.
    """

    def __init__(
        self,
        processed_dir: Path | str = PROCESSED_DATA_DIR,
        horizon_hours: int = DEFAULT_HORIZON_HOURS,
        expected_feature_count: int | None = EXPECTED_FEATURE_COUNT,
    ) -> None:
        self.processed_dir = Path(processed_dir)
        self.horizon_hours = horizon_hours
        self.expected_feature_count = expected_feature_count

        self._target_builder = ForecastTargetBuilder()
        self.target_col = self._target_builder.target_column_name(horizon_hours)
        
        # All horizon target columns EXCEPT the one we're training for
        self._other_target_cols = [
            c for c in self._target_builder.all_target_columns() if c != self.target_col
        ]

    def _fetch_data_from_hopsworks(self) -> pd.DataFrame:
        """Connects to Hopsworks Feature Store and reads the feature group dataframe."""
        load_dotenv()
        api_key = os.getenv("HOPSWORKS_API_KEY")

        logger.info("🔌 Connecting to Hopsworks Feature Store...")
        project = hopsworks.login(project="mlopsaqi123", api_key_value=api_key)
        fs = project.get_feature_store()

        logger.info("📥 Fetching feature group 'aqi_features' (version 1)...")
        aqi_fg = fs.get_feature_group(name="aqi_features", version=1)
        df = aqi_fg.read()
        
        return df

    def _validate_and_clean_df(self, df: pd.DataFrame, split_name: str) -> pd.DataFrame:
        """Run integrity checks and handle target NaNs."""
        if df.empty:
            error_msg = f"Loaded DataFrame for split '{split_name}' is empty."
            logger.error(error_msg)
            raise ValueError(error_msg)

        if df.columns.has_duplicates:
            duplicates = df.columns[df.columns.duplicated()].unique().tolist()
            error_msg = f"Duplicate columns detected in '{split_name}' split: {duplicates}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if self.target_col not in df.columns:
            error_msg = f"Target column '{self.target_col}' missing from '{split_name}' split."
            logger.error(error_msg)
            raise KeyError(error_msg)

        before = len(df)
        df = df.dropna(subset=[self.target_col]).reset_index(drop=True)
        dropped = before - len(df)
        if dropped:
            logger.info(
                "Dropped %d row(s) with no valid %dh-ahead target in '%s' split (%d -> %d).",
                dropped, self.horizon_hours, split_name, before, len(df),
            )

        if df.empty:
            error_msg = (
                f"'{split_name}' split has NO rows with a valid {self.horizon_hours}h-ahead "
                f"target after dropping NaNs."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        return df

    def _separate_features_and_target(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """Separate target variable and drop metadata + OTHER-horizon target columns."""
        y = df[self.target_col].copy()

        cols_to_drop = [c for c in METADATA_COLUMNS if c in df.columns]
        cols_to_drop.append(self.target_col)
        cols_to_drop.extend(c for c in self._other_target_cols if c in df.columns)

        X = df.drop(columns=cols_to_drop, errors="ignore")
        feature_names = list(X.columns)

        return X, y, feature_names

    def get_splits(self) -> DatasetSplits:
        """
        Pulls data from Hopsworks, creates temporal train/val/test splits,
        runs validation, and returns DatasetSplits.
        """
        df = self._fetch_data_from_hopsworks()
        
        # Ensure data is sorted by timestamp if available for temporal splitting
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp").reset_index(drop=True)

        # Temporal split simulation from single feature group dataframe (70% Train, 15% Val, 15% Test)
        train_idx = int(len(df) * 0.7)
        val_idx = int(len(df) * 0.85)

        train_df = df.iloc[:train_idx].copy()
        val_df = df.iloc[train_idx:val_idx].copy()
        test_df = df.iloc[val_idx:].copy()

        train_df = self._validate_and_clean_df(train_df, "Train")
        val_df = self._validate_and_clean_df(val_df, "Validation")
        test_df = self._validate_and_clean_df(test_df, "Test")

        X_train, y_train, train_features = self._separate_features_and_target(train_df)
        X_val, y_val, _ = self._separate_features_and_target(val_df)
        X_test, y_test, _ = self._separate_features_and_target(test_df)

        extracted_feature_count = len(train_features)
        if self.expected_feature_count is not None and extracted_feature_count != self.expected_feature_count:
            logger.warning(
                f"Extracted feature count mismatch: Expected {self.expected_feature_count}, "
                f"but got {extracted_feature_count}. Proceeding with extracted features."
            )

        logger.info(
            "Feature extraction validated successfully (horizon=%dh). Total feature count: %d",
            self.horizon_hours, extracted_feature_count,
        )

        splits = DatasetSplits(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            feature_names=train_features,
            horizon_hours=self.horizon_hours,
        )

        logger.info(
            "Dataset Loading Completed (horizon=%dh). Train shape: %s | Val shape: %s | Test shape: %s",
            self.horizon_hours, splits.train_shape[0], splits.val_shape[0], splits.test_shape[0],
        )

        return splits


def load_prepared_splits(
    processed_dir: Path | str = PROCESSED_DATA_DIR,
    horizon_hours: int = DEFAULT_HORIZON_HOURS,
) -> DatasetSplits:
    """
    Public utility function to fetch pre-split train/val/test data from Hopsworks
    for a SPECIFIC forecast horizon.
    """
    loader = AQIDatasetLoader(processed_dir=processed_dir, horizon_hours=horizon_hours)
    return loader.get_splits()


if __name__ == "__main__":
    try:
        for horizon in (24, 48, 72):
            dataset_splits = load_prepared_splits(horizon_hours=horizon)
            print(f"\n=== Dataset Load Successful from Hopsworks ({horizon}h horizon) ===")
            print("Summary:", dataset_splits.summary())
    except Exception as err:
        logger.exception("Failed to run dataset.py verification test.")
        raise
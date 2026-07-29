"""
Suggested path: src/training/dataset.py

SINGLE RESPONSIBILITY: Load pre-split or locally stored feature datasets (Train/Val/Test),
validate schema and data integrity, isolate features and target, and serve
DatasetSplits for model training and evaluation routines.

UPDATED: Completely removed Hopsworks integration and reads local parquet splits directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

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
    Manages loading of features from local disk storage, strict schema validation,
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

    def _fetch_data_locally(self, split_name: str) -> pd.DataFrame:
        """Loads pre-saved split dataframe directly from local disk directory."""
        # Map formal split names to file suffixes
        suffix_map = {
            "Train": "train",
            "Validation": "val",
            "Test": "test"
        }
        file_suffix = suffix_map.get(split_name, split_name.lower())
        
        file_path = Path("data/training") / f"features_{file_suffix}.parquet"
        logger.info("📂 Loading '%s' split locally from disk: %s", split_name, file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Local feature split file not found: {file_path}")
            
        return pd.read_parquet(file_path)

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
            # Fallback check for alternative column formats
            possible_targets = [c for c in df.columns if f"target" in c or f"{self.horizon_hours}h" in c]
            if possible_targets:
                self.target_col = possible_targets[0]
                logger.warning("Target column mapped fallback to: '%s'", self.target_col)
            else:
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
        
        # Also drop non-feature tracking columns if present
        for extra_drop in ["date", "city", "split"]:
            if extra_drop in df.columns and extra_drop not in cols_to_drop:
                cols_to_drop.append(extra_drop)

        X = df.drop(columns=cols_to_drop, errors="ignore")
        feature_names = list(X.columns)

        return X, y, feature_names

    def get_splits(self) -> DatasetSplits:
        """
        Pulls data from local storage, validates schema, aligns features, and returns DatasetSplits.
        """
        train_df = self._fetch_data_locally("Train")
        val_df = self._fetch_data_locally("Validation")
        test_df = self._fetch_data_locally("Test")

        train_df = self._validate_and_clean_df(train_df, "Train")
        val_df = self._validate_and_clean_df(val_df, "Validation")
        test_df = self._validate_and_clean_df(test_df, "Test")

        X_train, y_train, train_features = self._separate_features_and_target(train_df)
        X_val, y_val, val_features = self._separate_features_and_target(val_df)
        X_test, y_test, test_features = self._separate_features_and_target(test_df)

        # Ensure all splits share the exact same set of feature columns in the exact same order
        common_features = [f for f in train_features if f in val_features and f in test_features]
        
        X_train = X_train[common_features].fillna(0)
        X_val = X_val[common_features].fillna(0)
        X_test = X_test[common_features].fillna(0)

        extracted_feature_count = len(common_features)
        logger.info(
            "Feature alignment completed successfully (horizon=%dh). Common feature count: %d",
            self.horizon_hours, extracted_feature_count,
        )

        splits = DatasetSplits(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            feature_names=common_features,
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
    Public utility function to fetch pre-split train/val/test data locally
    for a SPECIFIC forecast horizon.
    """
    loader = AQIDatasetLoader(processed_dir=processed_dir, horizon_hours=horizon_hours)
    return loader.get_splits()


if __name__ == "__main__":
    try:
        for horizon in (24, 48, 72):
            dataset_splits = load_prepared_splits(horizon_hours=horizon)
            print(f"\n=== Dataset Load Successful Locally ({horizon}h horizon) ===")
            print("Summary:", dataset_splits.summary())
    except Exception as err:
        logger.exception("Failed to run dataset.py verification test.")
        raise
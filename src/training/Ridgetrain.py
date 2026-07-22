"""
Suggested path: src/training/train.py

SINGLE RESPONSIBILITY: Train a Ridge Regression baseline model on pre-split features,
evaluate its metrics on the validation set, log results, and save the baseline model artifact
alongside structured JSON metrics and metadata.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.training.dataset import DatasetSplits, load_prepared_splits
from src.utils.constants import MODELS_DIR, TARGET_COL
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BaselineMetrics:
    """Container holding model evaluation metrics."""

    rmse: float
    mae: float
    r2: float
    training_time_sec: float
    val_samples: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rmse": round(self.rmse, 4),
            "mae": round(self.mae, 4),
            "r2": round(self.r2, 4),
            "training_time_sec": round(self.training_time_sec, 4),
            "val_samples": self.val_samples,
        }


class RidgeBaselineTrainer:
    """
    Enterprise Baseline Model Engine (Ridge Regression).
    Handles model fitting, validation benchmarking, metadata tracking, and artifact serialization.
    """

    def __init__(
        self,
        alpha: float = 10.0,
        models_dir: Path | str = MODELS_DIR,
        model_version: str = "1.0.0",
    ) -> None:
        self.alpha = alpha
        self.models_dir = Path(models_dir)
        self.registry_dir = self.models_dir / "registry"
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.model_version = model_version
        self.model = Ridge(alpha=self.alpha, random_state=42)
        self.metrics: BaselineMetrics | None = None

    def train_and_evaluate(self, splits: DatasetSplits) -> BaselineMetrics:
        """Train Ridge Regression and evaluate performance on Validation Split."""
        logger.info("Starting Ridge Regression baseline training (alpha=%.2f)...", self.alpha)

        start_time = time.time()
        self.model.fit(splits.X_train, splits.y_train)
        elapsed_time = time.time() - start_time

        logger.info("Ridge model trained successfully in %.4f seconds.", elapsed_time)

        # Validation Split Prediction
        y_val_pred = self.model.predict(splits.X_val)

        # Calculate Metrics
        mse = mean_squared_error(splits.y_val, y_val_pred)
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(splits.y_val, y_val_pred))
        r2 = float(r2_score(splits.y_val, y_val_pred))

        self.metrics = BaselineMetrics(
            rmse=rmse,
            mae=mae,
            r2=r2,
            training_time_sec=elapsed_time,
            val_samples=len(splits.X_val),
        )

        logger.info(
            "Ridge Baseline Validation Results -> RMSE: %.4f | MAE: %.4f | R²: %.4f",
            rmse,
            mae,
            r2,
        )

        return self.metrics

    def save_model(self, filename: str = "ridge_baseline.joblib") -> Path:
        """Save trained Ridge model artifact to disk using joblib."""
        save_path = self.registry_dir / filename
        joblib.dump(self.model, save_path)
        logger.info("Saved Ridge baseline model artifact to: %s", save_path)
        return save_path

    def save_metrics(
        self, metrics: BaselineMetrics, filename: str = "ridge_metrics.json"
    ) -> Path:
        """Save evaluation metrics as JSON alongside model artifact."""
        save_path = self.registry_dir / filename
        metrics_payload = metrics.to_dict()

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(metrics_payload, f, indent=4)

        logger.info("Saved Ridge model metrics to: %s", save_path)
        return save_path

    def save_metadata(
        self,
        splits: DatasetSplits,
        metrics: BaselineMetrics,
        filename: str = "ridge_metadata.json",
    ) -> Path:
        """Save comprehensive training and environment metadata as JSON alongside model artifact."""
        save_path = self.registry_dir / filename
        metadata_payload: Dict[str, Any] = {
            "model_name": "Ridge Baseline",
            "algorithm": "Ridge Regression",
            "training_timestamp": datetime.now(timezone.utc).isoformat(),
            "feature_count": len(splits.feature_names),
            "feature_names": splits.feature_names,
            "target_column": TARGET_COL,
            "training_samples": len(splits.X_train),
            "validation_samples": len(splits.X_val),
            "test_samples": len(splits.X_test),
            "hyperparameters": {
                "alpha": self.alpha,
                "random_state": 42,
            },
            "model_version": self.model_version,
            "framework": f"scikit-learn=={sklearn.__version__}",
            "python_version": sys.version,
        }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(metadata_payload, f, indent=4)

        logger.info("Saved Ridge model metadata to: %s", save_path)
        return save_path


def run_ridge_baseline() -> Tuple[RidgeBaselineTrainer, BaselineMetrics]:
    """
    Public entrypoint function to execute Ridge baseline training and artifact logging.
    """
    logger.info("--- Starting Ridge Baseline Training Phase ---")

    # Load pre-split dataset
    splits = load_prepared_splits()

    # Train and Evaluate
    trainer = RidgeBaselineTrainer(alpha=10.0)
    metrics = trainer.train_and_evaluate(splits)

    # Save artifacts, metrics, and metadata
    trainer.save_model("ridge_baseline.joblib")
    trainer.save_metrics(metrics, "ridge_metrics.json")
    trainer.save_metadata(splits, metrics, "ridge_metadata.json")

    logger.info("--- Ridge Baseline Training Phase Completed Successfully ---")
    return trainer, metrics


if __name__ == "__main__":
    try:
        _, metrics = run_ridge_baseline()
        print("\n=== Ridge Regression Baseline Metrics ===")
        for key, val in metrics.to_dict().items():
            print(f"  {key.upper()}: {val}")
    except Exception as err:
        logger.exception("Ridge baseline execution failed: %s", err)
        raise
"""
Suggested path: src/training/train_xgboost.py

SINGLE RESPONSIBILITY: Train an XGBoost Regressor for AQI forecasting, 
evaluate performance, log runs to MLflow Tracking, and register the model 
into MLflow Model Registry & local disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Tuple

import joblib
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from src.training.dataset import DatasetSplits, load_prepared_splits
from src.utils.constants import MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

# MLflow Experiment Configuration
EXPERIMENT_NAME = "AQI_Forecasting_Model_Comparison"
REGISTERED_MODEL_NAME = "AQI_Forecaster_Production"


class XGBoostTrainer:
    """
    Automated XGBoost Training and MLflow Registry Pipeline.
    """

    def __init__(self, experiment_name: str = EXPERIMENT_NAME) -> None:
        self.experiment_name = experiment_name
        mlflow.set_experiment(self.experiment_name)

    def _evaluate_model(self, model: XGBRegressor, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        """Calculates standard regression evaluation metrics for XGBoost forecasts."""
        preds = model.predict(X)
        mse = mean_squared_error(y, preds)
        return {
            "rmse": float(np.sqrt(mse)),
            "mae": float(mean_absolute_error(y, preds)),
            "r2": float(r2_score(y, preds)),
        }

    def train_and_evaluate(self, splits: DatasetSplits) -> Tuple[str, float, Any]:
        """
        Trains XGBoost model, logs metrics/params to MLflow, and registers the model.
        """
        logger.info("Starting XGBoost Model Training & MLflow Tracking...")

        model_name = "XGBoost_Regressor"

        with mlflow.start_run(run_name=model_name) as run:
            # 1. Initialize and Fit XGBoost Regressor
            model = XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                n_jobs=-1
            )

            model.fit(
                splits.X_train, splits.y_train,
                eval_set=[(splits.X_val, splits.y_val)],
                verbose=False
            )

            # 2. Evaluate on Validation & Test Sets
            val_metrics = self._evaluate_model(model, splits.X_val, splits.y_val)
            test_metrics = self._evaluate_model(model, splits.X_test, splits.y_test)

            logger.info("[%s] Val RMSE: %.4f | Test RMSE: %.4f | Test R²: %.4f", 
                        model_name, val_metrics["rmse"], test_metrics["rmse"], test_metrics["r2"])

            # 3. Log Parameters and Metrics to MLflow
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("learning_rate", 0.1)
            mlflow.log_param("max_depth", 6)

            mlflow.log_metric("val_rmse", val_metrics["rmse"])
            mlflow.log_metric("val_mae", val_metrics["mae"])
            mlflow.log_metric("val_r2", val_metrics["r2"])
            mlflow.log_metric("test_rmse", test_metrics["rmse"])
            mlflow.log_metric("test_mae", test_metrics["mae"])
            mlflow.log_metric("test_r2", test_metrics["r2"])

            # 4. Log Model Artifact to MLflow
            mlflow.xgboost.log_model(
                xgb_model=model,
                artifact_path="model"
            )

            run_id = run.info.run_id
            test_rmse = test_metrics["rmse"]

        # 5. Register Model in MLflow Registry
        self._register_best_model(run_id, model_name)

        # 6. Save Model Locally
        self._save_model_locally(model, model_name, splits, test_rmse)

        return model_name, test_rmse, model

    def _register_best_model(self, run_id: str, model_name: str) -> None:
        """Registers the XGBoost model run into MLflow Model Registry."""
        model_uri = f"runs:/{run_id}/model"
        logger.info("Registering '%s' (Run ID: %s) to MLflow Model Registry under name '%s'...",
                    model_name, run_id, REGISTERED_MODEL_NAME)

        try:
            registered_model = mlflow.register_model(
                model_uri=model_uri,
                name=REGISTERED_MODEL_NAME,
            )
            logger.info("Successfully registered XGBoost model version %s in MLflow Model Registry!", registered_model.version)
        except Exception as err:
            logger.error("Failed to register model in MLflow Registry: %s", err)

    def _save_model_locally(
        self, model: XGBRegressor, model_name: str, splits: DatasetSplits, test_rmse: float
    ) -> None:
        """Saves XGBoost model locally using joblib and metadata JSON."""
        registry_dir = MODELS_DIR / "registry"
        registry_dir.mkdir(parents=True, exist_ok=True)

        model_path = registry_dir / "xgboost_aqi_model.joblib"
        metadata_path = registry_dir / "xgboost_metadata.json"

        joblib.dump(model, model_path)

        metadata = {
            "model_name": model_name,
            "model_version": "1.0.0",
            "algorithm": "XGBoost",
            "best_test_rmse": round(test_rmse, 4),
            "feature_count": len(splits.feature_names),
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        logger.info("Saved local XGBoost production model artifact and metadata to: %s", registry_dir)


if __name__ == "__main__":
    try:
        splits = load_prepared_splits()
        trainer = XGBoostTrainer()
        winner_name, winner_rmse, _ = trainer.train_and_evaluate(splits)
        print(f"\n🎉 XGBoost Training & Registration Complete! Model: {winner_name} (RMSE: {winner_rmse:.4f})")
    except Exception as err:
        logger.exception("XGBoost training failed: %s", err)
        raise
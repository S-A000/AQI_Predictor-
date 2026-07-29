"""
Suggested path: src/training/train_multi_models.py

SINGLE RESPONSIBILITY: Train multiple candidate algorithms (Ridge, Random Forest, 
Gradient Boosting), evaluate performance, log runs to MLflow Tracking, compare 
metrics, and auto-register the best performing model into MLflow Model Registry.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
# 🔥 FAST Multi-threaded Imports
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.training.dataset import DatasetSplits, load_prepared_splits
from src.utils.constants import MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

# MLflow Base Configuration Templates
EXPERIMENT_NAME_TEMPLATE = "AQI_Model_Comparison_{horizon_hours}H"
REGISTERED_MODEL_NAME_TEMPLATE = "AQI_Forecaster_{horizon_hours}H"


class MultiModelTrainer:
    """
    Automated Multi-Model Training and MLflow Registry Pipeline.
    """

    def __init__(self, experiment_name: str | None = None) -> None:
        self.experiment_name = experiment_name

    def _get_candidate_models(self) -> Dict[str, Any]:
        """Returns optimized candidate model architectures for fast execution."""
        return {
            "Ridge_Regression": Ridge(alpha=10.0, random_state=42),
            "Random_Forest": RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1),
            "Gradient_Boosting": HistGradientBoostingRegressor(max_iter=100, learning_rate=0.1, max_depth=6, random_state=42),
        }

    def _evaluate_model(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Calculates standard regression evaluation metrics."""
        preds = model.predict(X)
        mse = mean_squared_error(y, preds)
        return {
            "rmse": float(np.sqrt(mse)),
            "mae": float(mean_absolute_error(y, preds)),
            "r2": float(r2_score(y, preds)),
        }

    def train_and_evaluate_all(
        self, splits: DatasetSplits, horizon_hours: int
    ) -> Tuple[str, float, Any]:
        """
        Trains all candidate models, logs runs to MLflow, compares performance,
        and identifies the best performing model for a specific horizon.
        """
        exp_name = self.experiment_name or EXPERIMENT_NAME_TEMPLATE.format(horizon_hours=horizon_hours)
        mlflow.set_experiment(exp_name)

        registered_model_name = REGISTERED_MODEL_NAME_TEMPLATE.format(horizon_hours=horizon_hours)

        logger.info("Starting Multi-Model Benchmark & MLflow Experiment Tracking for %dH Horizon...", horizon_hours)
        candidate_models = self._get_candidate_models()

        best_model_name = ""
        best_test_rmse = float("inf")
        best_model_artifact = None
        best_run_id = ""

        for model_name, model_instance in candidate_models.items():
            logger.info("--------------------------------------------------")
            logger.info("Training Candidate Model: %s (%dH Horizon)", model_name, horizon_hours)

            with mlflow.start_run(run_name=f"{model_name}_{horizon_hours}H") as run:
                # 1. Fit Model
                model_instance.fit(splits.X_train, splits.y_train)

                # 2. Evaluate on Validation & Test Sets
                val_metrics = self._evaluate_model(model_instance, splits.X_val, splits.y_val)
                test_metrics = self._evaluate_model(model_instance, splits.X_test, splits.y_test)

                logger.info(
                    "[%s - %dH] Val RMSE: %.4f | Test RMSE: %.4f | Test R²: %.4f", 
                    model_name, horizon_hours, val_metrics["rmse"], test_metrics["rmse"], test_metrics["r2"]
                )

                # 3. Log Parameters and Metrics to MLflow
                if hasattr(model_instance, "get_params"):
                    mlflow.log_params(model_instance.get_params())

                mlflow.log_param("horizon_hours", horizon_hours)

                mlflow.log_metric("val_rmse", val_metrics["rmse"])
                mlflow.log_metric("val_mae", val_metrics["mae"])
                mlflow.log_metric("val_r2", val_metrics["r2"])
                mlflow.log_metric("test_rmse", test_metrics["rmse"])
                mlflow.log_metric("test_mae", test_metrics["mae"])
                mlflow.log_metric("test_r2", test_metrics["r2"])

                # 4. Infer Model Signature and Log Model Artifact
                val_preds = model_instance.predict(splits.X_val)
                signature = infer_signature(splits.X_val, val_preds)

                mlflow.sklearn.log_model(
                sk_model=model_instance,
                name=f"{model_name}_{horizon_hours}h_model",
                signature=signature,
                input_example=splits.X_val.head(5),
                serialization_format="cloudpickle",
                )

                # 5. Check if this is the best model so far
                if test_metrics["rmse"] < best_test_rmse:
                    best_test_rmse = test_metrics["rmse"]
                    best_model_name = model_name
                    best_model_artifact = model_instance
                    best_run_id = run.info.run_id

        logger.info("==================================================")
        logger.info("WINNING MODEL [%dH]: %s with Test RMSE: %.4f", horizon_hours, best_model_name, best_test_rmse)
        logger.info("==================================================")

        # Step 6: Register the Best Model in MLflow Model Registry
        self._register_best_model(best_run_id, best_model_name, registered_model_name)

        # Step 7: Save Best Model locally to models/registry/
        self._save_best_model_locally(best_model_artifact, best_model_name, splits, best_test_rmse, horizon_hours)

        return best_model_name, best_test_rmse, best_model_artifact

    def _register_best_model(self, run_id: str, model_name: str, registered_model_name: str) -> None:
        """Registers the winning model run into MLflow Model Registry."""
        model_uri = f"runs:/{run_id}/model"
        logger.info(
            "Registering '%s' (Run ID: %s) to MLflow Model Registry under name '%s'...",
            model_name, run_id, registered_model_name
        )

        try:
            registered_model = mlflow.register_model(
                model_uri=model_uri,
                name=registered_model_name,
            )
            logger.info("Successfully registered model version %s in MLflow Model Registry!", registered_model.version)
        except Exception as err:
            logger.error("Failed to register model in MLflow Registry: %s", err)

    def _save_best_model_locally(
        self, model: Any, model_name: str, splits: DatasetSplits, test_rmse: float, horizon_hours: int
    ) -> None:
        """Saves winning model and metadata to local disk directory for predictor compatibility."""
        registry_dir = MODELS_DIR / "registry"
        registry_dir.mkdir(parents=True, exist_ok=True)

        model_path = registry_dir / f"{horizon_hours}h_model.joblib"
        metadata_path = registry_dir / f"{horizon_hours}h_metadata.json"

        joblib.dump(model, model_path)

        metadata = {
            "model_version": "1.0.0",
            "algorithm": model_name,
            "horizon_hours": horizon_hours,
            "feature_names": splits.feature_names,
            "feature_count": len(splits.feature_names),
            "best_test_rmse": round(test_rmse, 4),
            "training_timestamp": datetime.now(timezone.utc).isoformat(),
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        logger.info("Saved local production model artifact and metadata for %dH to: %s", horizon_hours, registry_dir)


if __name__ == "__main__":
    try:
        for horizon in [24, 48, 72]:
            logger.info("==================================================")
            logger.info("Executing Pipeline for Horizon: %dH", horizon)
            logger.info("==================================================")
            splits = load_prepared_splits(horizon_hours=horizon)
            trainer = MultiModelTrainer()
            winner_name, winner_rmse, _ = trainer.train_and_evaluate_all(splits, horizon_hours=horizon)
            print(f"\n🎉 Multi-Model Training Complete for {horizon}H! Winner: {winner_name} (RMSE: {winner_rmse:.4f})")
    except Exception as err:
        logger.exception("Multi-model training failed: %s", err)
        raise
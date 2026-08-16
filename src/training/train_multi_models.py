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
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.training.dataset import DatasetSplits, load_prepared_splits
from src.training.model_bundle import ModelBundle, save_model_bundle
from src.utils.constants import MODELS_DIR, PROCESSED_DATA_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

EXPERIMENT_NAME_TEMPLATE = "AQI_Model_Comparison_{horizon_hours}H"
REGISTERED_MODEL_NAME_TEMPLATE = "AQI_Forecaster_{horizon_hours}H"


class MultiModelTrainer:
    """Automated Multi-Model Training and MLflow Registry Pipeline."""

    def __init__(
        self,
        experiment_name: str | None = None,
        preprocessor: Any | None = None,
        preprocessor_path: Path | str | None = None,
    ) -> None:
        self.experiment_name = experiment_name
        self.preprocessor = preprocessor
        self.preprocessor_path = Path(
            preprocessor_path or (PROCESSED_DATA_DIR / "scaler.joblib")
        )
        self.selected_test_metrics_: Dict[str, float] = {}

    def _get_candidate_models(self) -> Dict[str, Any]:
        """Returns optimized candidate model architectures for fast execution."""
        return {
            "Ridge_Regression": Ridge(alpha=10.0, random_state=42),
            "Random_Forest": RandomForestRegressor(
                n_estimators=50,
                max_depth=8,
                random_state=42,
                n_jobs=-1,
            ),
            "Gradient_Boosting": HistGradientBoostingRegressor(
                max_iter=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
            ),
        }

    def _evaluate_model(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Dict[str, float]:
        """Calculate standard regression evaluation metrics."""
        predictions = model.predict(X)
        mse = mean_squared_error(y, predictions)
        return {
            "rmse": float(np.sqrt(mse)),
            "mae": float(mean_absolute_error(y, predictions)),
            "r2": float(r2_score(y, predictions)),
        }

    def train_and_evaluate_all(
        self,
        splits: DatasetSplits,
        horizon_hours: int,
    ) -> Tuple[str, float, Any]:
        """Select by validation RMSE, then evaluate the winner once on test."""
        experiment_name = self.experiment_name or EXPERIMENT_NAME_TEMPLATE.format(
            horizon_hours=horizon_hours
        )
        mlflow.set_experiment(experiment_name)
        registered_model_name = REGISTERED_MODEL_NAME_TEMPLATE.format(
            horizon_hours=horizon_hours
        )

        logger.info(
            "Starting Multi-Model Benchmark & MLflow Experiment Tracking for %dH Horizon...",
            horizon_hours,
        )

        best_model_name = ""
        best_validation_rmse = float("inf")
        best_model_artifact: Any | None = None
        best_run_id = ""
        best_artifact_name = ""
        best_train_metrics: Dict[str, float] = {}
        best_validation_metrics: Dict[str, float] = {}

        for model_name, model_instance in self._get_candidate_models().items():
            logger.info("--------------------------------------------------")
            logger.info(
                "Training Candidate Model: %s (%dH Horizon)",
                model_name,
                horizon_hours,
            )

            with mlflow.start_run(
                run_name=f"{model_name}_{horizon_hours}H"
            ) as run:
                model_instance.fit(splits.X_train, splits.y_train)

                train_metrics = self._evaluate_model(
                    model_instance, splits.X_train, splits.y_train
                )
                validation_metrics = self._evaluate_model(
                    model_instance, splits.X_val, splits.y_val
                )

                logger.info(
                    "[%s - %dH] Train RMSE: %.4f | Validation RMSE: %.4f",
                    model_name,
                    horizon_hours,
                    train_metrics["rmse"],
                    validation_metrics["rmse"],
                )

                if hasattr(model_instance, "get_params"):
                    mlflow.log_params(model_instance.get_params())
                mlflow.log_param("horizon_hours", horizon_hours)
                mlflow.log_param("selection_metric", "validation_rmse")
                for split_name, metrics in (
                    ("train", train_metrics),
                    ("val", validation_metrics),
                ):
                    for metric_name, metric_value in metrics.items():
                        mlflow.log_metric(
                            f"{split_name}_{metric_name}", metric_value
                        )

                validation_predictions = model_instance.predict(splits.X_val)
                signature = infer_signature(
                    splits.X_val, validation_predictions
                )
                artifact_name = f"{model_name}_{horizon_hours}h_model"
                mlflow.sklearn.log_model(
                    sk_model=model_instance,
                    name=artifact_name,
                    signature=signature,
                    input_example=splits.X_val.head(5),
                    serialization_format="cloudpickle",
                )

                # Iteration order is the deterministic tie-breaker. Test data
                # is never evaluated or consulted during candidate ranking.
                if validation_metrics["rmse"] < best_validation_rmse:
                    best_validation_rmse = validation_metrics["rmse"]
                    best_model_name = model_name
                    best_model_artifact = model_instance
                    best_run_id = run.info.run_id
                    best_artifact_name = artifact_name
                    best_train_metrics = train_metrics
                    best_validation_metrics = validation_metrics

        if best_model_artifact is None:
            raise RuntimeError("No candidate model completed training successfully.")

        test_metrics = self._evaluate_model(
            best_model_artifact, splits.X_test, splits.y_test
        )
        self.selected_test_metrics_ = dict(test_metrics)

        with mlflow.start_run(run_id=best_run_id):
            for metric_name, metric_value in test_metrics.items():
                mlflow.log_metric(f"test_{metric_name}", metric_value)
            mlflow.log_metric(
                "selected_validation_rmse", best_validation_rmse
            )

        logger.info("==================================================")
        logger.info(
            "WINNING MODEL [%dH]: %s with Validation RMSE: %.4f | Final Test RMSE: %.4f",
            horizon_hours,
            best_model_name,
            best_validation_rmse,
            test_metrics["rmse"],
        )
        logger.info("==================================================")

        self._register_best_model(
            best_run_id,
            best_model_name,
            registered_model_name,
            best_artifact_name,
        )
        self._save_best_model_locally(
            best_model_artifact,
            best_model_name,
            splits,
            train_metrics=best_train_metrics,
            validation_metrics=best_validation_metrics,
            test_metrics=test_metrics,
            horizon_hours=horizon_hours,
            run_id=best_run_id,
        )

        # The historical return contract exposes the final test RMSE.
        return best_model_name, test_metrics["rmse"], best_model_artifact

    def _register_best_model(
        self,
        run_id: str,
        model_name: str,
        registered_model_name: str,
        artifact_name: str,
    ) -> None:
        """Register the validation-selected model in MLflow Model Registry."""
        model_uri = f"runs:/{run_id}/{artifact_name}"
        logger.info(
            "Registering '%s' (Run ID: %s) to MLflow Model Registry under name '%s'...",
            model_name,
            run_id,
            registered_model_name,
        )

        try:
            registered_model = mlflow.register_model(
                model_uri=model_uri,
                name=registered_model_name,
            )
            logger.info(
                "Successfully registered model version %s in MLflow Model Registry!",
                registered_model.version,
            )
        except Exception as err:
            logger.error("Failed to register model in MLflow Registry: %s", err)

    def _get_fitted_preprocessor(self) -> Any:
        if self.preprocessor is not None:
            return self.preprocessor
        if not self.preprocessor_path.exists():
            raise FileNotFoundError(
                f"Fitted preprocessing artifact missing at: {self.preprocessor_path}"
            )
        return joblib.load(self.preprocessor_path)

    def _save_best_model_locally(
        self,
        model: Any,
        model_name: str,
        splits: DatasetSplits,
        *,
        train_metrics: Dict[str, float],
        validation_metrics: Dict[str, float],
        test_metrics: Dict[str, float],
        horizon_hours: int,
        run_id: str,
    ) -> None:
        """Save the winner, metadata, and complete inference bundle locally."""
        registry_dir = MODELS_DIR / "registry"
        registry_dir.mkdir(parents=True, exist_ok=True)

        model_path = registry_dir / f"{horizon_hours}h_model.joblib"
        metadata_path = registry_dir / f"{horizon_hours}h_metadata.json"
        bundle_path = registry_dir / f"{horizon_hours}h_bundle.joblib"
        joblib.dump(model, model_path)

        training_timestamp = datetime.now(timezone.utc).isoformat()
        metadata = {
            "model_version": "1.0.0",
            "algorithm": model_name,
            "horizon_hours": horizon_hours,
            "feature_names": splits.feature_names,
            "feature_count": len(splits.feature_names),
            "selection_metric": "validation_rmse",
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
            "best_test_rmse": round(test_metrics["rmse"], 4),
            "mlflow_run_id": run_id,
            "training_timestamp": training_timestamp,
        }

        with open(metadata_path, "w", encoding="utf-8") as file_handle:
            json.dump(metadata, file_handle, indent=4)

        preprocessor = self._get_fitted_preprocessor()
        bundle = ModelBundle.create(
            model=model,
            transformer=preprocessor,
            feature_names=splits.feature_names,
            horizon_hours=horizon_hours,
            model_name=model_name,
            model_version="1.0.0",
            run_metadata={"mlflow_run_id": run_id},
            training_metadata=metadata,
        )
        save_model_bundle(bundle, bundle_path)

        logger.info(
            "Saved local model, metadata, and bundle for %dH to: %s",
            horizon_hours,
            registry_dir,
        )


if __name__ == "__main__":
    try:
        for horizon in [24, 48, 72]:
            logger.info("==================================================")
            logger.info("Executing Pipeline for Horizon: %dH", horizon)
            logger.info("==================================================")
            splits = load_prepared_splits(horizon_hours=horizon)
            trainer = MultiModelTrainer()
            winner_name, winner_rmse, _ = trainer.train_and_evaluate_all(
                splits, horizon_hours=horizon
            )
            print(
                f"\nMulti-Model Training Complete for {horizon}H! "
                f"Winner: {winner_name} (RMSE: {winner_rmse:.4f})"
            )
    except Exception as err:
        logger.exception("Multi-model training failed: %s", err)
        raise

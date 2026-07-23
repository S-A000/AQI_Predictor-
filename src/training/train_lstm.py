"""
Suggested path: src/training/train_lstm.py

SINGLE RESPONSIBILITY: Train a deep learning LSTM model for AQI forecasting, 
handle 3D tensor reshaping, evaluate performance, log runs to MLflow Tracking, 
and register the model into MLflow Model Registry & local disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import mlflow
import mlflow.tensorflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

from src.training.dataset import DatasetSplits, load_prepared_splits
from src.utils.constants import MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

# MLflow Experiment Configuration
EXPERIMENT_NAME = "AQI_Forecasting_Model_Comparison"
REGISTERED_MODEL_NAME = "AQI_Forecaster_Production"


class LSTMTrainer:
    """
    Automated LSTM Deep Learning Training and MLflow Registry Pipeline.
    """

    def __init__(self, experiment_name: str = EXPERIMENT_NAME) -> None:
        self.experiment_name = experiment_name
        mlflow.set_experiment(self.experiment_name)

    def _reshape_for_lstm(self, X: pd.DataFrame) -> np.ndarray:
        """
        Reshapes 2D tabular features (Samples, Features) into 
        3D tensor (Samples, TimeSteps=1, Features) required by LSTM.
        """
        X_arr = X.to_numpy()
        return np.reshape(X_arr, (X_arr.shape[0], 1, X_arr.shape[1]))

    def _build_lstm_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """Builds a robust LSTM neural network architecture for regression."""
        model = Sequential([
            LSTM(64, input_shape=input_shape, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(1, activation="linear")  # Continuous AQI regression output
        ])
        model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])
        return model

    def _evaluate_model(self, model: tf.keras.Model, X_3d: np.ndarray, y: pd.Series) -> Dict[str, float]:
        """Calculates standard regression metrics for LSTM predictions."""
        preds = model.predict(X_3d, verbose=0).flatten()
        mse = mean_squared_error(y, preds)
        return {
            "rmse": float(np.sqrt(mse)),
            "mae": float(mean_absolute_error(y, preds)),
            "r2": float(r2_score(y, preds)),
        }

    def train_and_evaluate(self, splits: DatasetSplits) -> Tuple[str, float, Any]:
        """
        Prepares 3D data, trains LSTM, logs metrics/artifacts to MLflow, 
        and registers the model.
        """
        logger.info("Starting LSTM Deep Learning Training & MLflow Tracking...")

        # 1. Reshape 2D tabular splits to 3D tensors for LSTM
        X_train_3d = self._reshape_for_lstm(splits.X_train)
        X_val_3d = self._reshape_for_lstm(splits.X_val)
        X_test_3d = self._reshape_for_lstm(splits.X_test)

        model_name = "LSTM_DeepLearning"

        with mlflow.start_run(run_name=model_name) as run:
            # 2. Build and Train Model
            input_shape = (X_train_3d.shape[1], X_train_3d.shape[2])
            model = self._build_lstm_model(input_shape)

            # Early stopping to prevent overfitting
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True
            )

            history = model.fit(
                X_train_3d, splits.y_train,
                validation_data=(X_val_3d, splits.y_val),
                epochs=15,
                batch_size=64,
                callbacks=[early_stopping],
                verbose=1
            )

            # 3. Evaluate on Validation & Test Sets
            val_metrics = self._evaluate_model(model, X_val_3d, splits.y_val)
            test_metrics = self._evaluate_model(model, X_test_3d, splits.y_test)

            logger.info("[%s] Val RMSE: %.4f | Test RMSE: %.4f | Test R²: %.4f", 
                        model_name, val_metrics["rmse"], test_metrics["rmse"], test_metrics["r2"])

            # 4. Log Parameters and Metrics to MLflow
            mlflow.log_param("epochs", 15)
            mlflow.log_param("batch_size", 64)
            mlflow.log_param("hidden_units", 64)

            mlflow.log_metric("val_rmse", val_metrics["rmse"])
            mlflow.log_metric("val_mae", val_metrics["mae"])
            mlflow.log_metric("val_r2", val_metrics["r2"])
            mlflow.log_metric("test_rmse", test_metrics["rmse"])
            mlflow.log_metric("test_mae", test_metrics["mae"])
            mlflow.log_metric("test_r2", test_metrics["r2"])

            # 5. Log TensorFlow Model Artifact to MLflow
            mlflow.tensorflow.log_model(
                model=model,
                artifact_path="model"
            )

            run_id = run.info.run_id
            test_rmse = test_metrics["rmse"]

        # 6. Register Model in MLflow Registry
        self._register_best_model(run_id, model_name)

        # 7. Save Model Locally for Production Predictor
        self._save_model_locally(model, model_name, splits, test_rmse)

        return model_name, test_rmse, model

    def _register_best_model(self, run_id: str, model_name: str) -> None:
        """Registers the LSTM model run into MLflow Model Registry."""
        model_uri = f"runs:/{run_id}/model"
        logger.info("Registering '%s' (Run ID: %s) to MLflow Model Registry under name '%s'...",
                    model_name, run_id, REGISTERED_MODEL_NAME)

        try:
            registered_model = mlflow.register_model(
                model_uri=model_uri,
                name=REGISTERED_MODEL_NAME,
            )
            logger.info("Successfully registered LSTM model version %s in MLflow Model Registry!", registered_model.version)
        except Exception as err:
            logger.error("Failed to register model in MLflow Registry: %s", err)

    def _save_model_locally(
        self, model: tf.keras.Model, model_name: str, splits: DatasetSplits, test_rmse: float
    ) -> None:
        """Saves Keras model locally using TensorFlow native format and metadata JSON."""
        registry_dir = MODELS_DIR / "registry"
        registry_dir.mkdir(parents=True, exist_ok=True)

        model_path = registry_dir / "lstm_aqi_model.keras"
        metadata_path = registry_dir / "lstm_metadata.json"

        # Save keras model file
        model.save(model_path)

        metadata = {
            "model_name": model_name,
            "model_version": "1.0.0",
            "algorithm": "LSTM",
            "best_test_rmse": round(test_rmse, 4),
            "feature_names": splits.feature_names,
            "feature_count": len(splits.feature_names),
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        logger.info("Saved local LSTM production model artifact and metadata to: %s", registry_dir)


if __name__ == "__main__":
    try:
        splits = load_prepared_splits()
        trainer = LSTMTrainer()
        winner_name, winner_rmse, _ = trainer.train_and_evaluate(splits)
        print(f"\n🎉 LSTM Training & Registration Complete! Model: {winner_name} (RMSE: {winner_rmse:.4f})")
    except Exception as err:
        logger.exception("LSTM training failed: %s", err)
        raise
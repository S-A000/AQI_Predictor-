"""
Suggested path: src/training/evaluate.py

SINGLE RESPONSIBILITY: Load trained baseline model artifact from registry,
strictly verify model version and feature schema, evaluate performance on 
validation and test splits, compute regression and residual metrics, generate plots,
and optionally log metrics/artifacts to MLflow in an enterprise-grade MLOps setup.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")  # Enforce non-interactive backend for headless enterprise environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    explained_variance_score,
    max_error,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)

# Optional MLflow integration import with fail-safe fallback
try:
    import mlflow
    from mlflow.models import infer_signature
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

from src.training.dataset import DatasetSplits, load_prepared_splits
from src.utils.constants import EXPECTED_FEATURE_COUNT, MODELS_DIR, TARGET_COL
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelVersionError(Exception):
    """Raised when model version or registry metadata validation fails."""
    pass


class SchemaValidationError(Exception):
    """Raised when input feature schema fails strict verification checks."""
    pass


@dataclass(frozen=True)
class SplitMetrics:
    """Container holding comprehensive evaluation metrics for a single dataset split."""

    rmse: float
    mae: float
    mse: float
    r2: float
    explained_variance: float
    median_absolute_error: float
    max_error: float
    residual_mean: float
    residual_std: float
    residual_min: float
    residual_max: float

    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary with standardized precision rounding."""
        return {
            "rmse": round(self.rmse, 4),
            "mae": round(self.mae, 4),
            "mse": round(self.mse, 4),
            "r2": round(self.r2, 4),
            "explained_variance": round(self.explained_variance, 4),
            "median_absolute_error": round(self.median_absolute_error, 4),
            "max_error": round(self.max_error, 4),
            "residual_mean": round(self.residual_mean, 4),
            "residual_std": round(self.residual_std, 4),
            "residual_min": round(self.residual_min, 4),
            "residual_max": round(self.residual_max, 4),
        }


@dataclass(frozen=True)
class EvaluationResults:
    """Container holding evaluation metrics for both Validation and Test splits."""

    validation: SplitMetrics
    test: SplitMetrics

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        """Return combined evaluation metrics dictionary."""
        return {
            "validation": self.validation.to_dict(),
            "test": self.test.to_dict(),
        }


class ModelEvaluator:
    """
    Enterprise Model Evaluation Engine.
    Executes model version verification, feature schema validation, offline benchmark inference,
    residual statistical analysis, diagnostic plot generation, and MLflow-ready artifact logging.
    """

    def __init__(
        self,
        model_path: Path | str = MODELS_DIR / "registry" / "ridge_baseline.joblib",
        metadata_path: Path | str = MODELS_DIR / "registry" / "ridge_metadata.json",
        evaluation_dir: Path | str = MODELS_DIR / "registry" / "evaluation",
        expected_model_version: str = "1.0.0",
        expected_algorithm: str = "Ridge Regression",
        enable_mlflow: bool = False,
    ) -> None:
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self.evaluation_dir = Path(evaluation_dir)
        self.evaluation_dir.mkdir(parents=True, exist_ok=True)
        
        self.expected_model_version = expected_model_version
        self.expected_algorithm = expected_algorithm
        self.enable_mlflow = enable_mlflow and HAS_MLFLOW

        # Load metadata and model artifact with validation
        self.metadata = self._verify_and_load_metadata()
        self.model = self._load_model()

    def _verify_and_load_metadata(self) -> Dict[str, Any]:
        """Verify model metadata file existence and validate model version compatibility."""
        if not self.metadata_path.exists():
            error_msg = f"Model metadata JSON missing at: {self.metadata_path}. Evaluation aborted."
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        logger.info("Loading model metadata from: %s", self.metadata_path)
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        actual_version = metadata.get("model_version")
        actual_algorithm = metadata.get("algorithm")

        # 1. Model Version Verification
        if actual_version != self.expected_model_version:
            error_msg = (
                f"Model Version Verification Failed! Expected '{self.expected_model_version}', "
                f"but found '{actual_version}' in registry metadata."
            )
            logger.error(error_msg)
            raise ModelVersionError(error_msg)

        # 2. Algorithm Type Verification
        if actual_algorithm != self.expected_algorithm:
            error_msg = (
                f"Algorithm Mismatch! Expected '{self.expected_algorithm}', "
                f"but found '{actual_algorithm}' in metadata."
            )
            logger.error(error_msg)
            raise ModelVersionError(error_msg)

        logger.info(
            "Model Metadata Verified Successfully [Name: %s | Version: %s | Algorithm: %s]",
            metadata.get("model_name"),
            actual_version,
            actual_algorithm,
        )
        return metadata

    def _load_model(self) -> Any:
        """Load trained joblib model artifact from registry."""
        if not self.model_path.exists():
            error_msg = f"Trained model artifact not found at path: {self.model_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        logger.info("Loading model artifact from: %s", self.model_path)
        try:
            return joblib.load(self.model_path)
        except Exception as err:
            logger.exception("Failed to load model artifact: %s", err)
            raise

    def verify_feature_schema(self, X: pd.DataFrame, split_name: str) -> None:
        """
        Executes strict Feature Schema Verification against metadata specifications:
        - Exact Feature Count
        - Feature Column Name Matching
        - Column Ordering
        - Numeric Data Type Integrity
        """
        logger.info("Verifying Feature Schema for '%s' split...", split_name)
        expected_features: List[str] = self.metadata.get("feature_names", [])

        # Check 1: Feature Count Validation
        if len(X.columns) != len(expected_features):
            error_msg = (
                f"Schema Error [{split_name}]: Feature count mismatch. "
                f"Expected {len(expected_features)} features, but received {len(X.columns)}."
            )
            logger.error(error_msg)
            raise SchemaValidationError(error_msg)

        # Check 2: Missing Features Validation
        missing_features = set(expected_features) - set(X.columns)
        if missing_features:
            error_msg = f"Schema Error [{split_name}]: Missing required feature columns: {missing_features}"
            logger.error(error_msg)
            raise SchemaValidationError(error_msg)

        # Check 3: Unexpected Extra Features
        extra_features = set(X.columns) - set(expected_features)
        if extra_features:
            error_msg = f"Schema Error [{split_name}]: Unexpected extra feature columns found: {extra_features}"
            logger.error(error_msg)
            raise SchemaValidationError(error_msg)

        # Check 4: Exact Feature Column Ordering Verification
        if list(X.columns) != expected_features:
            error_msg = (
                f"Schema Error [{split_name}]: Column ordering mismatch! "
                f"Input features do not match expected training column order."
            )
            logger.error(error_msg)
            raise SchemaValidationError(error_msg)

        # Check 5: Numeric Data Type Validation
        non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric_cols:
            error_msg = (
                f"Schema Error [{split_name}]: Non-numeric feature dtypes detected in columns: {non_numeric_cols}"
            )
            logger.error(error_msg)
            raise SchemaValidationError(error_msg)

        logger.info("Feature Schema for '%s' split validated successfully.", split_name)

    def _compute_split_metrics(
        self, y_true: pd.Series, y_pred: np.ndarray
    ) -> Tuple[SplitMetrics, pd.DataFrame]:
        """Compute regression performance metrics, residual statistics, and predictions DataFrame."""
        residuals = y_true.values - y_pred

        mse = mean_squared_error(y_true, y_pred)
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        evs = float(explained_variance_score(y_true, y_pred))
        medae = float(median_absolute_error(y_true, y_pred))
        me = float(max_error(y_true, y_pred))

        res_mean = float(np.mean(residuals))
        res_std = float(np.std(residuals))
        res_min = float(np.min(residuals))
        res_max = float(np.max(residuals))

        metrics = SplitMetrics(
            rmse=rmse,
            mae=mae,
            mse=float(mse),
            r2=r2,
            explained_variance=evs,
            median_absolute_error=medae,
            max_error=me,
            residual_mean=res_mean,
            residual_std=res_std,
            residual_min=res_min,
            residual_max=res_max,
        )

        predictions_df = pd.DataFrame(
            {
                "actual": y_true.values,
                "predicted": y_pred,
                "residual": residuals,
            },
            index=y_true.index,
        )

        return metrics, predictions_df

    def _generate_plots(
        self, val_pred_df: pd.DataFrame, test_pred_df: pd.DataFrame
    ) -> List[Path]:
        """Generate and save visual diagnostic residual plots."""
        logger.info("Generating evaluation visual plots...")
        plot_paths: List[Path] = []

        # 1. Residual Histogram Plot
        plt.figure(figsize=(10, 5))
        plt.hist(
            val_pred_df["residual"],
            bins=50,
            alpha=0.6,
            label="Validation Residuals",
            color="#1f77b4",
            edgecolor="black",
        )
        plt.hist(
            test_pred_df["residual"],
            bins=50,
            alpha=0.6,
            label="Test Residuals",
            color="#2ca02c",
            edgecolor="black",
        )
        plt.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero Error")
        plt.title("Residual Distribution (Actual - Predicted)")
        plt.xlabel("Residual Value")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        hist_path = self.evaluation_dir / "residual_histogram.png"
        plt.savefig(hist_path, dpi=300)
        plt.close()
        plot_paths.append(hist_path)

        # 2. Residual Scatter Plot
        plt.figure(figsize=(10, 5))
        plt.scatter(
            val_pred_df["predicted"],
            val_pred_df["residual"],
            alpha=0.4,
            label="Validation",
            color="#1f77b4",
            s=15,
        )
        plt.scatter(
            test_pred_df["predicted"],
            test_pred_df["residual"],
            alpha=0.4,
            label="Test",
            color="#2ca02c",
            s=15,
        )
        plt.axhline(0, color="red", linestyle="--", linewidth=1.5)
        plt.title("Residuals vs Predicted Values")
        plt.xlabel("Predicted AQI")
        plt.ylabel("Residuals")
        plt.legend()
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        scatter_path = self.evaluation_dir / "residual_scatter.png"
        plt.savefig(scatter_path, dpi=300)
        plt.close()
        plot_paths.append(scatter_path)

        # 3. Actual vs Predicted Plot
        plt.figure(figsize=(10, 5))
        plt.scatter(
            val_pred_df["actual"],
            val_pred_df["predicted"],
            alpha=0.4,
            label="Validation",
            color="#1f77b4",
            s=15,
        )
        plt.scatter(
            test_pred_df["actual"],
            test_pred_df["predicted"],
            alpha=0.4,
            label="Test",
            color="#2ca02c",
            s=15,
        )

        min_val = min(val_pred_df["actual"].min(), test_pred_df["actual"].min())
        max_val = max(val_pred_df["actual"].max(), test_pred_df["actual"].max())
        plt.plot(
            [min_val, max_val],
            [min_val, max_val],
            color="red",
            linestyle="--",
            label="Ideal Fit (y=x)",
        )

        plt.title("Actual vs Predicted AQI")
        plt.xlabel("Actual AQI")
        plt.ylabel("Predicted AQI")
        plt.legend()
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        act_vs_pred_path = self.evaluation_dir / "actual_vs_predicted.png"
        plt.savefig(act_vs_pred_path, dpi=300)
        plt.close()
        plot_paths.append(act_vs_pred_path)

        logger.info("Saved all diagnostic plots to directory: %s", self.evaluation_dir)
        return plot_paths

    def _save_predictions(
        self, val_pred_df: pd.DataFrame, test_pred_df: pd.DataFrame
    ) -> Tuple[Path, Path]:
        """Save validation and test predictions to disk as Parquet files."""
        val_path = self.evaluation_dir / "validation_predictions.parquet"
        test_path = self.evaluation_dir / "test_predictions.parquet"

        val_pred_df.to_parquet(val_path, index=False)
        test_pred_df.to_parquet(test_path, index=False)

        logger.info("Saved predictions parquet artifacts.")
        return val_path, test_path

    def _save_metrics(self, results: EvaluationResults) -> Path:
        """Save evaluation metrics as JSON file."""
        metrics_path = self.evaluation_dir / "evaluation_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(results.to_dict(), f, indent=4)

        logger.info("Saved evaluation metrics JSON to: %s", metrics_path)
        return metrics_path

    def _save_metadata(self, splits: DatasetSplits) -> Path:
        """Save evaluation metadata JSON artifact."""
        metadata_path = self.evaluation_dir / "evaluation_metadata.json"
        metadata_payload: Dict[str, Any] = {
            "model_name": self.metadata.get("model_name", "Ridge Baseline"),
            "model_version": self.expected_model_version,
            "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
            "feature_count": len(splits.feature_names),
            "validation_samples": len(splits.X_val),
            "test_samples": len(splits.X_test),
            "sklearn_version": sklearn.__version__,
            "python_version": sys.version,
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata_payload, f, indent=4)

        logger.info("Saved evaluation metadata JSON to: %s", metadata_path)
        return metadata_path

    def _log_to_mlflow(
        self,
        splits: DatasetSplits,
        results: EvaluationResults,
        val_preds: np.ndarray,
        plot_paths: List[Path],
    ) -> None:
        """
        Log metrics, parameters, model signature, and evaluation artifacts to MLflow experiment tracking.
        """
        if not self.enable_mlflow:
            return

        logger.info("MLflow Integration Active: Logging evaluation metrics and artifacts...")
        try:
            # Flatten metrics for MLflow logging
            for split_name, split_metric_obj in [("val", results.validation), ("test", results.test)]:
                for metric_key, metric_val in split_metric_obj.to_dict().items():
                    mlflow.log_metric(f"{split_name}_{metric_key}", metric_val)

            # Log parameters
            mlflow.log_param("model_version", self.expected_model_version)
            mlflow.log_param("algorithm", self.expected_algorithm)
            mlflow.log_param("feature_count", len(splits.feature_names))

            # Infer Model Signature (MLflow feature inputs & output schema)
            signature = infer_signature(splits.X_val, val_preds)
            
            # Log Model Artifact with Signature
            mlflow.sklearn.log_model(
                sk_model=self.model,
                artifact_path="model",
                signature=signature,
            )

            # Log Evaluation Plots and Artifact JSONs
            for plot_path in plot_paths:
                mlflow.log_artifact(str(plot_path), artifact_path="evaluation_plots")
                
            mlflow.log_artifact(str(self.evaluation_dir / "evaluation_metrics.json"), artifact_path="metrics")
            mlflow.log_artifact(str(self.evaluation_dir / "evaluation_metadata.json"), artifact_path="metadata")

            logger.info("Successfully logged all metrics, model signature, and artifacts to MLflow.")
        except Exception as err:
            logger.error("Failed to log to MLflow tracking server: %s", err)

    def evaluate(self, splits: Optional[DatasetSplits] = None) -> EvaluationResults:
        """
        Main orchestration routine:
        1. Verify Feature Schemas (Validation & Test)
        2. Run Benchmark Inference
        3. Compute Regression & Residual Metrics
        4. Save Evaluation Predictions, Metrics, & Metadata
        5. Generate Residual Plots
        6. Log to MLflow (if enabled)
        """
        logger.info("--- Starting Model Evaluation Phase ---")

        if splits is None:
            splits = load_prepared_splits()

        # Step 1: Execute Strict Feature Schema Verification
        self.verify_feature_schema(splits.X_val, "Validation")
        self.verify_feature_schema(splits.X_test, "Test")

        # Step 2: Run Inference
        logger.info("Running inference on Validation dataset (%d samples)...", len(splits.X_val))
        val_preds = self.model.predict(splits.X_val)

        logger.info("Running inference on Test dataset (%d samples)...", len(splits.X_test))
        test_preds = self.model.predict(splits.X_test)

        # Step 3: Calculate Metrics & Residual DataFrames
        val_metrics, val_pred_df = self._compute_split_metrics(splits.y_val, val_preds)
        test_metrics, test_pred_df = self._compute_split_metrics(splits.y_test, test_preds)

        results = EvaluationResults(validation=val_metrics, test=test_metrics)

        # Step 4: Save Output Artifacts
        self._save_predictions(val_pred_df, test_pred_df)
        self._save_metrics(results)
        self._save_metadata(splits)
        plot_paths = self._generate_plots(val_pred_df, test_pred_df)

        # Step 5: Optional MLflow Logging Hook
        self._log_to_mlflow(splits, results, val_preds, plot_paths)

        logger.info(
            "Validation Set Results -> RMSE: %.4f | MAE: %.4f | R²: %.4f",
            val_metrics.rmse,
            val_metrics.mae,
            val_metrics.r2,
        )
        logger.info(
            "Test Set Results -> RMSE: %.4f | MAE: %.4f | R²: %.4f",
            test_metrics.rmse,
            test_metrics.mae,
            test_metrics.r2,
        )
        logger.info("--- Model Evaluation Phase Completed Successfully ---")

        return results


def run_evaluation(enable_mlflow: bool = False) -> EvaluationResults:
    """
    Public entrypoint function to execute model evaluation routine.
    """
    evaluator = ModelEvaluator(enable_mlflow=enable_mlflow)
    return evaluator.evaluate()


if __name__ == "__main__":
    try:
        results = run_evaluation(enable_mlflow=False)
        print("\n=== Model Evaluation Complete ===")
        print(json.dumps(results.to_dict(), indent=2))
    except Exception as err:
        logger.exception("Model evaluation execution failed: %s", err)
        raise

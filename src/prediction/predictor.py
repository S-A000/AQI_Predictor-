"""
Suggested path: src/prediction/predictor.py

SINGLE RESPONSIBILITY: Orchestrate real-time and batch AQI predictions.
Integrates payload validation (PredictionValidator), feature engineering and
scaler transformation (PredictionFeaturePipeline), feature alignment verification,
and model execution. Supports DIRECT MULTI-HORIZON FORECASTING (24h, 48h, 72h).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from src.prediction.feature_pipeline import PredictionFeaturePipeline
from src.prediction.load_model import LoadedModelArtifact, get_production_model
from src.prediction.validator import PredictionPayload, PredictionValidator
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AQIPredictor:
    """
    Production-ready Predictor Engine integrating Validation, Feature Engineering,
    Scaling Transformation, Schema Alignment, and Model Inference for Multiple Horizons.
    """

    def __init__(
        self,
        artifact: LoadedModelArtifact | None = None,
        feature_pipeline: PredictionFeaturePipeline | None = None,
        validator: PredictionValidator | None = None,
    ) -> None:
        self.validator = validator or PredictionValidator()
        self._resources = {}
        
        # Load default 24h model for backward compatibility of instance attributes
        if artifact is None:
            artifact = get_production_model(horizon_hours=24)

        self.artifact = artifact
        self.model = self.artifact.model
        self.expected_features = self.artifact.feature_names
        
        # 🐛 FIX: Corrected the PredictionFeaturePipeline instantiation
        # Now passing exactly what it expects: the fitted scaler_engineer instance
        self.feature_pipeline = feature_pipeline or PredictionFeaturePipeline(
            scaler_engineer=self.artifact.scaler_engineer
        )

        # Cache the initialized 24h default resources
        self._resources[24] = {
            "artifact": self.artifact,
            "pipeline": self.feature_pipeline,
            "model": self.model,
            "expected_features": self.expected_features
        }

    def _get_horizon_resources(self, horizon_hours: int) -> dict:
        """Dynamically loads and caches model resources for the requested horizon."""
        if horizon_hours not in [24, 48, 72]:
            raise ValueError(f"Unsupported horizon_hours: {horizon_hours}. Must be 24, 48, or 72.")

        if horizon_hours not in self._resources:
            logger.info("Dynamically loading model and metadata for %sh horizon...", horizon_hours)
            
            artifact = get_production_model(horizon_hours=horizon_hours)
            pipeline = PredictionFeaturePipeline(
                scaler_engineer=artifact.scaler_engineer
            )
            
            self._resources[horizon_hours] = {
                "artifact": artifact,
                "pipeline": pipeline,
                "model": artifact.model,
                "expected_features": artifact.feature_names
            }

        return self._resources[horizon_hours]

    def _align_and_validate(self, df: pd.DataFrame, expected_features: List[str]) -> pd.DataFrame:
        """Validates feature presence and strictly aligns column order with training schema."""
        missing = set(expected_features) - set(df.columns)
        if missing:
            error_msg = f"Input dataset missing required features: {missing}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Enforce exact training column order
        return df[expected_features].copy()

    def predict_single(
        self,
        payload: Dict[str, Any] | PredictionPayload,
        context_df: Optional[pd.DataFrame] = None,
        horizon_hours: int = 24,
    ) -> Dict[str, Any]:
        """
        Executes real-time inference on a single prediction payload for a specific horizon.
        Flow: Payload -> Validation -> Feature Pipeline (FE + Scaler) -> Alignment -> Model.
        """
        logger.info("Executing predict_single inference for %sh horizon...", horizon_hours)
        resources = self._get_horizon_resources(horizon_hours)

        # Step 1-3: Validation, Feature Engineering, and Scaler Transform inside feature pipeline
        features_df = resources["pipeline"].build_features(payload, context_df=context_df)

        # Step 4: Final Feature Alignment Verification
        aligned_df = self._align_and_validate(features_df, resources["expected_features"])

        # Step 5: Execute Model Prediction
        prediction = resources["model"].predict(aligned_df)[0]

        return {
            "predicted_aqi": round(float(prediction), 2),
            "model_version": resources["artifact"].model_version,
        }

    def predict_batch(
        self,
        features_input: Union[List[Dict[str, Any]], List[PredictionPayload], pd.DataFrame],
        context_df: Optional[pd.DataFrame] = None,
        horizon_hours: int = 24,
    ) -> pd.DataFrame:
        """
        Executes batch inference on prediction payloads or feature DataFrames for a specific horizon.
        Flow: Payloads -> Validation -> Feature Pipeline (FE + Scaler) -> Alignment -> Model.
        """
        logger.info("Executing predict_batch inference for %sh horizon...", horizon_hours)
        resources = self._get_horizon_resources(horizon_hours)
        expected_features = resources["expected_features"]

        # Backward compatibility check: if input is a DataFrame with already aligned expected features
        if isinstance(features_input, pd.DataFrame) and set(expected_features).issubset(set(features_input.columns)):
            logger.info("Direct feature DataFrame provided for batch prediction.")
            aligned_df = self._align_and_validate(features_input, expected_features)
            predictions = resources["model"].predict(aligned_df)
            results = features_input.copy()
            results["predicted_aqi"] = np.round(predictions, 2)
            return results

        # Step 1-3: Validation, Feature Engineering, and Scaler Transform
        features_df = resources["pipeline"].build_batch_features(features_input, context_df=context_df)

        # Step 4: Final Feature Alignment Verification
        aligned_df = self._align_and_validate(features_df, expected_features)

        # Step 5: Execute Model Prediction
        predictions = resources["model"].predict(aligned_df)

        results = features_df.copy()
        results["predicted_aqi"] = np.round(predictions, 2)
        return results
"""
Suggested path: src/prediction/predictor.py

SINGLE RESPONSIBILITY: Validate input schema, align feature order, and execute 
batch or single-instance AQI predictions.
"""

from __future__ import annotations

from typing import Any, Dict
import numpy as np
import pandas as pd

from src.prediction.load_model import LoadedModelArtifact, get_production_model
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AQIPredictor:
    """
    Production-ready Predictor Engine.
    """

    def __init__(self, artifact: LoadedModelArtifact | None = None) -> None:
        self.artifact = artifact or get_production_model()
        self.model = self.artifact.model
        self.expected_features = self.artifact.feature_names

    def _align_and_validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validates feature presence and strictly aligns column order."""
        missing = set(self.expected_features) - set(df.columns)
        if missing:
            error_msg = f"Input dataset missing required features: {missing}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Enforce exact training column order
        return df[self.expected_features].copy()

    def predict_batch(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Executes batch inference and attaches predicted AQI column."""
        logger.info("Running batch prediction on %d samples...", len(features_df))
        aligned_df = self._align_and_validate(features_df)
        
        predictions = self.model.predict(aligned_df)
        
        results = features_df.copy()
        results["predicted_aqi"] = np.round(predictions, 2)
        return results

    def predict_single(self, feature_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Executes real-time inference on a single dictionary payload."""
        df = pd.DataFrame([feature_dict])
        aligned_df = self._align_and_validate(df)
        
        prediction = self.model.predict(aligned_df)[0]
        
        return {
            "predicted_aqi": round(float(prediction), 2),
            "model_version": self.artifact.model_version,
        }
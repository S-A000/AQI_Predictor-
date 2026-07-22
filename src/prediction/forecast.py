"""
Suggested path: src/prediction/forecast.py

SINGLE RESPONSIBILITY: Orchestrate multi-step forward horizon time-series forecasts 
using latest loaded features and predictor engine.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from src.prediction.load_features import FeatureLoader
from src.prediction.predictor import AQIPredictor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AQIForecaster:
    """
    Time-Series AQI Forecasting Orchestrator.
    """

    def __init__(
        self,
        predictor: AQIPredictor | None = None,
        feature_loader: FeatureLoader | None = None,
    ) -> None:
        self.predictor = predictor or AQIPredictor()
        self.feature_loader = feature_loader or FeatureLoader()

    def generate_forecast(self, horizon_hours: int = 24) -> List[Dict[str, Any]]:
        """
        Loads latest feature window and generates forward horizon predictions.
        """
        logger.info("Starting AQI forecast generation for %d-hour horizon...", horizon_hours)
        
        # Step 1: Fetch recent features
        features_df = self.feature_loader.load_latest_features(num_rows=horizon_hours)
        
        # Step 2: Run batch inference
        predictions_df = self.predictor.predict_batch(features_df)
        
        # Step 3: Format timestamped forecast horizon output
        base_time = datetime.now(timezone.utc)
        forecast_payload = []

        for idx, pred in enumerate(predictions_df["predicted_aqi"]):
            forecast_time = base_time + timedelta(hours=idx + 1)
            forecast_payload.append({
                "horizon_step": idx + 1,
                "timestamp": forecast_time.isoformat(),
                "predicted_aqi": float(pred),
                "model_version": self.predictor.artifact.model_version,
            })

        logger.info("Successfully generated %d forecast data points.", len(forecast_payload))
        return forecast_payload


if __name__ == "__main__":
    try:
        forecaster = AQIForecaster()
        forecasts = forecaster.generate_forecast(horizon_hours=5)
        print("\n=== Forecast Horizon Output Sample ===")
        for f in forecasts:
            print(f)
    except Exception as err:
        logger.exception("Forecast execution failed: %s", err)
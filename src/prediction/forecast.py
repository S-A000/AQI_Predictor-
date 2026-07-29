from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from src.prediction.load_features import FeatureLoader
from src.prediction.predictor import AQIPredictor
from src.prediction.validator import PredictionPayload
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AQIForecaster:
    """
    Time-Series AQI Forecasting Orchestrator using Direct Multi-Horizon Forecasting.
    Generates forward horizon predictions via AQIPredictor and FeaturePipeline selecting
    the optimal direct model (24h, 48h, 72h) based on requested forecast horizon.
    """

    def __init__(
        self,
        predictor: AQIPredictor | None = None,
        feature_loader: FeatureLoader | None = None,
    ) -> None:
        self.predictor = predictor or AQIPredictor()
        self.feature_loader = feature_loader or FeatureLoader()

    def _resolve_model_horizon(self, horizon_hours: int) -> int:
        if not isinstance(horizon_hours, int) or horizon_hours < 1 or horizon_hours > 72:
            error_msg = (
                f"Invalid forecast horizon '{horizon_hours}'. "
                "Supported horizon_hours must be an integer between 1 and 72."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        if horizon_hours <= 24:
            return 24
        elif horizon_hours <= 48:
            return 48
        else:
            return 72

    def generate_forecast(
        self,
        horizon_hours: int = 24,
        payloads: Optional[List[Dict[str, Any] | PredictionPayload]] = None,
        save_predictions: bool = True,  # 👈 Auto Save Flag Added
    ) -> List[Dict[str, Any]]:
        """
        Generates forward horizon predictions using the appropriate direct multi-horizon model.
        """
        logger.info("Starting AQI forecast generation for %d-hour horizon...", horizon_hours)

        model_horizon = self._resolve_model_horizon(horizon_hours)
        logger.info(
            "Mapped requested horizon of %d hours to direct model target: %dH",
            horizon_hours,
            model_horizon,
        )

        context_df = self.feature_loader.load_latest_features(num_rows=72)

        if payloads is None:
            base_time = datetime.now(timezone.utc)
            payloads = []

            last_row = context_df.iloc[-1] if not context_df.empty else {}

            for h in range(1, horizon_hours + 1):
                forecast_time = base_time + timedelta(hours=h)
                payload_item = {
                    "timestamp": forecast_time.isoformat(),
                    "latitude": float(last_row.get("latitude", 24.8607)),
                    "longitude": float(last_row.get("longitude", 67.0011)),
                    "temperature": float(last_row.get("temperature", 25.0)),
                    "humidity": float(last_row.get("humidity", 60.0)),
                    "pressure": float(last_row.get("pressure", 1013.0)),
                    "wind_speed": float(last_row.get("wind_speed", 3.5)),
                    "wind_deg": float(last_row.get("wind_degree", last_row.get("wind_direction", 180.0))),
                    "cloudiness": float(last_row.get("cloudiness", 10.0)),
                    "visibility": float(last_row.get("visibility", 10000.0)),
                    "pm25": float(last_row.get("pm25")) if pd.notna(last_row.get("pm25")) else None,
                    "pm10": float(last_row.get("pm10")) if pd.notna(last_row.get("pm10")) else None,
                    "no2": float(last_row.get("no2")) if pd.notna(last_row.get("no2")) else None,
                    "so2": float(last_row.get("so2")) if pd.notna(last_row.get("so2")) else None,
                    "co": float(last_row.get("co")) if pd.notna(last_row.get("co")) else None,
                    "o3": float(last_row.get("o3")) if pd.notna(last_row.get("o3")) else None,
                    "city": str(last_row.get("city", "Karachi")),
                }
                payloads.append(payload_item)

        try:
            predictions_df = self.predictor.predict_batch(
                payloads, context_df=context_df, horizon_hours=model_horizon
            )
            horizon_resources = self.predictor._get_horizon_resources(model_horizon)
            model_version = horizon_resources["artifact"].model_version
        except FileNotFoundError as err:
            logger.error("Missing model artifact for horizon %dH: %s", model_horizon, err)
            raise FileNotFoundError(f"Model for horizon {model_horizon}h is not available in registry.") from err
        except Exception as err:
            logger.exception("Failed executing prediction inference for horizon %dH: %s", model_horizon, err)
            raise

        forecast_payload = []
        for idx, row in enumerate(predictions_df.itertuples()):
            pred_aqi = getattr(row, "predicted_aqi")
            ts = getattr(row, "timestamp", None)
            if not isinstance(ts, str):
                ts = (datetime.now(timezone.utc) + timedelta(hours=idx + 1)).isoformat()

            forecast_payload.append({
                "horizon_step": idx + 1,
                "timestamp": ts,
                "predicted_aqi": float(pred_aqi),
                "model_version": model_version,
            })

        logger.info("Successfully generated %d forecast data points using %dH model.", len(forecast_payload), model_horizon)

        # ------------------------------------------------------------------
        # 📂 Exporting Forecast Output to Disk
        # ------------------------------------------------------------------
        if save_predictions and forecast_payload:
            output_dir = os.path.join("data", "predictions")
            os.makedirs(output_dir, exist_ok=True)

            export_df = pd.DataFrame(forecast_payload)
            csv_path = os.path.join(output_dir, f"forecast_{horizon_hours}h.csv")
            parquet_path = os.path.join(output_dir, f"forecast_{horizon_hours}h.parquet")

            export_df.to_csv(csv_path, index=False)
            export_df.to_parquet(parquet_path, index=False)
            logger.info("Saved forecast output to %s and %s", csv_path, parquet_path)

        return forecast_payload


if __name__ == "__main__":
    try:
        forecaster = AQIForecaster()
        for h in [12, 24, 48, 72]:
            forecasts = forecaster.generate_forecast(horizon_hours=h, save_predictions=True)
            print(f"\n=== Forecast Horizon Output Sample ({h} Hours) ===")
            for f in forecasts[:3]:
                print(f)
    except Exception as err:
        logger.exception("Forecast execution failed: %s", err)
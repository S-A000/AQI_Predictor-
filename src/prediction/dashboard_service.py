from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from src.explainability.dashboard_explainer import DashboardExplainer
from src.prediction.load_features import FeatureLoader
from src.prediction.predictor import AQIPredictor
from src.utils.logger import get_logger
from src.prediction.live_data_service import LiveDataService
from src.alerts.aqi_alert_service import AQIAlertService

logger = get_logger(__name__)

SUPPORTED_CITIES = ["Islamabad", "Karachi", "Lahore"]
SUPPORTED_HORIZONS = [24, 48, 72]


class DashboardForecastService:
    """
    Production-grade dashboard service.

    Responsibilities:
    - Load latest raw city-wise AQI/weather context.
    - Generate 24h, 48h, 72h forecasts.
    - Compute AQI category and message.
    - Compute trend.
    - Detect dominant pollutant.
    - Build historical AQI series.
    - Add forecast confidence using model RMSE.
    - Add SHAP/fallback model explainability.
    """

    def __init__(
        self,
        predictor: Optional[AQIPredictor] = None,
        feature_loader: Optional[FeatureLoader] = None,
        explainer: Optional[DashboardExplainer] = None,
    ) -> None:
        self.predictor = predictor or AQIPredictor()
        self.feature_loader = feature_loader or FeatureLoader()
        self.explainer = explainer or DashboardExplainer(predictor=self.predictor)
        self.live_data_service = LiveDataService()
        self.aqi_alert_service = AQIAlertService()
        self.use_live_api = True

    @staticmethod
    def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
        if value is None:
            return default

        try:
            if pd.isna(value):
                return default
            return float(value)
        except Exception:
            return default

    @staticmethod
    def _safe_str(value: Any, default: str = "") -> str:
        if value is None:
            return default

        try:
            if pd.isna(value):
                return default
        except Exception:
            pass

        return str(value)

    @staticmethod
    def _normalise_city(city: str) -> str:
        city_clean = city.strip().lower()

        mapping = {
            "islamabad": "Islamabad",
            "karachi": "Karachi",
            "lahore": "Lahore",
        }

        if city_clean not in mapping:
            raise ValueError(
                f"Unsupported city '{city}'. Supported cities: Islamabad, Karachi, Lahore."
            )

        return mapping[city_clean]

    @staticmethod
    def _aqi_category(aqi: float) -> str:
        if aqi <= 50:
            return "Good"
        if aqi <= 100:
            return "Moderate"
        if aqi <= 150:
            return "Unhealthy for Sensitive Groups"
        if aqi <= 200:
            return "Unhealthy"
        if aqi <= 300:
            return "Very Unhealthy"
        return "Hazardous"

    @staticmethod
    def _aqi_message(aqi: float) -> str:
        if aqi <= 50:
            return "Air quality is good. Normal outdoor activity is acceptable."
        if aqi <= 100:
            return "Air quality is moderate. Sensitive people should monitor symptoms."
        if aqi <= 150:
            return "Air may affect sensitive groups. Reduce long outdoor exposure."
        if aqi <= 200:
            return "Air quality is unhealthy. Limit outdoor activity where possible."
        if aqi <= 300:
            return "Air quality is very unhealthy. Avoid prolonged outdoor exposure."
        return "Air quality is hazardous. Avoid outdoor exposure as much as possible."

    @staticmethod
    def _confidence_from_rmse(rmse: Optional[float]) -> str:
        if rmse is None:
            return "Unknown"
        if rmse <= 15:
            return "High"
        if rmse <= 30:
            return "Medium"
        return "Low"

    def _load_full_context(self) -> pd.DataFrame:
        """
        Load full raw training dataset.

        Important:
        We do not use tail(72) directly here because that may only contain
        one city. We load full context first and later select last 72 rows
        per city.
        """
        df = self.feature_loader.load_latest_features(num_rows=None)

        if df.empty:
            raise ValueError("training_dataset.parquet is empty.")

        if "city" not in df.columns:
            raise ValueError("Column 'city' not found in training_dataset.parquet.")

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.sort_values("timestamp")

        city_filter = [city.lower() for city in SUPPORTED_CITIES]

        df = df[
            df["city"].astype(str).str.lower().isin(city_filter)
        ].copy()

        if df.empty:
            raise ValueError(
                "No records found for Islamabad, Karachi, or Lahore in training_dataset.parquet."
            )

        return df

    def _load_prediction_context(self) -> pd.DataFrame:
        """
        Keep last 72 rows per city for lag/rolling feature generation.
        """
        full_df = self._load_full_context()

        context_df = (
            full_df.groupby(full_df["city"].astype(str).str.lower(), group_keys=False)
            .tail(72)
            .copy()
        )

        if context_df.empty:
            raise ValueError("Failed to build prediction context.")

        return context_df

    def _get_city_df(self, df: pd.DataFrame, city: str) -> pd.DataFrame:
        city = self._normalise_city(city)

        city_df = df[
            df["city"].astype(str).str.lower() == city.lower()
        ].copy()

        if city_df.empty:
            raise ValueError(f"No data found for city: {city}")

        if "timestamp" in city_df.columns:
            city_df = city_df.sort_values("timestamp")

        return city_df

    def _get_latest_city_row(self, context_df: pd.DataFrame, city: str) -> pd.Series:
        city_df = self._get_city_df(context_df, city)
        return city_df.iloc[-1]

    def _build_payload(self, row: pd.Series, city: str) -> Dict[str, Any]:
        """
        Build valid prediction payload from latest city row.
        """
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "latitude": self._safe_float(row.get("latitude"), 0.0),
            "longitude": self._safe_float(row.get("longitude"), 0.0),
            "temperature": self._safe_float(row.get("temperature"), 25.0),
            "humidity": self._safe_float(row.get("humidity"), 60.0),
            "pressure": self._safe_float(row.get("pressure"), 1013.0),
            "wind_speed": self._safe_float(row.get("wind_speed"), 3.5),
            "wind_deg": self._safe_float(
                row.get("wind_degree", row.get("wind_direction")),
                180.0,
            ),
            "cloudiness": self._safe_float(row.get("cloudiness"), 10.0),
            "visibility": self._safe_float(row.get("visibility"), 10000.0),
            "pm25": self._safe_float(row.get("pm25")),
            "pm10": self._safe_float(row.get("pm10")),
            "no2": self._safe_float(row.get("no2")),
            "so2": self._safe_float(row.get("so2")),
            "co": self._safe_float(row.get("co")),
            "o3": self._safe_float(row.get("o3")),
            "aqi": self._safe_float(row.get("aqi")),
            "city": city,
        }

    def _build_trend(self, city_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute AQI trend using latest and previous available AQI.
        """
        if "aqi" not in city_df.columns or len(city_df) < 2:
            return {
                "value": 0.0,
                "direction": "flat",
                "label": "No previous reading available",
            }

        latest_aqi = self._safe_float(city_df.iloc[-1].get("aqi"), 0.0) or 0.0
        previous_aqi = (
            self._safe_float(city_df.iloc[-2].get("aqi"), latest_aqi)
            or latest_aqi
        )

        diff = round(latest_aqi - previous_aqi, 2)

        if diff > 0:
            return {
                "value": diff,
                "direction": "up",
                "label": f"▲ +{diff} since last reading",
            }

        if diff < 0:
            return {
                "value": diff,
                "direction": "down",
                "label": f"▼ {diff} since last reading",
            }

        return {
            "value": 0.0,
            "direction": "flat",
            "label": "No change since last reading",
        }

    def _dominant_pollutant(self, row: pd.Series) -> Dict[str, Any]:
        """
        Simple dominant pollutant detection.

        Note:
        This uses raw pollutant magnitude. A more scientific version would
        compute pollutant sub-indexes according to an AQI standard.
        """
        pollutants = {
            "PM2.5": self._safe_float(row.get("pm25")),
            "PM10": self._safe_float(row.get("pm10")),
            "NO₂": self._safe_float(row.get("no2")),
            "SO₂": self._safe_float(row.get("so2")),
            "CO": self._safe_float(row.get("co")),
            "O₃": self._safe_float(row.get("o3")),
        }

        valid_pollutants = {
            name: value for name, value in pollutants.items() if value is not None
        }

        if not valid_pollutants:
            return {
                "name": "Unknown",
                "value": None,
                "reason": "Pollutant values are unavailable.",
            }

        name, value = max(valid_pollutants.items(), key=lambda item: item[1])

        return {
            "name": name,
            "value": round(float(value), 2),
            "reason": f"{name} is currently the highest available pollutant reading.",
        }

    def _build_history(
        self,
        full_df: pd.DataFrame,
        city: str,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Build AQI history for dashboard charts.
        """
        city_df = self._get_city_df(full_df, city)

        if "timestamp" not in city_df.columns or "aqi" not in city_df.columns:
            return {
                "last_24h": [],
                "last_7d": [],
                "last_30d": [],
            }

        city_df = city_df.dropna(subset=["timestamp"]).copy()
        city_df["timestamp"] = pd.to_datetime(city_df["timestamp"], errors="coerce")
        city_df = city_df.dropna(subset=["timestamp"])
        city_df = city_df.sort_values("timestamp")

        latest_ts = city_df["timestamp"].max()

        def make_series(hours: int) -> List[Dict[str, Any]]:
            start_ts = latest_ts - pd.Timedelta(hours=hours)
            subset = city_df[city_df["timestamp"] >= start_ts].copy()

            return [
                {
                    "timestamp": row["timestamp"].isoformat(),
                    "aqi": self._safe_float(row.get("aqi"), 0.0),
                }
                for _, row in subset.iterrows()
            ]

        return {
            "last_24h": make_series(24),
            "last_7d": make_series(24 * 7),
            "last_30d": make_series(24 * 30),
        }

    def _prediction_confidence(self, horizon: int) -> Dict[str, Any]:
        """
        Use model metadata best_test_rmse to estimate confidence.
        """
        try:
            resources = self.predictor._get_horizon_resources(horizon)
            metadata = resources["artifact"].metadata
            rmse = metadata.get("best_test_rmse")
            rmse_float = self._safe_float(rmse)

            return {
                "rmse": rmse_float,
                "label": self._confidence_from_rmse(rmse_float),
            }

        except Exception as err:
            logger.warning(
                "Could not compute confidence for %sh model: %s",
                horizon,
                err,
            )

            return {
                "rmse": None,
                "label": "Unknown",
            }

    def _build_explainability(
        self,
        payload: Dict[str, Any],
        prediction_context_df: pd.DataFrame,
        horizon_hours: int = 24,
    ) -> Dict[str, Any]:
        """
        Build real explainability using SHAP. If SHAP fails, DashboardExplainer
        falls back to feature importance or coefficients.
        """
        return self.explainer.explain_prediction(
            payload=payload,
            context_df=prediction_context_df,
            horizon_hours=horizon_hours,
            top_k=5,
        )

    def get_city_dashboard_data(
        self,
        city: str,
        full_context_df: Optional[pd.DataFrame] = None,
        prediction_context_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        city = self._normalise_city(city)

        full_context_df = (
            full_context_df
            if full_context_df is not None
            else self._load_full_context()
        )

        prediction_context_df = (
            prediction_context_df
            if prediction_context_df is not None
            else self._load_prediction_context()
        )

        city_full_df = self._get_city_df(full_context_df, city)
        latest_row = self._get_latest_city_row(prediction_context_df, city)

        # Default fallback payload from latest stored row.
        payload = self._build_payload(latest_row, city)
        current_source = "stored_latest_row"

        # Real-time mode:
        # Try to fetch fresh AQICN/OpenWeather payload.
        # If live API fails, fallback to latest stored row.
        if self.use_live_api:
            try:
                live_payload = self.live_data_service.fetch_city_live_data(city)

                if live_payload:
                    payload = live_payload
                    current_source = "live_api"

            except Exception as err:
                logger.warning(
                    "Live data fetch failed for city=%s. Falling back to stored latest row. Error: %s",
                    city,
                    err,
                )

        predictions: Dict[str, float] = {}
        forecast_categories: Dict[str, str] = {}
        forecast_confidence: Dict[str, Dict[str, Any]] = {}
        model_versions: Dict[str, str] = {}

        for horizon in SUPPORTED_HORIZONS:
            prediction_result = self.predictor.predict_single(
                payload=payload,
                context_df=prediction_context_df,
                horizon_hours=horizon,
                strict_alignment=False,
                min_completeness=0.90,
            )

            predicted_aqi = float(prediction_result["predicted_aqi"])
            key = f"{horizon}h"

            # Send an email only when the forecast reaches the Hazardous range.
            # Alert failure must never stop dashboard prediction generation.
            if predicted_aqi > 300:
                try:
                    self.aqi_alert_service.send_hazardous_alert(
                        city=city,
                        horizon=key,
                        predicted_aqi=predicted_aqi,
                    )
                except Exception as err:
                    logger.warning(
                        "Could not send hazardous AQI email alert for "
                        "city=%s horizon=%s: %s",
                        city,
                        key,
                        err,
                    )

            predictions[key] = predicted_aqi
            forecast_categories[key] = self._aqi_category(predicted_aqi)
            forecast_confidence[key] = self._prediction_confidence(horizon)
            model_versions[key] = str(
                prediction_result.get("model_version", "unknown")
            )

        # IMPORTANT:
        # Current AQI/weather/pollutants should come from payload.
        # If live API worked, this is live data.
        # If live API failed, this is latest stored row fallback.
        current_aqi = self._safe_float(payload.get("aqi"), 0.0) or 0.0

        trend = self._build_trend(city_full_df)
        dominant = self._dominant_pollutant(pd.Series(payload))
        history = self._build_history(full_context_df, city)

        explainability = self._build_explainability(
            payload=payload,
            prediction_context_df=prediction_context_df,
            horizon_hours=24,
        )

        return {
            "city": city,
            "last_updated": self._safe_str(
                payload.get("timestamp"),
                datetime.now(timezone.utc).isoformat(),
            ),
            "current": {
                "aqi": round(current_aqi, 2),
                "category": self._aqi_category(current_aqi),
                "message": self._aqi_message(current_aqi),
                "temperature": self._safe_float(payload.get("temperature")),
                "humidity": self._safe_float(payload.get("humidity")),
                "pressure": self._safe_float(payload.get("pressure")),
                "wind_speed": self._safe_float(payload.get("wind_speed")),
                "pm25": self._safe_float(payload.get("pm25")),
                "pm10": self._safe_float(payload.get("pm10")),
                "no2": self._safe_float(payload.get("no2")),
                "so2": self._safe_float(payload.get("so2")),
                "co": self._safe_float(payload.get("co")),
                "o3": self._safe_float(payload.get("o3")),
                "source": current_source,
            },
            "trend": trend,
            "dominant_pollutant": dominant,
            "forecast": predictions,
            "forecast_categories": forecast_categories,
            "forecast_confidence": forecast_confidence,
            "model_versions": model_versions,
            "history": history,
            "explainability": explainability,
        }

    def get_all_cities_dashboard_data(self) -> List[Dict[str, Any]]:
        full_context_df = self._load_full_context()
        prediction_context_df = self._load_prediction_context()

        results: List[Dict[str, Any]] = []

        for city in SUPPORTED_CITIES:
            try:
                results.append(
                    self.get_city_dashboard_data(
                        city=city,
                        full_context_df=full_context_df,
                        prediction_context_df=prediction_context_df,
                    )
                )

            except Exception as err:
                logger.exception(
                    "Dashboard data generation failed for city=%s: %s",
                    city,
                    err,
                )

                results.append(
                    {
                        "city": city,
                        "error": str(err),
                        "current": None,
                        "trend": None,
                        "dominant_pollutant": None,
                        "forecast": None,
                        "forecast_categories": None,
                        "forecast_confidence": None,
                        "model_versions": None,
                        "history": None,
                        "explainability": None,
                    }
                )

        return results
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from src.explainability.dashboard_explainer import (
    DashboardExplainer,
)
from src.feature_store.bigquery_feature_store import (
    BigQueryFeatureStore,
)
from src.prediction.live_data_service import (
    LiveDataService,
)
from src.prediction.load_features import (
    FeatureLoader,
)
from src.prediction.predictor import (
    AQIPredictor,
)
from src.prediction.validator import (
    PredictionValidator,
)
from src.utils.logger import get_logger


logger = get_logger(__name__)


SUPPORTED_CITIES = [
    "Islamabad",
    "Karachi",
    "Lahore",
]

SUPPORTED_HORIZONS = [
    24,
    48,
    72,
]

CONTEXT_ROWS_PER_CITY = 168


PREDICTION_CONTEXT_COLUMNS = (
    "timestamp",
    "city",
    "station_id",
    "latitude",
    "longitude",
    "aqi",
    "temperature",
    "feels_like",
    "humidity",
    "pressure",
    "wind_speed",
    "wind_deg",
    "wind_degree",
    "wind_direction",
    "cloudiness",
    "visibility",
    "pm25",
    "pm10",
    "no2",
    "so2",
    "co",
    "o3",
    "dominant_pollutant",
)


def _env_flag(
    name: str,
    default: bool,
) -> bool:
    value = os.getenv(
        name,
        str(default),
    )

    return (
        value
        .strip()
        .lower()
        in {
            "1",
            "true",
            "yes",
            "on",
        }
    )


class DashboardForecastService:
    """
    Dashboard orchestration service.

    Responsibilities:
    - Load historical and prediction context.
    - Prefer BigQuery for recent prediction context.
    - Optionally enrich current readings from live APIs.
    - Run genuine direct 24h, 48h, and 72h models.
    - Build dashboard history/trend/current condition objects.
    - Generate explainability without breaking forecasts if SHAP fails.

    IMPORTANT:
    Arbitrary environmental defaults are never fabricated.
    """

    def __init__(
        self,
        predictor: Optional[AQIPredictor] = None,
        feature_loader: Optional[FeatureLoader] = None,
        explainer: Optional[DashboardExplainer] = None,
        live_data_service: Optional[LiveDataService] = None,
        bigquery_store: Optional[BigQueryFeatureStore] = None,
    ) -> None:

        self.predictor = (
            predictor
            or AQIPredictor()
        )

        self.feature_loader = (
            feature_loader
            or FeatureLoader()
        )

        self.explainer = (
            explainer
            or DashboardExplainer(
                predictor=self.predictor
            )
        )

        self.use_live_api = _env_flag(
            "DASHBOARD_USE_LIVE_API",
            True,
        )

        self.use_bigquery = _env_flag(
            "DASHBOARD_USE_BIGQUERY",
            True,
        )

        # --------------------------------------------------------------
        # Live API dependency
        # --------------------------------------------------------------

        self.live_data_service = (
            live_data_service
        )

        if (
            self.use_live_api
            and self.live_data_service is None
        ):
            try:
                self.live_data_service = (
                    LiveDataService()
                )

            except Exception as err:
                logger.warning(
                    "Live data service initialization failed. "
                    "Stored data fallback will be used. Error: %s",
                    err,
                )

                self.use_live_api = False

        # --------------------------------------------------------------
        # BigQuery dependency
        # --------------------------------------------------------------

        self.bigquery_store = (
            bigquery_store
        )

        if (
            self.use_bigquery
            and self.bigquery_store is None
        ):
            try:
                self.bigquery_store = (
                    BigQueryFeatureStore()
                )

            except Exception as err:
                logger.warning(
                    "BigQuery initialization failed. "
                    "Local Parquet fallback will be used. Error: %s",
                    err,
                )

                self.use_bigquery = False

    # ==================================================================
    # Safe conversion helpers
    # ==================================================================

    @staticmethod
    def _safe_float(
        value: Any,
        default: Optional[float] = None,
    ) -> Optional[float]:

        if value is None:
            return default

        try:
            if pd.isna(value):
                return default

            return float(value)

        except (
            TypeError,
            ValueError,
        ):
            return default

    @staticmethod
    def _safe_str(
        value: Any,
        default: str = "",
    ) -> str:

        if value is None:
            return default

        try:
            if pd.isna(value):
                return default
        except (
            TypeError,
            ValueError,
        ):
            pass

        return str(value)

    # ==================================================================
    # City normalization
    # ==================================================================

    @staticmethod
    def _normalise_city(
        city: str,
    ) -> str:

        city_clean = (
            city
            .strip()
            .lower()
        )

        mapping = {
            "islamabad": "Islamabad",
            "karachi": "Karachi",
            "lahore": "Lahore",
        }

        if city_clean not in mapping:
            raise ValueError(
                f"Unsupported city '{city}'. "
                "Supported cities: Islamabad, Karachi, Lahore."
            )

        return mapping[
            city_clean
        ]

    # ==================================================================
    # AQI metadata
    # ==================================================================

    @staticmethod
    def _aqi_category(
        aqi: Optional[float],
    ) -> str:

        if aqi is None:
            return "Unavailable"

        if aqi <= 50:
            return "Good"

        if aqi <= 100:
            return "Moderate"

        if aqi <= 150:
            return (
                "Unhealthy for Sensitive Groups"
            )

        if aqi <= 200:
            return "Unhealthy"

        if aqi <= 300:
            return "Very Unhealthy"

        return "Hazardous"

    @staticmethod
    def _aqi_message(
        aqi: Optional[float],
    ) -> str:

        if aqi is None:
            return (
                "Current AQI reading is unavailable."
            )

        if aqi <= 50:
            return (
                "Air quality is good. "
                "Normal outdoor activity is acceptable."
            )

        if aqi <= 100:
            return (
                "Air quality is moderate. "
                "Sensitive people should monitor symptoms."
            )

        if aqi <= 150:
            return (
                "Air may affect sensitive groups. "
                "Reduce long outdoor exposure."
            )

        if aqi <= 200:
            return (
                "Air quality is unhealthy. "
                "Limit outdoor activity where possible."
            )

        if aqi <= 300:
            return (
                "Air quality is very unhealthy. "
                "Avoid prolonged outdoor exposure."
            )

        return (
            "Air quality is hazardous. "
            "Avoid outdoor exposure as much as possible."
        )

    # ==================================================================
    # Context normalization
    # ==================================================================

    @staticmethod
    def _raw_prediction_context(
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        if df.empty:
            raise ValueError(
                "Prediction context is empty."
            )

        columns = [
            column
            for column in PREDICTION_CONTEXT_COLUMNS
            if column in df.columns
        ]

        result = (
            df[
                columns
            ]
            .copy()
        )

        if (
            "timestamp"
            not in result.columns
            or "city"
            not in result.columns
        ):
            raise ValueError(
                "Prediction context must contain city and timestamp."
            )

        result[
            "timestamp"
        ] = pd.to_datetime(
            result["timestamp"],
            utc=True,
            errors="coerce",
        )

        result = result.dropna(
            subset=[
                "timestamp",
                "city",
            ]
        )

        result["timestamp"] = (
            result["timestamp"]
            .dt.floor("h")
        )

        result["city"] = (
            result["city"]
            .astype(str)
            .str.strip()
            .str.title()
        )

        result = (
            result
            .drop_duplicates(
                subset=[
                    "city",
                    "timestamp",
                ],
                keep="last",
            )
            .sort_values(
                [
                    "city",
                    "timestamp",
                ]
            )
            .reset_index(
                drop=True
            )
        )

        return result

    # ==================================================================
    # Historical context
    # ==================================================================

    def _load_full_context(
        self,
    ) -> pd.DataFrame:

        df = (
            self.feature_loader
            .load_latest_features(
                num_rows=None
            )
        )

        if df.empty:
            raise ValueError(
                "training_dataset.parquet is empty."
            )

        if "city" not in df.columns:
            raise ValueError(
                "Column 'city' not found in training_dataset.parquet."
            )

        if "timestamp" in df.columns:
            df = df.copy()

            df["timestamp"] = pd.to_datetime(
                df["timestamp"],
                utc=True,
                errors="coerce",
            )

            df = df.dropna(
                subset=[
                    "timestamp",
                ]
            )

            df = df.sort_values(
                "timestamp"
            )

        allowed = {
            city.lower()
            for city in SUPPORTED_CITIES
        }

        df = df[
            df["city"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(allowed)
        ].copy()

        if df.empty:
            raise ValueError(
                "No supported-city observations found."
            )

        return df

    # ==================================================================
    # Prediction context
    # ==================================================================

    def _load_prediction_context(
        self,
    ) -> pd.DataFrame:

        # --------------------------------------------------------------
        # BigQuery primary path
        # --------------------------------------------------------------

        if (
            self.use_bigquery
            and self.bigquery_store is not None
        ):
            try:
                city_frames: List[
                    pd.DataFrame
                ] = []

                for city in SUPPORTED_CITIES:
                    city_context = (
                        self.bigquery_store
                        .get_latest_context(
                            city=city,
                            rows=CONTEXT_ROWS_PER_CITY,
                        )
                    )

                    if (
                        city_context is not None
                        and not city_context.empty
                    ):
                        city_frames.append(
                            city_context
                        )

                if not city_frames:
                    raise ValueError(
                        "BigQuery returned no prediction context."
                    )

                context_df = pd.concat(
                    city_frames,
                    ignore_index=True,
                )

                context_df = (
                    self._raw_prediction_context(
                        context_df
                    )
                )

                logger.info(
                    "Prediction context loaded from BigQuery | "
                    "cities=%s | rows=%s",
                    SUPPORTED_CITIES,
                    len(context_df),
                )

                return context_df

            except Exception as err:
                logger.warning(
                    "BigQuery context loading failed. "
                    "Falling back to local Parquet. Error: %s",
                    err,
                )

        # --------------------------------------------------------------
        # Local fallback
        # --------------------------------------------------------------

        context_df = (
            self.feature_loader
            .load_latest_features(
                num_rows=CONTEXT_ROWS_PER_CITY,
                per_city=True,
            )
        )

        context_df = (
            self._raw_prediction_context(
                context_df
            )
        )

        logger.info(
            "Prediction context loaded from local Parquet | rows=%s",
            len(context_df),
        )

        return context_df

    # ==================================================================
    # City context
    # ==================================================================

    def _get_city_df(
        self,
        df: pd.DataFrame,
        city: str,
    ) -> pd.DataFrame:

        city = (
            self._normalise_city(
                city
            )
        )

        city_df = df[
            df["city"]
            .astype(str)
            .str.strip()
            .str.lower()
            == city.lower()
        ].copy()

        if city_df.empty:
            raise ValueError(
                f"No data found for city: {city}"
            )

        if "timestamp" in city_df.columns:
            city_df[
                "timestamp"
            ] = pd.to_datetime(
                city_df["timestamp"],
                utc=True,
                errors="coerce",
            )

            city_df = (
                city_df
                .dropna(
                    subset=[
                        "timestamp",
                    ]
                )
                .sort_values(
                    "timestamp"
                )
            )

        return city_df

    # ==================================================================
    # Stored payload construction
    # ==================================================================

    def _build_payload(
        self,
        context_df: pd.DataFrame,
        city: str,
    ) -> Dict[str, Any]:
        """
        Build a prediction payload using real observations.

        Missing environmental fields remain None.

        No:
        - fake temperature
        - fake humidity
        - fake pressure
        - fake wind
        - fake AQI
        - fake timestamp
        """

        city = self._normalise_city(city)
        city_df = self._get_city_df(context_df, city)

        if "timestamp" not in city_df.columns:
            raise ValueError(
                "Stored prediction context is missing timestamp."
            )

        latest_timestamp = city_df["timestamp"].max()

        if pd.isna(latest_timestamp):
            raise ValueError(
                f"No valid observation timestamp for city={city}."
            )

        def latest_value(
            *columns: str,
            required: bool = False,
        ) -> Any:
            for column in columns:
                if column not in city_df.columns:
                    continue

                valid = city_df[column].dropna()

                if not valid.empty:
                    return valid.iloc[-1]

            if required:
                raise ValueError(
                    "No valid real observation for required "
                    f"field(s)={columns} city={city}"
                )

            return None

        def latest_float(
            *columns: str,
            required: bool = False,
        ) -> Optional[float]:
            value = latest_value(
                *columns,
                required=required,
            )

            value = self._safe_float(value)

            if required and value is None:
                raise ValueError(
                    f"Invalid required numeric field={columns} "
                    f"for city={city}"
                )

            return value

        return {
            "timestamp": latest_timestamp.isoformat(),
            "city": city,

            "latitude": latest_float(
                "latitude",
                required=True,
            ),
            "longitude": latest_float(
                "longitude",
                required=True,
            ),

            "temperature": latest_float("temperature"),
            "humidity": latest_float("humidity"),
            "pressure": latest_float("pressure"),
            "wind_speed": latest_float("wind_speed"),

            "wind_degree": latest_float(
                "wind_degree",
                "wind_deg",
                "wind_direction",
            ),

            "cloudiness": latest_float("cloudiness"),
            "visibility": latest_float("visibility"),

            "pm25": latest_float("pm25"),
            "pm10": latest_float("pm10"),
            "no2": latest_float("no2"),
            "so2": latest_float("so2"),
            "co": latest_float("co"),
            "o3": latest_float("o3"),
            "aqi": latest_float("aqi"),

            # Dashboard metadata. This is not a model feature, but it must
            # survive payload construction so the UI can show it.
            "dominant_pollutant": latest_value(
                "dominant_pollutant"
            ),
        }


    # ==================================================================
    # Live payload
    # ==================================================================

    def _merge_live_payload(
        self,
        stored_payload: Dict[str, Any],
        live_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Prefer real live observations while retaining real stored values
        for fields that the provider did not return.

        Dashboard-only metadata such as dominant_pollutant is preserved
        across PredictionValidator canonicalization.
        """

        merged = stored_payload.copy()

        for key, value in live_payload.items():
            if value is None:
                continue

            try:
                if pd.isna(value):
                    continue
            except (TypeError, ValueError):
                pass

            merged[key] = value

        # Support both the normalized internal name and AQICN's raw
        # "dominentpol" spelling if any adapter passes it through.
        dominant_pollutant = (
            merged.get("dominant_pollutant")
            or merged.get("dominentpol")
        )

        validated = PredictionValidator.validate(merged)
        canonical = PredictionValidator.to_feature_dict(validated)

        if dominant_pollutant is not None:
            try:
                is_missing = bool(pd.isna(dominant_pollutant))
            except (TypeError, ValueError):
                is_missing = False

            if not is_missing:
                canonical["dominant_pollutant"] = str(
                    dominant_pollutant
                ).strip()

        return canonical


    # ==================================================================
    # Trend
    # ==================================================================

    def _build_trend(
        self,
        city_df: pd.DataFrame,
    ) -> Dict[str, Any]:

        if "aqi" not in city_df.columns:
            return {
                "value": None,
                "direction": "unknown",
                "label": "AQI history unavailable",
            }

        valid_aqi = (
            pd.to_numeric(
                city_df["aqi"],
                errors="coerce",
            )
            .dropna()
        )

        if len(valid_aqi) < 2:
            return {
                "value": None,
                "direction": "unknown",
                "label": "Insufficient AQI history",
            }

        latest_aqi = float(
            valid_aqi.iloc[-1]
        )

        previous_aqi = float(
            valid_aqi.iloc[-2]
        )

        diff = round(
            latest_aqi
            - previous_aqi,
            2,
        )

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

    # ==================================================================
    # Dominant pollutant
    # ==================================================================

    def _dominant_pollutant(
        self,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Render the source-reported dominant pollutant.

        Raw pollutant concentrations are deliberately NOT compared with
        each other because they use different concentration scales/units.
        """

        raw_name = self._safe_str(
            payload.get("dominant_pollutant")
        ).strip().lower()

        normalized = (
            raw_name
            .replace(".", "")
            .replace("_", "")
            .replace("-", "")
            .replace(" ", "")
        )

        mapping = {
            "pm25": ("PM2.5", "pm25"),
            "pm10": ("PM10", "pm10"),
            "no2": ("NO₂", "no2"),
            "so2": ("SO₂", "so2"),
            "co": ("CO", "co"),
            "o3": ("O₃", "o3"),
        }

        if normalized in mapping:
            display_name, column = mapping[normalized]

            return {
                "name": display_name,
                "value": self._safe_float(
                    payload.get(column)
                ),
                "reason": (
                    "Reported as the dominant pollutant "
                    "by the source observation."
                ),
            }

        return {
            "name": "Unknown",
            "value": None,
            "reason": (
                "The current source did not provide a reliable "
                "dominant-pollutant indicator."
            ),
        }


    # ==================================================================
    # History
    # ==================================================================

    def _build_history(
        self,
        full_df: pd.DataFrame,
        city: str,
    ) -> Dict[
        str,
        List[
            Dict[str, Any]
        ],
    ]:

        city_df = (
            self._get_city_df(
                full_df,
                city,
            )
        )

        if (
            "timestamp"
            not in city_df.columns
            or "aqi"
            not in city_df.columns
        ):
            return {
                "last_24h": [],
                "last_7d": [],
                "last_30d": [],
            }

        city_df = (
            city_df
            .dropna(
                subset=[
                    "timestamp",
                ]
            )
            .copy()
        )

        city_df["timestamp"] = (
            pd.to_datetime(
                city_df["timestamp"],
                utc=True,
                errors="coerce",
            )
        )

        city_df = (
            city_df
            .dropna(
                subset=[
                    "timestamp",
                ]
            )
            .sort_values(
                "timestamp"
            )
        )

        if city_df.empty:
            return {
                "last_24h": [],
                "last_7d": [],
                "last_30d": [],
            }

        latest_ts = (
            city_df[
                "timestamp"
            ]
            .max()
        )

        def make_series(
            hours: int,
        ) -> List[
            Dict[str, Any]
        ]:

            start_ts = (
                latest_ts
                - pd.Timedelta(
                    hours=hours
                )
            )

            subset = city_df[
                city_df["timestamp"]
                >= start_ts
            ]

            return [
                {
                    "timestamp": (
                        row["timestamp"]
                        .isoformat()
                    ),
                    "aqi": self._safe_float(
                        row.get("aqi")
                    ),
                }
                for _, row in subset.iterrows()
            ]

        return {
            "last_24h": make_series(
                24
            ),
            "last_7d": make_series(
                24 * 7
            ),
            "last_30d": make_series(
                24 * 30
            ),
        }

    # ==================================================================
    # Confidence
    # ==================================================================

    def _prediction_confidence(
        self,
        horizon: int,
    ) -> Dict[str, Any]:
        """
        Return the registered model's held-out TEST RMSE.

        The API key remains `forecast_confidence` for backwards
        compatibility, but the value is explicitly labelled as an
        evaluation metric rather than a probabilistic confidence score.
        """

        try:
            resources = self.predictor._get_horizon_resources(
                horizon
            )

            artifact = resources["artifact"]
            metadata = getattr(
                artifact,
                "metadata",
                {},
            ) or {}

            test_metrics = (
                metadata.get("test_metrics")
                or {}
            )

            rmse = self._safe_float(
                test_metrics.get("rmse")
            )

            if rmse is None:
                rmse = self._safe_float(
                    metadata.get("best_test_rmse")
                )

            if rmse is None:
                return {
                    "rmse": None,
                    "label": "Evaluation unavailable",
                }

            return {
                "rmse": round(rmse, 2),
                "label": "Held-out test RMSE",
            }

        except Exception as err:
            logger.warning(
                "Could not load evaluation RMSE for %sh model: %s",
                horizon,
                err,
            )

            return {
                "rmse": None,
                "label": "Evaluation unavailable",
            }


    # ==================================================================
    # Explainability
    # ==================================================================

    def _build_explainability(
        self,
        payload: Dict[str, Any],
        prediction_context_df: pd.DataFrame,
        horizon_hours: int = 24,
    ) -> Dict[str, Any]:

        try:
            return (
                self.explainer
                .explain_prediction(
                    payload=payload,
                    context_df=prediction_context_df,
                    horizon_hours=horizon_hours,
                    top_k=5,
                )
            )

        except Exception as err:
            logger.warning(
                "Explainability failed for horizon=%sh: %s",
                horizon_hours,
                err,
            )

            # Explainability is secondary.
            # Its failure must not destroy a valid forecast response.
            return {
                "method": "unavailable",
                "note": (
                    "Explainability is temporarily unavailable."
                ),
                "top_factors": [],
            }

    # ==================================================================
    # One city
    # ==================================================================

    def get_city_dashboard_data(
        self,
        city: str,
        full_context_df: Optional[pd.DataFrame] = None,
        prediction_context_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:

        city = (
            self._normalise_city(
                city
            )
        )

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

        prediction_context_df = (
            self._raw_prediction_context(
                prediction_context_df
            )
        )

        city_full_df = (
            self._get_city_df(
                full_context_df,
                city,
            )
        )

        # --------------------------------------------------------------
        # Real stored fallback
        # --------------------------------------------------------------

        stored_payload = (
            self._build_payload(
                prediction_context_df,
                city,
            )
        )

        payload = (
            stored_payload.copy()
        )

        current_source = (
            "stored_latest_observation"
        )

        # --------------------------------------------------------------
        # Optional live enrichment
        # --------------------------------------------------------------

        if (
            self.use_live_api
            and self.live_data_service is not None
        ):
            try:
                live_payload = (
                    self.live_data_service
                    .fetch_city_live_data(
                        city
                    )
                )

                if live_payload:
                    payload = (
                        self._merge_live_payload(
                            stored_payload,
                            live_payload,
                        )
                    )

                    current_source = (
                        "live_api_with_stored_fallback"
                    )

            except Exception as err:
                logger.warning(
                    "Live data fetch failed for city=%s. "
                    "Using latest stored observation. Error: %s",
                    city,
                    err,
                )

        # --------------------------------------------------------------
        # Direct models
        # --------------------------------------------------------------

        predictions: Dict[
            str,
            float,
        ] = {}

        forecast_categories: Dict[
            str,
            str,
        ] = {}

        forecast_confidence: Dict[
            str,
            Dict[str, Any],
        ] = {}

        model_versions: Dict[
            str,
            str,
        ] = {}

        for horizon in SUPPORTED_HORIZONS:

            prediction_result = (
                self.predictor
                .predict_single(
                    payload=payload,
                    context_df=prediction_context_df,
                    horizon_hours=horizon,
                    strict_alignment=True,
                    min_completeness=0.90,
                )
            )

            predicted_aqi = float(
                prediction_result[
                    "predicted_aqi"
                ]
            )

            key = (
                f"{horizon}h"
            )

            predictions[
                key
            ] = predicted_aqi

            forecast_categories[
                key
            ] = (
                self._aqi_category(
                    predicted_aqi
                )
            )

            forecast_confidence[
                key
            ] = (
                self._prediction_confidence(
                    horizon
                )
            )

            model_versions[
                key
            ] = str(
                prediction_result.get(
                    "model_version",
                    "unknown",
                )
            )

        # --------------------------------------------------------------
        # Current data
        # --------------------------------------------------------------

        current_aqi = (
            self._safe_float(
                payload.get(
                    "aqi"
                )
            )
        )

        trend = (
            self._build_trend(
                city_full_df
            )
        )

        dominant = (
            self._dominant_pollutant(
                payload
            )
        )

        history = (
            self._build_history(
                full_context_df,
                city,
            )
        )

        explainability = (
            self._build_explainability(
                payload=payload,
                prediction_context_df=prediction_context_df,
                horizon_hours=24,
            )
        )

        return {
            "city": city,

            "last_updated": (
                self._safe_str(
                    payload.get(
                        "timestamp"
                    )
                )
                or None
            ),

            "current": {
                "aqi": current_aqi,

                "category": (
                    self._aqi_category(
                        current_aqi
                    )
                ),

                "message": (
                    self._aqi_message(
                        current_aqi
                    )
                ),

                "source": (
                    current_source
                ),

                "temperature": (
                    self._safe_float(
                        payload.get(
                            "temperature"
                        )
                    )
                ),

                "humidity": (
                    self._safe_float(
                        payload.get(
                            "humidity"
                        )
                    )
                ),

                "pressure": (
                    self._safe_float(
                        payload.get(
                            "pressure"
                        )
                    )
                ),

                "wind_speed": (
                    self._safe_float(
                        payload.get(
                            "wind_speed"
                        )
                    )
                ),

                "pm25": (
                    self._safe_float(
                        payload.get(
                            "pm25"
                        )
                    )
                ),

                "pm10": (
                    self._safe_float(
                        payload.get(
                            "pm10"
                        )
                    )
                ),

                "no2": (
                    self._safe_float(
                        payload.get(
                            "no2"
                        )
                    )
                ),

                "so2": (
                    self._safe_float(
                        payload.get(
                            "so2"
                        )
                    )
                ),

                "co": (
                    self._safe_float(
                        payload.get(
                            "co"
                        )
                    )
                ),

                "o3": (
                    self._safe_float(
                        payload.get(
                            "o3"
                        )
                    )
                ),
            },

            "trend": trend,

            "dominant_pollutant": (
                dominant
            ),

            "forecast": (
                predictions
            ),

            "forecast_categories": (
                forecast_categories
            ),

            "forecast_confidence": (
                forecast_confidence
            ),

            "model_versions": (
                model_versions
            ),

            "history": (
                history
            ),

            "explainability": (
                explainability
            ),
        }

    # ==================================================================
    # All cities
    # ==================================================================

    def get_all_cities_dashboard_data(
        self,
    ) -> List[
        Dict[str, Any]
    ]:

        full_context_df = (
            self._load_full_context()
        )

        prediction_context_df = (
            self._load_prediction_context()
        )

        results: List[
            Dict[str, Any]
        ] = []

        for city in SUPPORTED_CITIES:

            try:
                result = (
                    self.get_city_dashboard_data(
                        city=city,
                        full_context_df=full_context_df,
                        prediction_context_df=prediction_context_df,
                    )
                )

                results.append(
                    result
                )

            except Exception as err:
                logger.exception(
                    "Dashboard data generation failed "
                    "for city=%s: %s",
                    city,
                    err,
                )

                # A single city's failure must not destroy the complete
                # three-city dashboard response.
                results.append(
                    {
                        "city": city,
                        "error": (
                            "Dashboard data is temporarily unavailable."
                        ),
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
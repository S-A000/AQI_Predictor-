"""
forecast.py
===========

Production direct AQI forecasting for 24h, 48h, and 72h horizons.

Responsibilities:
- Load sufficient raw historical context per city.
- Build a prediction payload from the latest REAL city observations.
- Never fabricate environmental readings with arbitrary defaults.
- Route prediction to the correct horizon-specific model bundle.
- Produce only genuine direct 24h / 48h / 72h forecasts.
- Persist forecast outputs to CSV and Parquet.

The default forecast path uses the latest available observation timestamp
as the forecast origin. It does NOT pretend that an old observation was
measured at the current wall-clock time.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.prediction.load_features import FeatureLoader
from src.prediction.predictor import AQIPredictor
from src.prediction.validator import PredictionPayload
from src.utils.logger import get_logger


logger = get_logger(__name__)


# ------------------------------------------------------------------
# Production configuration
# ------------------------------------------------------------------

DIRECT_HORIZONS = (
    24,
    48,
    72,
)

# Longest rolling window used by the feature pipeline.
CONTEXT_ROWS_PER_CITY = 168


RAW_CONTEXT_COLUMNS = (
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


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREDICTIONS_DIR = (
    PROJECT_ROOT
    / "data"
    / "predictions"
)


class AQIForecaster:
    """
    Generate genuine direct AQI forecasts.

    Available model horizons:

        24 hours
        48 hours
        72 hours

    No interpolation or repeated 24h prediction is used to fabricate
    intermediate hourly forecasts.
    """

    def __init__(
        self,
        predictor: AQIPredictor | None = None,
        feature_loader: FeatureLoader | None = None,
    ) -> None:
        self.predictor = (
            predictor
            or AQIPredictor()
        )

        self.feature_loader = (
            feature_loader
            or FeatureLoader()
        )

    # ------------------------------------------------------------------
    # Horizon validation
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_model_horizon(
        horizon_hours: int,
    ) -> int:
        """
        Validate that a genuine direct model exists for the horizon.
        """

        if horizon_hours not in DIRECT_HORIZONS:
            raise ValueError(
                "Direct forecasts are available only at "
                "24h, 48h, and 72h."
            )

        return horizon_hours

    # ------------------------------------------------------------------
    # Raw prediction context
    # ------------------------------------------------------------------

    @staticmethod
    def _raw_context(
        context_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Keep only raw/source columns required by prediction feature
        engineering.

        Precomputed or scaled model features are intentionally excluded.
        """

        if context_df.empty:
            raise ValueError(
                "Prediction context is empty."
            )

        columns = [
            column
            for column in RAW_CONTEXT_COLUMNS
            if column in context_df.columns
        ]

        required_columns = {
            "timestamp",
            "city",
        }

        missing = (
            required_columns
            - set(columns)
        )

        if missing:
            raise ValueError(
                "Prediction context is missing required raw "
                f"column(s): {sorted(missing)}"
            )

        result = (
            context_df[
                columns
            ]
            .copy()
        )

        result["timestamp"] = pd.to_datetime(
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

        result["city"] = (
            result["city"]
            .astype(str)
            .str.strip()
            .str.title()
        )

        result = result[
            result["city"].ne("")
        ].copy()

        result["timestamp"] = (
            result["timestamp"]
            .dt.floor("h")
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

        if result.empty:
            raise ValueError(
                "No valid raw prediction context remained after "
                "normalization."
            )

        return result

    # ------------------------------------------------------------------
    # Default payload construction
    # ------------------------------------------------------------------

    @staticmethod
    def _default_payload(
        context_df: pd.DataFrame,
        city: str = "Karachi",
    ) -> Dict[str, Any]:
        """
        Build a prediction payload using the latest REAL observations
        available for one city.

        Production rules:
        - no arbitrary environmental defaults
        - no float(pd.NA)
        - latest valid historical reading may be used for an optional field
        - required spatial fields must actually exist
        - observation timestamp comes from real context data
        """

        if context_df.empty:
            raise ValueError(
                "Cannot build prediction payload because context is empty."
            )

        if "city" not in context_df.columns:
            raise ValueError(
                "Prediction context is missing required column: city"
            )

        if "timestamp" not in context_df.columns:
            raise ValueError(
                "Prediction context is missing required column: timestamp"
            )

        df = context_df.copy()

        # --------------------------------------------------------------
        # Normalize city
        # --------------------------------------------------------------

        df["city"] = (
            df["city"]
            .astype(str)
            .str.strip()
            .str.title()
        )

        normalized_city = (
            str(city)
            .strip()
            .title()
        )

        if not normalized_city:
            raise ValueError(
                "Prediction city cannot be empty."
            )

        city_df = (
            df[
                df["city"]
                == normalized_city
            ]
            .copy()
        )

        if city_df.empty:
            available_cities = sorted(
                df["city"]
                .dropna()
                .unique()
                .tolist()
            )

            raise ValueError(
                f"No prediction context available for city="
                f"{normalized_city}. "
                f"Available cities={available_cities}"
            )

        # --------------------------------------------------------------
        # Normalize timestamps
        # --------------------------------------------------------------

        city_df["timestamp"] = pd.to_datetime(
            city_df["timestamp"],
            utc=True,
            errors="coerce",
        )

        city_df = city_df.dropna(
            subset=[
                "timestamp",
            ]
        )

        if city_df.empty:
            raise ValueError(
                f"No valid timestamps available for city="
                f"{normalized_city}."
            )

        city_df["timestamp"] = (
            city_df["timestamp"]
            .dt.floor("h")
        )

        city_df = (
            city_df
            .sort_values(
                "timestamp"
            )
            .reset_index(
                drop=True
            )
        )

        # IMPORTANT:
        # Use the latest REAL observation timestamp rather than
        # datetime.now(). Using "now" with old measurements would invent
        # a sensor observation that never actually happened.
        observation_timestamp = (
            city_df["timestamp"]
            .iloc[-1]
        )

        # --------------------------------------------------------------
        # Latest valid value helpers
        # --------------------------------------------------------------

        def latest_value(
            *columns: str,
            required: bool = False,
        ) -> Any:
            """
            Search backwards for the latest valid value among one or more
            possible source-column names.
            """

            for column in columns:
                if column not in city_df.columns:
                    continue

                series = city_df[column]

                valid_mask = (
                    series.notna()
                )

                valid_values = (
                    series.loc[
                        valid_mask
                    ]
                )

                if not valid_values.empty:
                    return (
                        valid_values.iloc[-1]
                    )

            if required:
                raise ValueError(
                    "No valid historical value available for required "
                    f"field(s): {columns} | city={normalized_city}"
                )

            return None

        def latest_float(
            *columns: str,
            required: bool = False,
        ) -> float | None:
            """
            Return the latest valid numeric value without allowing
            pd.NA/None/non-numeric strings to crash float conversion.
            """

            value = latest_value(
                *columns,
                required=required,
            )

            if (
                value is None
                or pd.isna(value)
            ):
                if required:
                    raise ValueError(
                        "Required numeric field has no valid value: "
                        f"{columns}"
                    )

                return None

            numeric_value = pd.to_numeric(
                pd.Series(
                    [value]
                ),
                errors="coerce",
            ).iloc[0]

            if pd.isna(
                numeric_value
            ):
                if required:
                    raise ValueError(
                        "Required numeric field contains an invalid value: "
                        f"fields={columns}, value={value!r}"
                    )

                logger.warning(
                    "Ignoring invalid optional numeric value | "
                    "city=%s | fields=%s | value=%r",
                    normalized_city,
                    columns,
                    value,
                )

                return None

            return float(
                numeric_value
            )

        # --------------------------------------------------------------
        # Construct raw prediction payload
        # --------------------------------------------------------------

        payload: Dict[str, Any] = {
            "timestamp": (
                observation_timestamp
                .isoformat()
            ),

            "city": normalized_city,

            # Spatial coordinates are required because they represent
            # the location being forecast.
            "latitude": latest_float(
                "latitude",
                required=True,
            ),

            "longitude": latest_float(
                "longitude",
                required=True,
            ),

            # Weather values remain optional.
            "temperature": latest_float(
                "temperature",
            ),

            "humidity": latest_float(
                "humidity",
            ),

            "pressure": latest_float(
                "pressure",
            ),

            "wind_speed": latest_float(
                "wind_speed",
            ),

            # Different providers have used different names historically.
            "wind_deg": latest_float(
                "wind_degree",
                "wind_deg",
                "wind_direction",
            ),

            "cloudiness": latest_float(
                "cloudiness",
            ),

            "visibility": latest_float(
                "visibility",
            ),

            # Pollutants remain optional. Missing values should flow into
            # the persisted train-fitted preprocessing/imputation logic,
            # not be replaced with fabricated environmental measurements.
            "pm25": latest_float(
                "pm25",
            ),

            "pm10": latest_float(
                "pm10",
            ),

            "no2": latest_float(
                "no2",
            ),

            "so2": latest_float(
                "so2",
            ),

            "co": latest_float(
                "co",
            ),

            "o3": latest_float(
                "o3",
            ),

            "aqi": latest_float(
                "aqi",
            ),
        }

        logger.info(
            "Built default prediction payload from latest real "
            "context | city=%s | observation_time=%s",
            normalized_city,
            observation_timestamp.isoformat(),
        )

        return payload

    # ------------------------------------------------------------------
    # Forecast generation
    # ------------------------------------------------------------------

    def generate_forecast(
        self,
        horizon_hours: int = 24,
        payloads: Optional[
            List[
                Dict[str, Any]
                | PredictionPayload
            ]
        ] = None,
        save_predictions: bool = True,
        city: str = "Karachi",
    ) -> List[Dict[str, Any]]:
        """
        Generate genuine direct forecast points.

        Examples:

            horizon_hours=24
                -> 24h

            horizon_hours=48
                -> 24h + 48h

            horizon_hours=72
                -> 24h + 48h + 72h

        No intermediate hourly predictions are fabricated.
        """

        self._resolve_model_horizon(
            horizon_hours
        )

        requested_horizons = [
            horizon
            for horizon in DIRECT_HORIZONS
            if horizon <= horizon_hours
        ]

        # --------------------------------------------------------------
        # Load sufficient history PER CITY
        # --------------------------------------------------------------

        context_df = (
            self.feature_loader
            .load_latest_features(
                num_rows=CONTEXT_ROWS_PER_CITY,
                per_city=True,
            )
        )

        prediction_context = (
            self._raw_context(
                context_df
            )
        )

        # --------------------------------------------------------------
        # Prediction payload
        # --------------------------------------------------------------

        if payloads:
            input_payloads = payloads
        else:
            input_payloads = [
                self._default_payload(
                    prediction_context,
                    city=city,
                )
            ]

        forecast_payload: List[
            Dict[str, Any]
        ] = []

        # --------------------------------------------------------------
        # Direct horizon models
        # --------------------------------------------------------------

        for direct_horizon in requested_horizons:
            try:
                predictions_df = (
                    self.predictor
                    .predict_batch(
                        input_payloads,
                        context_df=prediction_context,
                        horizon_hours=direct_horizon,
                        strict_alignment=True,
                    )
                )

                resources = (
                    self.predictor
                    ._get_horizon_resources(
                        direct_horizon
                    )
                )

            except FileNotFoundError as err:
                logger.error(
                    "Missing model artifact for horizon %dH: %s",
                    direct_horizon,
                    err,
                )

                raise FileNotFoundError(
                    f"Model for horizon {direct_horizon}h "
                    "is not available in registry."
                ) from err

            if len(predictions_df) != len(
                input_payloads
            ):
                raise RuntimeError(
                    "Prediction output row count does not match "
                    "input payload count | "
                    f"horizon={direct_horizon} | "
                    f"inputs={len(input_payloads)} | "
                    f"predictions={len(predictions_df)}"
                )

            # ----------------------------------------------------------
            # Build forecast response
            # ----------------------------------------------------------

            for index, row in enumerate(
                predictions_df.itertuples()
            ):
                observation_time = getattr(
                    row,
                    "timestamp",
                    None,
                )

                parsed_time = pd.to_datetime(
                    observation_time,
                    utc=True,
                    errors="coerce",
                )

                # If transformed output no longer exposes timestamp,
                # use the original validated payload timestamp.
                if pd.isna(parsed_time):
                    payload = (
                        input_payloads[
                            index
                        ]
                    )

                    raw_timestamp = (
                        payload.timestamp
                        if isinstance(
                            payload,
                            PredictionPayload,
                        )
                        else payload.get(
                            "timestamp"
                        )
                    )

                    parsed_time = pd.to_datetime(
                        raw_timestamp,
                        utc=True,
                        errors="coerce",
                    )

                # Production code should not silently use "now" if both
                # timestamps are invalid.
                if pd.isna(parsed_time):
                    raise ValueError(
                        "Could not resolve the observation timestamp "
                        f"for forecast horizon={direct_horizon}h."
                    )

                forecast_timestamp = (
                    parsed_time
                    + timedelta(
                        hours=direct_horizon
                    )
                )

                forecast_payload.append(
                    {
                        "horizon_step": direct_horizon,
                        "horizon_hours": direct_horizon,
                        "timestamp": (
                            forecast_timestamp
                            .isoformat()
                        ),
                        "predicted_aqi": float(
                            row.predicted_aqi
                        ),
                        "model_version": (
                            resources[
                                "artifact"
                            ]
                            .model_version
                        ),
                    }
                )

        logger.info(
            "Generated %d genuine direct forecast point(s) "
            "for horizons=%s.",
            len(forecast_payload),
            requested_horizons,
        )

        # --------------------------------------------------------------
        # Persist prediction outputs
        # --------------------------------------------------------------

        if (
            save_predictions
            and forecast_payload
        ):
            PREDICTIONS_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )

            export_df = pd.DataFrame(
                forecast_payload
            )

            csv_path = (
                PREDICTIONS_DIR
                / f"forecast_{horizon_hours}h.csv"
            )

            parquet_path = (
                PREDICTIONS_DIR
                / f"forecast_{horizon_hours}h.parquet"
            )

            export_df.to_csv(
                csv_path,
                index=False,
            )

            export_df.to_parquet(
                parquet_path,
                index=False,
            )

            logger.info(
                "Saved forecast output to %s and %s",
                csv_path,
                parquet_path,
            )

        return forecast_payload


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

if __name__ == "__main__":
    try:
        forecaster = (
            AQIForecaster()
        )

        # One 72h request already produces the genuine:
        #
        #   24h
        #   48h
        #   72h
        #
        # forecasts, so there is no reason to run 24 + 48 + 72 pipelines
        # separately and repeat model inference.
        forecasts = (
            forecaster
            .generate_forecast(
                horizon_hours=72,
                save_predictions=True,
                city="Karachi",
            )
        )

        print(
            "\n=== Direct AQI Forecast Output ==="
        )

        for forecast in forecasts:
            print(
                forecast
            )

    except Exception as err:
        logger.exception(
            "Forecast execution failed: %s",
            err,
        )
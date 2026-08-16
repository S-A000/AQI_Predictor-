"""
src/prediction/validator.py

Production prediction-payload validation.

Responsibilities:
- Parse and validate external prediction payloads.
- Normalize missing values consistently.
- Normalize timestamp and city values.
- Support provider-specific wind-direction aliases.
- Enforce physical/range constraints.
- Allow genuinely missing weather/pollutant observations to flow into
  the persisted train-fitted preprocessing/imputation pipeline.

This module performs validation only.
It does not perform feature engineering or prediction.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)


class PredictionValidationError(Exception):
    """Raised when a prediction payload fails validation."""

    def __init__(
        self,
        message: str,
        errors: Optional[List[str]] = None,
    ) -> None:
        super().__init__(message)
        self.errors = errors or []


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

MISSING_STRING_VALUES = {
    "",
    "-",
    "n/a",
    "na",
    "null",
    "none",
    "nan",
}


def _looks_like_kelvin(value: float) -> bool:
    """
    Heuristic used for legacy/provider payloads that may accidentally
    supply temperature in Kelvin.
    """

    return value > 130.0


def _kelvin_to_celsius(value: float) -> float:
    """Convert Kelvin to Celsius."""

    return round(
        value - 273.15,
        2,
    )


def _is_missing(value: Any) -> bool:
    """
    Safely identify None / pd.NA / NaN values.
    """

    if value is None:
        return True

    try:
        return bool(
            pd.isna(value)
        )
    except (TypeError, ValueError):
        return False


# ------------------------------------------------------------------
# Prediction payload
# ------------------------------------------------------------------

class PredictionPayload(BaseModel):
    """
    Pydantic V2 payload contract for AQI prediction.

    Required:
    - timestamp
    - latitude
    - longitude
    - city

    Optional environmental observations:
    - weather fields
    - pollutant fields
    - current AQI

    Missing optional values are NOT replaced with arbitrary defaults.
    They are later handled by the persisted training-time preprocessing
    object.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="ignore",
        str_strip_whitespace=True,
        populate_by_name=True,
    )

    # ------------------------------------------------------------------
    # Required context
    # ------------------------------------------------------------------

    timestamp: datetime = Field(
        description="Observation timestamp",
    )

    latitude: float = Field(
        ge=-90.0,
        le=90.0,
        description="Latitude",
    )

    longitude: float = Field(
        ge=-180.0,
        le=180.0,
        description="Longitude",
    )

    city: str = Field(
        min_length=1,
        description="Target city name",
    )

    # ------------------------------------------------------------------
    # Optional weather observations
    # ------------------------------------------------------------------

    temperature: Optional[float] = Field(
        default=None,
        ge=-90.0,
        le=60.0,
        description="Temperature in Celsius",
    )

    humidity: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Humidity percentage",
    )

    pressure: Optional[float] = Field(
        default=None,
        ge=800.0,
        le=1100.0,
        description="Pressure in hPa",
    )

    wind_speed: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=120.0,
        description="Wind speed in m/s",
    )

    wind_direction: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=360.0,
        alias="wind_deg",
        description="Wind direction in degrees",
    )

    cloudiness: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Cloudiness percentage",
    )

    visibility: Optional[float] = Field(
        default=None,
        ge=-1.0,
        description="Visibility in meters",
    )

    # ------------------------------------------------------------------
    # Optional pollutant observations
    # ------------------------------------------------------------------

    pm25: Optional[float] = Field(
        default=None,
        ge=0.0,
    )

    pm10: Optional[float] = Field(
        default=None,
        ge=0.0,
    )

    no2: Optional[float] = Field(
        default=None,
        ge=0.0,
    )

    so2: Optional[float] = Field(
        default=None,
        ge=0.0,
    )

    co: Optional[float] = Field(
        default=None,
        ge=0.0,
    )

    o3: Optional[float] = Field(
        default=None,
        ge=0.0,
    )

    aqi: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=500.0,
    )

    # ------------------------------------------------------------------
    # Payload preprocessing
    # ------------------------------------------------------------------

    @model_validator(mode="before")
    @classmethod
    def preprocess_data(
        cls,
        data: Any,
    ) -> Any:
        """
        Normalize provider/user payloads before field validation.

        Handles:
        - missing-value strings
        - pd.NA / NaN
        - Kelvin temperature
        - wind aliases
        """

        if not isinstance(
            data,
            dict,
        ):
            return data

        cleaned: Dict[str, Any] = {}

        for key, value in data.items():

            # ----------------------------------------------------------
            # String cleanup
            # ----------------------------------------------------------

            if isinstance(
                value,
                str,
            ):
                stripped = (
                    value
                    .strip()
                )

                if (
                    stripped.lower()
                    in MISSING_STRING_VALUES
                ):
                    cleaned[key] = None
                else:
                    cleaned[key] = stripped

            # ----------------------------------------------------------
            # Native missing values
            # ----------------------------------------------------------

            elif _is_missing(
                value
            ):
                cleaned[key] = None

            else:
                cleaned[key] = value

        # --------------------------------------------------------------
        # Wind-direction aliases
        # --------------------------------------------------------------

        # Canonical Pydantic field is wind_direction.
        #
        # Input providers may use:
        # - wind_deg
        # - wind_degree
        # - wind_direction

        if (
            cleaned.get("wind_direction")
            is None
        ):
            for alias in (
                "wind_deg",
                "wind_degree",
            ):
                alias_value = cleaned.get(
                    alias
                )

                if alias_value is not None:
                    cleaned[
                        "wind_direction"
                    ] = alias_value
                    break

        # --------------------------------------------------------------
        # Temperature unit compatibility
        # --------------------------------------------------------------

        temperature = cleaned.get(
            "temperature"
        )

        if isinstance(
            temperature,
            (int, float),
        ):
            if _looks_like_kelvin(
                float(temperature)
            ):
                cleaned[
                    "temperature"
                ] = _kelvin_to_celsius(
                    float(temperature)
                )

        return cleaned

    # ------------------------------------------------------------------
    # Field normalization
    # ------------------------------------------------------------------

    @field_validator(
        "city",
        mode="after",
    )
    @classmethod
    def normalize_city(
        cls,
        value: str,
    ) -> str:
        """
        Normalize city names to the canonical production representation.
        """

        normalized = (
            value
            .strip()
            .title()
        )

        if not normalized:
            raise ValueError(
                "city cannot be empty"
            )

        return normalized

    @field_validator(
        "timestamp",
        mode="after",
    )
    @classmethod
    def normalize_timestamp(
        cls,
        value: datetime,
    ) -> datetime:
        """
        Ensure all prediction timestamps are timezone-aware UTC.
        """

        if value.tzinfo is None:
            return value.replace(
                tzinfo=timezone.utc
            )

        return value.astimezone(
            timezone.utc
        )


# ------------------------------------------------------------------
# Prediction validator
# ------------------------------------------------------------------

class PredictionValidator:
    """
    Validation façade for single and batch prediction payloads.

    No feature engineering.
    No preprocessing fitting.
    No prediction logic.
    """

    @staticmethod
    def validate(
        payload: Union[
            Dict[str, Any],
            PredictionPayload,
        ],
    ) -> PredictionPayload:
        """
        Validate one prediction payload.
        """

        if isinstance(
            payload,
            PredictionPayload,
        ):
            return payload

        try:
            return (
                PredictionPayload
                .model_validate(
                    payload
                )
            )

        except ValidationError as exc:
            errors = [
                (
                    f"{'.'.join(str(location) for location in error['loc'])}: "
                    f"{error['msg']}"
                )
                for error in exc.errors()
            ]

            raise PredictionValidationError(
                "Prediction payload validation failed",
                errors=errors,
            ) from exc

    @staticmethod
    def validate_batch(
        payloads: List[
            Union[
                Dict[str, Any],
                PredictionPayload,
            ]
        ],
    ) -> List[PredictionPayload]:
        """
        Validate a prediction batch while retaining the failing row index
        in the error message.
        """

        if not payloads:
            raise PredictionValidationError(
                "Prediction batch cannot be empty."
            )

        validated: List[
            PredictionPayload
        ] = []

        for index, payload in enumerate(
            payloads
        ):
            try:
                validated.append(
                    PredictionValidator.validate(
                        payload
                    )
                )

            except PredictionValidationError as exc:
                indexed_errors = [
                    f"row[{index}].{error}"
                    for error in exc.errors
                ]

                raise PredictionValidationError(
                    f"Prediction batch validation failed at row {index}",
                    errors=indexed_errors,
                ) from exc

        return validated

    @staticmethod
    def to_feature_dict(
    payload: PredictionPayload,
    ) -> Dict[str, Any]:
        
   

        data = payload.model_dump(
            
           by_alias=False,
        )

    # --------------------------------------------------
    # Timestamp
    # --------------------------------------------------

        timestamp = data.get("timestamp")

        if isinstance(timestamp, datetime):
             data["timestamp"] = (
             timestamp
             .astimezone(timezone.utc)
             .isoformat()
        )

    # --------------------------------------------------
    # Canonical wind column
    # --------------------------------------------------

        wind_direction = data.pop(
         "wind_direction",
          None,
        )

    # Training/raw feature schema uses `wind_degree`.
        data["wind_degree"] = wind_direction

    # Never expose provider aliases downstream.
        data.pop(
         "wind_deg",
         None,
        )
        
        return data
        

    
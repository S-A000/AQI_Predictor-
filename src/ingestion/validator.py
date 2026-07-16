"""
validator.py
============
Enterprise-grade validation layer for the AQI Forecasting MLOps project.
Author
------
Syed Abdullah
Description
-----------
Converts raw API JSON into validated Python objects.
This module is the single source of truth for
incoming API schemas.

Additional Capabilities
-----------------------
- Custom exceptions (AQIValidationError, WeatherValidationError)
- Generic ResponseValidator[T]
- Automatic schema versioning
- Data normalization ("-" -> None)
- Unit conversion (Kelvin <-> Celsius)
- Domain rules for impossible values
- Response checksum / integrity checks
- Rich validation reports
- Performance timing
- Metrics integration
- Warning collection (soft vs hard validation)
"""
from __future__ import annotations
import hashlib
import json
import time
from datetime import datetime, timezone
from typing import Any, ClassVar, Generic, TypeVar
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from src.utils.logger import get_logger
logger = get_logger(__name__)

# ============================================================
# Schema Versioning
# ============================================================

CURRENT_SCHEMA_VERSION = "1.0"

SUPPORTED_SCHEMA_VERSIONS: tuple[str, ...] = ("1.0",)


# ============================================================
# Custom Exceptions
# ============================================================

class SchemaValidationError(Exception):
    """
    Base exception for all schema validation failures.

    Carries the offending raw payload and the original pydantic
    error (if any) for easier debugging and observability.
    """

    def __init__(
        self,
        message: str,
        *,
        raw_data: dict | None = None,
        original_error: Exception | None = None,
    ):
        super().__init__(message)
        self.raw_data = raw_data
        self.original_error = original_error
        self.occurred_at = datetime.now(timezone.utc)


class WeatherValidationError(SchemaValidationError):
    """Raised when a WeatherResponse payload fails validation."""


class AQIValidationError(SchemaValidationError):
    """Raised when an AQIResponse payload fails validation."""


# ============================================================
# Normalization Helpers
# ============================================================

def _normalize_value(value: Any) -> Any:
    """
    Recursively normalize raw API values before validation:
        - "-" (a common "no data" placeholder) becomes None.
        - Empty strings become None.
        - Dicts and lists are normalized recursively.
    """

    if isinstance(value, str):
        stripped = value.strip()
        return None if stripped in ("-", "") else value

    if isinstance(value, dict):
        return {key: _normalize_value(val) for key, val in value.items()}

    if isinstance(value, list):
        return [_normalize_value(item) for item in value]

    return value


# ============================================================
# Unit Conversion Helpers
# ============================================================

def kelvin_to_celsius(value: float) -> float:
    """Convert a Kelvin temperature to Celsius."""
    return round(value - 273.15, 2)


def celsius_to_kelvin(value: float) -> float:
    """Convert a Celsius temperature to Kelvin."""
    return round(value + 273.15, 2)


def _looks_like_kelvin(value: float) -> bool:
    """
    Heuristic: OpenWeather in "standard" units returns Kelvin,
    which never dips below ~180K on Earth. Celsius readings are
    virtually never above 130. Anything above that threshold is
    treated as Kelvin and auto-converted.
    """
    return value > 130


# ============================================================
# Checksum / Integrity Helpers
# ============================================================

def compute_payload_checksum(data: dict) -> str:
    """
    Compute a stable SHA256 checksum for a raw API payload, used
    to detect duplicate or tampered responses.
    """

    encoded = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


# ============================================================
# Base Model
# ============================================================

class APIModel(BaseModel):
    """
    Base model used by all response schemas.
    """
    model_config = ConfigDict(
        frozen=True,
        extra="ignore",
        validate_assignment=True,
        str_strip_whitespace=True,
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_before_validation(cls, data: Any) -> Any:
        if isinstance(data, dict):
            return _normalize_value(data)
        return data


# ============================================================
# AQI Models
# ============================================================

class AQICity(APIModel):
    name: str
    geo: list[float]


class AQIData(APIModel):
    aqi: int = Field(ge=0, le=500)
    idx: int
    dominentpol: str
    city: AQICity
    time: dict[str, Any]
    iaqi: dict[str, Any]
    forecast: dict[str, Any] | None = None

    @field_validator("aqi")
    @classmethod
    def validate_aqi(cls, value: int):
        if value > 500:
            raise ValueError("AQI cannot exceed 500")
        return value

    @field_validator("dominentpol")
    @classmethod
    def validate_dominant_pollutant(cls, value: str):
        if not value:
            raise ValueError("dominentpol must not be empty")
        return value


class AQIResponse(APIModel):
    schema_version: str = CURRENT_SCHEMA_VERSION
    status: str
    data: AQIData

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: str):
        if value.lower() != "ok":
            raise ValueError("AQICN returned an error response.")
        return value


# ============================================================
# Weather Models
# ============================================================

class Coordinates(APIModel):
    lon: float
    lat: float


class MainWeather(APIModel):
    temp: float
    feels_like: float
    pressure: int
    humidity: int = Field(ge=0, le=100)
    temp_min: float
    temp_max: float

    @model_validator(mode="before")
    @classmethod
    def _convert_kelvin_if_needed(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        temp = data.get("temp")

        if isinstance(temp, (int, float)) and _looks_like_kelvin(temp):
            for field in ("temp", "feels_like", "temp_min", "temp_max"):
                value = data.get(field)
                if isinstance(value, (int, float)):
                    data[field] = kelvin_to_celsius(value)

        return data

    @field_validator("pressure")
    @classmethod
    def validate_pressure(cls, value: int):
        if not (800 <= value <= 1100):
            logger.warning(
                "Pressure %d hPa is outside the typical sea-level range.",
                value,
            )
        return value


class Wind(APIModel):
    speed: float = Field(ge=0)
    deg: int = Field(ge=0, le=360)
    gust: float | None = None

    @field_validator("speed")
    @classmethod
    def validate_speed(cls, value: float):
        if value > 120:
            logger.warning(
                "Wind speed %.1f m/s exceeds any recorded surface wind speed.",
                value,
            )
        return value


class Clouds(APIModel):
    all: int = Field(ge=0, le=100)


class WeatherDescription(APIModel):
    id: int
    main: str
    description: str
    icon: str


class Sys(APIModel):
    sunrise: int
    sunset: int
    country: str


class WeatherResponse(APIModel):
    schema_version: str = CURRENT_SCHEMA_VERSION
    coord: Coordinates
    weather: list[WeatherDescription]
    main: MainWeather
    wind: Wind
    clouds: Clouds
    visibility: int
    dt: int
    timezone: int
    name: str
    sys: Sys

    @property
    def timestamp(self) -> datetime:
        return datetime.utcfromtimestamp(self.dt)

    @field_validator("visibility")
    @classmethod
    def validate_visibility(cls, value: int):
        if value > 10_000:
            logger.warning(
                "Visibility %d m exceeds OpenWeather's reported max (10000 m).",
                value,
            )
        return value


# ============================================================
# Validation Report
# ============================================================

class ValidationReport(BaseModel):
    """
    Rich, structured record of a single validation attempt —
    suitable for logging, dashboards, or metrics export.
    """

    model_name: str
    success: bool
    schema_version: str = CURRENT_SCHEMA_VERSION
    checksum: str | None = None
    duration_ms: float
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    validated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# ============================================================
# Metrics Integration
# ============================================================

class ValidationMetrics:
    """
    Lightweight in-memory metrics collector for validation runs.
    Swap this out for a real metrics backend (Prometheus,
    StatsD, CloudWatch, ...) in production if needed.
    """

    _counts: ClassVar[dict[str, dict[str, int]]] = {}
    _durations_ms: ClassVar[dict[str, list[float]]] = {}

    @classmethod
    def record(cls, model_name: str, success: bool, duration_ms: float) -> None:
        bucket = cls._counts.setdefault(
            model_name, {"success": 0, "failure": 0}
        )
        bucket["success" if success else "failure"] += 1
        cls._durations_ms.setdefault(model_name, []).append(duration_ms)

    @classmethod
    def get_metrics(cls) -> dict[str, Any]:
        metrics: dict[str, Any] = {}

        for model_name, bucket in cls._counts.items():
            durations = cls._durations_ms.get(model_name, [])
            metrics[model_name] = {
                "success": bucket["success"],
                "failure": bucket["failure"],
                "total": bucket["success"] + bucket["failure"],
                "avg_duration_ms": (
                    sum(durations) / len(durations) if durations else 0.0
                ),
            }

        return metrics

    @classmethod
    def reset(cls) -> None:
        cls._counts.clear()
        cls._durations_ms.clear()


# ============================================================
# Domain Rules (soft checks, collected as warnings)
# ============================================================

def _weather_domain_warnings(weather: WeatherResponse) -> list[str]:
    warnings: list[str] = []

    if not (-90 <= weather.main.temp <= 60):
        warnings.append(
            f"Temperature {weather.main.temp}C is outside Earth's recorded range."
        )

    if weather.main.humidity == 100 and weather.main.temp > 45:
        warnings.append(
            "100% humidity combined with extreme heat is physically unusual."
        )

    if weather.wind.speed > 120:
        warnings.append(
            f"Wind speed {weather.wind.speed} m/s exceeds realistic surface winds."
        )

    return warnings


def _aqi_domain_warnings(aqi: AQIResponse) -> list[str]:
    warnings: list[str] = []

    if aqi.data.aqi >= 400:
        warnings.append(
            f"AQI {aqi.data.aqi} is in the hazardous range; verify station health."
        )

    if not aqi.data.iaqi:
        warnings.append("No pollutant breakdown (iaqi) present in response.")

    return warnings


# ============================================================
# Generic Response Validator
# ============================================================

T = TypeVar("T", bound=APIModel)


class BaseResponseValidator(Generic[T]):
    """
    Generic base class for a validator bound to a specific
    APIModel subtype. Concrete validators set `model` and
    `exception` and optionally override `_domain_warnings`.
    """

    model: ClassVar[type[APIModel]]
    exception: ClassVar[type[SchemaValidationError]] = SchemaValidationError

    @classmethod
    def _domain_warnings(cls, instance: T) -> list[str]:
        return []

    @classmethod
    def validate(cls, data: dict, *, strict: bool = True) -> T:
        """
        Validate raw data into an instance of `cls.model`.

        strict=True (hard validation): raises `cls.exception` on
        any pydantic ValidationError.
        strict=False (soft validation): still raises on pydantic
        errors (structurally invalid data can't be salvaged), but
        callers typically pair this with `validate_with_report`
        to also see non-fatal domain warnings.
        """

        instance, _ = cls.validate_with_report(data, strict=strict)

        if instance is None:
            raise cls.exception(
                f"{cls.model.__name__} failed validation.",
                raw_data=data,
            )

        return instance

    @classmethod
    def validate_with_report(
        cls,
        data: dict,
        *,
        strict: bool = True,
    ) -> tuple[T | None, ValidationReport]:
        """
        Validate raw data and return both the instance (or None
        on failure) and a rich ValidationReport.
        """

        model_name = cls.model.__name__
        checksum = compute_payload_checksum(data)
        start = time.perf_counter()

        errors: list[str] = []
        warnings: list[str] = []
        instance: T | None = None

        try:
            instance = cls.model.model_validate(data)  # type: ignore[assignment]
            warnings = cls._domain_warnings(instance)

        except ValidationError as exc:
            logger.exception(exc)
            errors = [str(error) for error in exc.errors()]

            duration_ms = (time.perf_counter() - start) * 1000
            ValidationMetrics.record(model_name, False, duration_ms)

            report = ValidationReport(
                model_name=model_name,
                success=False,
                checksum=checksum,
                duration_ms=duration_ms,
                errors=errors,
                warnings=warnings,
            )

            if strict:
                raise cls.exception(
                    f"{model_name} failed validation.",
                    raw_data=data,
                    original_error=exc,
                ) from exc

            return None, report

        duration_ms = (time.perf_counter() - start) * 1000
        ValidationMetrics.record(model_name, True, duration_ms)

        for warning in warnings:
            logger.warning("[%s] %s", model_name, warning)

        report = ValidationReport(
            model_name=model_name,
            success=True,
            checksum=checksum,
            duration_ms=duration_ms,
            errors=errors,
            warnings=warnings,
        )

        return instance, report


class WeatherResponseValidator(BaseResponseValidator[WeatherResponse]):
    model = WeatherResponse
    exception = WeatherValidationError

    @classmethod
    def _domain_warnings(cls, instance: WeatherResponse) -> list[str]:
        return _weather_domain_warnings(instance)


class AQIResponseValidator(BaseResponseValidator[AQIResponse]):
    model = AQIResponse
    exception = AQIValidationError

    @classmethod
    def _domain_warnings(cls, instance: AQIResponse) -> list[str]:
        return _aqi_domain_warnings(instance)


# ============================================================
# Validator (backwards-compatible facade)
# ============================================================

class ResponseValidator:
    """
    Enterprise response validator.

    Thin, backwards-compatible facade over
    WeatherResponseValidator / AQIResponseValidator, preserving
    the original `validate_weather` / `validate_aqi` call sites
    while exposing the richer generic API alongside them.
    """

    @staticmethod
    def validate_weather(data: dict) -> WeatherResponse:
        logger.info("Validating weather response.")
        return WeatherResponseValidator.validate(data)

    @staticmethod
    def validate_aqi(data: dict) -> AQIResponse:
        logger.info("Validating AQI response.")
        return AQIResponseValidator.validate(data)

    @staticmethod
    def validate_weather_with_report(
        data: dict,
        *,
        strict: bool = True,
    ) -> tuple[WeatherResponse | None, ValidationReport]:
        return WeatherResponseValidator.validate_with_report(data, strict=strict)

    @staticmethod
    def validate_aqi_with_report(
        data: dict,
        *,
        strict: bool = True,
    ) -> tuple[AQIResponse | None, ValidationReport]:
        return AQIResponseValidator.validate_with_report(data, strict=strict)

    @staticmethod
    def get_metrics() -> dict[str, Any]:
        return ValidationMetrics.get_metrics()
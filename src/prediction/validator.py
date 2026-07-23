"""
src/prediction/validator.py

1. Why it was modified: Extracted validation logic to a dedicated module to adhere to SRP.
2. Architecture before: No formal validation in prediction path; dummy payloads were generated.
3. Architecture after: Pydantic V2 based `PredictionValidator` handles all payload parsing, range, type, and field validation.
4. Exact code: See below.
5. Every changed function: N/A (new file).
6. Every new class: `PredictionPayload`, `PredictionValidator`.
7. Why the change follows SOLID: Single Responsibility Principle - validation is completely decoupled from feature engineering and prediction logic.
8. Why it removes training-serving skew: Ensures the prediction pipeline receives data in the exact expected types and ranges before feature engineering begins.
9. Why it is production-safe: Uses strict Pydantic models to reject malformed data early, preventing silent failures or invalid model inferences.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


class PredictionValidationError(Exception):
    """Exception raised for errors in the prediction payload validation."""
    def __init__(self, message: str, errors: Optional[List[str]] = None):
        super().__init__(message)
        self.errors = errors or []


def _looks_like_kelvin(value: float) -> bool:
    return value > 130.0


def _kelvin_to_celsius(value: float) -> float:
    return round(value - 273.15, 2)


class PredictionPayload(BaseModel):
    """
    Pydantic V2 model for prediction payload validation.
    Enforces strict types, ranges, and required/optional fields.
    """
    model_config = ConfigDict(frozen=True, extra="ignore", str_strip_whitespace=True)

    timestamp: datetime = Field(description="Observation timestamp")
    latitude: float = Field(ge=-90.0, le=90.0, description="Latitude")
    longitude: float = Field(ge=-180.0, le=180.0, description="Longitude")
    temperature: float = Field(ge=-90.0, le=60.0, description="Temperature in Celsius")
    humidity: float = Field(ge=0.0, le=100.0, description="Humidity percentage")
    pressure: float = Field(ge=800.0, le=1100.0, description="Pressure in hPa")
    wind_speed: float = Field(ge=0.0, le=120.0, description="Wind speed in m/s")
    wind_direction: float = Field(ge=0.0, le=360.0, alias="wind_deg", description="Wind direction")
    cloudiness: float = Field(ge=0.0, le=100.0, description="Cloudiness percentage")
    visibility: float = Field(ge=0.0, description="Visibility in meters")
    
    pm25: Optional[float] = Field(default=None, ge=0.0)
    pm10: Optional[float] = Field(default=None, ge=0.0)
    no2: Optional[float] = Field(default=None, ge=0.0)
    so2: Optional[float] = Field(default=None, ge=0.0)
    co: Optional[float] = Field(default=None, ge=0.0)
    o3: Optional[float] = Field(default=None, ge=0.0)
    aqi: Optional[float] = Field(default=None, ge=0.0, le=500.0)
    
    city: str = Field(min_length=1, description="Target city name")

    @model_validator(mode="before")
    @classmethod
    def preprocess_data(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        
        cleaned = {}
        for k, v in data.items():
            if isinstance(v, str):
                v_stripped = v.strip()
                cleaned[k] = None if v_stripped in ("-", "", "N/A", "null", "None") else v_stripped
            else:
                cleaned[k] = v

        if "temperature" in cleaned:
            temp = cleaned["temperature"]
            if isinstance(temp, (int, float)) and _looks_like_kelvin(temp):
                cleaned["temperature"] = _kelvin_to_celsius(temp)
                
        if "wind_direction" not in cleaned and "wind_deg" in cleaned:
            cleaned["wind_direction"] = cleaned["wind_deg"]
            
        return cleaned


class PredictionValidator:
    """
    PredictionValidator owns ALL validation.
    No feature engineering. No prediction.
    """
    @staticmethod
    def validate(payload: Union[Dict[str, Any], PredictionPayload]) -> PredictionPayload:
        if isinstance(payload, PredictionPayload):
            return payload
        try:
            return PredictionPayload.model_validate(payload)
        except ValidationError as e:
            errors = [f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}" for err in e.errors()]
            raise PredictionValidationError("Validation failed", errors=errors) from e

    @staticmethod
    def validate_batch(payloads: List[Union[Dict[str, Any], PredictionPayload]]) -> List[PredictionPayload]:
        validated = []
        for p in payloads:
            validated.append(PredictionValidator.validate(p))
        return validated

    @staticmethod
    def to_feature_dict(payload: PredictionPayload) -> Dict[str, Any]:
        data = payload.model_dump()
        if isinstance(data.get("timestamp"), datetime):
            data["timestamp"] = data["timestamp"].isoformat()
        if "wind_direction" in data:
            data["wind_degree"] = data["wind_direction"]
        return data

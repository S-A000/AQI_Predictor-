from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ------------------------------------------------------------------
# Current conditions
# ------------------------------------------------------------------

class CurrentAQIResponse(BaseModel):
    aqi: Optional[float] = None

    category: str

    message: Optional[str] = None

    source: Optional[str] = None

    temperature: Optional[float] = None
    humidity: Optional[float] = None
    pressure: Optional[float] = None
    wind_speed: Optional[float] = None

    pm25: Optional[float] = None
    pm10: Optional[float] = None
    no2: Optional[float] = None
    so2: Optional[float] = None
    co: Optional[float] = None
    o3: Optional[float] = None


# ------------------------------------------------------------------
# Trend
# ------------------------------------------------------------------

class TrendResponse(BaseModel):
    value: Optional[float] = None
    direction: str
    label: str


# ------------------------------------------------------------------
# Pollutant
# ------------------------------------------------------------------

class DominantPollutantResponse(BaseModel):
    name: str
    value: Optional[float] = None
    reason: Optional[str] = None


# ------------------------------------------------------------------
# Forecast confidence / quality metadata
# ------------------------------------------------------------------

class ForecastConfidenceItem(BaseModel):
    rmse: Optional[float] = None
    label: str


# ------------------------------------------------------------------
# Historical data
# ------------------------------------------------------------------

class HistoryPoint(BaseModel):
    timestamp: str
    aqi: Optional[float] = None


class HistoryResponse(BaseModel):
    last_24h: List[HistoryPoint] = Field(
        default_factory=list
    )

    last_7d: List[HistoryPoint] = Field(
        default_factory=list
    )

    last_30d: List[HistoryPoint] = Field(
        default_factory=list
    )


# ------------------------------------------------------------------
# Explainability
# ------------------------------------------------------------------

class ExplainabilityFactor(BaseModel):
    feature: str
    impact: str

    reason: Optional[str] = None

    raw_feature: Optional[str] = None

    contribution: Optional[float] = None

    direction: Optional[str] = None


class ExplainabilityResponse(BaseModel):
    method: str

    note: Optional[str] = None

    top_factors: List[
        ExplainabilityFactor
    ] = Field(
        default_factory=list
    )


# ------------------------------------------------------------------
# City dashboard
# ------------------------------------------------------------------

class CityDashboardResponse(BaseModel):
    city: str

    last_updated: Optional[str] = None

    current: Optional[
        CurrentAQIResponse
    ] = None

    trend: Optional[
        TrendResponse
    ] = None

    dominant_pollutant: Optional[
        DominantPollutantResponse
    ] = None

    forecast: Optional[
        Dict[str, float]
    ] = None

    forecast_categories: Optional[
        Dict[str, str]
    ] = None

    forecast_confidence: Optional[
        Dict[
            str,
            ForecastConfidenceItem,
        ]
    ] = None

    model_versions: Optional[
        Dict[str, str]
    ] = None

    history: Optional[
        HistoryResponse
    ] = None

    explainability: Optional[
        ExplainabilityResponse
    ] = None

    error: Optional[str] = None


class DashboardResponse(BaseModel):
    cities: List[
        CityDashboardResponse
    ]


# ------------------------------------------------------------------
# Health
# ------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    service: str
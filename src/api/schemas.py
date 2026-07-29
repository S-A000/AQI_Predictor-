from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class CurrentAQIResponse(BaseModel):
    aqi: float
    category: str
    message: Optional[str] = None

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


class TrendResponse(BaseModel):
    value: float
    direction: str
    label: str


class DominantPollutantResponse(BaseModel):
    name: str
    value: Optional[float] = None
    reason: Optional[str] = None


class ForecastConfidenceItem(BaseModel):
    rmse: Optional[float] = None
    label: str


class HistoryPoint(BaseModel):
    timestamp: str
    aqi: Optional[float] = None


class HistoryResponse(BaseModel):
    last_24h: List[HistoryPoint] = Field(default_factory=list)
    last_7d: List[HistoryPoint] = Field(default_factory=list)
    last_30d: List[HistoryPoint] = Field(default_factory=list)


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
    top_factors: List[ExplainabilityFactor] = Field(default_factory=list)


class CityDashboardResponse(BaseModel):
    city: str
    last_updated: Optional[str] = None

    current: Optional[CurrentAQIResponse] = None
    trend: Optional[TrendResponse] = None
    dominant_pollutant: Optional[DominantPollutantResponse] = None

    forecast: Optional[Dict[str, float]] = None
    forecast_categories: Optional[Dict[str, str]] = None
    forecast_confidence: Optional[Dict[str, ForecastConfidenceItem]] = None
    model_versions: Optional[Dict[str, str]] = None

    history: Optional[HistoryResponse] = None
    explainability: Optional[ExplainabilityResponse] = None

    error: Optional[str] = None


class DashboardResponse(BaseModel):
    cities: List[CityDashboardResponse]


class HealthResponse(BaseModel):
    status: str
    service: str
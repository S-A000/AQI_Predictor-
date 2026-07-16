"""
weather_client.py
=================

Enterprise OpenWeather API Client for AQI Forecasting.
Inherits from AdvancedAPIClient.

Features
--------
✔ Pydantic Response Models (Geo, Weather, AQI)
✔ Custom Exceptions
✔ Automatic Coordinate Lookup
✔ Async Batch Requests
✔ Forecast Filtering (Noon, Tomorrow, Next 3 Days)
✔ Timezone Conversion
✔ TTL Caching Decorator
✔ Pandas DataFrame Export
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone
from functools import wraps
from typing import Any, Dict, List, Literal

import pandas as pd
from pydantic import BaseModel, Field, field_validator

from src.configs.settings import settings
from src.ingestion.api_client import AdvancedAPIClient
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ==========================================================
# Custom Exceptions
# ==========================================================
class WeatherAPIError(Exception): """Base OpenWeather API error."""
class WeatherTimeoutError(WeatherAPIError): """Request timed out."""
class WeatherParsingError(WeatherAPIError): """Failed to parse response."""


# ==========================================================
# Caching Decorator
# ==========================================================
def async_ttl_cache(ttl_seconds: int = 3600):
    """In-Memory Async Cache (Default: 1 Hour)."""
    cache: Dict[str, Dict[str, Any]] = {}

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}_{args}_{kwargs}"
            now = time.time()
            if cache_key in cache and (now - cache[cache_key]["timestamp"]) < ttl_seconds:
                logger.debug("Cache HIT: %s", cache_key)
                return cache[cache_key]["data"]
            
            result = await func(*args, **kwargs)
            cache[cache_key] = {"timestamp": now, "data": result}
            return result
        return wrapper
    return decorator


# ==========================================================
# Pydantic Response Models (Validators)
# ==========================================================
class GeoResponse(BaseModel):
    name: str
    lat: float
    lon: float
    country: str
    state: str | None = None

class AirPollutionResponse(BaseModel):
    aqi: int
    co: float
    no2: float
    o3: float
    pm2_5: float
    pm10: float

class WeatherSummary(BaseModel):
    dt: datetime
    temperature: float = Field(alias="temp")
    humidity: int
    pressure: int
    wind_speed: float
    rain_mm: float = 0.0
    clouds_percent: int = Field(alias="clouds")
    timezone_offset: int = 0

    @field_validator("dt", mode="before")
    def parse_unix_time(cls, v):
        if isinstance(v, (int, float)):
            return datetime.fromtimestamp(v, tz=timezone.utc)
        return v
        
    def to_local_time(self) -> datetime:
        """Timezone Conversion using API offset."""
        return self.dt + timedelta(seconds=self.timezone_offset)

class ForecastResponse(BaseModel):
    city_name: str
    summaries: List[WeatherSummary]


# ==========================================================
# Weather Client
# ==========================================================
class WeatherClient(AdvancedAPIClient):
    """Enterprise OpenWeather Client."""

    BASE_URL = "https://api.openweathermap.org"

    def __init__(self):
        super().__init__(
            base_url=self.BASE_URL,
            timeout=settings.api.timeout,
        )
        self.base_params = {"appid": self.api_key, "units": "metric"}

    @property
    def api_key(self) -> str:
        return settings.openweather_api_key_revealed

    # ---------------------------------------------------------
    # Geocoding (With Cache)
    # ---------------------------------------------------------
    @async_ttl_cache(ttl_seconds=86400) # Cache coords for 24 hours
    async def geocode(self, city: str) -> GeoResponse:
        logger.info("Geocoding %s", city)
        params = {**self.base_params, "q": city, "limit": 1}
        
        data = await self.get("/geo/1.0/direct", params=params)
        if not data:
            raise WeatherAPIError(f"Coordinates not found for {city}")
        return GeoResponse.model_validate(data[0])

    # ---------------------------------------------------------
    # Current Weather
    # ---------------------------------------------------------
    async def get_current_weather(self, city: str | None = None) -> WeatherSummary:
        city = city or settings.city
        logger.info("Fetching current weather for %s", city)
        
        params = {**self.base_params, "q": city}
        data = await self.get("/data/2.5/weather", params=params)
        
        try:
            return WeatherSummary(
                dt=data["dt"],
                temp=data["main"]["temp"],
                humidity=data["main"]["humidity"],
                pressure=data["main"]["pressure"],
                wind_speed=data["wind"]["speed"],
                rain=data.get("rain", {}).get("1h", 0.0),
                clouds=data.get("clouds", {}).get("all", 0),
                timezone_offset=data.get("timezone", 0)
            )
        except KeyError as e:
            raise WeatherParsingError(f"Missing key in current weather: {e}")

    # ---------------------------------------------------------
    # 5-Day Forecast
    # ---------------------------------------------------------
    async def get_forecast(self, city: str | None = None) -> ForecastResponse:
        city = city or settings.city
        logger.info("Fetching forecast for %s", city)
        
        params = {**self.base_params, "q": city}
        data = await self.get("/data/2.5/forecast", params=params)
        
        try:
            tz_offset = data.get("city", {}).get("timezone", 0)
            summaries = []
            
            for item in data.get("list", []):
                summaries.append(WeatherSummary(
                    dt=item["dt"],
                    temp=item["main"]["temp"],
                    humidity=item["main"]["humidity"],
                    pressure=item["main"]["pressure"],
                    wind_speed=item["wind"]["speed"],
                    rain=item.get("rain", {}).get("3h", 0.0),
                    clouds=item.get("clouds", {}).get("all", 0),
                    timezone_offset=tz_offset
                ))
            return ForecastResponse(city_name=city, summaries=summaries)
        except KeyError as e:
            raise WeatherParsingError(f"Failed to parse forecast data: {e}")

    # ---------------------------------------------------------
    # Air Pollution (With Auto-Coordinate Lookup)
    # ---------------------------------------------------------
    async def get_air_pollution(self, city: str | None = None) -> AirPollutionResponse:
        """Automatically looks up coordinates before fetching AQI."""
        city = city or settings.city
        geo = await self.geocode(city)
        
        logger.info("Fetching pollution for %s (%s, %s)", city, geo.lat, geo.lon)
        params = {**self.base_params, "lat": geo.lat, "lon": geo.lon}
        
        data = await self.get("/data/2.5/air_pollution", params=params)
        
        try:
            components = data["list"][0]["components"]
            return AirPollutionResponse(
                aqi=data["list"][0]["main"]["aqi"],
                co=components["co"],
                no2=components["no2"],
                o3=components["o3"],
                pm2_5=components["pm2_5"],
                pm10=components["pm10"]
            )
        except (KeyError, IndexError) as e:
            raise WeatherParsingError(f"Failed to parse pollution data: {e}")

    # ---------------------------------------------------------
    # Batch Requests
    # ---------------------------------------------------------
    async def get_batch_forecasts(self, cities: List[str]) -> Dict[str, ForecastResponse | None]:
        """Fetch forecasts for multiple cities concurrently."""
        async def fetch(city: str):
            try:
                return city, await self.get_forecast(city)
            except Exception as e:
                logger.error("Batch error for %s: %s", city, e)
                return city, None

        results = await asyncio.gather(*(fetch(city) for city in cities))
        return dict(results)

    # ---------------------------------------------------------
    # Utilities: Filters & DataFrame Export
    # ---------------------------------------------------------
    @staticmethod
    def filter_forecast(
        forecasts: List[WeatherSummary], 
        filter_type: Literal["noon", "tomorrow", "next_3_days"]
    ) -> List[WeatherSummary]:
        
        now = datetime.now(timezone.utc)
        filtered = []

        for f in forecasts:
            local_dt = f.to_local_time()
            days_ahead = (local_dt.date() - now.date()).days

            if filter_type == "noon" and local_dt.hour == 12:
                filtered.append(f)
            elif filter_type == "tomorrow" and days_ahead == 1:
                filtered.append(f)
            elif filter_type == "next_3_days" and 0 <= days_ahead <= 3:
                filtered.append(f)

        return filtered

    @staticmethod
    def to_dataframe(forecasts: List[WeatherSummary]) -> pd.DataFrame:
        """Normalizes and exports to Pandas DataFrame for EDA pipelines."""
        if not forecasts:
            return pd.DataFrame()
            
        data = []
        for f in forecasts:
            row = f.model_dump(by_alias=True)
            row["local_time"] = f.to_local_time().isoformat()
            data.append(row)
            
        df = pd.DataFrame(data)
        
        # Ensure consistent columns for MLOps
        if "rain_mm" in df.columns:
            df["rain_mm"] = df["rain_mm"].fillna(0.0)
            
        return df
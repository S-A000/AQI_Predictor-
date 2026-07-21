"""
test_weather_client.py
=======================
Tests for src/ingestion/weather_client.py

`WeatherClient.get` (inherited from AdvancedAPIClient) is monkeypatched
directly with an AsyncMock per test, so no real HTTP or event-hook /
circuit-breaker machinery is exercised here (that's covered in
test_api_client.py).
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from src.ingestion.weather_client import (
    AirPollutionResponse,
    ForecastResponse,
    GeoResponse,
    WeatherAPIError,
    WeatherClient,
    WeatherParsingError,
    WeatherSummary,
    async_ttl_cache,
)


@pytest.fixture(autouse=True)
def _mock_settings(monkeypatch):
    from src.configs.settings import settings

    class DummyAPI:
        timeout = 10

    class DummySecret:
        def get_secret_value(self) -> str:
            return "test-key"

    # `openweather_api_key_revealed` and `city` are read-only properties
    # derived from `openweather_api_key` (SecretStr) and `location.city`
    # respectively — patch the underlying source, not the property itself.
    monkeypatch.setattr(settings, "openweather_api_key", DummySecret(), raising=False)
    monkeypatch.setattr(settings, "api", DummyAPI(), raising=False)
    monkeypatch.setattr(settings.location, "city", "Karachi", raising=False)


@pytest.fixture
def weather_client():
    return WeatherClient()


# ============================================================
# async_ttl_cache
# ============================================================

class TestAsyncTtlCache:

    @pytest.mark.asyncio
    async def test_caches_result_within_ttl(self):
        calls = {"n": 0}

        @async_ttl_cache(ttl_seconds=3600)
        async def fetch(city):
            calls["n"] += 1
            return f"data-{city}"

        first = await fetch("Karachi")
        second = await fetch("Karachi")

        assert first == second == "data-Karachi"
        assert calls["n"] == 1

    @pytest.mark.asyncio
    async def test_different_args_bypass_cache(self):
        calls = {"n": 0}

        @async_ttl_cache(ttl_seconds=3600)
        async def fetch(city):
            calls["n"] += 1
            return f"data-{city}"

        await fetch("Karachi")
        await fetch("Lahore")
        assert calls["n"] == 2

    @pytest.mark.asyncio
    async def test_expired_entry_refetches(self, monkeypatch):
        calls = {"n": 0}

        @async_ttl_cache(ttl_seconds=1)
        async def fetch(city):
            calls["n"] += 1
            return calls["n"]

        current_time = {"t": 1000.0}
        monkeypatch.setattr("time.time", lambda: current_time["t"])

        await fetch("Karachi")
        current_time["t"] += 2  # advance past ttl
        await fetch("Karachi")

        assert calls["n"] == 2


# ============================================================
# geocode
# ============================================================

class TestGeocode:

    @pytest.mark.asyncio
    async def test_geocode_returns_geo_response(self, weather_client, monkeypatch):
        raw = [{"name": "Karachi", "lat": 24.86, "lon": 67.0, "country": "PK"}]
        weather_client.get = AsyncMock(return_value=raw)

        geo = await weather_client.geocode("Karachi")
        assert isinstance(geo, GeoResponse)
        assert geo.lat == 24.86

    @pytest.mark.asyncio
    async def test_geocode_raises_when_empty(self, weather_client):
        weather_client.get = AsyncMock(return_value=[])
        with pytest.raises(WeatherAPIError):
            await weather_client.geocode("Nowhereville")


# ============================================================
# get_current_weather
# ============================================================

class TestCurrentWeather:

    @pytest.mark.asyncio
    async def test_parses_valid_payload(self, weather_client):
        payload = {
            "dt": 1_721_000_000,
            "main": {"temp": 30.0, "humidity": 50, "pressure": 1005},
            "wind": {"speed": 3.0},
            "clouds": {"all": 20},
            "timezone": 18000,
        }
        weather_client.get = AsyncMock(return_value=payload)

        summary = await weather_client.get_current_weather("Karachi")
        assert isinstance(summary, WeatherSummary)
        assert summary.temperature == 30.0
        assert summary.humidity == 50

    @pytest.mark.asyncio
    async def test_missing_key_raises_parsing_error(self, weather_client):
        weather_client.get = AsyncMock(return_value={"dt": 123})  # missing "main" etc.
        with pytest.raises(WeatherParsingError):
            await weather_client.get_current_weather("Karachi")

    @pytest.mark.asyncio
    async def test_defaults_to_settings_city(self, weather_client):
        payload = {
            "dt": 1_721_000_000,
            "main": {"temp": 30.0, "humidity": 50, "pressure": 1005},
            "wind": {"speed": 3.0},
            "clouds": {"all": 20},
            "timezone": 0,
        }
        weather_client.get = AsyncMock(return_value=payload)
        await weather_client.get_current_weather()
        args, kwargs = weather_client.get.call_args
        assert kwargs["params"]["q"] == "Karachi"


# ============================================================
# get_forecast
# ============================================================

class TestForecast:

    @pytest.mark.asyncio
    async def test_parses_forecast_list(self, weather_client):
        payload = {
            "city": {"timezone": 0},
            "list": [
                {
                    "dt": 1_721_000_000,
                    "main": {"temp": 28.0, "humidity": 40, "pressure": 1000},
                    "wind": {"speed": 2.0},
                    "clouds": {"all": 5},
                },
                {
                    "dt": 1_721_003_600,
                    "main": {"temp": 29.0, "humidity": 42, "pressure": 1001},
                    "wind": {"speed": 2.5},
                    "clouds": {"all": 8},
                },
            ],
        }
        weather_client.get = AsyncMock(return_value=payload)

        forecast = await weather_client.get_forecast("Karachi")
        assert isinstance(forecast, ForecastResponse)
        assert len(forecast.summaries) == 2

    @pytest.mark.asyncio
    async def test_malformed_forecast_raises_parsing_error(self, weather_client):
        weather_client.get = AsyncMock(return_value={"list": [{"dt": 1}]})  # missing "main"
        with pytest.raises(WeatherParsingError):
            await weather_client.get_forecast("Karachi")


# ============================================================
# get_air_pollution
# ============================================================

class TestAirPollution:

    @pytest.mark.asyncio
    async def test_parses_pollution_response(self, weather_client, monkeypatch):
        geo = GeoResponse(name="Karachi", lat=24.86, lon=67.0, country="PK")
        weather_client.geocode = AsyncMock(return_value=geo)

        payload = {
            "list": [
                {
                    "main": {"aqi": 3},
                    "components": {"co": 200.0, "no2": 15.0, "o3": 30.0, "pm2_5": 25.0, "pm10": 40.0},
                }
            ]
        }
        weather_client.get = AsyncMock(return_value=payload)

        result = await weather_client.get_air_pollution("Karachi")
        assert isinstance(result, AirPollutionResponse)
        assert result.aqi == 3
        assert result.pm2_5 == 25.0

    @pytest.mark.asyncio
    async def test_missing_components_raises(self, weather_client):
        weather_client.geocode = AsyncMock(
            return_value=GeoResponse(name="Karachi", lat=24.86, lon=67.0, country="PK")
        )
        weather_client.get = AsyncMock(return_value={"list": []})
        with pytest.raises(WeatherParsingError):
            await weather_client.get_air_pollution("Karachi")


# ============================================================
# Batch requests
# ============================================================

class TestBatchForecasts:

    @pytest.mark.asyncio
    async def test_batch_returns_dict_keyed_by_city(self, weather_client):
        forecast = ForecastResponse(city_name="Karachi", summaries=[])
        weather_client.get_forecast = AsyncMock(return_value=forecast)

        result = await weather_client.get_batch_forecasts(["Karachi", "Lahore"])
        assert set(result.keys()) == {"Karachi", "Lahore"}

    @pytest.mark.asyncio
    async def test_batch_handles_individual_failures(self, weather_client):
        async def flaky(city):
            if city == "Bad City":
                raise WeatherAPIError("boom")
            return ForecastResponse(city_name=city, summaries=[])

        weather_client.get_forecast = flaky
        result = await weather_client.get_batch_forecasts(["Karachi", "Bad City"])
        assert result["Karachi"] is not None
        assert result["Bad City"] is None


# ============================================================
# filter_forecast / to_dataframe
# ============================================================

class TestFilterAndDataframe:

    def _summary(self, dt: datetime) -> WeatherSummary:
        return WeatherSummary(
            dt=dt, temp=25.0, humidity=50, pressure=1000, wind_speed=1.0, clouds=10, timezone_offset=0
        )

    def test_filter_noon(self):
        noon = self._summary(datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc))
        not_noon = self._summary(datetime(2026, 7, 16, 15, 0, tzinfo=timezone.utc))
        result = WeatherClient.filter_forecast([noon, not_noon], "noon")
        assert result == [noon]

    def test_to_dataframe_empty_list(self):
        pd = pytest.importorskip("pandas")
        df = WeatherClient.to_dataframe([])
        assert df.empty

    def test_to_dataframe_fills_missing_rain(self):
        pytest.importorskip("pandas")
        summary = self._summary(datetime.now(timezone.utc))
        df = WeatherClient.to_dataframe([summary])
        assert "rain_mm" in df.columns
        assert not df["rain_mm"].isna().any()

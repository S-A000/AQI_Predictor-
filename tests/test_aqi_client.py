"""
test_aqi_client.py
===================
Tests for src/ingestion/aqi_client.py (AQIClient / AQICN).

`AQIClient` now inherits from the async `AdvancedAPIClient` (the source
of truth in api_client.py), so `get_current_aqi` is a coroutine that
awaits `self.get(...)`. These tests mock `self.get` with an
`AsyncMock` and drive everything through `pytest.mark.asyncio`.
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.ingestion.aqi_client import AQIClient


@pytest.fixture(autouse=True)
def _mock_settings(monkeypatch):
    from src.configs.settings import settings

    class DummySecret:
        def get_secret_value(self) -> str:
            return "test-token"

    # `aqicn_api_key_revealed` and `city` are read-only properties derived
    # from `aqicn_api_key` (SecretStr) and `location.city` respectively —
    # patch the underlying source, not the property itself.
    monkeypatch.setattr(settings, "aqicn_api_key", DummySecret(), raising=False)
    monkeypatch.setattr(settings.location, "city", "Karachi", raising=False)


@pytest.fixture
def aqi_client():
    client = AQIClient()
    yield client


class TestGetCurrentAqi:

    @pytest.mark.asyncio
    async def test_calls_get_with_correct_endpoint(self, aqi_client):
        aqi_client.get = AsyncMock(return_value={"status": "ok"})
        await aqi_client.get_current_aqi("Lahore")

        _, kwargs = aqi_client.get.call_args
        assert kwargs["endpoint"] == "/feed/Lahore/"

    @pytest.mark.asyncio
    async def test_calls_get_with_token_param(self, aqi_client):
        aqi_client.get = AsyncMock(return_value={"status": "ok"})
        await aqi_client.get_current_aqi("Lahore")

        _, kwargs = aqi_client.get.call_args
        assert kwargs["params"] == {"token": "test-token"}

    @pytest.mark.asyncio
    async def test_defaults_to_settings_city_when_none_given(self, aqi_client):
        aqi_client.get = AsyncMock(return_value={"status": "ok"})
        await aqi_client.get_current_aqi()

        _, kwargs = aqi_client.get.call_args
        assert kwargs["endpoint"] == "/feed/Karachi/"

    @pytest.mark.asyncio
    async def test_returns_whatever_get_returns(self, aqi_client):
        expected = {"status": "ok", "data": {"aqi": 100}}
        aqi_client.get = AsyncMock(return_value=expected)

        result = await aqi_client.get_current_aqi("Karachi")
        assert result == expected

    @pytest.mark.asyncio
    async def test_get_current_aqi_is_a_coroutine_function(self, aqi_client):
        import inspect
        assert inspect.iscoroutinefunction(aqi_client.get_current_aqi)

    def test_base_url_is_waqi(self):
        assert AQIClient.BASE_URL == "https://api.waqi.info"

    def test_inherits_from_advanced_api_client(self):
        from src.ingestion.api_client import AdvancedAPIClient
        assert issubclass(AQIClient, AdvancedAPIClient)
"""
test_api_client.py
===================
Tests for src/ingestion/api_client.py

Network calls are never made for real: ``AdvancedAPIClient.client.request``
is monkeypatched with an ``AsyncMock`` per test. ``asyncio.sleep`` is
patched to a no-op everywhere in this module so retry/backoff logic
doesn't slow the suite down.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from pydantic import BaseModel

from src.ingestion.api_client import (
    APIClientError,
    APIKeyAuth,
    AdvancedAPIClient,
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitState,
    OAuth2Bearer,
    RateLimiter,
)


@pytest.fixture(autouse=True)
def _fast_sleep(monkeypatch):
    """Never actually wait during tests (rate limiter / retry backoff)."""
    async def _noop_sleep(*_args, **_kwargs):
        return None
    monkeypatch.setattr(asyncio, "sleep", _noop_sleep)


class DummyModel(BaseModel):
    ok: bool


def _make_response(status_code: int = 200, json_body: dict | None = None) -> httpx.Response:
    request = httpx.Request("GET", "https://example.com/thing")
    response = httpx.Response(status_code, json=json_body or {"ok": True}, request=request)
    return response


# ============================================================
# Auth strategies
# ============================================================

class TestAuthStrategies:

    def test_api_key_auth_sets_header(self):
        auth = APIKeyAuth("secret-key")
        request = httpx.Request("GET", "https://example.com")
        flow = auth.auth_flow(request)
        sent = next(flow)
        assert sent.headers["X-Api-Key"] == "secret-key"

    def test_api_key_auth_custom_header_name(self):
        auth = APIKeyAuth("secret-key", header_name="X-Custom")
        request = httpx.Request("GET", "https://example.com")
        sent = next(auth.auth_flow(request))
        assert sent.headers["X-Custom"] == "secret-key"

    def test_oauth2_bearer_sets_authorization_header(self):
        auth = OAuth2Bearer("tok123")
        request = httpx.Request("GET", "https://example.com")
        sent = next(auth.auth_flow(request))
        assert sent.headers["Authorization"] == "Bearer tok123"


# ============================================================
# RateLimiter
# ============================================================

class TestRateLimiter:

    @pytest.mark.asyncio
    async def test_acquire_consumes_a_token(self):
        limiter = RateLimiter(tokens_per_second=5)
        before = limiter.tokens
        await limiter.acquire()
        assert limiter.tokens <= before

    @pytest.mark.asyncio
    async def test_acquire_sleeps_when_bucket_empty(self, monkeypatch):
        limiter = RateLimiter(tokens_per_second=1)
        limiter.tokens = 0

        sleep_calls = []

        async def _track_sleep(seconds):
            sleep_calls.append(seconds)

        monkeypatch.setattr(asyncio, "sleep", _track_sleep)
        await limiter.acquire()
        assert len(sleep_calls) == 1
        assert sleep_calls[0] >= 0


# ============================================================
# CircuitBreaker
# ============================================================

class TestCircuitBreaker:

    def test_starts_closed(self):
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED
        cb.check_state()  # should not raise

    def test_opens_after_threshold_failures(self):
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_open_circuit_raises_on_check(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=1000)
        cb.record_failure()
        with pytest.raises(CircuitBreakerOpenError):
            cb.check_state()

    def test_success_resets_failures(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb.failures == 0
        assert cb.state == CircuitState.CLOSED

    def test_transitions_to_half_open_after_recovery_timeout(self, monkeypatch):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        cb.check_state()
        assert cb.state == CircuitState.HALF_OPEN


# ============================================================
# AdvancedAPIClient
# ============================================================

@pytest.fixture
def client():
    c = AdvancedAPIClient(base_url="https://example.com", rate_limit_tps=1000.0)
    return c


class TestAdvancedAPIClientRequest:

    @pytest.mark.asyncio
    async def test_successful_get_returns_response(self, client):
        client.client.request = AsyncMock(return_value=_make_response(200))
        response = await client.get("/thing")
        assert response.status_code == 200
        assert client.circuit_breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_get_with_response_model_parses(self, client):
        client.client.request = AsyncMock(return_value=_make_response(200, {"ok": True}))
        result = await client.get("/thing", response_model=DummyModel)
        assert isinstance(result, DummyModel)
        assert result.ok is True

    @pytest.mark.asyncio
    async def test_post_passes_json_body(self, client):
        mock_request = AsyncMock(return_value=_make_response(200))
        client.client.request = mock_request
        await client.post("/thing", json={"a": 1})
        _, kwargs = mock_request.call_args
        assert kwargs.get("json") == {"a": 1}

    @pytest.mark.asyncio
    async def test_failure_raises_api_client_error_and_records_failure(self, client):
        request_obj = httpx.Request("GET", "https://example.com/thing")
        error_response = httpx.Response(500, request=request_obj)

        async def _raise(*_a, **_kw):
            return error_response

        client.client.request = _raise
        # raise_for_status will raise HTTPStatusError, which the retry logic
        # will retry 3x then re-raise as APIClientError.
        with pytest.raises(APIClientError):
            await client.get("/thing")
        assert client.circuit_breaker.failures >= 1

    @pytest.mark.asyncio
    async def test_open_circuit_short_circuits_request(self, client):
        client.circuit_breaker.failure_threshold = 1
        client.circuit_breaker.recovery_timeout = 1000
        client.circuit_breaker.record_failure()

        with pytest.raises(CircuitBreakerOpenError):
            await client.get("/thing")


class TestPagination:

    @pytest.mark.asyncio
    async def test_paginate_get_yields_until_empty(self, client, monkeypatch):
        pages = [DummyModel(ok=True), DummyModel(ok=True), None]

        async def fake_get(endpoint, response_model=None, **kwargs):
            return pages.pop(0)

        monkeypatch.setattr(client, "get", fake_get)

        results = []
        async for item in client.paginate_get("/things", DummyModel):
            results.append(item)

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_paginate_get_stops_on_404(self, client, monkeypatch):
        call_count = {"n": 0}

        async def fake_get(endpoint, response_model=None, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return DummyModel(ok=True)
            request_obj = httpx.Request("GET", endpoint)
            response = httpx.Response(404, request=request_obj)
            raise httpx.HTTPStatusError("not found", request=request_obj, response=response)

        monkeypatch.setattr(client, "get", fake_get)

        results = [item async for item in client.paginate_get("/things", DummyModel)]
        assert len(results) == 1


class TestLifecycle:

    @pytest.mark.asyncio
    async def test_close_closes_underlying_client(self, client):
        client.client.aclose = AsyncMock()
        await client.close()
        client.client.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_context_manager_closes_on_exit(self):
        c = AdvancedAPIClient(base_url="https://example.com")
        c.client.aclose = AsyncMock()
        async with c as ctx:
            assert ctx is c
        c.client.aclose.assert_awaited_once()

"""
api_client.py
=============

Ultimate Enterprise-grade HTTP Client for AQI Forecasting MLOps.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Dict, Generic, Type, TypeVar

import httpx
import hishel
from hishel.httpx import AsyncCacheClient
from pydantic import BaseModel
from prometheus_client import Counter, Histogram
from opentelemetry import trace
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.utils.logger import get_logger, request_id_var

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)

T = TypeVar("T", bound=BaseModel)

HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total_aqirefactor",
    "Total HTTP requests made",
    ["method", "endpoint", "status_code"]
)
HTTP_REQUEST_DURATION = Histogram(
    "http_request_duration_seconds_aqirefactor",
    "HTTP request latency",
    ["method", "endpoint"]
)


class APIClientError(Exception): """Base exception."""
class CircuitBreakerOpenError(APIClientError): """Circuit breaker is OPEN."""
class RateLimitExceededError(APIClientError): """Rate limit exceeded."""
class APIResponseError(APIClientError): """Invalid response or parse error."""


class APIKeyAuth(httpx.Auth):
    def __init__(self, api_key: str, header_name: str = "X-Api-Key"):
        self.api_key = api_key
        self.header_name = header_name

    def auth_flow(self, request: httpx.Request):
        request.headers[self.header_name] = self.api_key
        yield request


class OAuth2Bearer(httpx.Auth):
    def __init__(self, token: str):
        self.token = token

    def auth_flow(self, request: httpx.Request):
        request.headers["Authorization"] = f"Bearer {self.token}"
        yield request


class RateLimiter:
    def __init__(self, tokens_per_second: float):
        self.capacity = tokens_per_second
        self.tokens = tokens_per_second
        self.last_update = time.monotonic()
        self.lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            self.tokens = min(self.capacity, self.tokens + elapsed * self.capacity)
            self.last_update = now
            if self.tokens < 1:
                wait_time = (1 - self.tokens) / self.capacity
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= 1


class CircuitState(Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 30):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.state = CircuitState.CLOSED
        self.last_failure_time = 0.0

    def record_success(self):
        self.failures = 0
        self.state = CircuitState.CLOSED

    def record_failure(self):
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self.state = CircuitState.OPEN
            self.last_failure_time = time.monotonic()

    def check_state(self):
        if self.state == CircuitState.OPEN:
            if time.monotonic() - self.last_failure_time >= self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
            else:
                raise CircuitBreakerOpenError("Circuit breaker is currently OPEN.")


class AdvancedAPIClient:
    def __init__(
        self,
        base_url: str,
        auth: httpx.Auth | None = None,
        proxies: Dict[str, str] | None = None,
        timeout: int = 30,
        rate_limit_tps: float = 10.0,
    ):
        self.base_url = base_url
        self.rate_limiter = RateLimiter(tokens_per_second=rate_limit_tps)
        self.circuit_breaker = CircuitBreaker()

        # hishel >=1.0 caches via a client subclass (RFC 9111: ETag,
        # Cache-Control) rather than a transport wrapper — AsyncCacheClient
        # is a drop-in subclass of httpx.AsyncClient, so it accepts the
        # exact same constructor kwargs.
        self.client = AsyncCacheClient(
            base_url=self.base_url,
            auth=auth,
            timeout=timeout,
            transport=httpx.AsyncHTTPTransport(retries=3),
            event_hooks={
                "request": [self._on_request],
                "response": [self._on_response]
            }
        )

    async def _on_request(self, request: httpx.Request):
        req_id = request_id_var.get(uuid.uuid4().hex)
        request.headers["X-Request-ID"] = req_id
        request.ext["start_time"] = time.perf_counter()

    async def _on_response(self, response: httpx.Response):
        start_time = response.request.ext.get("start_time", time.perf_counter())
        latency = time.perf_counter() - start_time
        endpoint = response.request.url.path
        HTTP_REQUEST_DURATION.labels(method=response.request.method, endpoint=endpoint).observe(latency)
        HTTP_REQUESTS_TOTAL.labels(
            method=response.request.method,
            endpoint=endpoint,
            status_code=response.status_code
        ).inc()

    async def request(
        self,
        method: str,
        endpoint: str,
        response_model: Type[T] | None = None,
        **kwargs
    ) -> httpx.Response | T:

        self.circuit_breaker.check_state()
        await self.rate_limiter.acquire()

        retry_strategy = AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
            reraise=True
        )

        with tracer.start_as_current_span(f"{method} {endpoint}") as span:
            try:
                async for attempt in retry_strategy:
                    with attempt:
                        response = await self.client.request(method, endpoint, **kwargs)
                        response.raise_for_status()
                        self.circuit_breaker.record_success()
                        if response_model:
                            return response_model.model_validate_json(response.read())
                        return response
            except Exception as e:
                self.circuit_breaker.record_failure()
                span.record_exception(e)
                raise APIClientError(str(e)) from e

    async def get(self, endpoint: str, response_model: Type[T] | None = None, **kwargs) -> Any:
        return await self.request("GET", endpoint, response_model=response_model, **kwargs)

    async def post(self, endpoint: str, json: dict | None = None, response_model: Type[T] | None = None, **kwargs) -> Any:
        return await self.request("POST", endpoint, json=json, response_model=response_model, **kwargs)

    async def paginate_get(
        self,
        endpoint: str,
        response_model: Type[T],
        page_param: str = "page",
        start_page: int = 1
    ) -> AsyncGenerator[T, None]:
        current_page = start_page
        while True:
            params = {page_param: current_page}
            try:
                data = await self.get(endpoint, params=params, response_model=response_model)
                if not data:
                    break
                yield data
                current_page += 1
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    break
                raise

    async def close(self) -> None:
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
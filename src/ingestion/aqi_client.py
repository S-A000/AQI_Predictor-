"""
aqi_client.py
=============
Client for fetching Air Quality Index (AQI) data from AQICN (waqi.info).

CHANGED: now inherits from `AdvancedAPIClient` (src/ingestion/api_client.py)
instead of hand-rolling its own httpx.AsyncClient. This gets AQIClient the
same circuit breaker, token-bucket rate limiter, retry/backoff (tenacity),
RFC 9111 HTTP caching (hishel), Prometheus metrics, and OpenTelemetry
tracing that `AdvancedAPIClient` already provides — and matches what
test_aqi_client.py asserts (`issubclass(AQIClient, AdvancedAPIClient)`).

Only AQICN-specific request shaping (the /feed/{city}/ endpoint and the
`token` query param) lives in this file. `.get()` itself, retries, circuit
breaking, and caching are all inherited unchanged from the parent — this
file must NOT redefine `.get()`, or it would silently bypass all of that.
"""

from __future__ import annotations

import os
from typing import Any

from src.configs.settings import settings
from src.ingestion.api_client import AdvancedAPIClient
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _resolve_aqicn_token() -> str:
    """
    Resolve the AQICN API token with a strict precedence order:
        1. `settings.aqicn_api_key` (a pydantic SecretStr), unwrapped.
        2. Environment variable fallback (AQICN_API_KEY, then WAQI_API_TOKEN).

    CHANGED — replaces a hardcoded `token = ""` that was here before. An
    empty token meant every AQI request silently failed (401 / invalid
    key) with nothing in the code signaling that this was the cause.
    Now raises loudly at construction time instead of failing quietly
    on the first HTTP request.
    """
    secret = getattr(settings, "aqicn_api_key", None)
    if secret is not None:
        value = secret.get_secret_value() if hasattr(secret, "get_secret_value") else secret
        if value:
            return value

    for env_var in ("AQICN_API_KEY", "WAQI_API_TOKEN"):
        value = os.getenv(env_var)
        if value:
            return value

    raise ValueError(
        "AQICN API key is missing. Please set AQICN_API_KEY (or "
        "WAQI_API_TOKEN) in the environment or .env file."
    )


class AQIClient(AdvancedAPIClient):
    """
    AQICN (waqi.info) client. Inherits circuit breaker, rate limiting,
    retry/backoff, HTTP caching, metrics, and tracing from
    `AdvancedAPIClient` — only AQICN-specific request shaping
    (endpoint + token param) is defined here.
    """

    BASE_URL = "https://api.waqi.info"

    def __init__(self, timeout: int = 10, token: str | None = None):
        # Resolve the token BEFORE calling super().__init__ — no network
        # or client setup should happen with a missing/invalid token.
        self.token: str = token or _resolve_aqicn_token()
        super().__init__(base_url=self.BASE_URL, timeout=timeout)

    async def get_current_aqi(self, city: str | None = None) -> dict[str, Any]:
        """Fetch the current AQI feed for a city (defaults to settings.city)."""
        city = city or settings.city
        logger.info("Fetching AQI for %s", city)

        endpoint = f"/feed/{city}/"
        params = {"token": self.token}

        # `.get()` is inherited from AdvancedAPIClient — it already goes
        # through the circuit breaker, rate limiter, retries, and cache.
        return await self.get(endpoint=endpoint, params=params)

    async def fetch(self, city: str) -> dict[str, Any]:
        """
        Alias matching the `AQIClient`/`WeatherClient` Protocol expected
        by `run_pipeline.py`'s `Pipeline` / `_fetch_with_retry`
        (`async def fetch(self, city: str) -> dict`).
        """
        return await self.get_current_aqi(city)

    async def aclose(self) -> None:
        """
        Alias to the parent's `.close()` — `run_pipeline.py`'s
        `Pipeline._aclose_clients` specifically looks for `.aclose()`.
        """
        await self.close()
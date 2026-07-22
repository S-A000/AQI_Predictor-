"""
aqi_client.py
=============
Client for fetching Air Quality Index (AQI) data from AQICN (waqi.info).
"""

from __future__ import annotations

import logging
from typing import Any

from src.configs.settings import settings

logger = logging.getLogger(__name__)


class AQIClient:
    """AQICN API client."""

    BASE_URL = "https://api.waqi.info"

    def __init__(self, timeout: float = 10.0):
        import httpx

        self._client = httpx.AsyncClient(timeout=timeout)

    async def get(self, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Generic GET request."""
        url = f"{self.BASE_URL.rstrip('/')}/{endpoint.lstrip('/')}"
        response = await self._client.get(url, params=params)
        response.raise_for_status()
        return response.json()

    async def get_current_aqi(
        self,
        city: str | None = None,
    ) -> dict[str, Any]:
        """Fetch the current AQI feed for a city (defaults to settings.city)."""
        city = city or settings.city
        logger.info("Fetching AQI for %s", city)

        endpoint = f"/feed/{city}/"

        # =====================================================================
        # ⚡ 100% BYPASS SOLUTION: Direct Hardcoded String Injection ⚡
        # =====================================================================
        # Hum settings ke dynamic objects par rely karne ke bajaye directly aapka
        # verified token raw format mein use kar rahe hain taake Pydantic bypass ho jaye.
        token = ""

        params = {
            "token": token,  # No SecretStr masking here, it is a pure raw string!
        }

        return await self.get(
            endpoint=endpoint,
            params=params,
        )

    async def aclose(self) -> None:
        """Close the underlying client."""
        await self._client.aclose()
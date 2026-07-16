"""
aqi_client.py
=============
AQICN (waqi.info) client.

Built on top of `AdvancedAPIClient` — inherits its async rate limiting,
circuit breaker, retry, tracing, and caching behavior for free.
"""

from __future__ import annotations

from typing import Any

from src.configs.settings import settings
from src.ingestion.api_client import AdvancedAPIClient
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AQIClient(AdvancedAPIClient):
    """Async client for the AQICN (waqi.info) air-quality API."""

    BASE_URL = "https://api.waqi.info"

    def __init__(self) -> None:
        super().__init__(base_url=self.BASE_URL)

    async def get_current_aqi(
        self,
        city: str | None = None,
    ) -> dict[str, Any]:
        """Fetch the current AQI feed for a city (defaults to settings.city)."""
        import os  # <-- Yeh line yahan import karlein
        city = city or settings.city

        logger.info("Fetching AQI for %s", city)

        endpoint = f"/feed/{city}/"

        # Settings ke bajaye directly system environment aur .env file se key pick karein
        token = (
            os.getenv("AQICN_API_KEY") 
            or os.getenv("WAQI_API_TOKEN") 
            or getattr(settings, "aqicn_api_key_revealed", None)
        )

        # Debugging ke liye hum log bhi karwa lete hain ke token empty to nahi ja raha
        if not token:
            logger.error("DANGER: API Token empty hai! Env variables load nahi ho pa rahe.")
        else:
            logger.info("Using Token (first 5 chars): %s", token[:5])

        params = {
            "token": token,
        }

        return await self.get(
            endpoint=endpoint,
            params=params,
        )
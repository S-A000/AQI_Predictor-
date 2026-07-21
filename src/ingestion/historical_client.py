"""
historical_client.py
=====================
Suggested path: src/ingestion/historical_client.py

Historical WEATHER + AIR QUALITY client for backfilling 2-5 years of
training data, using Open-Meteo (free, no API key — AQICN and
OpenWeather only offer historical data on paid tiers).

Design decision
---------------
Unlike `aqi_client.py` / `weather_client.py` (which return raw dicts
for a SINGLE live request, later validated + merged by
`validator.py` / `merger.py`), this client fetches BULK hourly data
and builds `MergedFeature` objects directly. Running thousands of
historical rows through the live per-request validation path would
be the wrong tool for the job — `MergedFeature` (from merger.py) is
already the canonical schema, so we target it directly here to keep
exactly ONE feature schema for both live and historical paths.

Known gaps vs. AQICN/OpenWeather (flagged explicitly, not silently
guessed):
    - station_id      -> no physical station in a reanalysis model;
                          set to -1 (sentinel, NOT a real AQICN idx).
    - dominant_pollutant -> not returned directly; derived from
                          Open-Meteo's per-pollutant US AQI sub-indices.
    - visibility      -> not reliably available in the historical
                          archive; set to -1 (sentinel). Decide in
                          Phase 5/6 whether to impute or drop this
                          column — do the SAME thing on the live side
                          so train/serve schemas don't diverge.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta, timezone
from typing import Any

from src.ingestion.merger import MergedFeature
from src.utils.logger import get_logger

logger = get_logger(__name__)


WEATHER_HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "apparent_temperature",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "cloud_cover",
]

AQ_HOURLY_VARS = [
    "pm10",
    "pm2_5",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "carbon_monoxide",
    "ozone",
    "us_aqi",
    "us_aqi_pm2_5",
    "us_aqi_pm10",
    "us_aqi_no2",
    "us_aqi_so2",
    "us_aqi_co",
    "us_aqi_ozone",
]

_SENTINEL_STATION_ID = -1
_SENTINEL_VISIBILITY = -1


class HistoricalClientError(Exception):
    """Base error for historical data fetches."""


class HistoricalClient:
    """
    Open-Meteo historical WEATHER + AIR QUALITY client.

    Mirrors the constructor/retry conventions used by AQICNClient /
    OpenWeatherClient in run_pipeline.py (httpx.AsyncClient, async
    fetch, aclose()), so it plugs into the same lifecycle.
    """

    WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
    AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

    def __init__(self, timeout: float = 60.0, max_attempts: int = 4, backoff: float = 5.0):
        import httpx

        self._client = httpx.AsyncClient(timeout=timeout)
        self.max_attempts = max_attempts
        self.backoff = backoff

    async def _get_with_retry(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        delay = self.backoff
        last_exc: Exception | None = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                response = await self._client.get(url, params=params)
                if response.status_code == 429:
                    logger.warning("Rate-limited by Open-Meteo, waiting %.1fs", delay)
                    await asyncio.sleep(delay)
                    delay *= 2
                    continue
                response.raise_for_status()
                return response.json()
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.warning(
                    "Historical fetch failed (attempt %d/%d): %s",
                    attempt, self.max_attempts, exc,
                )
                if attempt < self.max_attempts:
                    await asyncio.sleep(delay)
                    delay *= 2

        assert last_exc is not None
        raise HistoricalClientError(str(last_exc)) from last_exc

    async def _fetch_weather(self, lat: float, lon: float, start: str, end: str) -> dict[str, Any]:
        params = {
            "latitude": lat, "longitude": lon,
            "start_date": start, "end_date": end,
            "hourly": ",".join(WEATHER_HOURLY_VARS),
            "timezone": "auto",
        }
        return await self._get_with_retry(self.WEATHER_URL, params)

    async def _fetch_air_quality(self, lat: float, lon: float, start: str, end: str) -> dict[str, Any]:
        params = {
            "latitude": lat, "longitude": lon,
            "start_date": start, "end_date": end,
            "hourly": ",".join(AQ_HOURLY_VARS),
            "timezone": "auto",
            "domains": "cams_global",
        }
        return await self._get_with_retry(self.AQ_URL, params)

    @staticmethod
    def _dominant_pollutant(hourly_row: dict[str, float | None]) -> str:
        sub_indices = {
            "pm25": hourly_row.get("us_aqi_pm2_5"),
            "pm10": hourly_row.get("us_aqi_pm10"),
            "no2": hourly_row.get("us_aqi_no2"),
            "so2": hourly_row.get("us_aqi_so2"),
            "co": hourly_row.get("us_aqi_co"),
            "o3": hourly_row.get("us_aqi_ozone"),
        }
        sub_indices = {k: v for k, v in sub_indices.items() if v is not None}
        return max(sub_indices, key=sub_indices.get) if sub_indices else "unknown"

    async def fetch_range(
        self,
        *,
        city: str,
        country: str,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
    ) -> list[MergedFeature]:
        """
        Fetch and merge historical weather+AQ for one date range,
        returning a list of `MergedFeature` — the SAME canonical
        schema used by the live pipeline.

        Callers doing multi-year backfills should chunk into yearly
        (or smaller) ranges and call this repeatedly — see
        `historical_backfill.py`.
        """

        logger.info("Fetching historical data for %s: %s -> %s", city, start_date, end_date)

        weather_raw, aq_raw = await asyncio.gather(
            self._fetch_weather(latitude, longitude, start_date, end_date),
            self._fetch_air_quality(latitude, longitude, start_date, end_date),
        )

        weather_hourly = weather_raw.get("hourly", {})
        aq_hourly = aq_raw.get("hourly", {})

        weather_times = weather_hourly.get("time", [])
        aq_times = aq_hourly.get("time", [])

        # Index AQ rows by timestamp so we can inner-join against
        # weather timestamps (the two APIs can have slightly
        # different available hours near the edges of the range).
        aq_by_time: dict[str, dict[str, Any]] = {}
        for idx, ts in enumerate(aq_times):
            aq_by_time[ts] = {key: values[idx] for key, values in aq_hourly.items() if key != "time"}

        features: list[MergedFeature] = []
        skipped = 0

        for w_idx, ts in enumerate(weather_times):
            aq_row = aq_by_time.get(ts)
            if aq_row is None:
                skipped += 1
                continue

            try:
                features.append(MergedFeature(
                    city=city,
                    country=country,
                    latitude=latitude,
                    longitude=longitude,
                    timestamp=datetime.fromisoformat(ts).replace(tzinfo=timezone.utc),

                    temperature=weather_hourly["temperature_2m"][w_idx],
                    feels_like=weather_hourly["apparent_temperature"][w_idx],
                    humidity=int(weather_hourly["relative_humidity_2m"][w_idx]),
                    pressure=int(weather_hourly["surface_pressure"][w_idx]),
                    visibility=_SENTINEL_VISIBILITY,
                    wind_speed=weather_hourly["wind_speed_10m"][w_idx],
                    wind_degree=int(weather_hourly["wind_direction_10m"][w_idx]),
                    cloudiness=int(weather_hourly["cloud_cover"][w_idx]),

                    aqi=int(aq_row["us_aqi"]) if aq_row.get("us_aqi") is not None else 0,
                    dominant_pollutant=self._dominant_pollutant(aq_row),
                    station_id=_SENTINEL_STATION_ID,

                    pm25=aq_row.get("pm2_5"),
                    pm10=aq_row.get("pm10"),
                    no2=aq_row.get("nitrogen_dioxide"),
                    so2=aq_row.get("sulphur_dioxide"),
                    co=aq_row.get("carbon_monoxide"),
                    o3=aq_row.get("ozone"),
                    source="Open-Meteo (historical, CAMS reanalysis)",
                ))
            except (KeyError, TypeError) as exc:
                skipped += 1
                logger.debug("Skipped row %s for %s: %s", ts, city, exc)

        if skipped:
            logger.warning("Skipped %d row(s) with missing/misaligned data for %s", skipped, city)

        logger.info("Built %d MergedFeature record(s) for %s (%s -> %s)", len(features), city, start_date, end_date)
        return features

    async def fetch_years(
        self,
        *,
        city: str,
        country: str,
        latitude: float,
        longitude: float,
        start_date: date,
        end_date: date,
    ) -> list[MergedFeature]:
        """
        Convenience wrapper: chunk a multi-year range into calendar
        years and fetch each sequentially (avoids huge single
        requests timing out or hitting undocumented range limits).
        """
        all_features: list[MergedFeature] = []
        cursor = start_date

        while cursor <= end_date:
            chunk_end = min(date(cursor.year, 12, 31), end_date)
            chunk = await self.fetch_range(
                city=city, country=country, latitude=latitude, longitude=longitude,
                start_date=cursor.isoformat(), end_date=chunk_end.isoformat(),
            )
            all_features.extend(chunk)
            cursor = chunk_end + timedelta(days=1)
            await asyncio.sleep(1)  # be polite to the free API

        return all_features

    async def aclose(self) -> None:
        await self._client.aclose()
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import requests

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


CITY_COORDS = {
    "karachi": {
        "city": "Karachi",
        "latitude": 24.8607,
        "longitude": 67.0011,
    },
    "lahore": {
        "city": "Lahore",
        "latitude": 31.5204,
        "longitude": 74.3587,
    },
    "islamabad": {
        "city": "Islamabad",
        "latitude": 33.6844,
        "longitude": 73.0479,
    },
}


class LiveDataService:
    """
    Fetch latest live AQI + weather data for real-time inference.

    Data sources:
    - AQICN / WAQI: current AQI + pollutant IAQI
    - OpenWeather: current weather + optional air-pollution fallback
    """

    def __init__(self, timeout: int = 15) -> None:
        self.timeout = timeout

        self.aqicn_api_key = (
            os.getenv("AQICN_API_KEY")
            or os.getenv("WAQI_API_KEY")
            or os.getenv("AQICN_TOKEN")
        )

        self.openweather_api_key = (
            os.getenv("OPENWEATHER_API_KEY")
            or os.getenv("OPENWEATHERMAP_API_KEY")
            or os.getenv("OPENWEATHER_TOKEN")
        )

    @staticmethod
    def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
        try:
            if value is None or value == "":
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _get_nested_value(data: Dict[str, Any], key: str) -> Optional[float]:
        """
        AQICN iaqi format:
        {
            "iaqi": {
                "pm25": {"v": 44},
                "pm10": {"v": 80}
            }
        }
        """
        try:
            value = data.get("iaqi", {}).get(key, {}).get("v")
            return float(value) if value is not None else None
        except Exception:
            return None

    def fetch_city_live_data(self, city: str) -> Dict[str, Any]:
        city_key = city.strip().lower()

        if city_key not in CITY_COORDS:
            raise ValueError("Unsupported city. Use Karachi, Lahore, or Islamabad.")

        city_meta = CITY_COORDS[city_key]

        aqicn_data = self._fetch_aqicn_current(city_meta)
        weather_data = self._fetch_openweather_current(city_meta)
        ow_air_data = self._fetch_openweather_air_pollution(city_meta)

        return self._merge_payload(
            city_meta=city_meta,
            aqicn_data=aqicn_data,
            weather_data=weather_data,
            ow_air_data=ow_air_data,
        )

    def _fetch_aqicn_current(self, city_meta: Dict[str, Any]) -> Dict[str, Any]:
        if not self.aqicn_api_key:
            raise ValueError(
                "AQICN API key missing. Add AQICN_API_KEY to your .env file."
            )

        lat = city_meta["latitude"]
        lon = city_meta["longitude"]

        url = f"https://api.waqi.info/feed/geo:{lat};{lon}/"

        response = requests.get(
            url,
            params={"token": self.aqicn_api_key},
            timeout=self.timeout,
        )
        response.raise_for_status()

        raw = response.json()

        if raw.get("status") != "ok":
            raise ValueError(f"AQICN returned non-ok response: {raw}")

        data = raw.get("data", {})

        return {
            "aqi": self._safe_float(data.get("aqi")),
            "pm25": self._get_nested_value(data, "pm25"),
            "pm10": self._get_nested_value(data, "pm10"),
            "no2": self._get_nested_value(data, "no2"),
            "so2": self._get_nested_value(data, "so2"),
            "co": self._get_nested_value(data, "co"),
            "o3": self._get_nested_value(data, "o3"),
        }

    def _fetch_openweather_current(self, city_meta: Dict[str, Any]) -> Dict[str, Any]:
        if not self.openweather_api_key:
            return {}

        lat = city_meta["latitude"]
        lon = city_meta["longitude"]

        url = "https://api.openweathermap.org/data/2.5/weather"

        response = requests.get(
            url,
            params={
                "lat": lat,
                "lon": lon,
                "appid": self.openweather_api_key,
                "units": "metric",
            },
            timeout=self.timeout,
        )
        response.raise_for_status()

        raw = response.json()

        main = raw.get("main", {})
        wind = raw.get("wind", {})
        clouds = raw.get("clouds", {})

        return {
            "temperature": self._safe_float(main.get("temp")),
            "humidity": self._safe_float(main.get("humidity")),
            "pressure": self._safe_float(main.get("pressure")),
            "wind_speed": self._safe_float(wind.get("speed")),
            "wind_deg": self._safe_float(wind.get("deg")),
            "cloudiness": self._safe_float(clouds.get("all")),
            "visibility": self._safe_float(raw.get("visibility")),
        }

    def _fetch_openweather_air_pollution(
        self,
        city_meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Optional fallback for pollutant components.
        AQICN gives AQI. OpenWeather can help fill missing pollutant values.
        """
        if not self.openweather_api_key:
            return {}

        lat = city_meta["latitude"]
        lon = city_meta["longitude"]

        url = "https://api.openweathermap.org/data/2.5/air_pollution"

        response = requests.get(
            url,
            params={
                "lat": lat,
                "lon": lon,
                "appid": self.openweather_api_key,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()

        raw = response.json()
        items = raw.get("list", [])

        if not items:
            return {}

        components = items[0].get("components", {})

        return {
            "pm25": self._safe_float(components.get("pm2_5")),
            "pm10": self._safe_float(components.get("pm10")),
            "no2": self._safe_float(components.get("no2")),
            "so2": self._safe_float(components.get("so2")),
            "co": self._safe_float(components.get("co")),
            "o3": self._safe_float(components.get("o3")),
        }

    def _merge_payload(
        self,
        city_meta: Dict[str, Any],
        aqicn_data: Dict[str, Any],
        weather_data: Dict[str, Any],
        ow_air_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        AQI should primarily come from AQICN.
        Pollutants come from AQICN first, OpenWeather fallback second.
        Weather comes from OpenWeather.
        """

        def prefer(primary: Any, fallback: Any, default: Any = None) -> Any:
            if primary is not None and primary != "":
                return primary
            if fallback is not None and fallback != "":
                return fallback
            return default

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "city": city_meta["city"],
            "latitude": city_meta["latitude"],
            "longitude": city_meta["longitude"],

            # AQI from AQICN
            "aqi": self._safe_float(aqicn_data.get("aqi")),

            # Weather from OpenWeather
            "temperature": self._safe_float(weather_data.get("temperature"), 25.0),
            "humidity": self._safe_float(weather_data.get("humidity"), 60.0),
            "pressure": self._safe_float(weather_data.get("pressure"), 1013.0),
            "wind_speed": self._safe_float(weather_data.get("wind_speed"), 3.0),
            "wind_deg": self._safe_float(weather_data.get("wind_deg"), 180.0),
            "cloudiness": self._safe_float(weather_data.get("cloudiness"), 0.0),
            "visibility": self._safe_float(weather_data.get("visibility"), 10000.0),

            # Pollutants: AQICN first, OpenWeather fallback
            "pm25": self._safe_float(
                prefer(aqicn_data.get("pm25"), ow_air_data.get("pm25"))
            ),
            "pm10": self._safe_float(
                prefer(aqicn_data.get("pm10"), ow_air_data.get("pm10"))
            ),
            "no2": self._safe_float(
                prefer(aqicn_data.get("no2"), ow_air_data.get("no2"))
            ),
            "so2": self._safe_float(
                prefer(aqicn_data.get("so2"), ow_air_data.get("so2"))
            ),
            "co": self._safe_float(
                prefer(aqicn_data.get("co"), ow_air_data.get("co"))
            ),
            "o3": self._safe_float(
                prefer(aqicn_data.get("o3"), ow_air_data.get("o3"))
            ),
        }
from __future__ import annotations

from pprint import pprint

from src.prediction.live_data_service import LiveDataService


def test_live_data_service_payload() -> None:
    service = LiveDataService()

    cities = ["Karachi", "Lahore", "Islamabad"]

    for city in cities:
        print("\n" + "=" * 80)
        print(f"Testing live data for: {city}")
        print("=" * 80)

        payload = service.fetch_city_live_data(city)

        print("\nLIVE PAYLOAD:")
        pprint(payload)

        assert isinstance(payload, dict), "Payload must be a dictionary"

        required_fields = [
            "city",
            "timestamp",
            "latitude",
            "longitude",
            "aqi",
            "temperature",
            "humidity",
            "pressure",
            "wind_speed",
            "pm25",
            "pm10",
            "no2",
            "so2",
            "co",
            "o3",
        ]

        for field in required_fields:
            print(f"{field}: {payload.get(field)}")

        assert payload.get("city") == city
        assert payload.get("timestamp") is not None
        assert payload.get("latitude") is not None
        assert payload.get("longitude") is not None

        # Ye main check hai:
        assert payload.get("aqi") is not None, (
            f"AQI live API se nahi aa raha for {city}. "
            "Check _fetch_aqicn_current() in src/prediction/live_data_service.py"
        )
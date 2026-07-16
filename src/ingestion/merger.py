"""
merger.py
=========

Enterprise Feature Merger

Author:
    Syed Abdullah

Description
-----------
Merge WeatherResponse and AQIResponse into a single
feature object for downstream Feature Engineering
and Feature Store ingestion.
"""

from __future__ import annotations

import csv
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field

from src.ingestion.validator import (
    AQIResponse,
    WeatherResponse,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ==========================================================
# Unified Feature Model
# ==========================================================

class MergedFeature(BaseModel):
    """
    Unified feature record.

    One object == One timestamp.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="ignore",
    )

    # --------------------------------------------------
    # Location
    # --------------------------------------------------

    city: str

    country: str

    latitude: float

    longitude: float

    timestamp: datetime

    # --------------------------------------------------
    # Weather
    # --------------------------------------------------

    temperature: float

    feels_like: float

    humidity: int

    pressure: int

    visibility: int

    wind_speed: float

    wind_degree: int

    cloudiness: int

    # --------------------------------------------------
    # AQI
    # --------------------------------------------------

    aqi: int

    dominant_pollutant: str

    station_id: int

    # --------------------------------------------------
    # Pollutants
    # --------------------------------------------------

    pm25: float | None = None

    pm10: float | None = None

    no2: float | None = None

    so2: float | None = None

    co: float | None = None

    o3: float | None = None

    # --------------------------------------------------
    # Metadata
    # --------------------------------------------------

    source: str = "AQICN + OpenWeather"

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# Numeric fields that are eligible for missing-value handling
# and statistical aggregation.
_POLLUTANT_FIELDS: tuple[str, ...] = (
    "pm25",
    "pm10",
    "no2",
    "so2",
    "co",
    "o3",
)

_NUMERIC_FIELDS: tuple[str, ...] = (
    "temperature",
    "feels_like",
    "humidity",
    "pressure",
    "visibility",
    "wind_speed",
    "wind_degree",
    "cloudiness",
    "aqi",
) + _POLLUTANT_FIELDS


# ==========================================================
# Feature Merger
# ==========================================================

class FeatureMerger:
    """
    Merge multiple API responses
    into a single feature object.
    """

    # --------------------------------------------------

    @staticmethod
    def _extract_pollutant(
        iaqi: dict[str, Any],
        name: str,
    ) -> float | None:
        """
        Extract pollutant safely.

        AQICN format:

        iaqi:
            pm25:
                v: 34
        """

        if name not in iaqi:
            return None

        value = iaqi[name]

        if isinstance(value, dict):

            return value.get("v")

        return None

    # --------------------------------------------------

    @classmethod
    def merge(
        cls,
        weather: WeatherResponse,
        aqi: AQIResponse,
    ) -> MergedFeature:

        logger.info(
            "Merging WeatherResponse with AQIResponse..."
        )

        feature = MergedFeature(

            # --------------------------------------
            # Location
            # --------------------------------------

            city=weather.name,

            country=weather.sys.country,

            latitude=weather.coord.lat,

            longitude=weather.coord.lon,

            timestamp=datetime.fromtimestamp(
                weather.dt,
                tz=timezone.utc,
            ),

            # --------------------------------------
            # Weather
            # --------------------------------------

            temperature=weather.main.temp,

            feels_like=weather.main.feels_like,

            humidity=weather.main.humidity,

            pressure=weather.main.pressure,

            visibility=weather.visibility,

            wind_speed=weather.wind.speed,

            wind_degree=weather.wind.deg,

            cloudiness=weather.clouds.all,

            # --------------------------------------
            # AQI
            # --------------------------------------

            aqi=aqi.data.aqi,

            dominant_pollutant=aqi.data.dominentpol,

            station_id=aqi.data.idx,

            # --------------------------------------
            # Pollutants
            # --------------------------------------

            pm25=cls._extract_pollutant(
                aqi.data.iaqi,
                "pm25",
            ),

            pm10=cls._extract_pollutant(
                aqi.data.iaqi,
                "pm10",
            ),

            no2=cls._extract_pollutant(
                aqi.data.iaqi,
                "no2",
            ),

            so2=cls._extract_pollutant(
                aqi.data.iaqi,
                "so2",
            ),

            co=cls._extract_pollutant(
                aqi.data.iaqi,
                "co",
            ),

            o3=cls._extract_pollutant(
                aqi.data.iaqi,
                "o3",
            ),
        )

        logger.info(
            "Successfully merged feature for %s",
            feature.city,
        )

        return feature

    # --------------------------------------------------
    # Batch Merge
    # --------------------------------------------------

    @classmethod
    def merge_batch(
        cls,
        weathers: list[WeatherResponse],
        aqis: list[AQIResponse],
        *,
        skip_errors: bool = True,
    ) -> list[MergedFeature]:
        """
        Merge many WeatherResponse/AQIResponse pairs at once.

        Pairs are matched index-by-index, i.e. ``weathers[i]``
        is merged with ``aqis[i]``.

        Parameters
        ----------
        weathers, aqis:
            Equal-length lists of raw API responses.
        skip_errors:
            If True (default), a failure on a single pair is
            logged and skipped rather than aborting the whole
            batch.
        """

        if len(weathers) != len(aqis):
            raise ValueError(
                "weathers and aqis must be the same length "
                f"(got {len(weathers)} vs {len(aqis)})"
            )

        logger.info(
            "Merging batch of %d weather/aqi pairs...",
            len(weathers),
        )

        features: list[MergedFeature] = []

        for index, (weather, aqi) in enumerate(zip(weathers, aqis)):

            try:
                features.append(cls.merge(weather, aqi))

            except Exception as exc:  # noqa: BLE001

                logger.error(
                    "Failed to merge pair at index %d: %s",
                    index,
                    exc,
                )

                if not skip_errors:
                    raise

        logger.info(
            "Batch merge complete: %d/%d succeeded.",
            len(features),
            len(weathers),
        )

        return features

    # --------------------------------------------------
    # Serialization
    # --------------------------------------------------

    @staticmethod
    def to_dict(feature: MergedFeature) -> dict[str, Any]:
        """
        Convert a MergedFeature to a plain, JSON-safe dict.

        Datetimes are serialized to ISO-8601 strings.
        """

        data = feature.model_dump()

        for key in ("timestamp", "created_at"):

            value = data.get(key)

            if isinstance(value, datetime):
                data[key] = value.isoformat()

        return data

    # --------------------------------------------------

    @staticmethod
    def from_dict(data: dict[str, Any]) -> MergedFeature:
        """
        Rebuild a MergedFeature from a plain dict, e.g. one
        previously produced by ``to_dict``.
        """

        return MergedFeature(**data)

    # --------------------------------------------------

    @classmethod
    def to_dataframe(cls, features: Iterable[MergedFeature]):
        """
        Convert an iterable of MergedFeature into a pandas
        DataFrame, ready for Feature Store ingestion.

        Raises
        ------
        ImportError
            If pandas is not installed.
        """

        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "pandas is required for to_dataframe(); "
                "install it with `pip install pandas`."
            ) from exc

        rows = [cls.to_dict(feature) for feature in features]

        return pd.DataFrame(rows)

    # --------------------------------------------------
    # Missing Value Handling
    # --------------------------------------------------

    @classmethod
    def handle_missing_values(
        cls,
        features: list[MergedFeature],
        *,
        strategy: str = "mean",
        fields: tuple[str, ...] = _POLLUTANT_FIELDS,
    ) -> list[MergedFeature]:
        """
        Fill missing (None) numeric fields across a batch.

        Parameters
        ----------
        strategy:
            - "mean": fill with the field's batch mean.
            - "zero": fill with 0.
            - "drop": drop any feature that has a missing value
              in one of ``fields``.
        fields:
            Which fields to consider. Defaults to the pollutant
            fields, since those are the ones most often missing
            from AQICN.
        """

        if strategy not in ("mean", "zero", "drop"):
            raise ValueError(
                f"Unknown strategy '{strategy}'; "
                "expected 'mean', 'zero', or 'drop'."
            )

        if strategy == "drop":

            cleaned = [
                feature
                for feature in features
                if all(
                    getattr(feature, field) is not None
                    for field in fields
                )
            ]

            logger.info(
                "Dropped %d/%d features with missing values.",
                len(features) - len(cleaned),
                len(features),
            )

            return cleaned

        # Precompute per-field fill values for "mean" / "zero".
        fill_values: dict[str, float] = {}

        for field in fields:

            if strategy == "zero":
                fill_values[field] = 0.0
                continue

            observed = [
                getattr(feature, field)
                for feature in features
                if getattr(feature, field) is not None
            ]

            fill_values[field] = (
                statistics.mean(observed) if observed else 0.0
            )

        filled: list[MergedFeature] = []

        for feature in features:

            updates = {
                field: fill_values[field]
                for field in fields
                if getattr(feature, field) is None
            }

            if updates:
                filled.append(feature.model_copy(update=updates))
            else:
                filled.append(feature)

        logger.info(
            "Missing value handling complete (strategy=%s).",
            strategy,
        )

        return filled

    # --------------------------------------------------
    # Feature Statistics
    # --------------------------------------------------

    @classmethod
    def compute_statistics(
        cls,
        features: list[MergedFeature],
        *,
        fields: tuple[str, ...] = _NUMERIC_FIELDS,
    ) -> dict[str, dict[str, float]]:
        """
        Compute basic descriptive statistics (count, mean, min,
        max, stdev) for each numeric field across a batch.

        Fields with no observed (non-None) values are omitted.
        """

        stats: dict[str, dict[str, float]] = {}

        for field in fields:

            values = [
                getattr(feature, field)
                for feature in features
                if getattr(feature, field) is not None
            ]

            if not values:
                continue

            stats[field] = {
                "count": len(values),
                "mean": statistics.mean(values),
                "min": min(values),
                "max": max(values),
                "stdev": (
                    statistics.stdev(values)
                    if len(values) > 1
                    else 0.0
                ),
            }

        return stats

    # --------------------------------------------------
    # Duplicate Detection
    # --------------------------------------------------

    @staticmethod
    def detect_duplicates(
        features: list[MergedFeature],
        *,
        keys: tuple[str, ...] = ("city", "timestamp"),
    ) -> list[MergedFeature]:
        """
        Return the subset of features that are duplicates
        (i.e. every occurrence after the first) based on the
        given key fields. Defaults to (city, timestamp).
        """

        seen: set[tuple[Any, ...]] = set()
        duplicates: list[MergedFeature] = []

        for feature in features:

            identity = tuple(
                getattr(feature, key) for key in keys
            )

            if identity in seen:
                duplicates.append(feature)
            else:
                seen.add(identity)

        if duplicates:
            logger.warning(
                "Detected %d duplicate feature(s) based on %s.",
                len(duplicates),
                keys,
            )

        return duplicates

    # --------------------------------------------------

    @classmethod
    def drop_duplicates(
        cls,
        features: list[MergedFeature],
        *,
        keys: tuple[str, ...] = ("city", "timestamp"),
    ) -> list[MergedFeature]:
        """
        Return ``features`` with duplicates (based on ``keys``)
        removed, keeping the first occurrence of each.
        """

        seen: set[tuple[Any, ...]] = set()
        unique: list[MergedFeature] = []

        for feature in features:

            identity = tuple(
                getattr(feature, key) for key in keys
            )

            if identity not in seen:
                seen.add(identity)
                unique.append(feature)

        return unique

    # --------------------------------------------------
    # Export Helpers
    # --------------------------------------------------

    @classmethod
    def export_to_json(
        cls,
        features: list[MergedFeature],
        path: str | Path,
        *,
        indent: int = 2,
    ) -> Path:
        """
        Export a batch of features to a JSON file (a list of
        objects). Returns the path written to.
        """

        path = Path(path)

        rows = [cls.to_dict(feature) for feature in features]

        path.write_text(
            json.dumps(rows, indent=indent, default=str),
            encoding="utf-8",
        )

        logger.info(
            "Exported %d features to JSON at %s",
            len(features),
            path,
        )

        return path

    # --------------------------------------------------

    @classmethod
    def export_to_csv(
        cls,
        features: list[MergedFeature],
        path: str | Path,
    ) -> Path:
        """
        Export a batch of features to a CSV file. Returns the
        path written to.
        """

        path = Path(path)

        rows = [cls.to_dict(feature) for feature in features]

        if not rows:
            logger.warning(
                "export_to_csv called with an empty feature list; "
                "writing an empty file."
            )
            path.write_text("", encoding="utf-8")
            return path

        with path.open("w", newline="", encoding="utf-8") as handle:

            writer = csv.DictWriter(
                handle,
                fieldnames=list(rows[0].keys()),
            )

            writer.writeheader()
            writer.writerows(rows)

        logger.info(
            "Exported %d features to CSV at %s",
            len(features),
            path,
        )

        return path
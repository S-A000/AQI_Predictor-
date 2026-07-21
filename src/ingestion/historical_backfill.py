"""
historical_backfill.py
=======================
Suggested path: src/ingestion/historical_backfill.py  (or top-level scripts/)

Orchestrates Phase 3 (historical download) + shared feature
engineering + Phase 4 (Feast backfill), the same way run_pipeline.py
orchestrates the live hourly path.

    python -m src.ingestion.historical_backfill \
        --city Karachi --country PK --lat 24.8607 --lon 67.0011 \
        --start 2021-01-01 --end 2026-07-17

This is a ONE-TIME / occasional job (not hourly) — run manually, or
schedule it as a separate, infrequent GitHub Actions / Airflow job
(e.g. monthly, to top up recent history), distinct from the hourly
live pipeline.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import date

from src.feature_pipeline.feature_engineering import engineer_features
from src.feature_pipeline.feast_writer import write_to_feast_source
from src.ingestion.historical_client import HistoricalClient
from src.ingestion.storage import StorageManager
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Convenience lookup for cities already used elsewhere in this project
# (matches run_pipeline.py's default --cities list). Add more here as
# needed — for anything not in this table, pass --lat/--lon explicitly.
KNOWN_CITY_COORDS: dict[str, tuple[float, float, str]] = {
    "Karachi": (24.8607, 67.0011, "PK"),
    "Lahore": (31.5497, 74.3436, "PK"),
    "Islamabad": (33.6844, 73.0479, "PK"),
}


async def run_backfill(
    *,
    city: str,
    country: str,
    latitude: float,
    longitude: float,
    start_date: date,
    end_date: date,
    upload_to_feature_store: bool = False,
    feature_group_name: str = "aqi_weather_features",
) -> None:

    client = HistoricalClient()
    storage = StorageManager()

    try:
        logger.info("Starting historical backfill for %s: %s -> %s", city, start_date, end_date)

        raw_features = await client.fetch_years(
            city=city, country=country, latitude=latitude, longitude=longitude,
            start_date=start_date, end_date=end_date,
        )

        if not raw_features:
            logger.error("No historical data fetched for %s; aborting backfill.", city)
            return

        # Same audit-trail save as the live pipeline: the raw,
        # unmodified MergedFeature batch, versioned.
        batch_result = storage.save_batch(
            raw_features,
            formats=("parquet",),
            filename_stem=f"historical_{city.lower()}",
        )
        logger.info("Saved raw historical batch as version %s", batch_result["version"])

        # SAME feature-engineering function the live pipeline uses.
        engineered_df = engineer_features(raw_features)

        engineered_path = (
            storage.base_directory / batch_result["version"] / f"historical_{city.lower()}_features.parquet"
        )
        engineered_df.to_parquet(engineered_path, index=False)
        logger.info("Saved engineered feature set -> %s", engineered_path)

        # Append into the STABLE, consolidated Feast source that
        # feature_repo/data_source.py points to. This is what
        # actually makes the data visible to Feast (the versioned
        # engineered_path above is just an audit-trail copy).
        write_to_feast_source(engineered_df)

        if upload_to_feature_store:
            storage.upload_to_feature_store(raw_features, feature_group_name)
            logger.info("Uploaded raw batch to feature store group '%s'", feature_group_name)

    finally:
        await client.aclose()


async def run_backfill_for_cities(
    *,
    cities: list[str],
    start_date: date,
    end_date: date,
    upload_to_feature_store: bool = False,
    feature_group_name: str = "aqi_weather_features",
) -> None:
    """
    Runs `run_backfill` sequentially for each city in `cities`,
    looking up coordinates from `KNOWN_CITY_COORDS`. Sequential (not
    parallel) on purpose — this is an occasional bulk job, not the
    hourly live path, so there's no need to race multiple large
    multi-year fetches against Open-Meteo at once.
    """
    unknown = [city for city in cities if city not in KNOWN_CITY_COORDS]
    if unknown:
        raise ValueError(
            f"No known coordinates for: {unknown}. Add them to "
            f"KNOWN_CITY_COORDS in historical_backfill.py, or run "
            f"run_backfill() directly with explicit --lat/--lon for "
            f"a single city."
        )

    for city in cities:
        latitude, longitude, country = KNOWN_CITY_COORDS[city]
        await run_backfill(
            city=city, country=country, latitude=latitude, longitude=longitude,
            start_date=start_date, end_date=end_date,
            upload_to_feature_store=upload_to_feature_store,
            feature_group_name=feature_group_name,
        )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Historical AQI/weather backfill (Open-Meteo)")
    parser.add_argument(
        "--cities", nargs="+", default=None,
        help="One or more known city names (e.g. Karachi Lahore Islamabad). "
             "Coordinates are looked up automatically. Mutually exclusive with --city.",
    )
    parser.add_argument("--city", help="Single city name (use with --lat/--lon for cities not in the known list).")
    parser.add_argument("--country", default="PK")
    parser.add_argument("--lat", type=float)
    parser.add_argument("--lon", type=float)
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--feature-store", action="store_true")
    parser.add_argument("--feature-group", default="aqi_weather_features")
    return parser


def _main() -> None:
    args = _build_arg_parser().parse_args()
    start_date = date.fromisoformat(args.start)
    end_date = date.fromisoformat(args.end)

    if args.cities:
        asyncio.run(run_backfill_for_cities(
            cities=args.cities,
            start_date=start_date, end_date=end_date,
            upload_to_feature_store=args.feature_store,
            feature_group_name=args.feature_group,
        ))
    elif args.city and args.lat is not None and args.lon is not None:
        asyncio.run(run_backfill(
            city=args.city, country=args.country,
            latitude=args.lat, longitude=args.lon,
            start_date=start_date, end_date=end_date,
            upload_to_feature_store=args.feature_store,
            feature_group_name=args.feature_group,
        ))
    else:
        raise SystemExit(
            "Provide either --cities (known cities, e.g. --cities Karachi Lahore Islamabad) "
            "or --city with --lat and --lon for a single custom city."
        )


if __name__ == "__main__":
    _main()
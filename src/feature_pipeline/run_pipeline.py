"""
run_pipeline.py
===============

Hourly AQI feature pipeline.

Flow:
    1. Load historical context from BigQuery.
    2. Fetch fresh AQI/weather data for supported cities.
    3. Combine live rows with historical context.
    4. Run the canonical feature-engineering pipeline.
    5. Select the newest engineered row for each city.
    6. Align output with the BigQuery table schema.
    7. Append the new rows to BigQuery.

This file does not duplicate feature-engineering logic.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.feature_engineering.pipeline_steps import (
    run_feature_engineering_steps,
)
from src.feature_store.bigquery_feature_store import (
    BigQueryFeatureStore,
)
from src.prediction.live_data_service import LiveDataService
from src.prediction.load_features import FeatureLoader
from src.utils.logger import get_logger


logger = get_logger(__name__)

SUPPORTED_CITIES = [
    "Islamabad",
    "Karachi",
    "Lahore",
]

# Your longest rolling window is 168 hours.
CONTEXT_ROWS_PER_CITY = 168

RAW_CONTEXT_COLUMNS = (
    "city",
    "timestamp",
    "event_hour",
    "country",
    "created_at",
    "source",
    "station_id",
    "latitude",
    "longitude",
    "aqi",
    "temperature",
    "feels_like",
    "humidity",
    "pressure",
    "wind_speed",
    "wind_deg",
    "wind_degree",
    "wind_direction",
    "cloudiness",
    "visibility",
    "pm25",
    "pm10",
    "no2",
    "so2",
    "co",
    "o3",
    "dominant_pollutant",
)


class HourlyFeaturePipeline:
    """
    Production hourly feature pipeline.

    BigQuery is the primary historical-context source.
    Local Parquet remains available as fallback.
    """

    def __init__(
        self,
        feature_store: BigQueryFeatureStore | None = None,
        live_data_service: LiveDataService | None = None,
        local_feature_loader: FeatureLoader | None = None,
    ) -> None:
        self.feature_store = (
            feature_store
            or BigQueryFeatureStore()
        )

        self.live_data_service = (
            live_data_service
            or LiveDataService()
        )

        self.local_feature_loader = (
            local_feature_loader
            or FeatureLoader()
        )

    @staticmethod
    def _normalise_live_payload(
        payload: Dict[str, Any],
        city: str,
    ) -> Dict[str, Any]:
        """
        Convert a live API payload into one canonical raw feature row.
        """

        row = dict(payload)

        row["city"] = city

        timestamp = row.get("timestamp")

        if timestamp is None:
            timestamp = datetime.now(
                timezone.utc
            ).isoformat()

        row["timestamp"] = pd.to_datetime(
            timestamp,
            utc=True,
            errors="coerce",
        )

        if pd.isna(row["timestamp"]):
            row["timestamp"] = pd.Timestamp.now(
                tz="UTC"
            )

        row["event_hour"] = row["timestamp"].floor("h")
        row["timestamp"] = row["event_hour"]

        return row

    def _load_bigquery_context(
        self,
    ) -> pd.DataFrame:
        """
        Load the latest 168 rows for every supported city.
        """

        city_frames: List[pd.DataFrame] = []

        for city in SUPPORTED_CITIES:
            city_df = self.feature_store.get_latest_context(
                city=city,
                rows=CONTEXT_ROWS_PER_CITY,
            )

            city_frames.append(city_df)

        context_df = pd.concat(
            city_frames,
            ignore_index=True,
        )

        if context_df.empty:
            raise ValueError(
                "BigQuery historical context is empty."
            )

        logger.info(
            "Historical context loaded from BigQuery | "
            "cities=%d | rows=%d",
            len(SUPPORTED_CITIES),
            len(context_df),
        )

        return context_df

    def _load_local_context(
        self,
    ) -> pd.DataFrame:
        """
        Load historical context from local Parquet as fallback.
        """

        full_df = (
            self.local_feature_loader
            .load_latest_features(num_rows=None)
        )

        if full_df.empty:
            raise ValueError(
                "Local historical feature dataset is empty."
            )

        if "city" not in full_df.columns:
            raise ValueError(
                "Local feature dataset has no 'city' column."
            )

        if "timestamp" not in full_df.columns:
            raise ValueError(
                "Local feature dataset has no 'timestamp' column."
            )

        full_df["timestamp"] = pd.to_datetime(
            full_df["timestamp"],
            utc=True,
            errors="coerce",
        )

        full_df = full_df.dropna(
            subset=["city", "timestamp"]
        )

        full_df["timestamp"] = full_df["timestamp"].dt.floor("h")
        full_df["event_hour"] = full_df["timestamp"]
        full_df = full_df.drop_duplicates(
            subset=["city", "event_hour"], keep="last"
        )

        full_df = full_df.sort_values(
            ["city", "timestamp"]
        )

        context_df = (
            full_df.groupby(
                full_df["city"]
                .astype(str)
                .str.lower(),
                group_keys=False,
            )
            .tail(CONTEXT_ROWS_PER_CITY)
            .copy()
        )

        if context_df.empty:
            raise ValueError(
                "Could not build local historical context."
            )

        logger.warning(
            "Historical context loaded from local fallback | "
            "rows=%d",
            len(context_df),
        )

        return context_df

    def load_historical_context(
        self,
    ) -> pd.DataFrame:
        """
        Primary source: BigQuery.
        Fallback source: local Parquet.
        """

        try:
            return self._load_bigquery_context()

        except Exception as err:
            logger.warning(
                "BigQuery context loading failed. "
                "Using local fallback. Error: %s",
                err,
            )

            return self._load_local_context()

    def fetch_live_rows(
        self,
    ) -> pd.DataFrame:
        """
        Fetch one fresh AQI/weather row for every available city.

        A failure for one city does not stop the remaining cities.
        """

        rows: List[Dict[str, Any]] = []

        for city in SUPPORTED_CITIES:
            try:
                payload = (
                    self.live_data_service
                    .fetch_city_live_data(city)
                )

                if not payload:
                    logger.warning(
                        "Live API returned no data for city=%s",
                        city,
                    )
                    continue

                rows.append(
                    self._normalise_live_payload(
                        payload=payload,
                        city=city,
                    )
                )

                logger.info(
                    "Live data fetched successfully | city=%s",
                    city,
                )

            except Exception as err:
                logger.exception(
                    "Live data fetch failed | city=%s | error=%s",
                    city,
                    err,
                )

        if not rows:
            raise RuntimeError(
                "Live APIs returned no usable city records."
            )

        live_df = pd.DataFrame(rows)

        logger.info(
            "Live batch created | cities=%d | rows=%d",
            live_df["city"].nunique(),
            len(live_df),
        )

        return live_df

    @staticmethod
    def combine_context_and_live_data(
        context_df: pd.DataFrame,
        live_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Combine historical records and fresh live rows.
        """

        required_columns = {
            "city",
            "timestamp",
        }

        for name, dataframe in {
            "context": context_df,
            "live": live_df,
        }.items():
            missing = (
                required_columns
                - set(dataframe.columns)
            )

            if missing:
                raise ValueError(
                    f"{name} DataFrame is missing columns: "
                    f"{sorted(missing)}"
                )

        combined_df = pd.concat(
            [context_df, live_df],
            ignore_index=True,
            sort=False,
        )

        combined_df = combined_df.dropna(subset=["city"])
        combined_df["city"] = (
            combined_df["city"]
            .astype(str)
            .str.strip()
            .str.title()
        )
        combined_df = combined_df[combined_df["city"].ne("")].copy()

        combined_df["timestamp"] = pd.to_datetime(
            combined_df["timestamp"],
            utc=True,
            errors="coerce",
        )

        combined_df = combined_df.dropna(
            subset=["city", "timestamp"]
        )

        combined_df["event_hour"] = combined_df["timestamp"].dt.floor("h")
        combined_df["timestamp"] = combined_df["event_hour"]

        combined_df = combined_df.drop_duplicates(
            subset=["city", "event_hour"],
            keep="last",
        )

        combined_df = combined_df.sort_values(
            ["city", "timestamp"]
        ).reset_index(drop=True)

        return combined_df

    @staticmethod
    def engineer_latest_rows(
        combined_df: pd.DataFrame,
        live_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Run the canonical feature pipeline and keep only new live rows.
        """

        # BigQuery context already contains engineered columns. Retain only
        # canonical source fields before rebuilding the latest row so features
        # are neither duplicated nor recursively engineered.
        raw_columns = [
            column for column in RAW_CONTEXT_COLUMNS if column in combined_df.columns
        ]
        raw_combined_df = combined_df[raw_columns].copy()

        engineered_df = (
            run_feature_engineering_steps(
                raw_combined_df
            )
        )

        if engineered_df.empty:
            raise ValueError(
                "Feature engineering returned an empty DataFrame."
            )

        live_keys = live_df[["city", "event_hour"]].copy()

        live_keys["event_hour"] = pd.to_datetime(
            live_keys["event_hour"],
            utc=True,
            errors="coerce",
        ).dt.floor("h")

        engineered_df["event_hour"] = pd.to_datetime(
            engineered_df["timestamp"], utc=True, errors="coerce"
        ).dt.floor("h")

        latest_rows = engineered_df.merge(
            live_keys,
            on=["city", "event_hour"],
            how="inner",
        )

        latest_rows = latest_rows.drop_duplicates(
            subset=["city", "event_hour"],
            keep="last",
        )

        # This feature depends on train-fitted normalization state. Store it
        # as NULL in the hourly table; the persisted training transformer
        # computes it consistently for train/validation/test/inference.
        if "pollution_index" in latest_rows.columns:
            latest_rows = latest_rows.drop(columns=["pollution_index"])

        if latest_rows.empty:
            raise ValueError(
                "No engineered live rows were produced."
            )

        logger.info(
            "Live feature engineering completed | "
            "rows=%d | columns=%d",
            len(latest_rows),
            len(latest_rows.columns),
        )

        return latest_rows

    def align_with_bigquery_schema(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Align the hourly rows with the existing BigQuery table.

        Missing target columns are stored as NULL because future AQI
        targets are not yet known at feature-generation time.
        """

        table = self.feature_store.client.get_table(
            self.feature_store.full_table_id
        )

        df = dataframe.copy()

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(
                df["timestamp"], utc=True, errors="coerce"
            ).dt.floor("h")
        if "event_hour" in df.columns:
            df["event_hour"] = pd.to_datetime(
                df["event_hour"], utc=True, errors="coerce"
            ).dt.floor("h")

        for field in table.schema:
            column = field.name

            if column in df.columns:
                continue

            field_type = field.field_type.upper()

            if field_type in {
                "INTEGER",
                "INT64",
                "FLOAT",
                "FLOAT64",
                "NUMERIC",
                "BIGNUMERIC",
            }:
                df[column] = np.nan

            elif field_type in {
                "TIMESTAMP",
                "DATETIME",
                "DATE",
            }:
                df[column] = pd.NaT

            else:
                df[column] = None

            logger.warning(
                "BigQuery column missing from hourly output; "
                "using NULL | column=%s",
                column,
            )

        table_columns = [
            field.name
            for field in table.schema
        ]

        df = df[table_columns]

        return df

    def run(self) -> int:
        """
        Execute one complete hourly feature-pipeline run.
        """

        logger.info(
            "Starting hourly AQI feature pipeline."
        )

        context_df = (
            self.load_historical_context()
        )

        live_df = self.fetch_live_rows()

        combined_df = (
            self.combine_context_and_live_data(
                context_df=context_df,
                live_df=live_df,
            )
        )

        latest_features_df = (
            self.engineer_latest_rows(
                combined_df=combined_df,
                live_df=live_df,
            )
        )

        aligned_df = (
            self.align_with_bigquery_schema(
                latest_features_df
            )
        )

        uploaded_rows = (
            self.feature_store.append_features(
                dataframe=aligned_df
            )
        )

        logger.info(
            "Hourly AQI feature pipeline completed | "
            "uploaded_rows=%d",
            uploaded_rows,
        )

        return uploaded_rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the hourly AQI feature pipeline "
            "and append new features to BigQuery."
        )
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Generate and validate features without "
            "uploading them to BigQuery."
        ),
    )

    args = parser.parse_args()

    pipeline = HourlyFeaturePipeline()

    try:
        if args.dry_run:
            context_df = (
                pipeline.load_historical_context()
            )

            live_df = pipeline.fetch_live_rows()

            combined_df = (
                pipeline.combine_context_and_live_data(
                    context_df=context_df,
                    live_df=live_df,
                )
            )

            latest_df = (
                pipeline.engineer_latest_rows(
                    combined_df=combined_df,
                    live_df=live_df,
                )
            )

            aligned_df = (
                pipeline.align_with_bigquery_schema(
                    latest_df
                )
            )

            logger.info(
                "Dry run successful | rows=%d | columns=%d",
                len(aligned_df),
                len(aligned_df.columns),
            )

            time_column = (
                "event_hour" if "event_hour" in aligned_df.columns else "timestamp"
            )
            print(
                aligned_df[["city", time_column]].to_string(index=False)
            )

            return 0

        uploaded_rows = pipeline.run()

        print(
            f"Hourly pipeline successful: "
            f"{uploaded_rows} row(s) uploaded."
        )

        return 0

    except Exception as err:
        logger.exception(
            "Hourly feature pipeline failed: %s",
            err,
        )

        return 1


if __name__ == "__main__":
    sys.exit(main())

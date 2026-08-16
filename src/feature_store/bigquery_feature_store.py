from __future__ import annotations

from datetime import datetime
import os
from typing import Optional, Sequence

import pandas as pd
from dotenv import load_dotenv
from google.cloud import bigquery

from src.utils.logger import get_logger

load_dotenv()

logger = get_logger(__name__)


class BigQueryFeatureStore:
    """Read and append AQI features in the BigQuery feature repository."""

    def __init__(
        self,
        project_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
        table_id: Optional[str] = None,
        location: Optional[str] = None,
    ) -> None:
        self.project_id = project_id or os.getenv("GCP_PROJECT_ID")
        self.dataset_id = dataset_id or os.getenv(
            "BIGQUERY_DATASET_ID", "aqi_feature_store"
        )
        self.table_id = table_id or os.getenv(
            "BIGQUERY_FEATURE_TABLE", "engineered_features"
        )
        self.location = location or os.getenv(
            "BIGQUERY_LOCATION", "asia-south1"
        )

        if not self.project_id:
            raise ValueError("GCP_PROJECT_ID is missing from .env")

        self.client = bigquery.Client(
            project=self.project_id,
            location=self.location,
        )

    @property
    def full_table_id(self) -> str:
        return f"{self.project_id}.{self.dataset_id}.{self.table_id}"

    @staticmethod
    def _as_utc_datetime(value: datetime | str | pd.Timestamp) -> datetime:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")
        return timestamp.to_pydatetime()

    @staticmethod
    def _time_column(table_columns: Sequence[str]) -> str:
        return "event_hour" if "event_hour" in table_columns else "timestamp"

    @staticmethod
    def _select_clause(
        columns: Sequence[str] | None,
        table_columns: Sequence[str],
    ) -> str:
        if columns is None:
            return "*"
        if not columns:
            raise ValueError("columns must contain at least one column name.")

        unknown = set(columns) - set(table_columns)
        if unknown:
            raise ValueError(f"Unknown BigQuery columns requested: {sorted(unknown)}")
        return ", ".join(f"`{column}`" for column in columns)

    def test_connection(self) -> int:
        """Return total table rows to confirm the connection."""
        query = f"""
        SELECT COUNT(*) AS total_rows
        FROM `{self.full_table_id}`
        """
        result = self.client.query(query, location=self.location).result()
        row = next(iter(result))
        return int(row.total_rows)

    def get_training_features(
        self,
        columns: Sequence[str] | None = None,
        *,
        start_time: datetime | str | pd.Timestamp | None = None,
        end_time: datetime | str | pd.Timestamp | None = None,
        cities: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Load historical features, optionally bounded by schema/time/city."""

        table = self.client.get_table(self.full_table_id)
        table_columns = [field.name for field in table.schema]
        time_column = self._time_column(table_columns)
        requested_columns = list(columns) if columns is not None else None
        select_clause = self._select_clause(requested_columns, table_columns)

        predicates: list[str] = []
        parameters: list[bigquery.QueryParameter] = []
        if start_time is not None:
            predicates.append(f"`{time_column}` >= @start_time")
            parameters.append(
                bigquery.ScalarQueryParameter(
                    "start_time", "TIMESTAMP", self._as_utc_datetime(start_time)
                )
            )
        if end_time is not None:
            predicates.append(f"`{time_column}` <= @end_time")
            parameters.append(
                bigquery.ScalarQueryParameter(
                    "end_time", "TIMESTAMP", self._as_utc_datetime(end_time)
                )
            )
        if cities:
            normalized_cities = sorted(
                {str(city).strip().lower() for city in cities if str(city).strip()}
            )
            predicates.append("LOWER(TRIM(city)) IN UNNEST(@cities)")
            parameters.append(
                bigquery.ArrayQueryParameter(
                    "cities", "STRING", normalized_cities
                )
            )

        where_clause = f"WHERE {' AND '.join(predicates)}" if predicates else ""
        query = f"""
        SELECT {select_clause}
        FROM `{self.full_table_id}`
        {where_clause}
        ORDER BY `{time_column}` ASC
        """
        job_config = (
            bigquery.QueryJobConfig(query_parameters=parameters)
            if parameters
            else None
        )
        dataframe = self.client.query(
            query,
            job_config=job_config,
            location=self.location,
        ).to_dataframe()
        if dataframe.empty:
            raise ValueError("BigQuery training dataset is empty.")
        if requested_columns is not None:
            dataframe = dataframe[requested_columns]
        return dataframe

    def get_latest_context(
        self,
        city: str,
        rows: int = 72,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Load the latest feature rows for one city."""

        if rows <= 0:
            raise ValueError("rows must be greater than zero.")

        table = self.client.get_table(self.full_table_id)
        table_columns = [field.name for field in table.schema]
        time_column = self._time_column(table_columns)
        requested_columns = list(columns) if columns is not None else None
        select_clause = self._select_clause(requested_columns, table_columns)
        query = f"""
        SELECT {select_clause}
        FROM `{self.full_table_id}`
        WHERE LOWER(TRIM(city)) = LOWER(TRIM(@city))
        ORDER BY `{time_column}` DESC
        LIMIT @row_limit
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("city", "STRING", city),
                bigquery.ScalarQueryParameter("row_limit", "INT64", rows),
            ]
        )
        dataframe = self.client.query(
            query,
            job_config=job_config,
            location=self.location,
        ).to_dataframe()
        if dataframe.empty:
            raise ValueError(f"No BigQuery records found for city: {city}")

        if time_column in dataframe.columns:
            dataframe = dataframe.sort_values(time_column).reset_index(drop=True)
        else:
            dataframe = dataframe.iloc[::-1].reset_index(drop=True)
        return dataframe

    def _get_existing_keys(
        self,
        *,
        cities: Sequence[str],
        start_hour: pd.Timestamp,
        end_hour: pd.Timestamp,
        time_column: str,
    ) -> set[tuple[str, pd.Timestamp]]:
        """Read only existing logical keys in the incoming city/time window."""

        query = f"""
        SELECT LOWER(TRIM(city)) AS city_key, `{time_column}` AS event_hour
        FROM `{self.full_table_id}`
        WHERE LOWER(TRIM(city)) IN UNNEST(@cities)
          AND `{time_column}` BETWEEN @start_hour AND @end_hour
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter(
                    "cities", "STRING", list(cities)
                ),
                bigquery.ScalarQueryParameter(
                    "start_hour", "TIMESTAMP", start_hour.to_pydatetime()
                ),
                bigquery.ScalarQueryParameter(
                    "end_hour", "TIMESTAMP", end_hour.to_pydatetime()
                ),
            ]
        )
        existing = self.client.query(
            query,
            job_config=job_config,
            location=self.location,
        ).to_dataframe()
        if existing.empty:
            return set()

        existing["event_hour"] = pd.to_datetime(
            existing["event_hour"], utc=True, errors="coerce"
        ).dt.floor("h")
        existing = existing.dropna(subset=["city_key", "event_hour"])
        return set(zip(existing["city_key"], existing["event_hour"]))

    def append_features(self, dataframe: pd.DataFrame) -> int:
        """Append only new canonical city/event-hour rows using a load job."""

        if dataframe.empty:
            raise ValueError("Cannot upload an empty feature DataFrame.")

        target_table = self.client.get_table(self.full_table_id)
        target_columns = [field.name for field in target_table.schema]
        time_column = self._time_column(target_columns)
        df = dataframe.copy()

        if "city" not in df.columns:
            raise ValueError("Required column is missing: city")
        if "event_hour" not in df.columns and "timestamp" not in df.columns:
            raise ValueError("Required timestamp/event_hour column is missing.")

        df = df.dropna(subset=["city"])
        df["city"] = df["city"].astype(str).str.strip()
        df["_city_key"] = df["city"].str.lower()
        event_hours = (
            pd.to_datetime(df["event_hour"], utc=True, errors="coerce")
            if "event_hour" in df.columns
            else pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
        )
        if "timestamp" in df.columns:
            event_hours = event_hours.fillna(
                pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            )
        df["_event_hour"] = event_hours.dt.floor("h")
        df = df[
            df["_city_key"].ne("") & df["_event_hour"].notna()
        ].copy()
        df = df.drop_duplicates(
            subset=["_city_key", "_event_hour"], keep="last"
        )
        if df.empty:
            raise ValueError("No valid feature rows remained after validation.")

        # Keep timestamp as the closest existing-schema equivalent and populate
        # event_hour as well when the table already supports that field.
        if "timestamp" in target_columns:
            df["timestamp"] = df["_event_hour"]
        if "event_hour" in target_columns:
            df["event_hour"] = df["_event_hour"]

        missing_target_columns = set(target_columns) - set(df.columns)
        if missing_target_columns:
            raise ValueError(
                "Incoming DataFrame is missing BigQuery columns: "
                f"{sorted(missing_target_columns)}"
            )

        incoming_cities = sorted(df["_city_key"].unique().tolist())
        start_hour = df["_event_hour"].min()
        end_hour = df["_event_hour"].max()
        existing_keys = self._get_existing_keys(
            cities=incoming_cities,
            start_hour=start_hour,
            end_hour=end_hour,
            time_column=time_column,
        )
        incoming_keys = list(zip(df["_city_key"], df["_event_hour"]))
        is_new = [key not in existing_keys for key in incoming_keys]
        new_df = df.loc[is_new].copy()

        if new_df.empty:
            logger.info(
                "BigQuery append skipped; all %d incoming city/event-hour key(s) already exist.",
                len(df),
            )
            return 0

        new_df = new_df[target_columns]
        job_config = bigquery.LoadJobConfig(
            schema=target_table.schema,
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        )
        load_job = self.client.load_table_from_dataframe(
            dataframe=new_df,
            destination=self.full_table_id,
            job_config=job_config,
            location=self.location,
        )
        load_job.result()

        logger.info(
            "BigQuery feature append completed | table=%s | rows=%s | skipped_existing=%s",
            self.full_table_id,
            len(new_df),
            len(df) - len(new_df),
        )
        return len(new_df)

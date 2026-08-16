from __future__ import annotations

from datetime import date, datetime
import os
from typing import Optional, Sequence

import pandas as pd
from dotenv import load_dotenv
from google.cloud import bigquery

from src.utils.logger import get_logger


load_dotenv()

logger = get_logger(__name__)


class BigQueryFeatureStore:
    """
    Read and append AQI observations/features in the BigQuery repository.

    The existing project table may contain its time key as a native
    TIMESTAMP/DATETIME/DATE or as an integer epoch value. This adapter
    detects the actual BigQuery schema and converts query parameters,
    read values, and append values consistently.

    This keeps the append path compatible with BigQuery Sandbox by using
    load jobs with WRITE_APPEND rather than SQL DML.
    """

    _INTEGER_TYPES = {"INT64", "INTEGER"}
    _SUPPORTED_TIME_TYPES = {
        "TIMESTAMP",
        "DATETIME",
        "DATE",
        "INT64",
        "INTEGER",
    }

    def __init__(
        self,
        project_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
        table_id: Optional[str] = None,
        location: Optional[str] = None,
    ) -> None:
        self.project_id = project_id or os.getenv("GCP_PROJECT_ID")
        self.dataset_id = dataset_id or os.getenv(
            "BIGQUERY_DATASET_ID",
            "aqi_feature_store",
        )
        self.table_id = table_id or os.getenv(
            "BIGQUERY_FEATURE_TABLE",
            "engineered_features",
        )
        self.location = location or os.getenv(
            "BIGQUERY_LOCATION",
            "asia-south1",
        )

        if not self.project_id:
            raise ValueError("GCP_PROJECT_ID is missing from .env")

        self.client = bigquery.Client(
            project=self.project_id,
            location=self.location,
        )

        # Cache inferred integer epoch unit so every operation in the same
        # process uses one consistent interpretation.
        self._integer_time_unit_cache: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Table metadata helpers
    # ------------------------------------------------------------------

    @property
    def full_table_id(self) -> str:
        return (
            f"{self.project_id}."
            f"{self.dataset_id}."
            f"{self.table_id}"
        )

    @staticmethod
    def _as_utc_timestamp(
        value: datetime | str | pd.Timestamp,
    ) -> pd.Timestamp:
        timestamp = pd.Timestamp(value)

        if pd.isna(timestamp):
            raise ValueError(
                f"Invalid timestamp value: {value!r}"
            )

        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")

        return timestamp

    @classmethod
    def _as_utc_datetime(
        cls,
        value: datetime | str | pd.Timestamp,
    ) -> datetime:
        return cls._as_utc_timestamp(
            value
        ).to_pydatetime()

    @staticmethod
    def _time_column(
        table_columns: Sequence[str],
    ) -> str:
        """
        Prefer event_hour when the table has it; otherwise use timestamp.
        """

        if "event_hour" in table_columns:
            return "event_hour"

        if "timestamp" in table_columns:
            return "timestamp"

        raise ValueError(
            "BigQuery table has neither 'event_hour' nor 'timestamp'."
        )

    @staticmethod
    def _select_clause(
        columns: Sequence[str] | None,
        table_columns: Sequence[str],
    ) -> str:
        if columns is None:
            return "*"

        if not columns:
            raise ValueError(
                "columns must contain at least one column name."
            )

        unknown = (
            set(columns)
            - set(table_columns)
        )

        if unknown:
            raise ValueError(
                "Unknown BigQuery columns requested: "
                f"{sorted(unknown)}"
            )

        return ", ".join(
            f"`{column}`"
            for column in columns
        )

    @staticmethod
    def _schema_field(
        table: bigquery.Table,
        column: str,
    ) -> bigquery.SchemaField:
        for field in table.schema:
            if field.name == column:
                return field

        raise ValueError(
            f"Column '{column}' is not present in BigQuery schema."
        )

    # ------------------------------------------------------------------
    # Integer time compatibility
    # ------------------------------------------------------------------

    def _infer_integer_time_unit(
        self,
        time_column: str,
    ) -> str:
        """
        Infer epoch unit for legacy INT64 time columns by magnitude.

        Typical modern epoch magnitudes:
            seconds      ~ 1e9
            milliseconds ~ 1e12
            microseconds ~ 1e15
            nanoseconds  ~ 1e18
        """

        cached = self._integer_time_unit_cache.get(
            time_column
        )

        if cached:
            return cached

        query = f"""
        SELECT ABS(CAST(`{time_column}` AS INT64)) AS time_value
        FROM `{self.full_table_id}`
        WHERE `{time_column}` IS NOT NULL
        LIMIT 1
        """

        rows = list(
            self.client.query(
                query,
                location=self.location,
            ).result()
        )

        if not rows:
            raise ValueError(
                "Cannot infer integer timestamp unit because "
                f"`{time_column}` contains no non-null values."
            )

        magnitude = abs(
            int(rows[0]["time_value"])
        )

        if magnitude >= 100_000_000_000_000_000:
            unit = "ns"
        elif magnitude >= 100_000_000_000_000:
            unit = "us"
        elif magnitude >= 100_000_000_000:
            unit = "ms"
        else:
            unit = "s"

        self._integer_time_unit_cache[
            time_column
        ] = unit

        logger.info(
            "Detected BigQuery integer time representation | "
            "column=%s | unit=%s | sample_magnitude=%s",
            time_column,
            unit,
            magnitude,
        )

        return unit

    @staticmethod
    def _timestamp_to_epoch_integer(
        timestamp: pd.Timestamp,
        unit: str,
    ) -> int:
        """
        Convert a UTC pandas Timestamp to the inferred integer epoch unit.
        """

        timestamp = BigQueryFeatureStore._as_utc_timestamp(
            timestamp
        )

        nanoseconds = int(
            timestamp.value
        )

        divisors = {
            "ns": 1,
            "us": 1_000,
            "ms": 1_000_000,
            "s": 1_000_000_000,
        }

        try:
            divisor = divisors[unit]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported integer epoch unit: {unit}"
            ) from exc

        return nanoseconds // divisor

    def _time_query_parameter(
        self,
        *,
        parameter_name: str,
        value: datetime | str | pd.Timestamp,
        field_type: str,
        time_column: str,
    ) -> bigquery.ScalarQueryParameter:
        """
        Build a BigQuery parameter with the SAME type as the table key.
        """

        field_type = field_type.upper()

        if field_type == "TIMESTAMP":
            return bigquery.ScalarQueryParameter(
                parameter_name,
                "TIMESTAMP",
                self._as_utc_datetime(value),
            )

        if field_type == "DATETIME":
            utc_dt = self._as_utc_datetime(
                value
            ).replace(
                tzinfo=None
            )

            return bigquery.ScalarQueryParameter(
                parameter_name,
                "DATETIME",
                utc_dt,
            )

        if field_type == "DATE":
            utc_date: date = self._as_utc_timestamp(
                value
            ).date()

            return bigquery.ScalarQueryParameter(
                parameter_name,
                "DATE",
                utc_date,
            )

        if field_type in self._INTEGER_TYPES:
            unit = self._infer_integer_time_unit(
                time_column
            )

            epoch_value = (
                self._timestamp_to_epoch_integer(
                    self._as_utc_timestamp(value),
                    unit,
                )
            )

            return bigquery.ScalarQueryParameter(
                parameter_name,
                "INT64",
                epoch_value,
            )

        raise TypeError(
            "Unsupported BigQuery time-column type "
            f"for '{time_column}': {field_type}. "
            f"Supported types={sorted(self._SUPPORTED_TIME_TYPES)}"
        )

    def _normalise_time_series(
        self,
        series: pd.Series,
        *,
        field_type: str,
        time_column: str,
    ) -> pd.Series:
        """
        Convert the table's physical time representation to UTC datetime.
        """

        field_type = field_type.upper()

        if field_type in self._INTEGER_TYPES:
            unit = self._infer_integer_time_unit(
                time_column
            )

            return pd.to_datetime(
                series,
                unit=unit,
                utc=True,
                errors="coerce",
            )

        return pd.to_datetime(
            series,
            utc=True,
            errors="coerce",
        )

    def _prepare_time_for_write(
        self,
        series: pd.Series,
        *,
        field_type: str,
        time_column: str,
    ) -> pd.Series:
        """
        Convert canonical UTC timestamps back to the table's physical type.
        """

        timestamps = pd.to_datetime(
            series,
            utc=True,
            errors="coerce",
        )

        if timestamps.isna().any():
            raise ValueError(
                f"Cannot write invalid/null values to time column '{time_column}'."
            )

        field_type = field_type.upper()

        if field_type == "TIMESTAMP":
            return timestamps

        if field_type == "DATETIME":
            return timestamps.dt.tz_convert(
                "UTC"
            ).dt.tz_localize(
                None
            )

        if field_type == "DATE":
            return timestamps.dt.date

        if field_type in self._INTEGER_TYPES:
            unit = self._infer_integer_time_unit(
                time_column
            )

            return timestamps.map(
                lambda value: self._timestamp_to_epoch_integer(
                    pd.Timestamp(value),
                    unit,
                )
            ).astype("int64")

        raise TypeError(
            "Unsupported BigQuery time-column type "
            f"for write '{time_column}': {field_type}"
        )

    # ------------------------------------------------------------------
    # Connectivity
    # ------------------------------------------------------------------

    def test_connection(self) -> int:
        """Return total table rows to confirm the connection."""

        query = f"""
        SELECT COUNT(*) AS total_rows
        FROM `{self.full_table_id}`
        """

        result = self.client.query(
            query,
            location=self.location,
        ).result()

        row = next(
            iter(result)
        )

        return int(
            row.total_rows
        )

    # ------------------------------------------------------------------
    # Training reads
    # ------------------------------------------------------------------

    def get_training_features(
        self,
        columns: Sequence[str] | None = None,
        *,
        start_time: datetime | str | pd.Timestamp | None = None,
        end_time: datetime | str | pd.Timestamp | None = None,
        cities: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """
        Load historical features, optionally bounded by time/city/schema.

        Returned time keys are normalized to timezone-aware UTC timestamps
        even if BigQuery physically stores them as INT64 epoch values.
        """

        table = self.client.get_table(
            self.full_table_id
        )

        table_columns = [
            field.name
            for field in table.schema
        ]

        time_column = self._time_column(
            table_columns
        )

        time_field = self._schema_field(
            table,
            time_column,
        )

        field_type = (
            time_field.field_type.upper()
        )

        requested_columns = (
            list(columns)
            if columns is not None
            else None
        )

        select_clause = self._select_clause(
            requested_columns,
            table_columns,
        )

        predicates: list[str] = []
        parameters: list[
            bigquery.QueryParameter
        ] = []

        if start_time is not None:
            predicates.append(
                f"`{time_column}` >= @start_time"
            )

            parameters.append(
                self._time_query_parameter(
                    parameter_name="start_time",
                    value=start_time,
                    field_type=field_type,
                    time_column=time_column,
                )
            )

        if end_time is not None:
            predicates.append(
                f"`{time_column}` <= @end_time"
            )

            parameters.append(
                self._time_query_parameter(
                    parameter_name="end_time",
                    value=end_time,
                    field_type=field_type,
                    time_column=time_column,
                )
            )

        if cities:
            normalized_cities = sorted(
                {
                    str(city).strip().lower()
                    for city in cities
                    if str(city).strip()
                }
            )

            predicates.append(
                "LOWER(TRIM(city)) IN UNNEST(@cities)"
            )

            parameters.append(
                bigquery.ArrayQueryParameter(
                    "cities",
                    "STRING",
                    normalized_cities,
                )
            )

        where_clause = (
            f"WHERE {' AND '.join(predicates)}"
            if predicates
            else ""
        )

        query = f"""
        SELECT {select_clause}
        FROM `{self.full_table_id}`
        {where_clause}
        ORDER BY `{time_column}` ASC
        """

        job_config = (
            bigquery.QueryJobConfig(
                query_parameters=parameters
            )
            if parameters
            else None
        )

        dataframe = self.client.query(
            query,
            job_config=job_config,
            location=self.location,
        ).to_dataframe()

        if dataframe.empty:
            raise ValueError(
                "BigQuery training dataset is empty."
            )

        if time_column in dataframe.columns:
            dataframe[time_column] = (
                self._normalise_time_series(
                    dataframe[time_column],
                    field_type=field_type,
                    time_column=time_column,
                )
            )

        if requested_columns is not None:
            dataframe = dataframe[
                requested_columns
            ]

        return dataframe

    # ------------------------------------------------------------------
    # Recent context
    # ------------------------------------------------------------------

    def get_latest_context(
        self,
        city: str,
        rows: int = 72,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """
        Load latest rows for one city and normalize the physical time key
        to UTC datetime.
        """

        if rows <= 0:
            raise ValueError(
                "rows must be greater than zero."
            )

        table = self.client.get_table(
            self.full_table_id
        )

        table_columns = [
            field.name
            for field in table.schema
        ]

        time_column = self._time_column(
            table_columns
        )

        time_field = self._schema_field(
            table,
            time_column,
        )

        field_type = (
            time_field.field_type.upper()
        )

        requested_columns = (
            list(columns)
            if columns is not None
            else None
        )

        select_clause = self._select_clause(
            requested_columns,
            table_columns,
        )

        query = f"""
        SELECT {select_clause}
        FROM `{self.full_table_id}`
        WHERE LOWER(TRIM(city)) = LOWER(TRIM(@city))
        ORDER BY `{time_column}` DESC
        LIMIT @row_limit
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter(
                    "city",
                    "STRING",
                    city,
                ),
                bigquery.ScalarQueryParameter(
                    "row_limit",
                    "INT64",
                    rows,
                ),
            ]
        )

        dataframe = self.client.query(
            query,
            job_config=job_config,
            location=self.location,
        ).to_dataframe()

        if dataframe.empty:
            raise ValueError(
                f"No BigQuery records found for city: {city}"
            )

        if time_column in dataframe.columns:
            dataframe[time_column] = (
                self._normalise_time_series(
                    dataframe[time_column],
                    field_type=field_type,
                    time_column=time_column,
                )
            )

            dataframe = (
                dataframe
                .sort_values(
                    time_column
                )
                .reset_index(
                    drop=True
                )
            )
        else:
            dataframe = (
                dataframe
                .iloc[::-1]
                .reset_index(
                    drop=True
                )
            )

        return dataframe

    # ------------------------------------------------------------------
    # Idempotency lookup
    # ------------------------------------------------------------------

    def _get_existing_keys(
        self,
        *,
        cities: Sequence[str],
        start_hour: pd.Timestamp,
        end_hour: pd.Timestamp,
        time_column: str,
        field_type: str,
    ) -> set[
        tuple[
            str,
            pd.Timestamp,
        ]
    ]:
        """
        Read existing logical city/hour keys only inside the incoming
        time window.

        The BETWEEN parameters are deliberately built with the same
        physical type as the BigQuery time column.
        """

        query = f"""
        SELECT
            LOWER(TRIM(city)) AS city_key,
            `{time_column}` AS event_hour
        FROM `{self.full_table_id}`
        WHERE LOWER(TRIM(city)) IN UNNEST(@cities)
          AND `{time_column}` BETWEEN @start_hour AND @end_hour
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter(
                    "cities",
                    "STRING",
                    list(cities),
                ),
                self._time_query_parameter(
                    parameter_name="start_hour",
                    value=start_hour,
                    field_type=field_type,
                    time_column=time_column,
                ),
                self._time_query_parameter(
                    parameter_name="end_hour",
                    value=end_hour,
                    field_type=field_type,
                    time_column=time_column,
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

        existing["event_hour"] = (
            self._normalise_time_series(
                existing["event_hour"],
                field_type=field_type,
                time_column=time_column,
            )
            .dt.floor("h")
        )

        existing = existing.dropna(
            subset=[
                "city_key",
                "event_hour",
            ]
        )

        return set(
            zip(
                existing["city_key"],
                existing["event_hour"],
            )
        )

    # ------------------------------------------------------------------
    # Sandbox-compatible append
    # ------------------------------------------------------------------

    def append_features(
        self,
        dataframe: pd.DataFrame,
    ) -> int:
        """
        Append only new canonical city/hour rows using a BigQuery load job.

        No MERGE/INSERT/UPDATE/DELETE is used, preserving BigQuery Sandbox
        compatibility.
        """

        if dataframe.empty:
            raise ValueError(
                "Cannot upload an empty feature DataFrame."
            )

        target_table = self.client.get_table(
            self.full_table_id
        )

        target_columns = [
            field.name
            for field in target_table.schema
        ]

        time_column = self._time_column(
            target_columns
        )

        time_field = self._schema_field(
            target_table,
            time_column,
        )

        time_field_type = (
            time_field.field_type.upper()
        )

        if (
            time_field_type
            not in self._SUPPORTED_TIME_TYPES
        ):
            raise TypeError(
                "Unsupported BigQuery time-column schema | "
                f"column={time_column} | type={time_field_type}"
            )

        df = dataframe.copy()

        if "city" not in df.columns:
            raise ValueError(
                "Required column is missing: city"
            )

        if (
            "event_hour" not in df.columns
            and "timestamp" not in df.columns
        ):
            raise ValueError(
                "Required timestamp/event_hour column is missing."
            )

        df = df.dropna(
            subset=[
                "city",
            ]
        )

        df["city"] = (
            df["city"]
            .astype(str)
            .str.strip()
        )

        df["_city_key"] = (
            df["city"]
            .str.lower()
        )

        event_hours = (
            pd.to_datetime(
                df["event_hour"],
                utc=True,
                errors="coerce",
            )
            if "event_hour" in df.columns
            else pd.Series(
                pd.NaT,
                index=df.index,
                dtype="datetime64[ns, UTC]",
            )
        )

        if "timestamp" in df.columns:
            fallback_timestamps = pd.to_datetime(
                df["timestamp"],
                utc=True,
                errors="coerce",
            )

            event_hours = event_hours.fillna(
                fallback_timestamps
            )

        df["_event_hour"] = (
            event_hours
            .dt.floor("h")
        )

        df = df[
            df["_city_key"].ne("")
            & df["_event_hour"].notna()
        ].copy()

        df = df.drop_duplicates(
            subset=[
                "_city_key",
                "_event_hour",
            ],
            keep="last",
        )

        if df.empty:
            raise ValueError(
                "No valid feature rows remained after validation."
            )

        incoming_cities = sorted(
            df["_city_key"]
            .unique()
            .tolist()
        )

        start_hour = (
            df["_event_hour"]
            .min()
        )

        end_hour = (
            df["_event_hour"]
            .max()
        )

        existing_keys = self._get_existing_keys(
            cities=incoming_cities,
            start_hour=start_hour,
            end_hour=end_hour,
            time_column=time_column,
            field_type=time_field_type,
        )

        incoming_keys = list(
            zip(
                df["_city_key"],
                df["_event_hour"],
            )
        )

        is_new = [
            key not in existing_keys
            for key in incoming_keys
        ]

        new_df = (
            df.loc[
                is_new
            ]
            .copy()
        )

        if new_df.empty:
            logger.info(
                "BigQuery append skipped; all %d incoming "
                "city/event-hour key(s) already exist.",
                len(df),
            )

            return 0

        # --------------------------------------------------------------
        # Convert canonical time to the table's ACTUAL schema type.
        # --------------------------------------------------------------

        if "timestamp" in target_columns:
            timestamp_field = self._schema_field(
                target_table,
                "timestamp",
            )

            new_df["timestamp"] = (
                self._prepare_time_for_write(
                    new_df["_event_hour"],
                    field_type=timestamp_field.field_type,
                    time_column="timestamp",
                )
            )

        if "event_hour" in target_columns:
            event_hour_field = self._schema_field(
                target_table,
                "event_hour",
            )

            new_df["event_hour"] = (
                self._prepare_time_for_write(
                    new_df["_event_hour"],
                    field_type=event_hour_field.field_type,
                    time_column="event_hour",
                )
            )

        missing_target_columns = (
            set(target_columns)
            - set(new_df.columns)
        )

        if missing_target_columns:
            raise ValueError(
                "Incoming DataFrame is missing BigQuery columns: "
                f"{sorted(missing_target_columns)}"
            )

        # Exact target schema/order only. Internal idempotency columns are
        # intentionally removed here.
        new_df = new_df[
            target_columns
        ]

        job_config = bigquery.LoadJobConfig(
            schema=target_table.schema,
            write_disposition=(
                bigquery.WriteDisposition.WRITE_APPEND
            ),
        )

        load_job = (
            self.client
            .load_table_from_dataframe(
                dataframe=new_df,
                destination=self.full_table_id,
                job_config=job_config,
                location=self.location,
            )
        )

        load_job.result()

        logger.info(
            "BigQuery feature append completed | "
            "table=%s | rows=%s | skipped_existing=%s | "
            "time_column=%s | time_type=%s",
            self.full_table_id,
            len(new_df),
            len(df) - len(new_df),
            time_column,
            time_field_type,
        )

        return len(
            new_df
        )
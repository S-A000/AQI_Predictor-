from __future__ import annotations

import os
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from google.cloud import bigquery

from src.utils.logger import get_logger

load_dotenv()

logger = get_logger(__name__)


class BigQueryFeatureStore:
    """Read AQI features from the BigQuery feature repository."""

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

    @property
    def full_table_id(self) -> str:
        return (
            f"{self.project_id}."
            f"{self.dataset_id}."
            f"{self.table_id}"
        )

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

        row = next(iter(result))
        return int(row.total_rows)

    def get_training_features(self) -> pd.DataFrame:
        """Load the complete historical training dataset."""

        query = f"""
        SELECT *
        FROM `{self.full_table_id}`
        ORDER BY timestamp ASC
        """

        dataframe = self.client.query(
            query,
            location=self.location,
        ).to_dataframe()

        if dataframe.empty:
            raise ValueError("BigQuery training dataset is empty.")

        return dataframe

    def get_latest_context(
        self,
        city: str,
        rows: int = 72,
    ) -> pd.DataFrame:
        """Load the latest feature rows for one city."""

        if rows <= 0:
            raise ValueError("rows must be greater than zero.")

        query = f"""
        SELECT *
        FROM `{self.full_table_id}`
        WHERE LOWER(city) = LOWER(@city)
        ORDER BY timestamp DESC
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

        return (
            dataframe
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

    def append_features(
        self,
        dataframe: pd.DataFrame,
    ) -> int:
        """
        Append new hourly feature rows using a BigQuery load job.

        This method works with BigQuery Sandbox because it does not use
        DML statements such as MERGE, UPDATE, INSERT, or DELETE.
        """

        if dataframe.empty:
            raise ValueError(
                "Cannot upload an empty feature DataFrame."
            )

        df = dataframe.copy()

        required_columns = {"city", "timestamp"}
        missing_columns = required_columns - set(df.columns)

        if missing_columns:
            raise ValueError(
                "Required columns are missing: "
                f"{sorted(missing_columns)}"
            )

        df["city"] = df["city"].astype(str).str.strip()

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            utc=True,
            errors="coerce",
        )

        df = df.dropna(
            subset=["city", "timestamp"]
        )

        df = df.drop_duplicates(
            subset=["city", "timestamp"],
            keep="last",
        )

        if df.empty:
            raise ValueError(
                "No valid feature rows remained after validation."
            )

        target_table = self.client.get_table(
            self.full_table_id
        )

        target_columns = [
            field.name
            for field in target_table.schema
        ]

        missing_target_columns = (
            set(target_columns) - set(df.columns)
        )

        if missing_target_columns:
            raise ValueError(
                "Incoming DataFrame is missing BigQuery columns: "
                f"{sorted(missing_target_columns)}"
            )

        # Keep exact BigQuery column order and discard unexpected columns.
        df = df[target_columns]

        job_config = bigquery.LoadJobConfig(
            schema=target_table.schema,
            write_disposition=(
                bigquery.WriteDisposition.WRITE_APPEND
            ),
        )

        load_job = self.client.load_table_from_dataframe(
            dataframe=df,
            destination=self.full_table_id,
            job_config=job_config,
            location=self.location,
        )

        load_job.result()

        logger.info(
            "BigQuery feature append completed | table=%s | rows=%s",
            self.full_table_id,
            len(df),
        )

        return len(df)
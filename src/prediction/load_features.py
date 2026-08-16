"""
load_features.py
================

Load recent RAW observations for online AQI inference.

This loader intentionally reads:

    data/training/training_dataset.parquet

and NOT:

    features_train.parquet
    features_val.parquet
    features_test.parquet

PredictionFeaturePipeline must receive raw observations so it can
rebuild lag/rolling/temporal/domain features using the exact same
feature-engineering pipeline used during training.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger


logger = get_logger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAINING_DIR = (
    PROJECT_ROOT
    / "data"
    / "training"
)


class FeatureLoader:
    """
    Load chronological raw observation context for prediction.

    When `per_city=True`, `num_rows` means rows PER CITY.

    Example:

        num_rows=168
        per_city=True

    with three supported cities can return up to:

        168 * 3 = 504 rows

    This is required because lag/rolling features are calculated
    independently within each city.
    """

    def __init__(
        self,
        data_dir: Path | str = TRAINING_DIR,
    ) -> None:
        self.data_dir = Path(data_dir)

    def load_latest_features(
        self,
        filename: str = "training_dataset.parquet",
        num_rows: int | None = None,
        timestamp_col: str = "timestamp",
        city_col: str = "city",
        *,
        per_city: bool = False,
    ) -> pd.DataFrame:
        """
        Load recent raw observations for online feature engineering.

        Parameters
        ----------
        filename:
            Raw observation parquet file.

        num_rows:
            Number of latest rows to retain.

            If per_city=False:
                number of rows globally.

            If per_city=True:
                number of rows for EACH city.

        timestamp_col:
            Timestamp column used for chronological ordering.

        city_col:
            City column used when per_city=True.

        per_city:
            Keep `num_rows` independently for every city.
        """

        file_path = (
            self.data_dir
            / filename
        )

        if not file_path.exists():
            error_message = (
                "Raw context feature file not found at: "
                f"{file_path}"
            )

            logger.error(
                error_message
            )

            raise FileNotFoundError(
                error_message
            )

        logger.info(
            "Loading latest raw context features from: %s",
            file_path,
        )

        df = pd.read_parquet(
            file_path
        )

        if df.empty:
            raise ValueError(
                "Raw prediction context dataset is empty."
            )

        # --------------------------------------------------
        # Timestamp validation
        # --------------------------------------------------

        if timestamp_col not in df.columns:
            raise ValueError(
                "Raw prediction context is missing required "
                f"timestamp column: {timestamp_col}"
            )

        df = df.copy()

        df[timestamp_col] = pd.to_datetime(
            df[timestamp_col],
            utc=True,
            errors="coerce",
        )

        invalid_timestamps = int(
            df[timestamp_col]
            .isna()
            .sum()
        )

        if invalid_timestamps:
            logger.warning(
                "Dropping %d context row(s) with invalid timestamps.",
                invalid_timestamps,
            )

            df = df.dropna(
                subset=[timestamp_col]
            )

        if df.empty:
            raise ValueError(
                "No valid timestamped context rows remain."
            )

        # Canonical hourly key.
        df[timestamp_col] = (
            df[timestamp_col]
            .dt.floor("h")
        )

        # --------------------------------------------------
        # City normalization
        # --------------------------------------------------

        if city_col in df.columns:
            df = df.dropna(
                subset=[city_col]
            )

            df[city_col] = (
                df[city_col]
                .astype(str)
                .str.strip()
                .str.title()
            )

            df = df[
                df[city_col].ne("")
            ].copy()

            # Prevent duplicate city/hour observations from influencing
            # rolling or lag feature calculations.
            df = df.drop_duplicates(
                subset=[
                    city_col,
                    timestamp_col,
                ],
                keep="last",
            )

        elif per_city:
            raise ValueError(
                "per_city=True requires city column "
                f"'{city_col}' in prediction context."
            )

        # --------------------------------------------------
        # Context selection
        # --------------------------------------------------

        if (
            per_city
            and num_rows is not None
            and num_rows > 0
        ):
            df = (
                df
                .sort_values(
                    [
                        city_col,
                        timestamp_col,
                    ]
                )
                .groupby(
                    city_col,
                    group_keys=False,
                )
                .tail(num_rows)
                .sort_values(
                    timestamp_col
                )
                .reset_index(
                    drop=True
                )
            )

        else:
            df = (
                df
                .sort_values(
                    timestamp_col
                )
            )

            if (
                num_rows is not None
                and num_rows > 0
            ):
                df = (
                    df
                    .tail(num_rows)
                )

            df = df.reset_index(
                drop=True
            )

        logger.info(
            "Successfully loaded %d raw context record(s)%s.",
            len(df),
            (
                f" ({num_rows} maximum per city)"
                if per_city and num_rows
                else ""
            ),
        )

        return df
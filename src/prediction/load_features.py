"""
Suggested path: src/prediction/load_features.py

SINGLE RESPONSIBILITY: Fetch, filter, and load live/latest feature records
for batch inference and forecasting pipelines.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Mirrors build_features.py's own path resolution (PROJECT_ROOT via
# __file__, not src.utils.constants.PROCESSED_DATA_DIR) so this always
# points at the SAME data/training/ folder build_features.py reads
# its raw input from and writes its scaled outputs to — regardless of
# whatever PROCESSED_DATA_DIR happens to resolve to elsewhere.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = PROJECT_ROOT / "data" / "training"


class FeatureLoader:
    """
    Feature Loader engine responsible for preparing context/history
    records for online (real-time) inference and forecasting.

    IMPORTANT: this loads RAW, pre-feature-engineering observations
    (build_features.py's own input — training_dataset.parquet), NOT
    the scaled features_*.parquet outputs. PredictionFeaturePipeline
    concatenates this context with a new raw payload and runs the
    FULL feature engineering + scaling chain on the combination
    (see feature_pipeline.py) — feeding it already-scaled data here
    would double-scale/corrupt that computation, and scaled physical
    values (e.g. negative humidity) also fail PredictionPayload's
    validation if ever routed there directly.
    """

    def __init__(self, data_dir: Path | str = TRAINING_DIR) -> None:
        self.data_dir = Path(data_dir)

    def load_latest_features(
        self,
        filename: str = "training_dataset.parquet",
        num_rows: int | None = None,
        timestamp_col: str = "timestamp",
    ) -> pd.DataFrame:
        """
        Loads the most recent raw observation rows for use as
        lag/rolling context in online feature engineering.
        """
        file_path = self.data_dir / filename
        if not file_path.exists():
            error_msg = f"Raw context feature file not found at: {file_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        logger.info("Loading latest raw context features from: %s", file_path)
        df = pd.read_parquet(file_path)

        if timestamp_col in df.columns:
            df = df.sort_values(timestamp_col)
        else:
            logger.warning(
                "'%s' column not found — cannot guarantee chronological order; "
                "'latest' rows may not actually be the most recent.",
                timestamp_col,
            )

        if num_rows and num_rows > 0:
            # .tail(), not .head(): we want the MOST RECENT rows for
            # lag/rolling context, not the oldest ones in the file.
            df = df.tail(num_rows)

        logger.info("Successfully loaded %d feature record(s).", len(df))
        return df
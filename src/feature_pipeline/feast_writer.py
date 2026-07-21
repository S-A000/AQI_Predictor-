"""
feast_writer.py
================
Suggested path: src/feature_pipeline/feast_writer.py

The single append/upsert point that BOTH the live pipeline
(run_pipeline.py) and the historical backfill (historical_backfill.py)
call after `engineer_features()`. Writes into ONE stable parquet path
that Feast's `data_source.py` always points to — this is what lets
historical (bulk, one-time) and live (small, hourly) writes accumulate
into the same offline store instead of one overwriting the other.

Dedup strategy: keyed on (city, timestamp). If the same city+hour is
written twice (e.g. a historical backfill re-run overlaps with live
data that already came in), the LATEST write wins — this keeps re-runs
idempotent/safe.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEAST_READY_DIR = PROJECT_ROOT / "data" / "feast_ready"
FEAST_READY_DIR.mkdir(parents=True, exist_ok=True)

FEAST_SOURCE_PATH = FEAST_READY_DIR / "aqi_features.parquet"


def write_to_feast_source(
    new_df: pd.DataFrame,
    *,
    key_cols: tuple[str, ...] = ("city", "timestamp"),
    path: Path = FEAST_SOURCE_PATH,
) -> Path:
    """
    Append `new_df` (output of `engineer_features()`) into the
    consolidated Feast-ready parquet file, de-duplicating on
    `key_cols` (latest write wins).

    Safe to call repeatedly from both the live hourly pipeline and
    occasional historical backfills.
    """
    new_df = new_df.copy()
    new_df["timestamp"] = pd.to_datetime(new_df["timestamp"], utc=True)

    if path.exists():
        existing_df = pd.read_parquet(path)
        existing_df["timestamp"] = pd.to_datetime(existing_df["timestamp"], utc=True)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined = new_df

    before = len(combined)
    combined = combined.drop_duplicates(subset=list(key_cols), keep="last")
    after = len(combined)

    combined = combined.sort_values(list(key_cols)).reset_index(drop=True)
    combined.to_parquet(path, index=False)

    logger.info(
        "Wrote %d row(s) to Feast source (%d new, %d duplicate(s) resolved) -> %s",
        after, len(new_df), before - after, path,
    )
    return path
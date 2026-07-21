"""
feature_engineering.py
=======================
Suggested path: src/feature_pipeline/feature_engineering.py

Shared feature-engineering logic. Called by BOTH:
    - the live hourly pipeline (run_pipeline.py, after storage.save_batch)
    - the historical backfill (historical_backfill.py)

This is deliberately the ONLY place that computes derived features
(time-based + AQI change rate + anything added later). Keeping this
logic in one module — rather than duplicating it per source — is
what prevents train/serve skew between live (AQICN/OpenWeather) and
historical (Open-Meteo) data.

Input: any iterable of `MergedFeature` (from src.ingestion.merger).
Output: a pandas DataFrame, ready to be written to the Feast
FileSource (offline store) or pushed to the online store.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from src.ingestion.merger import FeatureMerger, MergedFeature
from src.utils.logger import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------
# Time-based features
# --------------------------------------------------------------

def add_time_features(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """
    Adds hour, day, month, day_of_week, is_weekend from `timestamp_col`.
    Non-destructive: returns a new DataFrame.
    """
    df = df.copy()
    ts = pd.to_datetime(df[timestamp_col], utc=True)

    df["hour"] = ts.dt.hour
    df["day"] = ts.dt.day
    df["month"] = ts.dt.month
    df["day_of_week"] = ts.dt.dayofweek          # 0 = Monday
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    return df


# --------------------------------------------------------------
# AQI change rate (and other lag/rolling features)
# --------------------------------------------------------------

def add_aqi_change_rate(
    df: pd.DataFrame,
    *,
    city_col: str = "city",
    timestamp_col: str = "timestamp",
    aqi_col: str = "aqi",
) -> pd.DataFrame:
    """
    Adds `aqi_change_rate`: difference between this row's AQI and the
    previous timestamp's AQI, computed PER CITY (so cities don't leak
    into each other's deltas). Also adds `aqi_rolling_mean_3h` as a
    simple smoothing feature — remove if not needed downstream.

    Assumes roughly-hourly cadence; if there are gaps, the "previous
    row" is just whatever the last recorded reading was (not
    necessarily exactly 1 hour prior) — this matches how the live
    hourly pipeline naturally behaves when a run is missed/retried.
    """
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)
    df = df.sort_values([city_col, timestamp_col])

    grouped = df.groupby(city_col)[aqi_col]
    df["aqi_change_rate"] = grouped.diff().fillna(0.0)
    df["aqi_rolling_mean_3h"] = (
        grouped.transform(lambda s: s.rolling(window=3, min_periods=1).mean())
    )

    return df


# --------------------------------------------------------------
# Entry point: MergedFeature list -> feature-store-ready DataFrame
# --------------------------------------------------------------

def engineer_features(features: Iterable[MergedFeature]) -> pd.DataFrame:
    """
    Full feature-engineering pass over a batch of MergedFeature
    records (from either the live pipeline or the historical
    backfill). Returns a DataFrame ready for Feast ingestion.

    NOTE: MergedFeature itself is left untouched (frozen, canonical,
    audit-trail-safe). All derived columns are added at the
    DataFrame stage, which is also why `visibility=-1` /
    `station_id=-1` sentinels from historical data flow through
    unchanged here — decide how to handle them (impute/drop) in
    Phase 6 Feature Selection, applied identically to both sources.
    """
    features = list(features)
    if not features:
        logger.warning("engineer_features called with an empty batch.")
        return pd.DataFrame()

    df = FeatureMerger.to_dataframe(features)  # reuse existing serializer
    df = add_time_features(df)
    df = add_aqi_change_rate(df)

    logger.info(
        "Engineered %d feature row(s) across %d city/cities.",
        len(df), df["city"].nunique(),
    )
    return df
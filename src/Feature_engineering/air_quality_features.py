"""
air_quality_features.py
=========================
Suggested path: src/feature_engineering/air_quality_features.py

Phase 3, Part 6 — Air Quality Features.

SINGLE RESPONSIBILITY: derive domain-specific AQI/pollutant features
— category labels, inter-pollutant ratios, dominant-pollutant
encoding, and a composite pollution index. Does not touch temporal/
lag/rolling/trend/interaction/spatial features — see sibling modules.

No groupby-by-city needed: all of these are row-wise combinations of
columns already present in the SAME row (same reasoning as
interaction_features.py) — no cross-row leakage risk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# US EPA AQI breakpoints (standard 6-category scale) — matches the
# `us_aqi` values this project already uses (historical_client.py
# requests `us_aqi` from Open-Meteo; AQICN's `aqi` field is on the
# same 0-500 US EPA scale).
AQI_CATEGORY_BINS = (-1, 50, 100, 150, 200, 300, 501)
AQI_CATEGORY_LABELS = (
    "Good", "Moderate", "Unhealthy_Sensitive", "Unhealthy", "Very_Unhealthy", "Hazardous",
)

POLLUTANT_COLUMNS = ("pm25", "pm10", "no2", "so2", "co", "o3")


class AirQualityFeatureEngineer:
    """
    Adds AQI category, pollutant ratios, dominant-pollutant one-hot
    encoding, and a composite pollution index.
    """

    def __init__(
        self,
        *,
        aqi_col: str = "aqi",
        dominant_pollutant_col: str = "dominant_pollutant",
        pollutant_cols: tuple[str, ...] = POLLUTANT_COLUMNS,
    ):
        self.aqi_col = aqi_col
        self.dominant_pollutant_col = dominant_pollutant_col
        self.pollutant_cols = pollutant_cols

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------

    def _has_columns(self, df: pd.DataFrame, *columns: str) -> bool:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            logger.warning("Column(s) not found, skipping dependent feature(s): %s", missing)
            return False
        return True

    # --------------------------------------------------
    # AQI category (standard US EPA 6-band scale)
    # --------------------------------------------------

    def add_aqi_category(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._has_columns(df, self.aqi_col):
            return df
        df = df.copy()
        # Coerce to numeric first: at prediction time 'aqi' is Optional
        # and often arrives as Python None (not NaN) — e.g. a
        # single-row payload where the caller doesn't know the
        # current AQI yet. An object-dtype Series containing raw
        # None crashes pd.cut's internal searchsorted comparison
        # against the (numeric) bin edges; a proper float NaN does
        # not — pd.cut just assigns it a NaN category, which is the
        # correct/expected behavior here.
        aqi_numeric = pd.to_numeric(df[self.aqi_col], errors="coerce")
        df["aqi_category"] = pd.cut(
            aqi_numeric, bins=AQI_CATEGORY_BINS, labels=AQI_CATEGORY_LABELS,
        )
        return df

    # --------------------------------------------------
    # Pollutant ratios (relative composition of the pollutant mix)
    # --------------------------------------------------

    def add_pollutant_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ratios capture the pollution SOURCE/COMPOSITION, not just
        magnitude — e.g. a high pm25/pm10 ratio points to combustion
        (vehicles, industry) rather than dust/construction, which
        tends to skew coarser (pm10-heavy). Epsilon-guarded against
        divide-by-zero.
        """
        df = df.copy()
        eps = 0.1

        if self._has_columns(df, "pm25", "pm10"):
            df["pm25_pm10_ratio"] = df["pm25"] / (df["pm10"] + eps)

        if self._has_columns(df, "no2", "so2"):
            df["no2_so2_ratio"] = df["no2"] / (df["so2"] + eps)

        if self._has_columns(df, "co", "no2"):
            df["co_no2_ratio"] = df["co"] / (df["no2"] + eps)

        if self._has_columns(df, "pm25", "o3"):
            df["pm25_o3_ratio"] = df["pm25"] / (df["o3"] + eps)

        return df

    # --------------------------------------------------
    # Dominant pollutant one-hot encoding
    # --------------------------------------------------

    def add_dominant_pollutant_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._has_columns(df, self.dominant_pollutant_col):
            return df
        df = df.copy()
        dummies = pd.get_dummies(
            df[self.dominant_pollutant_col], prefix="dominant", dtype=int,
        )
        df = pd.concat([df, dummies], axis=1)
        logger.info("Dominant pollutant categories encoded: %s", list(dummies.columns))
        return df

    # --------------------------------------------------
    # Composite pollution index
    # --------------------------------------------------

    def add_pollution_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        A simple weighted composite of the available pollutants,
        each min-max normalized (0-1) before weighting, so no single
        pollutant's raw scale (e.g. co's raw values run in the
        hundreds vs pm25 in tens) dominates the index just because
        of its units. Weights reflect established health-impact
        weighting used in composite air-quality indices (particulates
        weighted highest, matching WHO guidance emphasis on PM2.5/PM10
        as the primary health-risk drivers).
        """
        available = [c for c in self.pollutant_cols if c in df.columns]
        if not available:
            logger.warning("No pollutant columns found; skipping pollution_index.")
            return df

        df = df.copy()

        weights = {"pm25": 0.35, "pm10": 0.25, "no2": 0.15, "so2": 0.10, "co": 0.10, "o3": 0.05}

        normalized_weighted_sum = pd.Series(0.0, index=df.index)
        total_weight_used = 0.0

        for col in available:
            col_min, col_max = df[col].min(), df[col].max()
            span = col_max - col_min
            normalized = (df[col] - col_min) / span if span > 0 else 0.0
            weight = weights.get(col, 0.1)
            normalized_weighted_sum += normalized * weight
            total_weight_used += weight

        df["pollution_index"] = normalized_weighted_sum / total_weight_used if total_weight_used > 0 else 0.0
        return df

    # --------------------------------------------------
    # Full Part 6 pipeline
    # --------------------------------------------------

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        before_cols = df.shape[1]

        df = self.add_aqi_category(df)
        df = self.add_pollutant_ratios(df)
        df = self.add_dominant_pollutant_encoding(df)
        df = self.add_pollution_index(df)

        after_cols = df.shape[1]
        logger.info(
            "Air quality features added: %d new column(s) (%d -> %d).",
            after_cols - before_cols, before_cols, after_cols,
        )
        return df
"""
lag_features.py
=================
Suggested path: src/feature_engineering/lag_features.py

Phase 3, Part 2 — Lag Features.

SINGLE RESPONSIBILITY: create lagged versions of AQI/pollutant/
weather columns. Does not touch temporal/rolling/trend/interaction/
air-quality/spatial features — see sibling modules.

Lag windows below are NOT arbitrary — they were chosen from the
EDA's `lag_feature_analysis` / `future_target_correlation` results
(top-correlating lags with future AQI were 3, 6, 12, 24, 48, 72
hours). Lag 1 is included too since it's the strongest single
predictor for short-horizon (t+1) forecasts even though it didn't
show up in the top-15 (it's the trivial "last known value").

CRITICAL — must be computed PER CITY:
    Rows for Karachi/Lahore/Islamabad are interleaved in the
    dataset. A naive `.shift()` on the whole DataFrame would leak
    Lahore's AQI into Karachi's lag_1 feature. Every method here
    groups by `city` before shifting.
"""

from __future__ import annotations

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_AQI_LAGS = (1, 3, 6, 12, 24, 48, 72)
DEFAULT_POLLUTANT_LAGS = (1, 3, 6, 24, 72)
DEFAULT_WEATHER_LAGS = (1, 3, 6, 24)

POLLUTANT_COLUMNS = ("pm25", "pm10", "no2", "so2", "co", "o3")
WEATHER_COLUMNS = ("temperature", "humidity", "pressure", "wind_speed")


class LagFeatureEngineer:
    """
    Adds lagged features for AQI, pollutants, and weather — each
    lag computed within its own city group, sorted by timestamp,
    so no cross-city or out-of-order leakage is possible.
    """

    def __init__(
        self,
        *,
        city_col: str = "city",
        timestamp_col: str = "timestamp",
        aqi_col: str = "aqi",
        aqi_lags: tuple[int, ...] = DEFAULT_AQI_LAGS,
        pollutant_lags: tuple[int, ...] = DEFAULT_POLLUTANT_LAGS,
        weather_lags: tuple[int, ...] = DEFAULT_WEATHER_LAGS,
        pollutant_cols: tuple[str, ...] = POLLUTANT_COLUMNS,
        weather_cols: tuple[str, ...] = WEATHER_COLUMNS,
    ):
        self.city_col = city_col
        self.timestamp_col = timestamp_col
        self.aqi_col = aqi_col
        self.aqi_lags = aqi_lags
        self.pollutant_lags = pollutant_lags
        self.weather_lags = weather_lags
        self.pollutant_cols = pollutant_cols
        self.weather_cols = weather_cols

    # --------------------------------------------------
    # Internal helper
    # --------------------------------------------------

    def _add_lags_for_column(self, df: pd.DataFrame, column: str, lags: tuple[int, ...]) -> pd.DataFrame:
        if column not in df.columns:
            logger.warning("Column '%s' not found; skipping its lag features.", column)
            return df

        df = df.copy()
        grouped = df.groupby(self.city_col)[column]

        for lag in lags:
            df[f"{column}_lag_{lag}"] = grouped.shift(lag)

        return df

    def _ensure_sorted(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lag correctness depends entirely on row order within each
        city group — enforce it rather than assume the caller did."""
        return df.sort_values([self.city_col, self.timestamp_col]).reset_index(drop=True)

    # --------------------------------------------------
    # Public per-group methods
    # --------------------------------------------------

    def add_aqi_lags(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._add_lags_for_column(df, self.aqi_col, self.aqi_lags)

    def add_pollutant_lags(self, df: pd.DataFrame) -> pd.DataFrame:
        for column in self.pollutant_cols:
            df = self._add_lags_for_column(df, column, self.pollutant_lags)
        return df

    def add_weather_lags(self, df: pd.DataFrame) -> pd.DataFrame:
        for column in self.weather_cols:
            df = self._add_lags_for_column(df, column, self.weather_lags)
        return df

    # --------------------------------------------------
    # Full Part 2 pipeline
    # --------------------------------------------------

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._ensure_sorted(df)

        before_cols = df.shape[1]
        df = self.add_aqi_lags(df)
        df = self.add_pollutant_lags(df)
        df = self.add_weather_lags(df)
        after_cols = df.shape[1]

        logger.info(
            "Lag features added: %d new column(s) (%d -> %d).",
            after_cols - before_cols, before_cols, after_cols,
        )
        return df
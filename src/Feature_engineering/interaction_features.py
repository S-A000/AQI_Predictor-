"""
interaction_features.py
=========================
Suggested path: src/feature_engineering/interaction_features.py

Phase 3, Part 5 — Weather Interaction Features.

SINGLE RESPONSIBILITY: combine two raw weather/pollutant columns into
a single derived feature that captures a real physical relationship
neither column alone expresses. Does not touch temporal/lag/rolling/
trend/air-quality/spatial features — see sibling modules.

Why these specific interactions matter for AQI (not arbitrary):
    - Temp x Humidity:   muggy/humid heat traps particulates near
                          the ground (poor vertical mixing).
    - Wind x PM2.5:       wind DISPERSES particulates — this
                          interaction lets the model learn that high
                          PM2.5 with low wind is a very different
                          situation than high PM2.5 with high wind.
    - Pressure change:    falling pressure often precedes weather
                          fronts that trap or clear pollution;
                          rate-of-change is more informative than
                          the raw pressure reading.
    - Dew point:          a physically grounded humidity measure
                          (unlike relative humidity, it's not
                          temperature-relative) — classic input for
                          fog/haze formation, which directly affects
                          PM2.5 readings.
    - Heat index:         "feels like" from temp+humidity combined;
                          `feels_like` already exists in this dataset
                          from the weather API, so this is computed
                          as an independent cross-check / fallback
                          feature, not a duplicate — see note below.

No groupby-by-city needed here: unlike lag/rolling/trend features,
these are all ROW-WISE combinations of columns already present in
the SAME row — there's no cross-row (and therefore no cross-city)
leakage risk to guard against.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class InteractionFeatureEngineer:
    """
    Adds weather x weather and weather x pollutant interaction
    features, all computed row-wise from existing columns.
    """

    def __init__(
        self,
        *,
        city_col: str = "city",
        timestamp_col: str = "timestamp",
        temperature_col: str = "temperature",
        humidity_col: str = "humidity",
        pressure_col: str = "pressure",
        wind_speed_col: str = "wind_speed",
        pm25_col: str = "pm25",
    ):
        self.city_col = city_col
        self.timestamp_col = timestamp_col
        self.temperature_col = temperature_col
        self.humidity_col = humidity_col
        self.pressure_col = pressure_col
        self.wind_speed_col = wind_speed_col
        self.pm25_col = pm25_col

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------

    def _has_columns(self, df: pd.DataFrame, *columns: str) -> bool:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            logger.warning("Column(s) not found, skipping dependent feature(s): %s", missing)
            return False
        return True

    def _ensure_sorted(self, df: pd.DataFrame) -> pd.DataFrame:
        """Only pressure_change needs row order (previous-hour lookup);
        kept consistent with the other Phase 3 modules regardless."""
        return df.sort_values([self.city_col, self.timestamp_col]).reset_index(drop=True)

    # --------------------------------------------------
    # Temp x Humidity
    # --------------------------------------------------

    def add_temp_humidity_interaction(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._has_columns(df, self.temperature_col, self.humidity_col):
            return df
        df = df.copy()
        df["temp_humidity_interaction"] = df[self.temperature_col] * df[self.humidity_col]
        return df

    # --------------------------------------------------
    # Wind x PM2.5
    # --------------------------------------------------

    def add_wind_pm25_interaction(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._has_columns(df, self.wind_speed_col, self.pm25_col):
            return df
        df = df.copy()
        df["wind_pm25_interaction"] = df[self.wind_speed_col] * df[self.pm25_col]
        # Also add the inverse relationship explicitly: PM2.5 per unit
        # wind (dispersion-adjusted pollution level). Guard divide-by-zero
        # (calm/no-wind hours) by adding a small epsilon rather than
        # producing inf.
        df["pm25_per_wind"] = df[self.pm25_col] / (df[self.wind_speed_col] + 0.1)
        return df

    # --------------------------------------------------
    # Pressure change (hour-over-hour, per city)
    # --------------------------------------------------

    def add_pressure_change(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._has_columns(df, self.pressure_col):
            return df
        df = self._ensure_sorted(df)
        df["pressure_change_1h"] = df.groupby(self.city_col)[self.pressure_col].transform(
            lambda s: s.diff(1)
        )
        return df

    # --------------------------------------------------
    # Dew point (Magnus-Tetens approximation, standard meteorological formula)
    # --------------------------------------------------

    def add_dew_point(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._has_columns(df, self.temperature_col, self.humidity_col):
            return df
        df = df.copy()

        a, b = 17.27, 237.7  # Magnus-Tetens constants (standard, valid for 0-60C)
        temp = df[self.temperature_col]
        rh = df[self.humidity_col].clip(lower=1, upper=100)  # avoid log(0)

        alpha = (a * temp) / (b + temp) + np.log(rh / 100.0)
        df["dew_point"] = (b * alpha) / (a - alpha)
        return df

    # --------------------------------------------------
    # Heat index (Rothfusz regression, NOAA standard formula)
    # --------------------------------------------------

    def add_heat_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computed independently from the API-provided `feels_like`
        column as a cross-check feature — NOAA's heat index formula
        is Fahrenheit-based and only valid above ~26.7C (80F); below
        that it's set equal to the raw temperature (NOAA convention),
        since "heat index" isn't a meaningful concept in cool weather.
        """
        if not self._has_columns(df, self.temperature_col, self.humidity_col):
            return df
        df = df.copy()

        temp_c = df[self.temperature_col]
        rh = df[self.humidity_col].clip(lower=0, upper=100)
        temp_f = temp_c * 9 / 5 + 32

        hi_f = (
            -42.379 + 2.04901523 * temp_f + 10.14333127 * rh
            - 0.22475541 * temp_f * rh - 0.00683783 * temp_f**2
            - 0.05481717 * rh**2 + 0.00122874 * temp_f**2 * rh
            + 0.00085282 * temp_f * rh**2 - 0.00000199 * temp_f**2 * rh**2
        )
        hi_c = (hi_f - 32) * 5 / 9

        # Below ~26.7C the Rothfusz regression isn't valid/meaningful —
        # fall back to plain temperature there (NOAA convention).
        df["heat_index"] = np.where(temp_c >= 26.7, hi_c, temp_c)
        return df

    # --------------------------------------------------
    # Full Part 5 pipeline
    # --------------------------------------------------

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        before_cols = df.shape[1]

        df = self.add_temp_humidity_interaction(df)
        df = self.add_wind_pm25_interaction(df)
        df = self.add_pressure_change(df)
        df = self.add_dew_point(df)
        df = self.add_heat_index(df)

        after_cols = df.shape[1]
        logger.info(
            "Interaction features added: %d new column(s) (%d -> %d).",
            after_cols - before_cols, before_cols, after_cols,
        )
        return df
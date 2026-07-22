"""
temporal_features.py
====================
Suggested path: src/feature_engineering/temporal_features.py

Phase 3, Part 1 — Temporal Features.

SINGLE RESPONSIBILITY: extract time-based features from the timestamp
column. Captures cyclical patterns (hour-of-day, day-of-week, month,
season), calendar events (holidays, working days), and time-since
markers. These features help models learn daily, weekly, and seasonal
rhythms in air quality.

No groupby-by-city needed: all operations are derived row-wise from
the timestamp column. Per-city grouping is irrelevant here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# South Asian season mapping (Pakistan/India)
SEASON_MAP = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring",
    5: "Summer", 6: "Summer",
    7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
    10: "Autumn", 11: "Autumn",
}

# Major Pakistani holidays (simplified — extend with full calendar)
PAKISTAN_HOLIDAYS = {
    # Fixed-date holidays (month, day)
    (3, 23): "Pakistan Day",
    (8, 14): "Independence Day",
    (12, 25): "Quaid-e-Azam Day",
    # Islamic holidays vary by lunar calendar — placeholder
}


class TemporalFeatureEngineer:
    """
    Extracts temporal features from the timestamp column.

    Features produced:
        - Raw time components: year, month, day, hour, minute
        - Cyclical encodings: hour_sin/cos, day_sin/cos, month_sin/cos
        - Calendar features: day_of_week, day_of_year, week_of_year
        - Season: season label + season_sin/cos
        - Business time: is_weekend, is_working_day, is_holiday
        - Time markers: is_morning, is_afternoon, is_evening, is_night
        - Quarter and half-year markers
    """

    def __init__(
        self,
        *,
        time_col: str = "timestamp",
        city_col: str = "city",
        compute_cyclical: bool = True,
        compute_calendar: bool = True,
        compute_business_time: bool = True,
        compute_time_markers: bool = True,
        compute_quarter_markers: bool = True,
        holiday_calendar: dict | None = None,
    ):
        self.time_col = time_col
        self.city_col = city_col
        self.compute_cyclical = compute_cyclical
        self.compute_calendar = compute_calendar
        self.compute_business_time = compute_business_time
        self.compute_time_markers = compute_time_markers
        self.compute_quarter_markers = compute_quarter_markers
        self.holiday_calendar = holiday_calendar or PAKISTAN_HOLIDAYS

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------

    def _has_columns(self, df: pd.DataFrame, *columns: str) -> bool:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            logger.warning("Column(s) not found, skipping dependent feature(s): %s", missing)
            return False
        return True

    def _ensure_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure the time column is proper datetime."""
        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df[self.time_col]):
            df[self.time_col] = pd.to_datetime(df[self.time_col], utc=True, errors="coerce")
            logger.info("Converted '%s' to datetime", self.time_col)
        return df

    @staticmethod
    def _cyclical_encode(value: pd.Series, period: int) -> tuple[pd.Series, pd.Series]:
        """
        Encode a periodic feature using sine/cosine.
        Maps [0, period) → unit circle, preserving continuity across wrap-around.
        """
        radians = 2 * np.pi * value / period
        return np.sin(radians), np.cos(radians)

    # --------------------------------------------------
    # 1. Raw time components
    # --------------------------------------------------

    def add_time_components(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract basic time components: year, month, day, hour, minute."""
        if not self._has_columns(df, self.time_col):
            return df

        df = df.copy()
        dt = df[self.time_col].dt

        df["year"] = dt.year
        df["month"] = dt.month
        df["day"] = dt.day
        df["hour"] = dt.hour
        df["minute"] = dt.minute
        df["day_of_year"] = dt.dayofyear
        df["week_of_year"] = dt.isocalendar().week.astype(int)

        logger.info("Time components added: year, month, day, hour, minute, day_of_year, week_of_year")
        return df

    # --------------------------------------------------
    # 2. Cyclical encodings
    # --------------------------------------------------

    def add_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode periodic features as sine/cosine pairs.
        This preserves the cyclical nature of time (e.g. hour 23 is
        close to hour 0, not far away).
        """
        if not self.compute_cyclical:
            return df
        if not self._has_columns(df, self.time_col):
            return df

        df = df.copy()
        dt = df[self.time_col].dt

        # Hour (24-hour cycle)
        hour_sin, hour_cos = self._cyclical_encode(dt.hour, 24)
        df["hour_sin"] = hour_sin
        df["hour_cos"] = hour_cos

        # Day of week (7-day cycle)
        day_sin, day_cos = self._cyclical_encode(dt.dayofweek, 7)
        df["day_of_week_sin"] = day_sin
        df["day_of_week_cos"] = day_cos

        # Month (12-month cycle)
        month_sin, month_cos = self._cyclical_encode(dt.month, 12)
        df["month_sin"] = month_sin
        df["month_cos"] = month_cos

        # Day of year (365-day cycle)
        day_of_year_sin, day_of_year_cos = self._cyclical_encode(dt.dayofyear, 365)
        df["day_of_year_sin"] = day_of_year_sin
        df["day_of_year_cos"] = day_of_year_cos

        logger.info("Cyclical features added: hour_sin/cos, day_sin/cos, month_sin/cos, day_of_year_sin/cos")
        return df

    # --------------------------------------------------
    # 3. Calendar features
    # --------------------------------------------------

    def add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calendar features: day of week, is_month_start, is_month_end,
        quarter, days_in_month.
        """
        if not self.compute_calendar:
            return df
        if not self._has_columns(df, self.time_col):
            return df

        df = df.copy()
        dt = df[self.time_col].dt

        df["day_of_week"] = dt.dayofweek  # 0=Monday
        df["day_of_week_name"] = dt.day_name()
        df["quarter"] = dt.quarter
        df["is_month_start"] = dt.is_month_start.astype(int)
        df["is_month_end"] = dt.is_month_end.astype(int)
        df["days_in_month"] = dt.days_in_month

        logger.info("Calendar features added: day_of_week, quarter, is_month_start/end, days_in_month")
        return df

    # --------------------------------------------------
    # 4. Season features
    # --------------------------------------------------

    def add_season_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Season label and cyclical encoding for South Asian seasons:
        Winter, Spring, Summer, Monsoon, Autumn.
        """
        if not self._has_columns(df, self.time_col):
            return df

        df = df.copy()
        dt = df[self.time_col].dt

        # Season label
        df["season"] = dt.month.map(SEASON_MAP)

        # Season as numeric (for cyclical encoding)
        season_numeric = df["season"].map({
            "Winter": 0, "Spring": 1, "Summer": 2, "Monsoon": 3, "Autumn": 4,
        })

        # Cyclical encoding for 5-season cycle
        season_sin, season_cos = self._cyclical_encode(season_numeric, 5)
        df["season_sin"] = season_sin
        df["season_cos"] = season_cos

        logger.info("Season features added: season, season_sin/cos")
        return df

    # --------------------------------------------------
    # 5. Business time features
    # --------------------------------------------------

    def add_business_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Weekend, working day, and holiday flags.
        Working day = not weekend and not holiday.
        """
        if not self.compute_business_time:
            return df
        if not self._has_columns(df, self.time_col):
            return df

        df = df.copy()
        dt = df[self.time_col].dt

        # Weekend flag (Saturday=5, Sunday=6)
        df["is_weekend"] = dt.dayofweek.isin([5, 6]).astype(int)

        # Holiday flag
        df["is_holiday"] = df.apply(
            lambda row: 1 if (row[self.time_col].month, row[self.time_col].day) in self.holiday_calendar else 0,
            axis=1,
        )

        # Working day = not weekend AND not holiday
        df["is_working_day"] = ((df["is_weekend"] == 0) & (df["is_holiday"] == 0)).astype(int)

        # Days since last weekend (for proximity effect)
        df["days_since_weekend"] = dt.dayofweek.apply(
            lambda x: min(x, 6 - x) if x < 5 else 0
        )

        logger.info("Business time features added: is_weekend, is_holiday, is_working_day, days_since_weekend")
        return df

    # --------------------------------------------------
    # 6. Time-of-day markers
    # --------------------------------------------------

    def add_time_markers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Time-of-day buckets: morning, afternoon, evening, night.
        Also rush hour flags (typical South Asian traffic patterns).
        """
        if not self.compute_time_markers:
            return df
        if not self._has_columns(df, self.time_col):
            return df

        df = df.copy()
        hour = df[self.time_col].dt.hour

        # Time-of-day buckets
        df["is_morning"] = ((hour >= 5) & (hour < 12)).astype(int)      # 5am-12pm
        df["is_afternoon"] = ((hour >= 12) & (hour < 17)).astype(int)  # 12pm-5pm
        df["is_evening"] = ((hour >= 17) & (hour < 21)).astype(int)   # 5pm-9pm
        df["is_night"] = ((hour >= 21) | (hour < 5)).astype(int)      # 9pm-5am

        # Rush hours (typical Pakistan traffic)
        df["is_morning_rush"] = ((hour >= 8) & (hour <= 10)).astype(int)
        df["is_evening_rush"] = ((hour >= 17) & (hour <= 19)).astype(int)
        df["is_rush_hour"] = ((df["is_morning_rush"] == 1) | (df["is_evening_rush"] == 1)).astype(int)

        logger.info("Time markers added: morning/afternoon/evening/night, rush hour flags")
        return df

    # --------------------------------------------------
    # 7. Quarter and half-year markers
    # --------------------------------------------------

    def add_quarter_markers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Quarter and half-year indicators for longer-term patterns.
        """
        if not self.compute_quarter_markers:
            return df
        if not self._has_columns(df, self.time_col):
            return df

        df = df.copy()
        dt = df[self.time_col].dt

        df["is_q1"] = (dt.quarter == 1).astype(int)
        df["is_q2"] = (dt.quarter == 2).astype(int)
        df["is_q3"] = (dt.quarter == 3).astype(int)
        df["is_q4"] = (dt.quarter == 4).astype(int)

        df["is_first_half"] = (dt.quarter <= 2).astype(int)
        df["is_second_half"] = (dt.quarter > 2).astype(int)

        logger.info("Quarter markers added: is_q1/q2/q3/q4, first/second half")
        return df

    # --------------------------------------------------
    # Full Part 1 pipeline
    # --------------------------------------------------

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete temporal features pipeline:
            1. Ensure datetime format
            2. Extract raw time components
            3. Add cyclical encodings (sin/cos)
            4. Add calendar features
            5. Add season features (South Asian seasons)
            6. Add business time features (weekend, holiday, working day)
            7. Add time-of-day markers (morning/afternoon/evening/night)
            8. Add quarter and half-year markers
        """
        before_cols = df.shape[1]

        # Ensure datetime
        df = self._ensure_datetime(df)

        # Build all temporal features
        df = self.add_time_components(df)
        df = self.add_cyclical_features(df)
        df = self.add_calendar_features(df)
        df = self.add_season_features(df)
        df = self.add_business_time_features(df)
        df = self.add_time_markers(df)
        df = self.add_quarter_markers(df)

        after_cols = df.shape[1]
        logger.info(
            "Temporal features added: %d new column(s) (%d -> %d).",
            after_cols - before_cols, before_cols, after_cols,
        )
        return df
"""
trend_features.py
===================
Suggested path: src/feature_engineering/trend_features.py

Phase 3, Part 4 — Trend Features.

SINGLE RESPONSIBILITY: capture DIRECTION and SPEED of change for AQI
and pollutants (is it rising, falling, how fast). Does not touch
temporal/lag/rolling/interaction/air-quality/spatial features — see
sibling modules.

Four feature types, each answering a different question:
    - difference:        "how much did it change?" (absolute, units)
    - pct_change:         "how much did it change, relatively?" (%)
    - rate_of_change:      "how fast is it changing per hour?" (slope)
    - momentum:            "is the trend itself accelerating?"
                           (change-of-change — second derivative)

CRITICAL — same rules as lag_features.py / rolling_features.py:
    Computed PER CITY (groupby), sorted by timestamp. All of these
    are backward-looking (based on current vs N-hours-ago), so no
    future leakage — but cross-city leakage is just as real a risk
    here as anywhere else, hence the same groupby discipline.
"""

from __future__ import annotations

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_PERIODS = (1, 3, 6, 24)
DEFAULT_MOMENTUM_PERIOD = 3

AQI_COLUMN = "aqi"
POLLUTANT_COLUMNS = ("pm25", "pm10", "no2", "so2", "co", "o3")


class TrendFeatureEngineer:
    """
    Adds difference / percentage-change / rate-of-change / momentum
    features for AQI and pollutants, each computed within its own
    city group.
    """

    def __init__(
        self,
        *,
        city_col: str = "city",
        timestamp_col: str = "timestamp",
        periods: tuple[int, ...] = DEFAULT_PERIODS,
        momentum_period: int = DEFAULT_MOMENTUM_PERIOD,
        aqi_col: str = AQI_COLUMN,
        pollutant_cols: tuple[str, ...] = POLLUTANT_COLUMNS,
    ):
        self.city_col = city_col
        self.timestamp_col = timestamp_col
        self.periods = periods
        self.momentum_period = momentum_period
        self.aqi_col = aqi_col
        self.pollutant_cols = pollutant_cols

    # --------------------------------------------------
    # Internal helpers
    # --------------------------------------------------

    def _ensure_sorted(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.city_col in df.columns and self.timestamp_col in df.columns:
            return df.sort_values([self.city_col, self.timestamp_col]).reset_index(drop=True)
        elif self.timestamp_col in df.columns:
            return df.sort_values([self.timestamp_col]).reset_index(drop=True)
        return df

    def _columns_to_process(self, df: pd.DataFrame) -> list[str]:
        columns = [self.aqi_col, *self.pollutant_cols]
        available = [c for c in columns if c in df.columns]
        missing = set(columns) - set(available)
        if missing:
            logger.warning("Column(s) not found, skipping trend features for them: %s", sorted(missing))
        return available

    def _prep_numeric_series(self, df: pd.DataFrame, column: str) -> pd.Series:
        """Coerces column to numeric float series to avoid NoneType calculation crashes."""
        return pd.to_numeric(df[column], errors="coerce").astype(float)

    # --------------------------------------------------
    # Difference (absolute change over N hours)
    # --------------------------------------------------

    def add_difference(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for column in self._columns_to_process(df):
            series = self._prep_numeric_series(df, column)
            if self.city_col in df.columns:
                grouped = series.groupby(df[self.city_col])
                for period in self.periods:
                    df[f"{column}_diff_{period}"] = grouped.transform(lambda s, p=period: s.diff(p))
            else:
                for period in self.periods:
                    df[f"{column}_diff_{period}"] = series.diff(period)
        return df

    # --------------------------------------------------
    # Percentage change over N hours
    # --------------------------------------------------

    def add_pct_change(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for column in self._columns_to_process(df):
            series = self._prep_numeric_series(df, column)
            if self.city_col in df.columns:
                grouped = series.groupby(df[self.city_col])
                for period in self.periods:
                    pct = grouped.transform(lambda s, p=period: s.pct_change(p, fill_method=None))
                    df[f"{column}_pctchange_{period}"] = pct.replace([float("inf"), float("-inf")], pd.NA)
            else:
                for period in self.periods:
                    pct = series.pct_change(period)
                    df[f"{column}_pctchange_{period}"] = pct.replace([float("inf"), float("-inf")], pd.NA)
        return df

    # --------------------------------------------------
    # Rate of change (slope per hour = difference / period)
    # --------------------------------------------------

    def add_rate_of_change(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for column in self._columns_to_process(df):
            series = self._prep_numeric_series(df, column)
            if self.city_col in df.columns:
                grouped = series.groupby(df[self.city_col])
                for period in self.periods:
                    diff = grouped.transform(lambda s, p=period: s.diff(p))
                    df[f"{column}_roc_{period}"] = diff / period
            else:
                for period in self.periods:
                    diff = series.diff(period)
                    df[f"{column}_roc_{period}"] = diff / period
        return df

    # --------------------------------------------------
    # Momentum (change of the change — is the trend accelerating?)
    # --------------------------------------------------

    def add_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        period = self.momentum_period
        for column in self._columns_to_process(df):
            series = self._prep_numeric_series(df, column)
            if self.city_col in df.columns:
                grouped = series.groupby(df[self.city_col])
                first_diff = grouped.transform(lambda s, p=period: s.diff(p))
                df[f"{column}_momentum_{period}"] = first_diff.groupby(df[self.city_col]).transform(
                    lambda s, p=period: s.diff(p)
                )
            else:
                first_diff = series.diff(period)
                df[f"{column}_momentum_{period}"] = first_diff.diff(period)
        return df

    # --------------------------------------------------
    # Full Part 4 pipeline
    # --------------------------------------------------

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._ensure_sorted(df)

        before_cols = df.shape[1]
        df = self.add_difference(df)
        df = self.add_pct_change(df)
        df = self.add_rate_of_change(df)
        df = self.add_momentum(df)
        after_cols = df.shape[1]

        logger.info(
            "Trend features added: %d new column(s) (%d -> %d).",
            after_cols - before_cols, before_cols, after_cols,
        )
        return df
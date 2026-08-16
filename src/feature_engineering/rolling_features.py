
from __future__ import annotations

import gc
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Default rolling windows (in hours) — tuned for hourly air quality data
DEFAULT_WINDOWS = (6, 12, 24, 48, 168)  # 6h, 12h, 1d, 2d, 1w

# Default columns to compute rolling features on
DEFAULT_TARGET_COLS = ("aqi",)
DEFAULT_POLLUTANT_COLS = ("pm25", "pm10", "no2", "so2", "co", "o3")
DEFAULT_WEATHER_COLS = ("temperature", "humidity", "wind_speed", "pressure")


class RollingFeatureEngineer:
    """
    Computes rolling window statistics for time-series features.

    Supports:
        - Simple rolling: mean, std, min, max, median
        - Exponential Moving Average (EMA)
        - Rolling quantiles (25th, 75th percentile)
        - Rolling range (max - min)

    All operations are forward-looking-safe: window is strictly
    historical (center=False, closed="left" behavior via shift).
    """

    def __init__(
        self,
        *,
        windows: tuple[int, ...] = DEFAULT_WINDOWS,
        target_cols: tuple[str, ...] = DEFAULT_TARGET_COLS,
        pollutant_cols: tuple[str, ...] = DEFAULT_POLLUTANT_COLS,
        weather_cols: tuple[str, ...] = DEFAULT_WEATHER_COLS,
        time_col: str = "timestamp",
        city_col: str = "city",
        compute_ema: bool = True,
        compute_quantiles: bool = False,
        compute_range: bool = True,
        min_periods: int = 1,
    ):
        self.windows = windows
        self.target_cols = target_cols
        self.pollutant_cols = pollutant_cols
        self.weather_cols = weather_cols
        self.time_col = time_col
        self.city_col = city_col
        self.compute_ema = compute_ema
        self.compute_quantiles = compute_quantiles
        self.compute_range = compute_range
        self.min_periods = min_periods

        self.all_source_cols = list(
            set(self.target_cols + self.pollutant_cols + self.weather_cols)
        )

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------

    def _has_columns(self, df: pd.DataFrame, *columns: str) -> bool:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            logger.warning("Column(s) not found, skipping dependent feature(s): %s", missing)
            return False
        return True

    def _get_available_cols(self, df: pd.DataFrame) -> list[str]:
        """Return source columns that actually exist in the DataFrame."""
        return [c for c in self.all_source_cols if c in df.columns]

    def _validate_sorted(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure DataFrame is sorted by time (and city if present)."""
        sort_cols = [self.city_col, self.time_col] if self.city_col in df.columns else [self.time_col]
        if not df[sort_cols].equals(df[sort_cols].sort_values(by=sort_cols)):
            logger.warning("DataFrame not sorted by %s — sorting now", sort_cols)
            df = df.sort_values(by=sort_cols).reset_index(drop=True)
        return df

    def _downcast_floats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Downcast float64 to float32 to cut RAM usage by 50%."""
        float64_cols = df.select_dtypes(include=["float64"]).columns
        if len(float64_cols) > 0:
            df[float64_cols] = df[float64_cols].astype(np.float32)
        return df

    # --------------------------------------------------
    # Core rolling statistics (Downcasting added `.astype(np.float32)`)
    # --------------------------------------------------

    def add_rolling_mean(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        """Rolling mean over each window."""
        df = df.copy()
        for col in cols:
            for window in self.windows:
                new_col = f"{col}_rollmean_{window}h"
                df[new_col] = df[col].rolling(window=window, min_periods=self.min_periods).mean().astype(np.float32)
        logger.info("Rolling mean added for %d columns × %d windows", len(cols), len(self.windows))
        return df

    def add_rolling_std(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        """Rolling standard deviation — measures volatility."""
        df = df.copy()
        for col in cols:
            for window in self.windows:
                new_col = f"{col}_rollstd_{window}h"
                df[new_col] = df[col].rolling(window=window, min_periods=self.min_periods).std().astype(np.float32)
        logger.info("Rolling std added for %d columns × %d windows", len(cols), len(self.windows))
        return df

    def add_rolling_min(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        """Rolling minimum — captures lower bound of recent range."""
        df = df.copy()
        for col in cols:
            for window in self.windows:
                new_col = f"{col}_rollmin_{window}h"
                df[new_col] = df[col].rolling(window=window, min_periods=self.min_periods).min().astype(np.float32)
        return df

    def add_rolling_max(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        """Rolling maximum — captures upper bound of recent range."""
        df = df.copy()
        for col in cols:
            for window in self.windows:
                new_col = f"{col}_rollmax_{window}h"
                df[new_col] = df[col].rolling(window=window, min_periods=self.min_periods).max().astype(np.float32)
        return df

    def add_rolling_median(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        """Rolling median — robust to outliers vs mean."""
        df = df.copy()
        for col in cols:
            for window in self.windows:
                new_col = f"{col}_rollmedian_{window}h"
                df[new_col] = df[col].rolling(window=window, min_periods=self.min_periods).median().astype(np.float32)
        logger.info("Rolling median added for %d columns × %d windows", len(cols), len(self.windows))
        return df

    # --------------------------------------------------
    # Exponential Moving Average (EMA)
    # --------------------------------------------------

    def add_ema(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        if not self.compute_ema:
            return df

        df = df.copy()
        for col in cols:
            for window in self.windows:
                new_col = f"{col}_ema_{window}h"
                df[new_col] = df[col].ewm(span=window, adjust=False, min_periods=self.min_periods).mean().astype(np.float32)
        logger.info("EMA added for %d columns × %d windows", len(cols), len(self.windows))
        return df

    # --------------------------------------------------
    # Rolling quantiles
    # --------------------------------------------------

    def add_rolling_quantiles(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        if not self.compute_quantiles:
            return df

        df = df.copy()
        for col in cols:
            for window in self.windows:
                q25_col = f"{col}_rollq25_{window}h"
                q75_col = f"{col}_rollq75_{window}h"
                df[q25_col] = df[col].rolling(window=window, min_periods=self.min_periods).quantile(0.25).astype(np.float32)
                df[q75_col] = df[col].rolling(window=window, min_periods=self.min_periods).quantile(0.75).astype(np.float32)
        logger.info("Rolling quantiles added for %d columns × %d windows", len(cols), len(self.windows))
        return df

    # --------------------------------------------------
    # Rolling range
    # --------------------------------------------------

    def add_rolling_range(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        if not self.compute_range:
            return df

        df = df.copy()
        for col in cols:
            for window in self.windows:
                new_col = f"{col}_rollrange_{window}h"
                roll_max = df[col].rolling(window=window, min_periods=self.min_periods).max()
                roll_min = df[col].rolling(window=window, min_periods=self.min_periods).min()
                df[new_col] = (roll_max - roll_min).astype(np.float32)
        logger.info("Rolling range added for %d columns × %d windows", len(cols), len(self.windows))
        return df

    # --------------------------------------------------
    # Per-city rolling (Memory Optimized)
    # --------------------------------------------------

    def _apply_per_city(
        self,
        df: pd.DataFrame,
        func,
        cols: list[str],
    ) -> pd.DataFrame:
        if self.city_col not in df.columns:
            return func(df, cols)

        result_dfs = []
        for city, group in df.groupby(self.city_col, sort=False, observed=True):
            group = group.sort_values(by=self.time_col)
            group = func(group, cols)
            group = self._downcast_floats(group)
            result_dfs.append(group)

        # Clear old reference to free memory before concat
        del df
        gc.collect()

        out_df = pd.concat(result_dfs, ignore_index=True).sort_values(
            by=[self.city_col, self.time_col]
        ).reset_index(drop=True)

        del result_dfs
        gc.collect()
        return out_df

    # --------------------------------------------------
    # Full Part 3 pipeline
    # --------------------------------------------------

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        before_cols = df.shape[1]

        # Validate
        df = self._validate_sorted(df)
        df = self._downcast_floats(df)  # Ensure initial df is float32
        
        available_cols = self._get_available_cols(df)

        if not available_cols:
            logger.warning("No source columns found for rolling features")
            return df

        logger.info("Building rolling features for: %s", available_cols)

        # Core rolling stats
        df = self._apply_per_city(df, self.add_rolling_mean, available_cols)
        df = self._apply_per_city(df, self.add_rolling_std, available_cols)
        df = self._apply_per_city(df, self.add_rolling_min, available_cols)
        df = self._apply_per_city(df, self.add_rolling_max, available_cols)
        df = self._apply_per_city(df, self.add_rolling_median, available_cols)

        # Optional features
        df = self._apply_per_city(df, self.add_ema, available_cols)
        df = self._apply_per_city(df, self.add_rolling_quantiles, available_cols)
        df = self._apply_per_city(df, self.add_rolling_range, available_cols)

        after_cols = df.shape[1]
        logger.info(
            "Rolling features added: %d new column(s) (%d -> %d).",
            after_cols - before_cols, before_cols, after_cols,
        )
        return df   
"""
forecast_targets.py
=====================
Suggested path: src/training/forecast_targets.py

SINGLE RESPONSIBILITY: create the TARGET columns needed for direct
multi-horizon forecasting — "what will AQI be N hours from now" —
as opposed to the existing `aqi` column, which is "what is AQI right
now" (used for nowcasting/training features, not as a future target).

Why this exists:
    The 6 models (Ridge, RF, GradientBoosting, XGBoost, Prophet,
    LSTM) trained so far predict CURRENT aqi from CURRENT features —
    useful for evaluating model fit, but NOT directly usable for
    "predict the next 3 days" without either (a) future weather data
    + recursive step-by-step prediction, or (b) training the model
    to predict a FUTURE aqi value directly from CURRENT features.
    This module implements the data-prep side of approach (b): exact
    elapsed-time matching within each city so each row's target
    becomes "aqi N hours from THIS row's timestamp", not "aqi at
    this row's timestamp". The features stay as-is (still "what do
    we know right now") — only the target changes.

CRITICAL — per-city, chronologically sorted, same discipline as
lag_features.py / rolling_features.py:
    Shifting must never pull a future value from a DIFFERENT city's
    row. Every method here groups by city first.

CRITICAL — missing future event hours remain unusable:
    A row without an exact city observation 72 hours later has no
    target to match — its target is NaN by construction, not
    a data-quality bug. These rows must be dropped before training
    (this module flags them; dropping happens in the training
    pipeline, not here, to keep this module's responsibility to
    target-creation only).

Naming: target columns are named `target_aqi_t+{horizon}` (not
"future_aqi..." or anything matching the leakage-pattern substrings
`scaling_encoding.py` already auto-drops — "future"/"lead" — since
these ARE meant to survive as targets, not be treated as leaky
features). Make sure `target_col` passed to
ScalingEncodingEngineer/dataset.py excludes ALL of these except the
one horizon actually being trained for.
"""

from __future__ import annotations

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_HORIZONS_HOURS = (24, 48, 72)  # day 1, day 2, day 3


class ForecastTargetBuilder:
    """
    Adds one target column per forecast horizon: `target_aqi_t+24`,
    `target_aqi_t+48`, `target_aqi_t+72` by default — each row's
    value is that city's `aqi` reading N hours after this row's own
    timestamp.
    """

    def __init__(
        self,
        *,
        city_col: str = "city",
        timestamp_col: str = "timestamp",
        aqi_col: str = "aqi",
        horizons_hours: tuple[int, ...] = DEFAULT_HORIZONS_HOURS,
    ):
        self.city_col = city_col
        self.timestamp_col = timestamp_col
        self.aqi_col = aqi_col
        self.horizons_hours = horizons_hours

    def target_column_name(self, horizon_hours: int) -> str:
        return f"target_aqi_t_{horizon_hours}"

    def _ensure_sorted(self, df: pd.DataFrame) -> pd.DataFrame:
        required_columns = {self.city_col, self.timestamp_col, self.aqi_col}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(
                "Cannot build forecast targets; missing columns: "
                f"{sorted(missing_columns)}"
            )

        result = df.copy()
        result = result.dropna(subset=[self.city_col])
        result[self.city_col] = (
            result[self.city_col].astype(str).str.strip().str.title()
        )
        result = result[result[self.city_col].ne("")]
        result[self.timestamp_col] = pd.to_datetime(
            result[self.timestamp_col], utc=True, errors="coerce"
        )

        invalid_timestamps = int(result[self.timestamp_col].isna().sum())
        if invalid_timestamps:
            logger.warning(
                "Dropping %d row(s) with invalid timestamps before target matching.",
                invalid_timestamps,
            )
            result = result.dropna(subset=[self.timestamp_col])

        before_deduplication = len(result)
        result = result.drop_duplicates(
            subset=[self.city_col, self.timestamp_col], keep="last"
        )
        removed_duplicates = before_deduplication - len(result)
        if removed_duplicates:
            logger.warning(
                "Removed %d duplicate city/timestamp row(s) before target matching.",
                removed_duplicates,
            )

        return result.sort_values(
            [self.city_col, self.timestamp_col]
        ).reset_index(drop=True)

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds one target column per horizon. Rows near the end of
        each city's timeline, or rows followed by collection gaps,
        will have NaN targets for the larger horizons. Call
        `drop_unusable_rows()` per-horizon before training that
        horizon's model.
        """
        df = self._ensure_sorted(df)

        for horizon in self.horizons_hours:
            col_name = self.target_column_name(horizon)
            # Shift the future observation's timestamp backward by the
            # horizon, then require an exact same-city timestamp join.
            future_values = df[
                [self.city_col, self.timestamp_col, self.aqi_col]
            ].copy()
            future_values[self.timestamp_col] = (
                future_values[self.timestamp_col]
                - pd.Timedelta(hours=horizon)
            )
            future_values = future_values.rename(
                columns={self.aqi_col: col_name}
            )

            if col_name in df.columns:
                df = df.drop(columns=[col_name])

            df = df.merge(
                future_values,
                on=[self.city_col, self.timestamp_col],
                how="left",
                sort=False,
                validate="one_to_one",
            )

            usable = df[col_name].notna().sum()
            logger.info(
                "Created target '%s': %d/%d row(s) usable (rest are the "
                "rows without an exact +%dh timestamp, correctly NaN).",
                col_name, usable, len(df), horizon,
            )

        return df

    def drop_unusable_rows(self, df: pd.DataFrame, horizon_hours: int) -> pd.DataFrame:
        """
        Returns a copy with rows dropped where the target for this
        SPECIFIC horizon is NaN. Call this separately per horizon
        right before training that horizon's model — don't drop
        based on ALL horizons at once, or you'd lose more rows than
        necessary for the shorter horizons (e.g. the last 24h of
        data is still perfectly usable for training the 24h-horizon
        model, even though its 72h target is NaN).
        """ 
        col_name = self.target_column_name(horizon_hours)
        if col_name not in df.columns:
            raise ValueError(f"Column '{col_name}' not found — call build() first.")

        before = len(df)
        result = df.dropna(subset=[col_name]).reset_index(drop=True)
        logger.info(
            "Dropped %d row(s) with no valid %dh-ahead target (%d -> %d).",
            before - len(result), horizon_hours, before, len(result),
        )
        return result

    def all_target_columns(self) -> list[str]:
        """All target column names this builder produces — useful for
        excluding them from feature sets / leakage checks elsewhere."""
        return [self.target_column_name(h) for h in self.horizons_hours]

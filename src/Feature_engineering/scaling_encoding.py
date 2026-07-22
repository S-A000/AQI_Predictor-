"""
scaling_encoding.py
===================
Suggested path: src/feature_engineering/scaling_encoding.py

Phase 3, Part 8 — Scaling & Encoding.

SINGLE RESPONSIBILITY: bring all features to a consistent numeric
scale and encode any remaining categoricals. Final feature
selection / dropping also lives here. Does not create new
features — see sibling modules (temporal, lag, rolling, trend,
interaction, air_quality, spatial).

This module is the LAST step in the feature engineering pipeline:
    raw_data → temporal → lag → rolling → trend → interaction
    → air_quality → spatial → SCALING_ENCODING → model_ready_data

No groupby-by-city needed: all operations are row-wise or column-wise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ScalingEncodingEngineer:
    """
    Scales numeric features, encodes remaining categoricals,
    handles missing values, and performs final feature selection.

    Supports three scaling strategies:
        - "standard"  : StandardScaler (z-score, mean=0, std=1)
        - "minmax"    : MinMaxScaler (0-1 range)
        - "robust"    : RobustScaler (median/IQR based, outlier-resistant)
    """

    def __init__(
        self,
        *,
        scaling_strategy: str = "standard",
        target_col: str = "aqi",
        time_col: str = "timestamp",
        city_col: str = "city",
        drop_leakage_features: bool = True,
        drop_high_vif_features: bool = True,
        vif_threshold: float = 10.0,
        fill_na_strategy: str = "median",  # "median", "mean", "zero", "forward"
    ):
        self.scaling_strategy = scaling_strategy.lower()
        self.target_col = target_col
        self.time_col = time_col
        self.city_col = city_col
        self.drop_leakage_features = drop_leakage_features
        self.drop_high_vif_features = drop_high_vif_features
        self.vif_threshold = vif_threshold
        self.fill_na_strategy = fill_na_strategy

        self.scaler = None
        self.feature_columns_: list[str] | None = None
        self.dropped_columns_: list[str] = []

        # Validate scaling strategy
        valid_strategies = {"standard", "minmax", "robust"}
        if self.scaling_strategy not in valid_strategies:
            raise ValueError(f"scaling_strategy must be one of {valid_strategies}")

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------

    def _has_columns(self, df: pd.DataFrame, *columns: str) -> bool:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            logger.warning("Column(s) not found, skipping dependent feature(s): %s", missing)
            return False
        return True

    def _get_numeric_columns(self, df: pd.DataFrame) -> list[str]:
        """Return numeric columns excluding target, time, and raw categoricals."""
        exclude = {self.target_col, self.time_col, self.city_col, "dominant_pollutant", "aqi_category"}
        return [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]

    def _get_categorical_columns(self, df: pd.DataFrame) -> list[str]:
        """Return remaining object/category columns that need encoding."""
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        # Exclude columns that are already handled or should be dropped
        exclude = {self.time_col, self.city_col}
        return [c for c in cat_cols if c not in exclude]

    # --------------------------------------------------
    # 1. Handle missing values
    # --------------------------------------------------

    def fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill NaN values in numeric columns before scaling.
        Target column NaNs are dropped, not filled.
        """
        df = df.copy()
        numeric_cols = self._get_numeric_columns(df)

        if self.fill_na_strategy == "median":
            for col in numeric_cols:
                if df[col].isna().any():
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    logger.info("Filled NaN in '%s' with median (%.4f)", col, median_val)

        elif self.fill_na_strategy == "mean":
            for col in numeric_cols:
                if df[col].isna().any():
                    mean_val = df[col].mean()
                    df[col] = df[col].fillna(mean_val)
                    logger.info("Filled NaN in '%s' with mean (%.4f)", col, mean_val)

        elif self.fill_na_strategy == "zero":
            df[numeric_cols] = df[numeric_cols].fillna(0)
            logger.info("Filled NaN in numeric columns with 0")

        elif self.fill_na_strategy == "forward":
            df[numeric_cols] = df[numeric_cols].fillna(method="ffill").fillna(method="bfill")
            logger.info("Filled NaN using forward/backward fill")

        # Drop rows where target is NaN (cannot train without target)
        target_na_before = df[self.target_col].isna().sum()
        if target_na_before > 0:
            df = df.dropna(subset=[self.target_col])
            logger.info("Dropped %d rows with missing target '%s'", target_na_before, self.target_col)

        return df

    # --------------------------------------------------
    # 2. Drop leakage / redundant features
    # --------------------------------------------------

    def drop_leakage_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop features that carry future information or are direct
        derivatives of the target (source-leakage risk).

        Includes:
            - Any column with "future", "lead", "target" in name
            - Raw pollutant columns if they are measured AFTER the
              AQI calculation (context-dependent — logged but not
              auto-dropped unless explicitly flagged)
            - Dominant pollutant raw string (already one-hot encoded
              by air_quality_features.py)
            - AQI category (already encoded or can be derived from target)
        """
        if not self.drop_leakage_features:
            return df

        df = df.copy()
        leakage_patterns = ["future", "lead", "target_shifted"]
        to_drop = []

        for col in df.columns:
            if any(p in col.lower() for p in leakage_patterns):
                to_drop.append(col)

        # Drop raw categoricals that have been one-hot encoded
        if "dominant_pollutant" in df.columns and any(c.startswith("dominant_") for c in df.columns):
            to_drop.append("dominant_pollutant")

        if "aqi_category" in df.columns:
            to_drop.append("aqi_category")

        # Drop city raw if city one-hot exists
        if self.city_col in df.columns and any(c.startswith("city_") for c in df.columns):
            to_drop.append(self.city_col)

        to_drop = list(set(to_drop))
        for col in to_drop:
            if col in df.columns:
                df = df.drop(columns=[col])
                self.dropped_columns_.append(col)

        if to_drop:
            logger.info("Dropped leakage/redundant columns: %s", to_drop)
        return df

    def drop_high_vif_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optionally drop features with VIF > threshold to reduce
        multicollinearity. Uses a simple iterative approach.
        NOTE: Requires statsmodels. Skipped if not installed.
        """
        if not self.drop_high_vif_features:
            return df

        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
        except ImportError:
            logger.warning("statsmodels not installed; skipping VIF-based dropping")
            return df

        df = df.copy()
        numeric_cols = self._get_numeric_columns(df)
        X = df[numeric_cols].dropna()

        if X.empty or len(numeric_cols) < 2:
            return df

        dropped = []
        max_iter = len(numeric_cols)

        for _ in range(max_iter):
            X_const = X.copy()
            X_const["const"] = 1

            vifs = []
            for i, col in enumerate(X.columns):
                try:
                    vif_val = variance_inflation_factor(X_const.values, i)
                    vifs.append((col, vif_val))
                except Exception:
                    vifs.append((col, np.inf))

            vifs.sort(key=lambda x: x[1], reverse=True)
            highest_vif_col, highest_vif = vifs[0]

            if highest_vif <= self.vif_threshold:
                break

            X = X.drop(columns=[highest_vif_col])
            dropped.append((highest_vif_col, round(float(highest_vif), 2)))
            self.dropped_columns_.append(highest_vif_col)

        if dropped:
            df = df.drop(columns=[c for c, _ in dropped])
            logger.info("Dropped high-VIF features (>%.1f): %s", self.vif_threshold, dropped)

        return df

    # --------------------------------------------------
    # 3. Encode remaining categoricals
    # --------------------------------------------------

    def encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        One-hot encode any remaining categorical columns.
        Should be minimal if previous modules handled city/dominant
        pollutant encoding already.
        """
        cat_cols = self._get_categorical_columns(df)
        if not cat_cols:
            logger.info("No remaining categorical columns to encode")
            return df

        df = df.copy()
        for col in cat_cols:
            dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(columns=[col])
            logger.info("One-hot encoded '%s' → %s", col, list(dummies.columns))

        return df

    # --------------------------------------------------
    # 4. Scale numeric features
    # --------------------------------------------------

    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale numeric features using the configured strategy.
        Target column is NEVER scaled.

        Args:
            fit: If True, fit the scaler on this data. Set to False
                 for transform-only (e.g. validation/test sets).
        """
        df = df.copy()
        numeric_cols = self._get_numeric_columns(df)

        if not numeric_cols:
            logger.warning("No numeric columns found to scale")
            return df

        # Initialize scaler
        if self.scaler is None:
            if self.scaling_strategy == "standard":
                self.scaler = StandardScaler()
            elif self.scaling_strategy == "minmax":
                self.scaler = MinMaxScaler()
            elif self.scaling_strategy == "robust":
                self.scaler = RobustScaler()

        if fit:
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
            logger.info("Fitted %s scaler on %d features", self.scaling_strategy, len(numeric_cols))
        else:
            df[numeric_cols] = self.scaler.transform(df[numeric_cols])
            logger.info("Transformed %d features using fitted %s scaler", len(numeric_cols), self.scaling_strategy)

        self.feature_columns_ = numeric_cols
        return df

    # --------------------------------------------------
    # 5. Final feature selection (optional)
    # --------------------------------------------------

    def select_features(self, df: pd.DataFrame, max_features: int | None = None) -> pd.DataFrame:
        """
        Optional final feature selection. Currently a placeholder
        for correlation-based or model-based selection (Part 8
        extension). Can be extended with SelectKBest, RFE, etc.
        """
        if max_features is None:
            return df

        # Simple variance-based selection as placeholder
        numeric_cols = self._get_numeric_columns(df)
        if len(numeric_cols) <= max_features:
            return df

        variances = df[numeric_cols].var().sort_values(ascending=False)
        selected = variances.head(max_features).index.tolist()
        dropped = [c for c in numeric_cols if c not in selected]

        df = df.drop(columns=dropped)
        self.dropped_columns_.extend(dropped)
        logger.info("Variance-based selection: kept %d features, dropped %d", len(selected), len(dropped))
        return df

    # --------------------------------------------------
    # 6. Fit / Transform split support
    # --------------------------------------------------

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit scaler and transform — use on TRAINING data."""
        return self.build(df, fit=True)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform only — use on VALIDATION/TEST data."""
        return self.build(df, fit=False)

    # --------------------------------------------------
    # Full Part 8 pipeline
    # --------------------------------------------------

    def build(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Complete scaling & encoding pipeline:
            1. Fill missing values
            2. Drop leakage/redundant features
            3. Drop high-VIF features (optional)
            4. Encode remaining categoricals
            5. Scale numeric features
            6. Final feature selection (optional)
        """
        before_cols = df.shape[1]

        df = self.fill_missing_values(df)
        df = self.drop_leakage_features(df)
        df = self.drop_high_vif_features(df)
        df = self.encode_categoricals(df)
        df = self.scale_features(df, fit=fit)

        after_cols = df.shape[1]
        logger.info(
            "Scaling & encoding complete: %d columns → %d columns "
            "(dropped: %s)",
            before_cols, after_cols, self.dropped_columns_,
        )
        return df
"""
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
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from src.feature_engineering.air_quality_features import AirQualityFeatureEngineer
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
        target_columns: list[str] | None = None,
        time_col: str = "timestamp",
        city_col: str = "city",
        drop_leakage_features: bool = True,
        drop_high_vif_features: bool = True,
        vif_threshold: float = 10.0,
        vif_columns: list[str] | None = None,
        fill_na_strategy: str = "median",  # "median", "mean", "zero", "forward"
    ):
        self.scaling_strategy = scaling_strategy.lower()
        self.target_col = target_col
        self.target_columns = set(target_columns) if target_columns else {target_col}
        self.time_col = time_col
        self.city_col = city_col
        self.fill_na_strategy = fill_na_strategy

        self._should_drop_leakage_features = drop_leakage_features
        self._should_drop_high_vif_features = drop_high_vif_features

        self.vif_threshold = vif_threshold
        self.vif_columns = vif_columns

        self.scaler = None
        self.feature_columns_: list[str] | None = None
        self.dropped_columns_: list[str] = []

        # Fitted state storage to prevent train/val mismatches
        self.fill_values_: dict[str, float] = {}
        self.categorical_columns_: list[str] = []
        self.categories_map_: dict[str, list[str]] = {}
        self.vif_dropped_cols_: list[str] = []
        self.pollution_engineer_ = AirQualityFeatureEngineer()
        self.model_feature_columns_: list[str] = []
        self.final_columns_: list[str] = []

        # NEW: tracks which expected features were missing on the most
        # recent transform() call (empty list = nothing was missing).
        # Callers (e.g. PredictionFeaturePipeline / AQIPredictor) can
        # inspect this instead of the gap being invisible.
        self.last_transform_missing_features_: list[str] = []

        valid_strategies = {"standard", "minmax", "robust"}
        if self.scaling_strategy not in valid_strategies:
            raise ValueError(f"scaling_strategy must be one of {valid_strategies}")

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------

    def _get_numeric_columns(self, df: pd.DataFrame) -> list[str]:
        """Return numeric columns excluding target, time, city, and raw categoricals."""
        exclude = self.target_columns | {self.time_col, self.city_col, "dominant_pollutant", "aqi_category","station_id"}
        return [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]

    def _get_categorical_columns(self, df: pd.DataFrame, *, max_cardinality: int = 50) -> list[str]:
        """Return remaining object/category columns that need encoding."""
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        exclude = {
            self.time_col,
            self.city_col,
            "country",
            "source",
            "created_at",
            "aqi_category",
            "station_id"
        }
        candidates = [c for c in cat_cols if c not in exclude]

        safe_cols = []
        for col in candidates:
            cardinality = df[col].nunique(dropna=True)
            if cardinality > max_cardinality:
                logger.warning(
                    "Column '%s' has %d unique values (> %d) — skipping one-hot encoding.",
                    col, cardinality, max_cardinality,
                )
                continue
            safe_cols.append(col)

        return safe_cols

    # --------------------------------------------------
    # 1. Handle missing values
    # --------------------------------------------------

    def fill_missing_values(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Fill NaN values using training statistics."""
        df = df.copy()
        numeric_cols = self._get_numeric_columns(df)

        if fit:
            if self.fill_na_strategy == "median":
                self.fill_values_ = df[numeric_cols].median().to_dict()
            elif self.fill_na_strategy == "mean":
                self.fill_values_ = df[numeric_cols].mean().to_dict()
            elif self.fill_na_strategy == "zero":
                self.fill_values_ = {c: 0.0 for c in numeric_cols}
            elif self.fill_na_strategy == "forward":
                self.fill_values_ = df[numeric_cols].median().to_dict()

        if self.fill_na_strategy in ("median", "mean", "zero"):
            for col, val in self.fill_values_.items():
                if col in df.columns and df[col].isna().any():
                    df[col] = df[col].fillna(val)
                    if fit:
                        logger.info("Filled NaN in '%s' with %s (%.4f)", col, self.fill_na_strategy, val)

        elif self.fill_na_strategy == "forward":
            sort_columns = [
                col for col in (self.city_col, self.time_col) if col in df.columns
            ]
            if sort_columns:
                df = df.sort_values(sort_columns).copy()

            if self.city_col in df.columns:
                df[numeric_cols] = df.groupby(
                    self.city_col, sort=False
                )[numeric_cols].ffill()
            else:
                df[numeric_cols] = df[numeric_cols].ffill()

            for col, value in self.fill_values_.items():
                if col in df.columns and df[col].isna().any():
                    df[col] = df[col].fillna(value)
            logger.info(
                "Filled NaN causally using forward fill and train-fitted medians."
            )

        unresolved = [
            col for col in numeric_cols if col in df.columns and df[col].isna().any()
        ]
        if unresolved:
            raise ValueError(
                "Numeric missing values remain after train-fitted imputation: "
                f"{unresolved}"
            )

        return df

    # --------------------------------------------------
    # 2. Drop leakage / redundant features
    # --------------------------------------------------

    def drop_leakage_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop features carrying future information or direct target derivatives."""
        if not self._should_drop_leakage_features:
            return df

        df = df.copy()
        leakage_patterns = ["future", "lead", "target_shifted"]
        to_drop = []

        for col in df.columns:
            if any(p in col.lower() for p in leakage_patterns):
                to_drop.append(col)

        if "dominant_pollutant" in df.columns and any(c.startswith("dominant_") for c in df.columns):
            to_drop.append("dominant_pollutant")

        if "aqi_category" in df.columns:
            to_drop.append("aqi_category")

        # ❌ Puraana code:
        # if self.city_col in df.columns and any(c.startswith("city_") for c in df.columns):
        #     to_drop.append(self.city_col)

        # ✅ REMOVE 'city' from to_drop list! City column Hopsworks Primary Key ke liye zaroori hai.

        to_drop = list(set(to_drop))
        for col in to_drop:
            if col in df.columns:
                df = df.drop(columns=[col])
                if col not in self.dropped_columns_:
                    self.dropped_columns_.append(col)

        if to_drop:
            logger.info("Dropped leakage/redundant columns: %s", to_drop)
        return df

    def drop_high_vif_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Drop features with VIF > threshold during fit, reuse drops during transform."""
        if not self._should_drop_high_vif_features:
            return df

        df = df.copy()

        if not fit:
            # During transform: simply drop what was dropped during fit
            cols_to_drop = [c for c in self.vif_dropped_cols_ if c in df.columns]
            return df.drop(columns=cols_to_drop)

        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
        except ImportError:
            logger.warning("statsmodels not installed; skipping VIF-based dropping")
            return df

        default_vif_columns = (
            "temperature", "feels_like", "humidity", "pressure", "wind_speed",
            "cloudiness", "pm25", "pm10", "no2", "so2", "co", "o3",
        )
        candidate_cols = self.vif_columns if self.vif_columns is not None else list(default_vif_columns)
        numeric_cols = [c for c in candidate_cols if c in df.columns]

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
            self.vif_dropped_cols_.append(highest_vif_col)
            self.dropped_columns_.append(highest_vif_col)

        if dropped:
            df = df.drop(columns=[c for c, _ in dropped])
            logger.info("Dropped high-VIF features (>%.1f): %s", self.vif_threshold, dropped)

        return df

    # --------------------------------------------------
    # 3. Encode remaining categoricals
    # --------------------------------------------------

    def encode_categoricals(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        One-hot encode remaining categorical columns using fixed categories
        from the training set.
        """
        df = df.copy()

        if fit:
            self.categorical_columns_ = self._get_categorical_columns(df)
            for col in self.categorical_columns_:
                self.categories_map_[col] = df[col].dropna().astype(str).unique().tolist()

        if not self.categorical_columns_:
            logger.info("No remaining categorical columns to encode")
            return df

        for col in self.categorical_columns_:
            if col not in df.columns:
                continue

            # Force pandas to use training categories to ensure identical one-hot columns
            cat_type = pd.CategoricalDtype(categories=self.categories_map_[col], ordered=False)
            col_series = df[col].astype(str).astype(cat_type)

            dummies = pd.get_dummies(col_series, prefix=col, dtype=int)
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(columns=[col])

        return df

    # --------------------------------------------------
    # 4. Scale numeric features
    # --------------------------------------------------

    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale numeric features using the configured strategy."""
        df = df.copy()

        if fit:
            numeric_cols = self._get_numeric_columns(df)
            self.feature_columns_ = numeric_cols

            if not numeric_cols:
                logger.warning("No numeric columns found to scale")
                return df

            if self.scaling_strategy == "standard":
                self.scaler = StandardScaler()
            elif self.scaling_strategy == "minmax":
                self.scaler = MinMaxScaler()
            elif self.scaling_strategy == "robust":
                self.scaler = RobustScaler()

            df[self.feature_columns_] = self.scaler.fit_transform(df[self.feature_columns_])
            logger.info("Fitted %s scaler on %d features", self.scaling_strategy, len(self.feature_columns_))
            self.last_transform_missing_features_ = []
        else:
            if self.scaler is None or self.feature_columns_ is None:
                raise RuntimeError("Scaler has not been fitted yet! Call with fit=True first.")

            # --------------------------------------------------------
            # CHANGED (Silent Zeros fix): previously this block did
            #   for col in self.feature_columns_:
            #       if col not in df.columns:
            #           df[col] = 0.0
            # which silently invented a "0.0" reading for any missing
            # expected feature — 0.0 is a plausible real value for many
            # AQI-scale features, so a genuine upstream bug (a dropped
            # pollutant column, a broken join) would look like clean,
            # valid data instead of an error.
            #
            # Now every missing fitted feature is logged by name and
            # rejected before sklearn sees the row. Missing values within
            # present columns are handled earlier with train-fitted stats.
            # Historical fallback behavior was based on this
            # column's TRAINING-TIME fallback value (median/mean/zero,
            # per fill_na_strategy — already computed during fit and
            # stored in fill_values_) instead of a bare 0.0, so the
            # fallback at least represents "typical", not "suspiciously
            # clean air".
            # --------------------------------------------------------
            missing_cols = [c for c in self.feature_columns_ if c not in df.columns]
            if missing_cols:
                logger.error(
                    "Transform-time schema mismatch: expected feature(s) missing: %s",
                    missing_cols,
                )
                raise ValueError(
                    "Transform input is missing fitted numeric features: "
                    f"{missing_cols}"
                )
            self.last_transform_missing_features_ = missing_cols

            df[self.feature_columns_] = self.scaler.transform(df[self.feature_columns_])
            logger.info("Transformed %d features using fitted %s scaler", len(self.feature_columns_), self.scaling_strategy)

        return df

    # --------------------------------------------------
    # 5. Final feature selection
    # --------------------------------------------------

    def select_features(self, df: pd.DataFrame, max_features: int | None = None) -> pd.DataFrame:
        """Optional final feature selection placeholder."""
        if max_features is None:
            return df

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
        """Complete scaling & encoding pipeline."""
        before_cols = df.shape[1]

        df = self.pollution_engineer_.add_pollution_index(df, fit=fit)
        df = self.fill_missing_values(df, fit=fit)
        df = self.drop_leakage_features(df)
        df = self.drop_high_vif_features(df, fit=fit)
        df = self.encode_categoricals(df, fit=fit)
        df = self.scale_features(df, fit=fit)

        after_cols = df.shape[1]
        if fit:
            self.final_columns_ = list(df.columns)
        logger.info(
            "Scaling & encoding complete: %d columns -> %d columns (dropped: %s)",
            before_cols, after_cols, self.dropped_columns_,
        )
        return df

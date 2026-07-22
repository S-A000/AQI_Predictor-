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

No groupby-by-city needed: all operations are row-wise or column-wise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ScalingEncodingReport:
    """
    Summary of one build() run — returned alongside the transformed
    DataFrame so the caller (build_features.py) can log/persist what
    happened without re-deriving it.
    """

    columns_before: int
    columns_after: int
    dropped_columns: list[str] = field(default_factory=list)
    feature_columns: list[str] = field(default_factory=list)
    scaling_strategy: str = "standard"
    fit: bool = True

    @property
    def final_feature_count(self) -> int:
        return len(self.feature_columns)

    def to_dict(self) -> dict:
        return {
            "columns_before": self.columns_before,
            "columns_after": self.columns_after,
            "dropped_columns": self.dropped_columns,
            "feature_columns": self.feature_columns,
            "scaling_strategy": self.scaling_strategy,
            "fit": self.fit,
            "final_feature_count": self.final_feature_count,
        }


class ScalingEncodingEngineer:
    """
    Scales numeric features, encodes remaining categoricals,
    handles missing values, and performs final feature selection.

    Supports three scaling strategies:
        - "standard"  : StandardScaler (z-score, mean=0, std=1)
        - "minmax"    : MinMaxScaler (0-1 range)
        - "robust"    : RobustScaler (median/IQR based, outlier-resistant)
    """

    # Pipeline bookkeeping / metadata columns — not physical signal,
    # never encoded, always dropped before one-hot encoding.
    METADATA_COLUMNS = {"created_at", "source"}

    # Safety guard: any categorical column with more unique values than
    # this is SKIPPED (dropped, not encoded) — no matter what else is
    # in cat_cols. This guards against a column with thousands of
    # unique values (e.g. a raw id/timestamp-like string) getting
    # one-hot encoded into tens of thousands of dummy columns and
    # blowing up memory.
    MAX_ENCODING_CARDINALITY = 50

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
        vif_columns: list[str] | None = None,
        fill_na_strategy: str = "median",  # "median", "mean", "zero", "forward"
    ):
        self.scaling_strategy = scaling_strategy.lower()
        self.target_col = target_col
        self.time_col = time_col
        self.city_col = city_col
        self.fill_na_strategy = fill_na_strategy

        # NOTE: these flags are intentionally named with a
        # `_should_` prefix, NOT the same name as the
        # drop_leakage_features()/drop_high_vif_features() methods
        # below. Giving an attribute the exact same name as a
        # method on the same class silently shadows the method —
        # `self.drop_leakage_features = True` would overwrite the
        # method itself, and calling `self.drop_leakage_features(df)`
        # later would try to call `True(df)` and crash with
        # `TypeError: 'bool' object is not callable`. Keep these
        # names distinct.
        self._should_drop_leakage_features = drop_leakage_features
        self._should_drop_high_vif_features = drop_high_vif_features

        self.vif_threshold = vif_threshold
        # Restrict VIF computation to a curated subset of columns.
        # Iterative VIF fits one OLS regression PER remaining column
        # PER iteration — on the ~380 columns this pipeline produces
        # after Parts 1-7, that is prohibitively slow (potentially
        # hours) and statistically not very meaningful anyway (lag/
        # rolling features are BY DESIGN highly collinear with each
        # other; VIF is more useful on the small set of raw/base
        # columns, matching what the EDA's own vif_analysis already
        # did on 14 base columns). If not given, defaults to the
        # 14 base weather+pollutant+aqi columns; pass an explicit
        # list to override, or pass all numeric columns yourself if
        # you accept the runtime cost.
        self.vif_columns = vif_columns

        self.scaler = None
        self.feature_columns_: list[str] | None = None
        self.dropped_columns_: list[str] = []

        # New state variables for deterministic transform alignment
        self.categorical_mappings_: dict[str, list] = {}
        self.encoded_columns_: list[str] = []
        self.final_columns_: list[str] | None = None
        self.leakage_dropped_columns_: list[str] = []
        self.vif_dropped_columns_: list[str] = []
        self.fill_values_: dict[str, float] = {}

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

    def fill_missing_values(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Fill NaN values in numeric columns before scaling.
        Target column NaNs are dropped, not filled.

        CAUTION: for lag/rolling features, the NaNs are concentrated
        in the first N rows of EACH city (not enough history yet —
        e.g. `aqi_lag_72` is NaN for the first 72 hours per city).
        Filling those with a global median/mean disguises "no
        history yet" as "an average reading", which is a real value
        judgement, not a neutral default — if that concerns you,
        drop those warm-up rows instead (see build_features.py's
        `drop_warmup_nans` option) rather than filling them here.
        """
        df = df.copy()
        numeric_cols = self._get_numeric_columns(df)

        if fit:
            self.fill_values_ = {}

        if self.fill_na_strategy == "median":
            for col in numeric_cols:
                if fit:
                    val = df[col].median()
                    self.fill_values_[col] = val if not pd.isna(val) else 0
                if df[col].isna().any():
                    fill_val = self.fill_values_.get(col, 0)
                    df[col] = df[col].fillna(fill_val)
                    logger.info("Filled NaN in '%s' with median (%.4f)", col, fill_val)

        elif self.fill_na_strategy == "mean":
            for col in numeric_cols:
                if fit:
                    val = df[col].mean()
                    self.fill_values_[col] = val if not pd.isna(val) else 0
                if df[col].isna().any():
                    fill_val = self.fill_values_.get(col, 0)
                    df[col] = df[col].fillna(fill_val)
                    logger.info("Filled NaN in '%s' with mean (%.4f)", col, fill_val)

        elif self.fill_na_strategy == "zero":
            df[numeric_cols] = df[numeric_cols].fillna(0)
            logger.info("Filled NaN in numeric columns with 0")

        elif self.fill_na_strategy == "forward":
            # .fillna(method="ffill"/"bfill") is deprecated in
            # pandas >= 2.1 — use the direct .ffill()/.bfill() methods.
            df[numeric_cols] = df[numeric_cols].ffill().bfill()
            logger.info("Filled NaN using forward/backward fill")

        # Drop rows where target is NaN (cannot train without target)
        if self.target_col in df.columns:
            target_na_before = df[self.target_col].isna().sum()
            if target_na_before > 0:
                df = df.dropna(subset=[self.target_col])
                logger.info("Dropped %d rows with missing target '%s'", target_na_before, self.target_col)

        return df

    # --------------------------------------------------
    # 2. Drop leakage / redundant features
    # --------------------------------------------------

    def drop_leakage_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
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
        if not self._should_drop_leakage_features:
            return df

        df = df.copy()

        if fit:
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
            self.leakage_dropped_columns_ = to_drop
            for col in to_drop:
                if col in df.columns:
                    df = df.drop(columns=[col])
                    self.dropped_columns_.append(col)

            if to_drop:
                logger.info("Dropped leakage/redundant columns: %s", to_drop)
        else:
            # In transform, rigidly mirror the drops made during fit
            to_drop = [c for c in self.leakage_dropped_columns_ if c in df.columns]
            if to_drop:
                df = df.drop(columns=to_drop)
                logger.info("Dropped leakage/redundant columns (transform): %s", to_drop)

        return df

    def drop_high_vif_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Optionally drop features with VIF > threshold to reduce
        multicollinearity. Uses a simple iterative approach.
        NOTE: Requires statsmodels. Skipped if not installed.

        Runs ONLY on `self.vif_columns` (defaults to the 14 base
        weather+pollutant+aqi columns, matching the EDA's own
        vif_analysis) — NOT on the full engineered feature set —
        for both runtime and statistical-validity reasons (see the
        constructor docstring for `vif_columns`).
        """
        if not self._should_drop_high_vif_features:
            return df

        df = df.copy()

        if not fit:
            # Skip recalculation on transform; just replicate what was dropped during fit
            to_drop = [c for c in self.vif_dropped_columns_ if c in df.columns]
            if to_drop:
                df = df.drop(columns=to_drop)
                logger.info("Dropped high-VIF features (transform): %s", to_drop)
            return df

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

        if len(numeric_cols) > 60:
            logger.warning(
                "VIF check requested on %d columns — this can be very slow "
                "(iterative VIF fits one OLS regression per remaining column "
                "per iteration). Consider passing a smaller `vif_columns` list.",
                len(numeric_cols),
            )

        X = df[numeric_cols].dropna()

        if X.empty or len(numeric_cols) < 2:
            return df

        dropped = []
        max_iter = len(numeric_cols)
        self.vif_dropped_columns_ = []

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
            self.vif_dropped_columns_.append(highest_vif_col)

        if dropped:
            df = df.drop(columns=[c for c, _ in dropped])
            logger.info("Dropped high-VIF features (>%.1f): %s", self.vif_threshold, dropped)

        return df

    # --------------------------------------------------
    # 3. Encode remaining categoricals
    # --------------------------------------------------

    def encode_categoricals(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        One-hot encode any remaining categorical columns deterministically.
        Should be minimal if previous modules handled city/dominant
        pollutant encoding already.

        Two safety measures run before encoding:
          1. METADATA_COLUMNS (created_at, source, ...) are dropped
             outright — pipeline bookkeeping, not physical signal.
          2. Any column with more than MAX_ENCODING_CARDINALITY unique
             values is dropped with a warning instead of encoded —
             a hard guard against a repeat of the memory-blowup bug,
             even if a rogue high-cardinality column slips into
             cat_cols in the future.
        """
        df = df.copy()

        # 1. Drop metadata columns outright — pipeline bookkeeping,
        #    not physical signal, never encoded.
        metadata_present = [c for c in self.METADATA_COLUMNS if c in df.columns]
        if metadata_present:
            df = df.drop(columns=metadata_present)
            if fit:
                self.dropped_columns_.extend(metadata_present)
            logger.info("Dropped metadata columns (not physical signal): %s", metadata_present)

        if fit:
            self.categorical_mappings_ = {}
            self.encoded_columns_ = []
            cat_cols = self._get_categorical_columns(df)

            if not cat_cols:
                logger.info("No remaining categorical columns to encode")
                return df

            # 2. Cardinality guard & deterministic category extraction
            for col in cat_cols:
                n_unique = df[col].nunique(dropna=True)
                if n_unique > self.MAX_ENCODING_CARDINALITY:
                    logger.warning(
                        "Skipping '%s': %d unique values exceeds "
                        "MAX_ENCODING_CARDINALITY=%d — one-hot encoding this "
                        "would explode into %d+ columns. Dropping instead. "
                        "If this column is genuinely needed, engineer it "
                        "into a lower-cardinality feature upstream (bucket/ "
                        "hash/frequency-encode) rather than raising this "
                        "threshold.",
                        col, n_unique, self.MAX_ENCODING_CARDINALITY, n_unique,
                    )
                    df = df.drop(columns=[col])
                    self.dropped_columns_.append(col)
                    continue

                # Save precise categories observed during fit
                categories = df[col].dropna().unique().tolist()
                self.categorical_mappings_[col] = categories

                # Categorize to enforce identical column generation
                df[col] = pd.Categorical(df[col], categories=categories)
                dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
                
                self.encoded_columns_.extend(dummies.columns.tolist())
                df = pd.concat([df, dummies], axis=1)
                df = df.drop(columns=[col])
                logger.info("One-hot encoded '%s' (%d categories) -> %s", col, n_unique, list(dummies.columns))

        else:
            # Transform mode: mirror fitted encodings completely
            if not hasattr(self, 'categorical_mappings_'):
                raise RuntimeError("Encoder not fitted. Call build(fit=True) first.")

            for col, categories in self.categorical_mappings_.items():
                if col in df.columns:
                    # pd.Categorical ensures exactly the dummy columns from fit are created, 
                    # dropping extra unseen categories and filling unseen training categories with 0.
                    df[col] = pd.Categorical(df[col], categories=categories)
                    dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
                    df = pd.concat([df, dummies], axis=1)
                    df = df.drop(columns=[col])
                else:
                    # If column is missing in transform entirely, explicitly inject zeroed dummies
                    for cat in categories:
                        df[f"{col}_{cat}"] = 0

            # Drop any new/unseen categorical columns present in transform data
            cat_cols_transform = self._get_categorical_columns(df)
            if cat_cols_transform:
                df = df.drop(columns=cat_cols_transform)

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

        if fit:
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

            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
            logger.info("Fitted %s scaler on %d features", self.scaling_strategy, len(numeric_cols))
            
            # Save the exact numeric feature order for strict transform alignment
            self.feature_columns_ = numeric_cols

        else:
            if self.feature_columns_ is None:
                raise RuntimeError("Cannot transform: no scaler has been fitted yet.")

            # Align feature lists rigorously before scaling
            missing_cols = [c for c in self.feature_columns_ if c not in df.columns]
            if missing_cols:
                logger.info("Adding missing feature columns filled with zeros: %s", missing_cols)
                for c in missing_cols:
                    df[c] = 0

            # Transform purely on the columns saved during fit
            df[self.feature_columns_] = self.scaler.transform(df[self.feature_columns_])
            logger.info("Transformed %d features using fitted %s scaler", len(self.feature_columns_), self.scaling_strategy)

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

    def fit_transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, ScalingEncodingReport]:
        """
        Fit scaler and transform — use on TRAINING data ONLY.
        Returns (transformed_df, report) — matches how
        build_features.py calls this:
            train_scaled, report = scaler_engineer.fit_transform(train_df)
        """
        return self.build(df, fit=True)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform only, using the scaler already fitted by
        fit_transform() — use on VALIDATION/TEST data. Returns just
        the transformed DataFrame (no report — the report from the
        fit_transform() call already describes what this scaler
        does), matching:
            val_scaled = scaler_engineer.transform(val_df)
        """
        df, _report = self.build(df, fit=False)
        return df

    def save(self, path: str | Path) -> None:
        """
        Persist the fitted scaler plus the metadata needed to
        reproduce feature handling at prediction time (Phase 9):
        which columns were scaled, which were dropped, and the
        column/target/time/city naming this instance used.

        Raises if called before fit_transform() (nothing fitted yet).
        """
        if self.scaler is None:
            raise RuntimeError("Cannot save: no scaler has been fitted yet. Call fit_transform() first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "scaler": self.scaler,
                "scaling_strategy": self.scaling_strategy,
                "feature_columns": self.feature_columns_,
                "dropped_columns": self.dropped_columns_,
                "target_col": self.target_col,
                "time_col": self.time_col,
                "city_col": self.city_col,
                "categorical_mappings": getattr(self, "categorical_mappings_", {}),
                "encoded_columns": getattr(self, "encoded_columns_", []),
                "final_columns": getattr(self, "final_columns_", []),
                "leakage_dropped_columns": getattr(self, "leakage_dropped_columns_", []),
                "vif_dropped_columns": getattr(self, "vif_dropped_columns_", []),
                "fill_values": getattr(self, "fill_values_", {}),
            },
            path,
        )
        logger.info("Saved fitted scaler + deterministic metadata -> %s", path)

    # --------------------------------------------------
    # Full Part 8 pipeline
    # --------------------------------------------------

    def build(self, df: pd.DataFrame, fit: bool = True) -> tuple[pd.DataFrame, ScalingEncodingReport]:
        """
        Complete scaling & encoding pipeline:
            1. Fill missing values
            2. Drop leakage/redundant features
            3. Drop high-VIF features (optional)
            4. Encode remaining categoricals
            5. Scale numeric features
            6. Final feature selection (optional)

        Always returns (transformed_df, report). fit_transform()
        passes both through; transform() unpacks and discards the
        report since it isn't a new fit.
        """
        before_cols = df.shape[1]

        df = self.fill_missing_values(df, fit=fit)
        df = self.drop_leakage_features(df, fit=fit)
        df = self.drop_high_vif_features(df, fit=fit)
        df = self.encode_categoricals(df, fit=fit)
        df = self.scale_features(df, fit=fit)

        # Enforce exact structural column alignment
        if fit:
            self.final_columns_ = df.columns.tolist()
        else:
            if getattr(self, "final_columns_", None) is None:
                raise RuntimeError("Pipeline not fitted. Call build(fit=True) first.")

            # Zero-fill any expected column completely missing
            missing = [c for c in self.final_columns_ if c not in df.columns]
            for c in missing:
                if c != self.target_col:
                    df[c] = 0

            # Restrict entirely to saved column structure maintaining exact order
            current_final = [c for c in self.final_columns_ if c in df.columns]
            df = df[current_final]

        after_cols = df.shape[1]
        report = ScalingEncodingReport(
            columns_before=before_cols,
            columns_after=after_cols,
            dropped_columns=list(self.dropped_columns_),
            feature_columns=list(self.feature_columns_) if self.feature_columns_ is not None else [],
            scaling_strategy=self.scaling_strategy,
            fit=fit,
        )

        logger.info(
            "Scaling & encoding complete: %d columns -> %d columns "
            "(dropped: %s)",
            before_cols, after_cols, self.dropped_columns_,
        )
        return df, report


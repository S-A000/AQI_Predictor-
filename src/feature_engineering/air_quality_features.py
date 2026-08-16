"""
air_quality_features.py
=======================

Suggested path:
    src/feature_engineering/air_quality_features.py

Phase 3, Part 6 — Air Quality Features.

SINGLE RESPONSIBILITY:
Derive domain-specific AQI/pollutant features:

- AQI category
- pollutant ratios
- composite pollution index

IMPORTANT:
`dominant_pollutant` one-hot encoding is intentionally NOT performed
inside the main `build()` pipeline.

Reason:
Running `pd.get_dummies()` independently on train, validation, test,
or inference data can create different columns depending on which
categories happen to be present in each split.

Example:

Train:
    co, no2, so2
    -> dominant_co, dominant_no2, dominant_so2

Validation:
    pm25, o3
    -> dominant_pm25, dominant_o3

That creates training-serving/schema mismatch.

Therefore the raw `dominant_pollutant` categorical column is preserved
here and its train-fitted categorical encoding is handled later by
`ScalingEncodingEngineer`, which learns categories from TRAIN and
reuses exactly the same categories for validation/test/inference.

The standalone `add_dominant_pollutant_encoding()` helper is retained
for backwards compatibility/manual use, but it is not called by the
production `build()` path.

No groupby-by-city is needed for the features created here because
they are row-wise combinations of values already belonging to the
same observation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logger import get_logger


logger = get_logger(__name__)


# ------------------------------------------------------------------
# AQI category configuration
# ------------------------------------------------------------------

# US EPA AQI breakpoints using the standard six-category scale.
AQI_CATEGORY_BINS = (
    -1,
    50,
    100,
    150,
    200,
    300,
    501,
)

AQI_CATEGORY_LABELS = (
    "Good",
    "Moderate",
    "Unhealthy_Sensitive",
    "Unhealthy",
    "Very_Unhealthy",
    "Hazardous",
)


# Pollutants used for the composite pollution index.
POLLUTANT_COLUMNS = (
    "pm25",
    "pm10",
    "no2",
    "so2",
    "co",
    "o3",
)


class AirQualityFeatureEngineer:
    """
    Build domain-specific air-quality features.

    Production pipeline responsibilities:

    1. AQI category
    2. Pollutant ratios
    3. Composite pollution index

    `dominant_pollutant` categorical encoding is intentionally deferred
    to ScalingEncodingEngineer so the categorical schema is fitted on
    training data and reused consistently during validation, testing,
    and inference.
    """

    def __init__(
        self,
        *,
        aqi_col: str = "aqi",
        dominant_pollutant_col: str = "dominant_pollutant",
        pollutant_cols: tuple[str, ...] = POLLUTANT_COLUMNS,
    ) -> None:
        self.aqi_col = aqi_col
        self.dominant_pollutant_col = dominant_pollutant_col
        self.pollutant_cols = pollutant_cols

        # Train-fitted normalization state for pollution_index.
        self.pollution_min_: dict[str, float] = {}
        self.pollution_max_: dict[str, float] = {}
        self.pollution_index_fitted_: bool = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _has_columns(
        self,
        df: pd.DataFrame,
        *columns: str,
    ) -> bool:
        """
        Check whether all required columns exist.

        Missing dependent inputs do not crash the whole feature pipeline;
        the related optional feature is skipped and logged.
        """

        missing = [
            column
            for column in columns
            if column not in df.columns
        ]

        if missing:
            logger.warning(
                "Column(s) not found, skipping dependent feature(s): %s",
                missing,
            )
            return False

        return True

    # ------------------------------------------------------------------
    # AQI category
    # ------------------------------------------------------------------

    def add_aqi_category(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add the standard US-EPA AQI category for the current AQI value.
        """

        if not self._has_columns(
            df,
            self.aqi_col,
        ):
            return df

        result = df.copy()

        # Convert objects / None safely to numeric values.
        #
        # At inference time AQI can occasionally arrive as Python None.
        # pd.cut handles float NaN safely, but object None can otherwise
        # create dtype-related comparison problems.
        aqi_numeric = pd.to_numeric(
            result[self.aqi_col],
            errors="coerce",
        )

        result["aqi_category"] = pd.cut(
            aqi_numeric,
            bins=AQI_CATEGORY_BINS,
            labels=AQI_CATEGORY_LABELS,
        )

        return result

    # ------------------------------------------------------------------
    # Pollutant ratios
    # ------------------------------------------------------------------

    def add_pollutant_ratios(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add pollutant composition ratios.

        Ratios can capture differences in pollutant composition rather
        than relying only on absolute pollutant concentrations.

        A small epsilon protects against division by zero.
        """

        result = df.copy()

        epsilon = 0.1

        if self._has_columns(
            result,
            "pm25",
            "pm10",
        ):
            result["pm25_pm10_ratio"] = (
                result["pm25"]
                / (result["pm10"] + epsilon)
            )

        if self._has_columns(
            result,
            "no2",
            "so2",
        ):
            result["no2_so2_ratio"] = (
                result["no2"]
                / (result["so2"] + epsilon)
            )

        if self._has_columns(
            result,
            "co",
            "no2",
        ):
            result["co_no2_ratio"] = (
                result["co"]
                / (result["no2"] + epsilon)
            )

        if self._has_columns(
            result,
            "pm25",
            "o3",
        ):
            result["pm25_o3_ratio"] = (
                result["pm25"]
                / (result["o3"] + epsilon)
            )

        return result

    # ------------------------------------------------------------------
    # Dominant pollutant encoding
    # ------------------------------------------------------------------

    def add_dominant_pollutant_encoding(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        One-hot encode dominant_pollutant.

        IMPORTANT:
        This helper remains available for backwards compatibility or
        standalone/manual use.

        It is intentionally NOT called by `build()` because independently
        applying `pd.get_dummies()` to train/validation/test/inference can
        produce different schemas.

        Production categorical encoding is handled by
        ScalingEncodingEngineer.
        """

        if not self._has_columns(
            df,
            self.dominant_pollutant_col,
        ):
            return df

        result = df.copy()

        dummies = pd.get_dummies(
            result[self.dominant_pollutant_col],
            prefix="dominant",
            dtype=int,
        )

        result = pd.concat(
            [
                result,
                dummies,
            ],
            axis=1,
        )

        logger.info(
            "Dominant pollutant categories encoded manually: %s",
            list(dummies.columns),
        )

        return result

    # ------------------------------------------------------------------
    # Composite pollution index
    # ------------------------------------------------------------------

    def fit_pollution_index(
        self,
        df: pd.DataFrame,
    ) -> "AirQualityFeatureEngineer":
        """
        Fit pollutant normalization bounds using training data only.

        These min/max values are persisted inside the fitted engineer and
        reused for validation, test, and inference data.

        This prevents train-serving skew and validation/test leakage.
        """

        available_pollutants = [
            column
            for column in self.pollutant_cols
            if column in df.columns
        ]

        if not available_pollutants:
            logger.warning(
                "No pollutant columns found; pollution index was not fitted."
            )

            self.pollution_min_ = {}
            self.pollution_max_ = {}
            self.pollution_index_fitted_ = True

            return self

        self.pollution_min_ = {}
        self.pollution_max_ = {}

        for column in available_pollutants:
            numeric = pd.to_numeric(
                df[column],
                errors="coerce",
            )

            if not numeric.notna().any():
                continue

            self.pollution_min_[column] = float(
                numeric.min()
            )

            self.pollution_max_[column] = float(
                numeric.max()
            )

        self.pollution_index_fitted_ = True

        logger.info(
            "Fitted pollution-index normalization on %d pollutant column(s).",
            len(self.pollution_min_),
        )

        return self

    def transform_pollution_index(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build pollution_index using previously fitted training bounds.
        """

        if not self.pollution_index_fitted_:
            raise RuntimeError(
                "Pollution-index normalization has not been fitted "
                "on training data."
            )

        # No valid pollutant columns existed during fitting.
        if not self.pollution_min_:
            return df

        missing_columns = [
            column
            for column in self.pollution_min_
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                "Pollution-index input is missing fitted pollutant columns: "
                f"{missing_columns}"
            )

        result = df.copy()

        weights = {
            "pm25": 0.35,
            "pm10": 0.25,
            "no2": 0.15,
            "so2": 0.10,
            "co": 0.10,
            "o3": 0.05,
        }

        normalized_weighted_sum = pd.Series(
            0.0,
            index=result.index,
            dtype=float,
        )

        total_weight_used = 0.0

        for column, column_minimum in self.pollution_min_.items():
            column_maximum = self.pollution_max_[column]

            numeric = pd.to_numeric(
                result[column],
                errors="coerce",
            )

            span = (
                column_maximum
                - column_minimum
            )

            if span > 0:
                normalized = (
                    numeric
                    - column_minimum
                ) / span
            else:
                # Constant pollutant during training.
                normalized = pd.Series(
                    0.0,
                    index=result.index,
                    dtype=float,
                )

            weight = weights.get(
                column,
                0.1,
            )

            normalized_weighted_sum = (
                normalized_weighted_sum
                + normalized * weight
            )

            total_weight_used += weight

        if total_weight_used > 0:
            result["pollution_index"] = (
                normalized_weighted_sum
                / total_weight_used
            )
        else:
            result["pollution_index"] = 0.0

        return result

    def add_pollution_index(
        self,
        df: pd.DataFrame,
        *,
        fit: bool | None = None,
    ) -> pd.DataFrame:
        """
        Add the composite pollution index.

        When `fit=True`, normalization bounds are fitted from the supplied
        training data.

        When `fit=False`, the previously fitted training normalization
        state is reused.

        If fit is omitted and no fitted state exists yet, fitting is
        performed for backwards compatibility.
        """

        if (
            fit is True
            or (
                fit is None
                and not self.pollution_index_fitted_
            )
        ):
            self.fit_pollution_index(df)

        return self.transform_pollution_index(df)

    # ------------------------------------------------------------------
    # Full Part 6 pipeline
    # ------------------------------------------------------------------

    def build(
        self,
        df: pd.DataFrame,
        *,
        fit_pollution_normalization: bool | None = None,
    ) -> pd.DataFrame:
        """
        Run the production air-quality feature pipeline.

        IMPORTANT:
        dominant_pollutant remains categorical here.

        It will later be encoded by ScalingEncodingEngineer using
        train-fitted categorical categories, ensuring that train,
        validation, test, and inference all receive the same schema.
        """

        before_columns = df.shape[1]

        # --------------------------------------------------------------
        # 1. AQI category
        # --------------------------------------------------------------

        df = self.add_aqi_category(df)

        # --------------------------------------------------------------
        # 2. Pollutant ratios
        # --------------------------------------------------------------

        df = self.add_pollutant_ratios(df)

        # --------------------------------------------------------------
        # 3. Dominant pollutant
        # --------------------------------------------------------------
        #
        # DO NOT call:
        #
        #     self.add_dominant_pollutant_encoding(df)
        #
        # here.
        #
        # Independent pd.get_dummies() calls create different schemas
        # between train/validation/test.
        #
        # ScalingEncodingEngineer performs the production encoding later
        # using categories learned strictly from the training split.
        # --------------------------------------------------------------

        # --------------------------------------------------------------
        # 4. Pollution index
        # --------------------------------------------------------------

        df = self.add_pollution_index(
            df,
            fit=fit_pollution_normalization,
        )

        after_columns = df.shape[1]

        logger.info(
            "Air quality features added: %d new column(s) (%d -> %d).",
            after_columns - before_columns,
            before_columns,
            after_columns,
        )

        return df
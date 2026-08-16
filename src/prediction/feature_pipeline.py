"""
feature_pipeline.py
===================

Production prediction feature pipeline.

Responsibilities:
- Validate prediction payloads
- Combine payloads with raw historical context
- Run the exact canonical feature-engineering pipeline
- Remove stale/precomputed legacy features before regeneration
- Apply ONLY the persisted train-fitted preprocessing state
- Preserve strict training-serving schema parity
- Never silently fabricate missing model features
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from src.feature_engineering.air_quality_features import (
    AirQualityFeatureEngineer,
)
from src.feature_engineering.interaction_features import (
    InteractionFeatureEngineer,
)
from src.feature_engineering.lag_features import (
    LagFeatureEngineer,
)
from src.feature_engineering.pipeline_steps import (
    run_feature_engineering_steps,
)
from src.feature_engineering.rolling_features import (
    RollingFeatureEngineer,
)
from src.feature_engineering.spatial_features import (
    SpatialFeatureEngineer,
)
from src.feature_engineering.temporal_features import (
    TemporalFeatureEngineer,
)
from src.feature_engineering.trend_features import (
    TrendFeatureEngineer,
)
from src.prediction.validator import (
    PredictionPayload,
    PredictionValidator,
)
from src.utils.logger import get_logger


logger = get_logger(__name__)


# BigQuery/raw training history currently contains some lightweight
# derived columns. Training removes these before canonical feature
# engineering; inference must do the same to maintain parity.
PRECOMPUTED_FEATURE_COLUMNS = (
    "hour",
    "day",
    "month",
    "day_of_week",
    "is_weekend",
    "aqi_change_rate",
    "aqi_rolling_mean_3h",
)


class PredictionFeaturePipeline:
    """
    Build production model-ready features from raw observations.

    The persisted ScalingEncodingEngineer from the selected model bundle
    owns all fitted preprocessing state:

    - imputation statistics
    - pollution-index normalization
    - categorical categories
    - VIF drops
    - scaler
    - numeric schema

    Prediction NEVER calls fit().
    """

    def __init__(
        self,
        scaler_engineer: Any,
    ) -> None:
        self.scaler_engineer = scaler_engineer

        self.validator = (
            PredictionValidator()
        )

        # Long-lived feature engineer instances.
        self.temporal_engineer = (
            TemporalFeatureEngineer()
        )

        self.lag_engineer = (
            LagFeatureEngineer()
        )

        self.rolling_engineer = (
            RollingFeatureEngineer()
        )

        self.trend_engineer = (
            TrendFeatureEngineer()
        )

        self.interaction_engineer = (
            InteractionFeatureEngineer()
        )

        self.air_quality_engineer = (
            AirQualityFeatureEngineer()
        )

        self.spatial_engineer = (
            SpatialFeatureEngineer()
        )

        # Feature order remains controlled centrally by
        # pipeline_steps.py.
        self._engineers: Dict[str, Any] = {
            "temporal": self.temporal_engineer,
            "lag": self.lag_engineer,
            "rolling": self.rolling_engineer,
            "trend": self.trend_engineer,
            "interaction": self.interaction_engineer,
            "air_quality": self.air_quality_engineer,
            "spatial": self.spatial_engineer,
        }

        self.last_transform_missing_features_: List[str] = []

    # --------------------------------------------------
    # Raw observation preparation
    # --------------------------------------------------

    @staticmethod
    def _prepare_raw_observations(
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Canonicalize raw observation rows before feature engineering.
        """

        if df.empty:
            raise ValueError(
                "Cannot engineer an empty prediction DataFrame."
            )

        result = df.copy()

        required = {
            "city",
            "timestamp",
        }

        missing = (
            required
            - set(result.columns)
        )

        if missing:
            raise ValueError(
                "Prediction feature input is missing required "
                f"column(s): {sorted(missing)}"
            )

        result["timestamp"] = pd.to_datetime(
            result["timestamp"],
            utc=True,
            errors="coerce",
        )

        result = result.dropna(
            subset=[
                "city",
                "timestamp",
            ]
        )

        result["timestamp"] = (
            result["timestamp"]
            .dt.floor("h")
        )

        result["city"] = (
            result["city"]
            .astype(str)
            .str.strip()
            .str.title()
        )

        result = result[
            result["city"].ne("")
        ].copy()

        # Remove legacy/precomputed columns exactly as training does.
        stale_columns = [
            column
            for column in PRECOMPUTED_FEATURE_COLUMNS
            if column in result.columns
        ]

        if stale_columns:
            logger.info(
                "Removing precomputed context feature(s) before "
                "canonical inference engineering: %s",
                stale_columns,
            )

            result = result.drop(
                columns=stale_columns
            )

        # Payload rows are appended after context, so keep="last"
        # ensures a new observation replaces an older same city/hour row.
        result = (
            result
            .drop_duplicates(
                subset=[
                    "city",
                    "timestamp",
                ],
                keep="last",
            )
            .sort_values(
                [
                    "city",
                    "timestamp",
                ]
            )
            .reset_index(
                drop=True
            )
        )

        return result

    # --------------------------------------------------
    # Canonical feature engineering
    # --------------------------------------------------

    def _apply_engineers(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Apply the same canonical feature-engineering order as training.
        """

        prepared_df = (
            self._prepare_raw_observations(
                df
            )
        )

        return run_feature_engineering_steps(
            prepared_df,
            engineers=self._engineers,
        )

    # --------------------------------------------------
    # Train-fitted preprocessing
    # --------------------------------------------------

    def _transform_scaler(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Apply persisted training preprocessing.

        IMPORTANT:
        Missing fitted features are NEVER silently created with zeroes.
        Any preprocessing/schema problem fails immediately.
        """

        obj = self.scaler_engineer

        # --------------------------------------------------
        # Legacy dictionary artifact support
        # --------------------------------------------------

        if isinstance(
            obj,
            dict,
        ):
            extracted = None

            for key in (
                "scaler_engineer",
                "pipeline",
                "scaler",
            ):
                if key in obj:
                    extracted = obj[key]
                    break

            if extracted is None:
                raise ValueError(
                    "Scaler artifact dictionary does not contain a "
                    "supported preprocessing object."
                )

            obj = extracted

        # --------------------------------------------------
        # Preferred bundled ScalingEncodingEngineer path
        # --------------------------------------------------

        if (
            hasattr(obj, "transform")
            and callable(obj.transform)
            and hasattr(
                obj,
                "model_feature_columns_",
            )
        ):
            # DO NOT catch and swallow exceptions here.
            result = obj.transform(
                df
            )

            self.last_transform_missing_features_ = list(
                getattr(
                    obj,
                    "last_transform_missing_features_",
                    [],
                )
            )

            return result

        # --------------------------------------------------
        # Strict raw sklearn-transformer compatibility path
        # --------------------------------------------------

        if (
            hasattr(obj, "feature_names_in_")
            and hasattr(obj, "transform")
        ):
            expected_features = list(
                obj.feature_names_in_
            )

            missing = [
                column
                for column in expected_features
                if column not in df.columns
            ]

            self.last_transform_missing_features_ = (
                missing
            )

            if missing:
                raise ValueError(
                    "Raw preprocessing artifact expected missing "
                    f"feature(s): {missing}"
                )

            aligned_df = (
                df.loc[
                    :,
                    expected_features,
                ]
                .copy()
            )

            aligned_df = aligned_df.replace(
                {
                    pd.NA: np.nan,
                    None: np.nan,
                }
            )

            unresolved = [
                column
                for column in aligned_df.columns
                if aligned_df[column].isna().any()
            ]

            if unresolved:
                raise ValueError(
                    "Raw preprocessing input contains unresolved "
                    f"missing values: {unresolved}"
                )

            scaled_array = obj.transform(
                aligned_df
            )

            return pd.DataFrame(
                scaled_array,
                columns=expected_features,
                index=aligned_df.index,
            )

        raise TypeError(
            "Provided preprocessing artifact does not expose a "
            "supported fitted transform contract. "
            f"type={type(obj)}"
        )

    # --------------------------------------------------
    # Target key helpers
    # --------------------------------------------------

    @staticmethod
    def _canonical_payload_keys(
        payload_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Return canonical city/hour keys for payload rows.
        """

        keys = payload_df[
            [
                "city",
                "timestamp",
            ]
        ].copy()

        keys["city"] = (
            keys["city"]
            .astype(str)
            .str.strip()
            .str.title()
        )

        keys["timestamp"] = (
            pd.to_datetime(
                keys["timestamp"],
                utc=True,
                errors="coerce",
            )
            .dt.floor("h")
        )

        if keys["timestamp"].isna().any():
            raise ValueError(
                "Prediction payload contains invalid timestamp value(s)."
            )

        return keys

    @staticmethod
    def _select_engineered_payload_rows(
        engineered_df: pd.DataFrame,
        payload_keys: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Select only engineered rows belonging to requested payload keys.

        Matching uses BOTH city and timestamp.
        """

        engineered = (
            engineered_df.copy()
        )

        engineered["city"] = (
            engineered["city"]
            .astype(str)
            .str.strip()
            .str.title()
        )

        engineered["timestamp"] = (
            pd.to_datetime(
                engineered["timestamp"],
                utc=True,
                errors="coerce",
            )
            .dt.floor("h")
        )

        key_frame = payload_keys.copy()

        key_frame["_payload_order"] = range(
            len(key_frame)
        )

        selected = engineered.merge(
            key_frame,
            on=[
                "city",
                "timestamp",
            ],
            how="inner",
            validate="many_to_one",
        )

        if selected.empty:
            raise ValueError(
                "Canonical feature engineering produced no rows "
                "matching the prediction payload."
            )

        selected = (
            selected
            .sort_values(
                "_payload_order"
            )
            .drop_duplicates(
                subset=[
                    "city",
                    "timestamp",
                ],
                keep="last",
            )
        )

        if len(selected) != len(
            key_frame.drop_duplicates(
                subset=[
                    "city",
                    "timestamp",
                ]
            )
        ):
            raise ValueError(
                "Prediction payload/features could not be aligned "
                "one-to-one by city and timestamp."
            )

        selected = selected.drop(
            columns=[
                "_payload_order",
            ]
        )

        return selected

    # --------------------------------------------------
    # Single prediction features
    # --------------------------------------------------

    def build_features(
        self,
        payload: Union[
            Dict[str, Any],
            PredictionPayload,
        ],
        context_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Build one model-ready prediction feature row.

        transform() is used; fit() is never called.
        """

        validated = (
            self.validator.validate(
                payload
            )
        )

        payload_dict = (
            self.validator.to_feature_dict(
                validated
            )
        )

        payload_df = pd.DataFrame(
            [
                payload_dict,
            ]
        )

        payload_keys = (
            self._canonical_payload_keys(
                payload_df
            )
        )

        if (
            context_df is not None
            and not context_df.empty
        ):
            combined_df = pd.concat(
                [
                    context_df,
                    payload_df,
                ],
                ignore_index=True,
                sort=False,
            )

        else:
            combined_df = (
                payload_df
            )

        engineered_df = (
            self._apply_engineers(
                combined_df
            )
        )

        target_engineered = (
            self._select_engineered_payload_rows(
                engineered_df,
                payload_keys,
            )
        )

        if len(target_engineered) != 1:
            raise ValueError(
                "Single prediction expected exactly one engineered row; "
                f"received {len(target_engineered)}."
            )

        scaled_df = (
            self._transform_scaler(
                target_engineered
            )
        )

        return scaled_df

    # --------------------------------------------------
    # Batch prediction features
    # --------------------------------------------------

    def build_batch_features(
        self,
        payloads: List[
            Union[
                Dict[str, Any],
                PredictionPayload,
            ]
        ],
        context_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Build model-ready features for multiple payloads.

        Matching uses city + timestamp, preventing different cities
        sharing the same timestamp from being mixed together.
        """

        if not payloads:
            raise ValueError(
                "Prediction batch cannot be empty."
            )

        validated_batch = (
            self.validator.validate_batch(
                payloads
            )
        )

        batch_dicts = [
            self.validator.to_feature_dict(
                payload
            )
            for payload in validated_batch
        ]

        payload_df = pd.DataFrame(
            batch_dicts
        )

        payload_keys = (
            self._canonical_payload_keys(
                payload_df
            )
        )

        duplicate_keys = (
            payload_keys
            .duplicated(
                subset=[
                    "city",
                    "timestamp",
                ],
                keep=False,
            )
        )

        if duplicate_keys.any():
            duplicates = (
                payload_keys.loc[
                    duplicate_keys,
                    [
                        "city",
                        "timestamp",
                    ],
                ]
                .astype(str)
                .to_dict(
                    orient="records"
                )
            )

            raise ValueError(
                "Prediction batch contains duplicate city/timestamp "
                f"keys: {duplicates}"
            )

        if (
            context_df is not None
            and not context_df.empty
        ):
            combined_df = pd.concat(
                [
                    context_df,
                    payload_df,
                ],
                ignore_index=True,
                sort=False,
            )

        else:
            combined_df = (
                payload_df
            )

        engineered_df = (
            self._apply_engineers(
                combined_df
            )
        )

        batch_engineered = (
            self._select_engineered_payload_rows(
                engineered_df,
                payload_keys,
            )
        )

        if len(batch_engineered) != len(
            payload_df
        ):
            raise ValueError(
                "Batch feature engineering row-count mismatch; "
                f"expected={len(payload_df)}, "
                f"received={len(batch_engineered)}"
            )

        scaled_batch_df = (
            self._transform_scaler(
                batch_engineered
            )
        )

        return scaled_batch_df
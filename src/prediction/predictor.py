"""
predictor.py
============

Production AQI prediction engine.

Responsibilities:
- Load horizon-specific model bundles
- Use each bundle's exact fitted preprocessing state
- Build features through PredictionFeaturePipeline
- Enforce exact ordered model schema
- Reject missing/unexpected model features
- Never fabricate missing feature values
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from src.prediction.feature_pipeline import (
    PredictionFeaturePipeline,
)
from src.prediction.load_model import (
    LoadedModelArtifact,
    get_production_model,
)
from src.prediction.validator import (
    PredictionPayload,
    PredictionValidator,
)
from src.utils.logger import get_logger


logger = get_logger(__name__)


DIRECT_HORIZONS = (
    24,
    48,
    72,
)


class AQIPredictor:
    """
    Prediction engine for direct 24h, 48h, and 72h AQI models.
    """

    def __init__(
        self,
        artifact: LoadedModelArtifact | None = None,
        feature_pipeline: PredictionFeaturePipeline | None = None,
        validator: PredictionValidator | None = None,
    ) -> None:
        self.validator = (
            validator
            or PredictionValidator()
        )

        self._resources: Dict[
            int,
            Dict[str, Any],
        ] = {}

        if artifact is None:
            artifact = (
                get_production_model(
                    horizon_hours=24
                )
            )

        self.artifact = artifact
        self.model = artifact.model

        self.expected_features = list(
            artifact.feature_names
        )

        self.feature_pipeline = (
            feature_pipeline
            or PredictionFeaturePipeline(
                scaler_engineer=artifact.scaler_engineer
            )
        )

        self._resources[
            artifact.horizon_hours
        ] = {
            "artifact": artifact,
            "pipeline": self.feature_pipeline,
            "model": self.model,
            "expected_features": self.expected_features,
        }

        self.last_alignment_report: Dict[
            str,
            Any,
        ] = {}

    # --------------------------------------------------
    # Horizon resources
    # --------------------------------------------------

    def _get_horizon_resources(
        self,
        horizon_hours: int,
    ) -> Dict[str, Any]:
        """
        Load and cache complete bundled resources for one direct horizon.
        """

        if horizon_hours not in DIRECT_HORIZONS:
            raise ValueError(
                f"Unsupported horizon_hours: {horizon_hours}. "
                "Must be 24, 48, or 72."
            )

        if horizon_hours not in self._resources:
            logger.info(
                "Loading bundled resources for %sh horizon...",
                horizon_hours,
            )

            artifact = (
                get_production_model(
                    horizon_hours=horizon_hours
                )
            )

            pipeline = (
                PredictionFeaturePipeline(
                    scaler_engineer=artifact.scaler_engineer
                )
            )

            self._resources[
                horizon_hours
            ] = {
                "artifact": artifact,
                "pipeline": pipeline,
                "model": artifact.model,
                "expected_features": list(
                    artifact.feature_names
                ),
            }

        return self._resources[
            horizon_hours
        ]

    # --------------------------------------------------
    # Exact schema alignment
    # --------------------------------------------------

    def _align_and_validate(
        self,
        df: pd.DataFrame,
        expected_features: List[str],
        *,
        min_completeness: float = 0.90,
        strict: bool = False,
    ) -> pd.DataFrame:
        """
        Enforce the exact persisted model feature contract.

        Missing model features are never fabricated.
        """

        # Retained only for backwards-compatible method signature.
        del min_completeness
        del strict

        if not expected_features:
            raise ValueError(
                "Loaded model artifact has an empty feature schema."
            )

        if (
            len(set(expected_features))
            != len(expected_features)
        ):
            raise ValueError(
                "Loaded model feature schema contains duplicates."
            )

        if df.columns.has_duplicates:
            duplicate_columns = (
                df.columns[
                    df.columns.duplicated()
                ]
                .unique()
                .tolist()
            )

            raise ValueError(
                "Prediction preprocessing produced duplicate "
                f"columns: {duplicate_columns}"
            )

        expected_set = set(
            expected_features
        )

        actual_set = set(
            df.columns
        )

        missing = [
            feature
            for feature in expected_features
            if feature not in actual_set
        ]

        # Columns that may legitimately survive preprocessing but are
        # metadata, identifiers, or targets rather than model inputs.
        allowed_non_features = {
            "timestamp",
            "event_hour",
            "city",
            "country",
            "created_at",
            "source",
            "date",
            "split",
            "aqi_category",
            "dominant_pollutant",
            "station_id",
        }

        allowed_non_features.update(
            column
            for column in df.columns
            if column.startswith(
                "target_aqi_t_"
            )
        )

        unexpected = sorted(
            actual_set
            - expected_set
            - allowed_non_features
        )

        completeness = (
            (
                len(expected_features)
                - len(missing)
            )
            / len(expected_features)
        )

        self.last_alignment_report = {
            "completeness": completeness,
            "missing_features": missing,
            "unexpected_features": unexpected,
            "expected_feature_count": len(
                expected_features
            ),
            "actual_feature_count": len(
                expected_set.intersection(
                    actual_set
                )
            ),
        }

        if missing or unexpected:
            message = (
                "Prediction feature contract mismatch; "
                f"missing={missing}, "
                f"unexpected={unexpected}"
            )

            logger.error(
                message
            )

            raise ValueError(
                message
            )

        aligned = (
            df.loc[
                :,
                expected_features,
            ]
            .copy()
        )

        if aligned.isna().any().any():
            nan_columns = (
                aligned.columns[
                    aligned.isna().any()
                ]
                .tolist()
            )

            raise ValueError(
                "Model-ready prediction features contain unresolved "
                f"NaN values: {nan_columns}"
            )

        return aligned

    # --------------------------------------------------
    # Preprocessing schema check
    # --------------------------------------------------

    @staticmethod
    def _reject_preprocessing_schema_gap(
        pipeline: PredictionFeaturePipeline,
    ) -> None:
        """
        Reject any missing feature detected by the fitted transformer.
        """

        missing = list(
            getattr(
                pipeline,
                "last_transform_missing_features_",
                [],
            )
        )

        if missing:
            raise ValueError(
                "Bundled preprocessing input is missing required "
                f"feature(s): {missing}"
            )

    # --------------------------------------------------
    # Single prediction
    # --------------------------------------------------

    def predict_single(
        self,
        payload: Dict[str, Any] | PredictionPayload,
        context_df: Optional[pd.DataFrame] = None,
        horizon_hours: int = 24,
        *,
        strict_alignment: bool = False,
        min_completeness: float = 0.90,
    ) -> Dict[str, Any]:
        """
        Run one genuine direct-horizon prediction.
        """

        logger.info(
            "Executing predict_single inference for %sh horizon...",
            horizon_hours,
        )

        resources = (
            self._get_horizon_resources(
                horizon_hours
            )
        )

        features_df = (
            resources["pipeline"]
            .build_features(
                payload,
                context_df=context_df,
            )
        )

        self._reject_preprocessing_schema_gap(
            resources["pipeline"]
        )

        aligned_df = (
            self._align_and_validate(
                features_df,
                resources["expected_features"],
                min_completeness=min_completeness,
                strict=strict_alignment,
            )
        )

        prediction = (
            resources["model"]
            .predict(
                aligned_df
            )[0]
        )

        return {
            "predicted_aqi": round(
                float(prediction),
                2,
            ),
            "horizon_hours": horizon_hours,
            "model_version": (
                resources["artifact"]
                .model_version
            ),
            "feature_completeness": round(
                self.last_alignment_report.get(
                    "completeness",
                    1.0,
                ),
                4,
            ),
        }

    # --------------------------------------------------
    # Batch prediction
    # --------------------------------------------------

    def predict_batch(
        self,
        features_input: Union[
            List[Dict[str, Any]],
            List[PredictionPayload],
            pd.DataFrame,
        ],
        context_df: Optional[pd.DataFrame] = None,
        horizon_hours: int = 24,
        *,
        strict_alignment: bool = False,
        min_completeness: float = 0.90,
    ) -> pd.DataFrame:
        """
        Run batch direct-horizon inference with exact feature enforcement.
        """

        logger.info(
            "Executing predict_batch inference for %sh horizon...",
            horizon_hours,
        )

        resources = (
            self._get_horizon_resources(
                horizon_hours
            )
        )

        expected_features = (
            resources[
                "expected_features"
            ]
        )

        # --------------------------------------------------
        # Already model-ready DataFrame
        # --------------------------------------------------

        if (
            isinstance(
                features_input,
                pd.DataFrame,
            )
            and set(expected_features).issubset(
                features_input.columns
            )
        ):
            aligned_df = (
                self._align_and_validate(
                    features_input,
                    expected_features,
                    min_completeness=min_completeness,
                    strict=strict_alignment,
                )
            )

            predictions = (
                resources["model"]
                .predict(
                    aligned_df
                )
            )

            results = (
                features_input.copy()
            )

        # --------------------------------------------------
        # Raw payload input
        # --------------------------------------------------

        else:
            if isinstance(
                features_input,
                pd.DataFrame,
            ):
                raise ValueError(
                    "DataFrame input does not satisfy the complete "
                    "persisted model feature schema."
                )

            features_df = (
                resources["pipeline"]
                .build_batch_features(
                    features_input,
                    context_df=context_df,
                )
            )

            self._reject_preprocessing_schema_gap(
                resources["pipeline"]
            )

            aligned_df = (
                self._align_and_validate(
                    features_df,
                    expected_features,
                    min_completeness=min_completeness,
                    strict=strict_alignment,
                )
            )

            predictions = (
                resources["model"]
                .predict(
                    aligned_df
                )
            )

            results = (
                features_df.copy()
            )

        results[
            "predicted_aqi"
        ] = np.round(
            predictions,
            2,
        )

        return results
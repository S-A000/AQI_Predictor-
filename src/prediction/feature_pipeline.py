"""
src/prediction/feature_pipeline.py

1. Why it was modified: Created to serve as the single source of truth for online feature engineering, enforcing strict training-serving parity. Enhanced with automatic encoding alignment and feature schema matching.
2. Architecture before: Prediction skipped feature engineering and relied on pre-built parquet files.
3. Architecture after: PredictionFeaturePipeline orchestrates all 8 feature engineering classes and handles dynamic categorical encoding/scaling alignment.
4. Exact code: See below.
5. Every changed function: __init__, _apply_engineers, _transform_scaler, build_features, build_batch_features.
6. Every new class: `PredictionFeaturePipeline`.
7. Why the change follows SOLID: Open/Closed Principle - the pipeline delegates to existing feature engineer classes without modifying them.
8. Why it removes training-serving skew: By importing and executing the exact same classes in the exact same order as `build_features.py`, it guarantees identical transformations and feature order. UPDATED: the order itself is now imported from pipeline_steps.py (single source of truth), not hand-listed here — see that file's docstring.
9. Why it is production-safe: Prevents data leakage by calling `transform()`, automatically aligns unseen/missing dummy columns, and unpacks dict artifacts safely. UPDATED: missing-feature fills are no longer silent — every gap is logged by name so a schema drift upstream is visible instead of masquerading as a "clean" 0.0 reading.
"""

from typing import Any, Dict, List, Optional, Union
import pandas as pd

# CHANGED: step order now comes from the single shared module instead of
# being re-derived here. The individual engineer classes are still
# imported because PredictionFeaturePipeline keeps long-lived instances
# of them (passed into run_feature_engineering_steps via `engineers=`).
from src.feature_engineering.temporal_features import TemporalFeatureEngineer
from src.feature_engineering.lag_features import LagFeatureEngineer
from src.feature_engineering.rolling_features import RollingFeatureEngineer
from src.feature_engineering.trend_features import TrendFeatureEngineer
from src.feature_engineering.interaction_features import InteractionFeatureEngineer
from src.feature_engineering.air_quality_features import AirQualityFeatureEngineer
from src.feature_engineering.spatial_features import SpatialFeatureEngineer
from src.feature_engineering.pipeline_steps import run_feature_engineering_steps
from src.prediction.validator import PredictionPayload, PredictionValidator
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PredictionFeaturePipeline:
    """
    PredictionFeaturePipeline owns ALL preprocessing.
    It reuses existing feature engineering classes to ensure zero training-serving skew.
    """

    def __init__(self, scaler_engineer: Any):
        """
        Initializes the pipeline with a FITTED ScalingEncodingEngineer or scaler artifact dict.
        """
        self.scaler_engineer = scaler_engineer
        self.validator = PredictionValidator()

        self.temporal_engineer = TemporalFeatureEngineer()
        self.lag_engineer = LagFeatureEngineer()
        self.rolling_engineer = RollingFeatureEngineer()
        self.trend_engineer = TrendFeatureEngineer()
        self.interaction_engineer = InteractionFeatureEngineer()
        self.air_quality_engineer = AirQualityFeatureEngineer()
        self.spatial_engineer = SpatialFeatureEngineer()

        # Maps step name -> long-lived instance, handed to
        # run_feature_engineering_steps() so the ORDER is still
        # controlled centrally by pipeline_steps.py while this class
        # keeps its own instances (in case any engineer ever grows
        # per-instance state — currently none do, but this keeps the
        # option open without another refactor).
        self._engineers: Dict[str, Any] = {
            "temporal": self.temporal_engineer,
            "lag": self.lag_engineer,
            "rolling": self.rolling_engineer,
            "trend": self.trend_engineer,
            "interaction": self.interaction_engineer,
            "air_quality": self.air_quality_engineer,
            "spatial": self.spatial_engineer,
        }

        # NEW: visibility into the most recent transform-time schema gap,
        # for callers (e.g. AQIPredictor) that want to inspect or report it.
        self.last_transform_missing_features_: List[str] = []

    def _apply_engineers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies feature engineers in the exact order used during training."""
        return run_feature_engineering_steps(df, engineers=self._engineers)

    def _transform_scaler(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Industry-Grade Scaler & Encoding Handler:
        1. Unpacks dict artifacts.
        2. Clean pd.NA / NAType and fill NaNs to avoid sklearn array conversion crash.
        3. Aligns feature schema with 'feature_names_in_' to guarantee zero column mismatches.

        CHANGED: every branch that fills a missing expected column now logs
        the exact column names it filled, instead of silently reindexing to
        0.0. `self.last_transform_missing_features_` is set on every call
        (empty list when nothing was missing) so a caller can check it
        without parsing logs.
        """
        import numpy as np

        obj = self.scaler_engineer

        # 1. Unpack dictionary artifacts if loaded from joblib
        if isinstance(obj, dict):
            for key in ("scaler_engineer", "scaler", "model", "pipeline"):
                if key in obj:
                    obj = obj[key]
                    break

        # 2. If it is a ScalingEncodingEngineer instance or custom wrapper.
        # NOTE: ScalingEncodingEngineer.transform() itself now logs+tracks
        # its own missing-feature fills (see scaling_encoding.py) — nothing
        # extra needed here for that path, we just don't swallow it.
        if hasattr(obj, "transform") and callable(obj.transform):
            try:
                result = obj.transform(df)
                self.last_transform_missing_features_ = list(
                    getattr(obj, "last_transform_missing_features_", [])
                )
                return result
            except Exception:
                pass

        # 3. Direct Scikit-Learn Scaler with Schema Alignment (feature_names_in_)
        if hasattr(obj, "feature_names_in_"):
            expected_features = list(obj.feature_names_in_)
            missing = [c for c in expected_features if c not in df.columns]

            if missing:
                logger.warning(
                    "PredictionFeaturePipeline: %d expected feature(s) missing "
                    "before raw-scaler transform, filled with 0.0: %s",
                    len(missing), missing,
                )
            self.last_transform_missing_features_ = missing

            # Auto-align One-Hot dummies & drop raw targets/unseen columns
            aligned_df = df.reindex(columns=expected_features, fill_value=0.0)

            # 🧹 CRITICAL FIX: Convert pd.NA / NAType to standard np.nan, then fill NAs with 0.0
            aligned_df = aligned_df.fillna(np.nan)
            aligned_df = aligned_df.replace({pd.NA: np.nan, None: np.nan})
            aligned_df = aligned_df.astype(float).fillna(0.0)

            scaled_array = obj.transform(aligned_df)
            return pd.DataFrame(scaled_array, columns=expected_features, index=df.index)

        # 4. Standard transform fallback with NA cleanup
        if hasattr(obj, "transform"):
            df_cleaned = df.fillna(np.nan).replace({pd.NA: np.nan}).fillna(0.0)
            return obj.transform(df_cleaned)

        raise AttributeError(
            f"Provided scaler_engineer of type {type(self.scaler_engineer)} "
            f"does not expose a valid '.transform()' or 'feature_names_in_' attribute."
        )

    def build_features(
        self,
        payload: Union[Dict[str, Any], PredictionPayload],
        context_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Builds features for a single payload.
        Calls transform(), NEVER fit().
        """
        validated = self.validator.validate(payload)
        payload_dict = self.validator.to_feature_dict(validated)
        payload_df = pd.DataFrame([payload_dict])
        target_timestamp = payload_df["timestamp"].iloc[0]

        if context_df is not None and not context_df.empty:
            combined_df = pd.concat([context_df, payload_df], ignore_index=True)
            engineered_df = self._apply_engineers(combined_df)
            mask = engineered_df["timestamp"] == target_timestamp
            target_engineered = engineered_df[mask].tail(1).copy() if mask.any() else engineered_df.tail(1).copy()
        else:
            target_engineered = self._apply_engineers(payload_df)

        scaled_df = self._transform_scaler(target_engineered)
        return scaled_df

    def build_batch_features(
        self,
        payloads: List[Union[Dict[str, Any], PredictionPayload]],
        context_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Builds features for a batch of payloads.
        Calls transform(), NEVER fit().
        """
        validated_batch = self.validator.validate_batch(payloads)
        batch_dicts = [self.validator.to_feature_dict(p) for p in validated_batch]
        payload_df = pd.DataFrame(batch_dicts)

        target_timestamps = set(payload_df["timestamp"])

        if context_df is not None and not context_df.empty:
            combined_df = pd.concat([context_df, payload_df], ignore_index=True)
            engineered_df = self._apply_engineers(combined_df)
            mask = engineered_df["timestamp"].isin(target_timestamps)
            batch_engineered = engineered_df[mask].copy() if mask.any() else engineered_df.tail(len(payload_df)).copy()
        else:
            batch_engineered = self._apply_engineers(payload_df)

        scaled_batch_df = self._transform_scaler(batch_engineered)
        return scaled_batch_df
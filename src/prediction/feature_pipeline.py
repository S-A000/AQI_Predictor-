"""
src/prediction/feature_pipeline.py

1. Why it was modified: Created to serve as the single source of truth for online feature engineering, enforcing parity with training.
2. Architecture before: Prediction skipped feature engineering and relied on pre-built parquet files.
3. Architecture after: PredictionFeaturePipeline orchestrates the exact same 8 feature engineering classes used during training.
4. Exact code: See below.
5. Every changed function: N/A (new file).
6. Every new class: `PredictionFeaturePipeline`.
7. Why the change follows SOLID: Open/Closed Principle - the pipeline delegates to existing feature engineer classes without modifying them.
8. Why it removes training-serving skew: By importing and executing the exact same classes in the exact same order as `build_features.py`, it guarantees identical transformations.
9. Why it is production-safe: It explicitly prevents data leakage by only calling `transform()` on the scaler, never `fit()`.
"""

from typing import Any, Dict, List, Optional, Union
import pandas as pd

from src.feature_engineering.temporal_features import TemporalFeatureEngineer
from src.feature_engineering.lag_features import LagFeatureEngineer
from src.feature_engineering.rolling_features import RollingFeatureEngineer
from src.feature_engineering.trend_features import TrendFeatureEngineer
from src.feature_engineering.interaction_features import InteractionFeatureEngineer
from src.feature_engineering.air_quality_features import AirQualityFeatureEngineer
from src.feature_engineering.spatial_features import SpatialFeatureEngineer
from src.feature_engineering.scaling_encoding import ScalingEncodingEngineer
from src.prediction.validator import PredictionPayload, PredictionValidator


class PredictionFeaturePipeline:
    """
    PredictionFeaturePipeline owns ALL preprocessing.
    It reuses existing feature engineering classes to ensure zero training-serving skew.
    """

    def __init__(self, scaler_engineer: ScalingEncodingEngineer):
        """
        Initializes the pipeline with a FITTED ScalingEncodingEngineer.
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

    def _apply_engineers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies feature engineers in the exact order used during training."""
        df = self.temporal_engineer.build(df)
        df = self.lag_engineer.build(df)
        df = self.rolling_engineer.build(df)
        df = self.trend_engineer.build(df)
        df = self.interaction_engineer.build(df)
        df = self.air_quality_engineer.build(df)
        df = self.spatial_engineer.build(df)
        return df

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

        scaled_df = self.scaler_engineer.transform(target_engineered)
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

        scaled_batch_df = self.scaler_engineer.transform(batch_engineered)
        return scaled_batch_df

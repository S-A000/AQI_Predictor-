"""
pipeline_steps.py
==================
Suggested path: src/feature_engineering/pipeline_steps.py

SINGLE RESPONSIBILITY: define the ONE canonical ORDER of feature
engineering steps, shared by both the offline/batch pipeline
(src/feature_engineering/build_features.py) and the online/real-time
pipeline (src/prediction/feature_pipeline.py).

WHY THIS FILE EXISTS (fixes Training-Serving Skew structurally):
    Before this file, `build_features.py` and `feature_pipeline.py`
    each hand-wrote their own sequence of `.build()` calls. They
    happened to match today — but nothing enforced that they'd keep
    matching. Change the order, add a new engineer, or remove one in
    just ONE of the two files, and training/serving silently diverge.

    Now both files import FEATURE_ENGINEERING_STEPS (or call
    run_feature_engineering_steps) from HERE instead of writing their
    own list. There is exactly one place left to edit the pipeline
    order — it is now structurally impossible for the two paths to
    drift apart, instead of just "currently correct by discipline".

This file does NOT change what any feature engineer does — it only
centralizes the ORDER in which they run. No behavior change for
existing callers who already had the order right.
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from src.feature_engineering.air_quality_features import AirQualityFeatureEngineer
from src.feature_engineering.interaction_features import InteractionFeatureEngineer
from src.feature_engineering.lag_features import LagFeatureEngineer
from src.feature_engineering.rolling_features import RollingFeatureEngineer
from src.feature_engineering.spatial_features import SpatialFeatureEngineer
from src.feature_engineering.temporal_features import TemporalFeatureEngineer
from src.feature_engineering.trend_features import TrendFeatureEngineer

# Order matters: rolling/trend features consume lag features' output,
# interaction/air_quality consume rolling+trend outputs, spatial runs
# last. This exact order is what BOTH build_features.py (batch/train)
# and feature_pipeline.py (real-time/serve) must run — identically.
FEATURE_ENGINEERING_STEPS = (
    ("temporal", TemporalFeatureEngineer),
    ("lag", LagFeatureEngineer),
    ("rolling", RollingFeatureEngineer),
    ("trend", TrendFeatureEngineer),
    ("interaction", InteractionFeatureEngineer),
    ("air_quality", AirQualityFeatureEngineer),
    ("spatial", SpatialFeatureEngineer),
)


def run_feature_engineering_steps(
    df: pd.DataFrame,
    *,
    engineers: Optional[Dict[str, object]] = None,
) -> pd.DataFrame:
    """
    Runs every step in FEATURE_ENGINEERING_STEPS, in order, against df.

    `engineers`: optional dict of {step_name: instance} letting a caller
    reuse already-constructed engineer instances (e.g. PredictionFeaturePipeline
    holds its own instances). If a step name isn't supplied, a fresh
    instance of that step's default class is constructed on the spot —
    this is what build_features.py's batch path does, since it has no
    reason to keep long-lived instances around.
    """
    engineers = engineers or {}
    for step_name, engineer_cls in FEATURE_ENGINEERING_STEPS:
        engineer = engineers.get(step_name) or engineer_cls()
        df = engineer.build(df)
    return df
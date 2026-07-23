"""
build_features.py
===================
Suggested path: src/feature_engineering/build_features.py

Phase 3 orchestrator — runs all 8 feature-engineering parts, THEN
creates the multi-horizon forecast targets, THEN splits and scales.

ORDER:
    1-7. Temporal, Lag, Rolling, Trend, Interaction, Air Quality, Spatial
    8.   Forecast targets (target_aqi_t+24, t+48, t+72 — see
         src/training/forecast_targets.py)
    9.   Chronological split (70/15/15)
    10.  Scaling/Encoding — fit on train only

IMPORTANT — warm-up NaN dropping vs target NaN dropping (DO NOT MERGE):
    Lag/rolling features are NaN for the first ~168 rows per city
    (not enough history yet) — those rows are genuinely unusable and
    dropped here.
    Forecast targets are NaN for the LAST up-to-72 rows per city
    (not enough FUTURE data yet for the largest horizon) — those
    rows are only unusable for the LARGER horizons, not all of them,
    so they are deliberately NOT dropped here. Per-horizon target-NaN
    dropping happens later, in src/training/dataset.py, at load time
    — once we know which specific horizon is being trained for. If
    this file dropped rows with ANY target NaN, we'd needlessly lose
    valid 24h-horizon training data just because the 72h target
    wasn't available for the same row.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src.feature_engineering.air_quality_features import AirQualityFeatureEngineer
from src.feature_engineering.interaction_features import InteractionFeatureEngineer
from src.feature_engineering.lag_features import LagFeatureEngineer
from src.feature_engineering.rolling_features import RollingFeatureEngineer
from src.feature_engineering.scaling_encoding import ScalingEncodingEngineer
from src.feature_engineering.spatial_features import SpatialFeatureEngineer
from src.feature_engineering.temporal_features import TemporalFeatureEngineer
from src.feature_engineering.trend_features import TrendFeatureEngineer
from src.training.forecast_targets import ForecastTargetBuilder
from src.utils.logger import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = PROJECT_ROOT / "data" / "training"
INPUT_PATH = TRAINING_DIR / "training_dataset.parquet"


def _chronological_split(
    df: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    train_pct: float = 0.70,
    val_pct: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_pct)
    val_end = int(n * (train_pct + val_pct))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    logger.info(
        "Chronological split: train=%d, val=%d, test=%d", len(train_df), len(val_df), len(test_df),
    )
    return train_df, val_df, test_df


def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Runs Parts 1-7 (feature engineering only, no targets)."""
    logger.info("Starting Phase 3 feature engineering: %d row(s), %d column(s) in.", *df.shape)

    df = TemporalFeatureEngineer().build(df)
    df = LagFeatureEngineer().build(df)
    df = RollingFeatureEngineer().build(df)
    df = TrendFeatureEngineer().build(df)
    df = InteractionFeatureEngineer().build(df)
    df = AirQualityFeatureEngineer().build(df)
    df = SpatialFeatureEngineer().build(df)

    logger.info("Parts 1-7 complete: %d row(s), %d column(s) out.", *df.shape)
    return df


def build_features_pipeline(
    *,
    input_path: Path = INPUT_PATH,
    output_dir: Path = TRAINING_DIR,
    drop_warmup_nans: bool = True,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading %s", input_path)
    df = pd.read_parquet(input_path)

    # Parts 1-7
    df = build_all_features(df)

    # Multi-horizon forecast targets (target_aqi_t+24/48/72)
    target_builder = ForecastTargetBuilder()
    df = target_builder.build(df)
    target_columns = target_builder.all_target_columns()

    # Chronological split (targets included as normal columns for now —
    # horizon selection happens later in dataset.py)
    train_df, val_df, test_df = _chronological_split(df)

    if drop_warmup_nans:
        # CRITICAL: only drop rows with NaN in FEATURE columns (lag/
        # rolling warm-up), never based on the target columns — see
        # module docstring. Explicitly exclude target_columns from
        # the NaN check.
        before = len(train_df)
        train_df = train_df.dropna(subset=[c for c in train_df.columns if c not in target_columns])
        logger.info("Dropped %d warm-up/NaN row(s) from train split (%d -> %d).", before - len(train_df), before, len(train_df))

        val_df = val_df.dropna(subset=[c for c in val_df.columns if c not in target_columns])
        test_df = test_df.dropna(subset=[c for c in test_df.columns if c not in target_columns])

    # Part 8 (Scaling) — fit ONLY on train, transform all three splits.
    # ScalingEncodingEngineer must be told about ALL target columns so
    # none of them get scaled — none of the 3 are "the" target at this
    # stage, since horizon selection happens later at load time.
    scaler_engineer = ScalingEncodingEngineer(target_columns=target_columns)
    train_scaled = scaler_engineer.fit_transform(train_df)
    val_scaled = scaler_engineer.transform(val_df)
    test_scaled = scaler_engineer.transform(test_df)

    paths = {
        "train": output_dir / "features_train.parquet",
        "val": output_dir / "features_val.parquet",
        "test": output_dir / "features_test.parquet",
    }

    train_scaled.to_parquet(paths["train"], index=False)
    val_scaled.to_parquet(paths["val"], index=False)
    test_scaled.to_parquet(paths["test"], index=False)

    summary = {
        "target_columns": target_columns,
        "split_sizes": {"train": len(train_scaled), "val": len(val_scaled), "test": len(test_scaled)},
        "final_columns": len(train_scaled.columns),
    }
    (output_dir / "feature_build_report.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    logger.info(
        "Phase 3 complete: train=%d, val=%d, test=%d, targets=%s. Saved to %s",
        len(train_scaled), len(val_scaled), len(test_scaled), target_columns, output_dir,
    )

    return paths


def _main() -> int:
    parser = argparse.ArgumentParser(description="Build final Phase 3 features + multi-horizon targets")
    parser.add_argument("--keep-warmup-nans", action="store_true", help="Don't drop lag/rolling warm-up NaN rows.")
    args = parser.parse_args()

    try:
        build_features_pipeline(drop_warmup_nans=not args.keep_warmup_nans)
    except FileNotFoundError as exc:
        logger.error("Failed to build features: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(_main())
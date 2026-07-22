"""
build_features.py
===================
Suggested path: src/feature_engineering/build_features.py

Phase 3 orchestrator — runs all 8 parts in the correct order and
produces the final model-ready dataset, mirroring how
build_training_dataset.py orchestrates Phase 5.

ORDER MATTERS:
    1. Temporal       — no dependencies
    2. Lag             — no dependencies (needs sorted city+timestamp,
                         which every module already enforces itself)
    3. Rolling         — no dependencies
    4. Trend           — no dependencies
    5. Interaction     — no dependencies (row-wise on raw columns)
    6. Air Quality     — no dependencies (row-wise on raw columns)
    7. Spatial         — no dependencies (row-wise on raw columns)
    8. Scaling/Encoding — MUST run LAST: it scales/drops columns
                         produced by parts 1-7, and its scaler must
                         be fit ONLY on the training split.

Chronological train/val/test split (70/15/15, matching the split
your EDA already validated) happens BEFORE scaling — scaling needs
to know which rows are "training" to fit on.

Output (data/training/):
    features_train.parquet   <- scaled, model-ready
    features_val.parquet
    features_test.parquet
    scaler.joblib             <- fitted on train only, reusable at
                                 prediction time (Phase 9)
    feature_build_report.json <- what was scaled/dropped and why
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
    """
    Splits chronologically (NOT randomly) — same 70/15/15 split your
    EDA report already used. Random splitting would leak future
    information into training via nearby-in-time rows ending up on
    both sides of the split; a strict time cutoff avoids that.
    """
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_pct)
    val_end = int(n * (train_pct + val_pct))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    logger.info(
        "Chronological split: train=%d (%s -> %s), val=%d (%s -> %s), test=%d (%s -> %s)",
        len(train_df), train_df[timestamp_col].min(), train_df[timestamp_col].max(),
        len(val_df), val_df[timestamp_col].min(), val_df[timestamp_col].max(),
        len(test_df), test_df[timestamp_col].min(), test_df[timestamp_col].max(),
    )

    return train_df, val_df, test_df


def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Runs Parts 1-7 (everything except scaling) in order."""
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

    # --------------------------------------------------
    # Parts 1-7
    # --------------------------------------------------
    df = build_all_features(df)

    # --------------------------------------------------
    # Chronological split (before scaling, so the scaler only
    # ever sees the training portion)
    # --------------------------------------------------
    train_df, val_df, test_df = _chronological_split(df)

    if drop_warmup_nans:
        # Lag/rolling features are NaN for the first N rows per city
        # (not enough history yet) — drop rows with ANY NaN in the
        # engineered feature columns before fitting/scaling, rather
        # than silently imputing invented values into what should be
        # real historical signal.
        before = len(train_df)
        train_df = train_df.dropna()
        logger.info("Dropped %d warm-up/NaN row(s) from train split (%d -> %d).", before - len(train_df), before, len(train_df))
        val_df = val_df.dropna()
        test_df = test_df.dropna()

    # --------------------------------------------------
    # Part 8 — fit ONLY on train, transform all three splits
    # --------------------------------------------------
    scaler_engineer = ScalingEncodingEngineer()
    train_scaled, report = scaler_engineer.fit_transform(train_df)
    val_scaled = scaler_engineer.transform(val_df)
    test_scaled = scaler_engineer.transform(test_df)

    # --------------------------------------------------
    # Save outputs
    # --------------------------------------------------
    paths = {
        "train": output_dir / "features_train.parquet",
        "val": output_dir / "features_val.parquet",
        "test": output_dir / "features_test.parquet",
        "scaler": output_dir / "scaler.joblib",
        "report": output_dir / "feature_build_report.json",
    }

    train_scaled.to_parquet(paths["train"], index=False)
    val_scaled.to_parquet(paths["val"], index=False)
    test_scaled.to_parquet(paths["test"], index=False)
    scaler_engineer.save(paths["scaler"])

    report_dict = report.to_dict()
    report_dict["split_sizes"] = {"train": len(train_scaled), "val": len(val_scaled), "test": len(test_scaled)}
    paths["report"].write_text(json.dumps(report_dict, indent=2, default=str), encoding="utf-8")

    logger.info(
        "Phase 3 complete: train=%d, val=%d, test=%d, %d final feature(s). Saved to %s",
        len(train_scaled), len(val_scaled), len(test_scaled), report.final_feature_count, output_dir,
    )

    return paths


def _main() -> int:
    parser = argparse.ArgumentParser(description="Build final Phase 3 features (all 8 parts)")
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
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# CHANGED: no longer imports each individual feature-engineering class.
# The step order now lives in ONE shared place (pipeline_steps.py) that
# both this file and src/prediction/feature_pipeline.py import from —
# see that file's docstring for why (Training-Serving Skew fix).
from src.feature_engineering.pipeline_steps import run_feature_engineering_steps
from src.feature_engineering.scaling_encoding import ScalingEncodingEngineer
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

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    logger.info(
        "Chronological split (Raw): train=%d, val=%d, test=%d", len(train_df), len(val_df), len(test_df),
    )
    return train_df, val_df, test_df


def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs Parts 1-7 (feature engineering only, no targets).

    CHANGED: delegates to run_feature_engineering_steps() from
    pipeline_steps.py instead of hand-calling each engineer in sequence.
    Output is IDENTICAL to before — same 7 classes, same order — this
    change only removes the ability for this file's order to silently
    drift away from src/prediction/feature_pipeline.py's order.
    """
    return run_feature_engineering_steps(df)


def build_features_pipeline(
    *,
    input_path: Path = INPUT_PATH,
    output_dir: Path = TRAINING_DIR,
    drop_warmup_nans: bool = True,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading %s", input_path)
    df = pd.read_parquet(input_path)

    # 1. CRITICAL FIX: Split FIRST before any feature engineering or scaling
    train_df, val_df, test_df = _chronological_split(df)

    # 2. Build targets on each split independently (or build on full before split, but targets don't leak features)
    target_builder = ForecastTargetBuilder()
    train_df = target_builder.build(train_df)
    val_df = target_builder.build(val_df)
    test_df = target_builder.build(test_df)
    target_columns = target_builder.all_target_columns()

    # 3. Run Parts 1-7 Feature Engineering SEPARATELY on each split
    # (To prevent rolling/lag windows from crossing split boundaries)
    logger.info("Running feature engineering on Train split...")
    train_df = build_all_features(train_df)

    logger.info("Running feature engineering on Validation split...")
    val_df = build_all_features(val_df)

    logger.info("Running feature engineering on Test split...")
    test_df = build_all_features(test_df)

    if drop_warmup_nans:
        before = len(train_df)
        train_df = train_df.dropna(subset=[c for c in train_df.columns if c not in target_columns])
        logger.info("Dropped %d warm-up/NaN row(s) from train split (%d -> %d).", before - len(train_df), before, len(train_df))

        val_df = val_df.dropna(subset=[c for c in val_df.columns if c not in target_columns])
        test_df = test_df.dropna(subset=[c for c in test_df.columns if c not in target_columns])

    # 4. Scaling & Encoding — FIT strictly on train, TRANSFORM val and test
    scaler_engineer = ScalingEncodingEngineer(target_columns=target_columns)

    logger.info("Fitting scaler on Train split and transforming...")
    train_scaled = scaler_engineer.fit_transform(train_df)

    logger.info("Transforming Validation split...")
    val_scaled = scaler_engineer.transform(val_df)

    logger.info("Transforming Test split...")
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
        "Phase 3 complete (Leakage-Free): train=%d, val=%d, test=%d. Saved to %s",
        len(train_scaled), len(val_scaled), len(test_scaled), output_dir,
    )

    return paths


def _main() -> int:
    parser = argparse.ArgumentParser(description="Build leakage-free Phase 3 features + multi-horizon targets")
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
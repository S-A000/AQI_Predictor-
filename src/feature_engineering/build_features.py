from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

# CHANGED: no longer imports each individual feature-engineering class.
# The step order now lives in ONE shared place (pipeline_steps.py) that
# both this file and src/prediction/feature_pipeline.py import from —
# see that file's docstring for why (Training-Serving Skew fix).
from src.feature_engineering.pipeline_steps import run_feature_engineering_steps
from src.feature_engineering.scaling_encoding import ScalingEncodingEngineer
from src.training.forecast_targets import ForecastTargetBuilder
from src.utils.constants import METADATA_COLUMNS
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
    df = df.sort_values([timestamp_col, "city"]).reset_index(drop=True)
    unique_timestamps = df[timestamp_col].drop_duplicates().sort_values()
    n_timestamps = len(unique_timestamps)
    train_end = int(n_timestamps * train_pct)
    val_end = int(n_timestamps * (train_pct + val_pct))

    if train_end <= 0 or val_end <= train_end or val_end >= n_timestamps:
        raise ValueError(
            "Not enough distinct timestamps for chronological train/validation/test splits."
        )

    train_times = set(unique_timestamps.iloc[:train_end])
    validation_times = set(unique_timestamps.iloc[train_end:val_end])
    test_times = set(unique_timestamps.iloc[val_end:])

    train_df = df[df[timestamp_col].isin(train_times)].copy()
    val_df = df[df[timestamp_col].isin(validation_times)].copy()
    test_df = df[df[timestamp_col].isin(test_times)].copy()

    logger.info(
        "Chronological split (Raw): train=%d, val=%d, test=%d", len(train_df), len(val_df), len(test_df),
    )
    return train_df, val_df, test_df


def _canonicalize_observations(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the production city/hour key before targets or splitting."""

    required_columns = {"city", "timestamp", "aqi"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Training data is missing required columns: {sorted(missing_columns)}"
        )

    result = df.copy()
    if "event_hour" in result.columns:
        event_hours = pd.to_datetime(
            result["event_hour"], utc=True, errors="coerce"
        )
        result["timestamp"] = event_hours.fillna(
            pd.to_datetime(result["timestamp"], utc=True, errors="coerce")
        )
        result = result.drop(columns=["event_hour"])
    else:
        result["timestamp"] = pd.to_datetime(
            result["timestamp"], utc=True, errors="coerce"
        )

    result["timestamp"] = result["timestamp"].dt.floor("h")
    result = result.dropna(subset=["city"])
    result["city"] = result["city"].astype(str).str.strip().str.title()
    result = result[
        result["city"].ne("") & result["timestamp"].notna()
    ].copy()

    before = len(result)
    result = result.drop_duplicates(
        subset=["city", "timestamp"], keep="last"
    )
    removed = before - len(result)
    if removed:
        logger.warning(
            "Removed %d duplicate city/event-hour row(s) before target creation.",
            removed,
        )

    return result.sort_values(["city", "timestamp"]).reset_index(drop=True)


def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the canonical feature-engineering pipeline once.

    BigQuery currently contains raw/source columns plus a small number
    of legacy/lightweight derived columns. Those derived columns are
    removed first so the canonical pipeline regenerates them cleanly.
    """

    df = df.copy()

    precomputed_features = [
        "hour",
        "day",
        "month",
        "day_of_week",
        "is_weekend",
        "aqi_change_rate",
        "aqi_rolling_mean_3h",
    ]

    columns_to_drop = [
        col for col in precomputed_features
        if col in df.columns
    ]

    if columns_to_drop:
        logger.info(
            "Removing %d precomputed BigQuery feature(s) before canonical "
            "feature engineering: %s",
            len(columns_to_drop),
            columns_to_drop,
        )

        df = df.drop(columns=columns_to_drop)

    return run_feature_engineering_steps(df)


def build_features_pipeline(
    *,
    input_path: Path = INPUT_PATH,
    output_dir: Path = TRAINING_DIR,
    drop_warmup_nans: bool = True,
    already_engineered: bool = False,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading %s", input_path)
    df = pd.read_parquet(input_path)
    df = _canonicalize_observations(df)

    # Exact elapsed-time targets are constructed on the complete canonical
    # timeline so valid labels at split boundaries are not discarded.
    target_builder = ForecastTargetBuilder()
    df = target_builder.build(df)
    target_columns = target_builder.all_target_columns()

    train_df, val_df, test_df = _chronological_split(df)

    if already_engineered:
        logger.info(
            "Input is the engineered BigQuery feature table; skipping duplicate feature engineering."
        )
    else:
        logger.info("Running feature engineering on Train split...")
        train_df = build_all_features(train_df)

        logger.info("Running feature engineering on Validation split...")
        val_df = build_all_features(val_df)

        logger.info("Running feature engineering on Test split...")
        test_df = build_all_features(test_df)

    if drop_warmup_nans and not already_engineered:
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

    excluded_columns = set(target_columns) | set(METADATA_COLUMNS) | {
        "date",
        "city",
        "split",
        "event_hour",
    }
    ordered_feature_names = [
        column for column in train_scaled.columns if column not in excluded_columns
    ]
    if not ordered_feature_names:
        raise ValueError("Preprocessing produced an empty model feature schema.")

    for split_name, split_df in {
        "validation": val_scaled,
        "test": test_scaled,
    }.items():
        split_features = [
            column for column in split_df.columns if column not in excluded_columns
        ]
        if split_features != ordered_feature_names:
            missing = [c for c in ordered_feature_names if c not in split_features]
            unexpected = [c for c in split_features if c not in ordered_feature_names]
            raise ValueError(
                f"{split_name} preprocessing schema mismatch; "
                f"missing={missing}, unexpected={unexpected}"
            )

    scaler_engineer.model_feature_columns_ = list(ordered_feature_names)
    scaler_engineer.final_columns_ = list(train_scaled.columns)
    scaler_path = output_dir / "scaler.joblib"
    joblib.dump(scaler_engineer, scaler_path)
    logger.info("Persisted fitted preprocessing state to %s", scaler_path)

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
        "ordered_feature_names": ordered_feature_names,
        "preprocessor_path": str(scaler_path),
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

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict

from sklearn.metrics import r2_score

# ------------------------------------------------------------------
# Project path configuration
# ------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))


# ------------------------------------------------------------------
# Enterprise pipeline dependencies
# ------------------------------------------------------------------

from src.feature_engineering.build_features import build_features_pipeline
from src.feature_store.bigquery_feature_store import BigQueryFeatureStore
from src.training.dataset import load_prepared_splits
from src.training.train_multi_models import MultiModelTrainer
from src.utils.logger import get_logger


logger = get_logger(__name__)

TRAINING_DATA_PATH = (
    ROOT_DIR
    / "data"
    / "training"
    / "training_dataset.parquet"
)

TARGET_HORIZONS = [24, 48, 72]


def sync_training_data_from_bigquery() -> Path:
    """
    Download the complete raw training dataset from BigQuery.

    The dataset is saved to the local Parquet path expected by
    build_features_pipeline().

    BigQuery remains the primary feature repository while the generated
    local Parquet file acts as the training pipeline's runtime input.

    Returns:
        Path to the downloaded training dataset.
    """

    logger.info(
        "Connecting to BigQuery feature repository..."
    )

    feature_store = BigQueryFeatureStore()

    training_df = feature_store.get_training_features()

    if training_df.empty:
        raise ValueError(
            "BigQuery returned an empty training dataset."
        )

    required_columns = {
        "city",
        "timestamp",
        "aqi",
    }

    missing_columns = (
        required_columns
        - set(training_df.columns)
    )

    if missing_columns:
        raise ValueError(
            "BigQuery training dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    before_deduplication = len(training_df)

    # Protect training from repeated hourly Sandbox appends.
    training_df = (
        training_df
        .drop_duplicates(
            subset=["city", "timestamp"],
            keep="last",
        )
        .sort_values(
            ["city", "timestamp"],
        )
        .reset_index(drop=True)
    )

    removed_duplicates = (
        before_deduplication
        - len(training_df)
    )

    if removed_duplicates:
        logger.warning(
            "Removed %d duplicate city/timestamp row(s) "
            "from BigQuery training data.",
            removed_duplicates,
        )

    TRAINING_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    training_df.to_parquet(
        TRAINING_DATA_PATH,
        index=False,
    )

    logger.info(
        "Training dataset synced from BigQuery | "
        "rows=%d | columns=%d | path=%s",
        len(training_df),
        len(training_df.columns),
        TRAINING_DATA_PATH,
    )

    return TRAINING_DATA_PATH


def run_end_to_end_pipeline() -> Dict[int, Dict[str, Any]]:
    """
    Execute the direct multi-horizon AQI forecasting MLOps pipeline.

    Workflow:
        0. Download raw training data from BigQuery.
        1. Build leakage-safe features and forecast targets.
        2. Prepare train, validation and test datasets.
        3. Train candidate models for 24h, 48h and 72h horizons.
        4. Evaluate candidates and register the best model.
        5. Output final leaderboard and execution timings.

    Returns:
        Mapping of forecast horizon to winning model metrics.
    """

    logger.info(
        "=================================================="
    )
    logger.info(
        "STARTING MULTI-HORIZON MLOPS PIPELINE"
    )
    logger.info(
        "=================================================="
    )

    pipeline_start_time = time.time()

    timings: Dict[str, float] = {}
    results: Dict[int, Dict[str, Any]] = {}

    try:
        # ----------------------------------------------------------
        # STEP 0: Sync raw training dataset from BigQuery
        # ----------------------------------------------------------

        logger.info(
            "Step 0: Syncing training data from BigQuery..."
        )

        sync_start = time.time()

        training_path = (
            sync_training_data_from_bigquery()
        )

        sync_duration = (
            time.time()
            - sync_start
        )

        timings["BigQuery Sync"] = sync_duration

        logger.info(
            "-> BigQuery training sync completed in %.0f seconds | "
            "path=%s",
            sync_duration,
            training_path,
        )

        # ----------------------------------------------------------
        # STEP 1: Feature engineering and target generation
        # ----------------------------------------------------------

        logger.info(
            "Step 1: Starting Feature Engineering "
            "& Target Generation..."
        )

        feature_start = time.time()

        # Generates:
        # - features_train.parquet
        # - features_val.parquet
        # - features_test.parquet
        build_features_pipeline(
            input_path=training_path,
        )

        feature_duration = (
            time.time()
            - feature_start
        )

        timings["Feature Engineering"] = (
            feature_duration
        )

        logger.info(
            "-> Feature engineering completed in %.0f seconds.",
            feature_duration,
        )

        # ----------------------------------------------------------
        # STEP 2: Train a direct model for each horizon
        # ----------------------------------------------------------

        for horizon in TARGET_HORIZONS:
            logger.info(
                "--------------------------------------------------"
            )
            logger.info(
                "Step 2: Training Direct Forecast Model "
                "for %dh Horizon",
                horizon,
            )
            logger.info(
                "--------------------------------------------------"
            )

            horizon_start = time.time()

            # Load strictly aligned splits for this horizon.
            splits = load_prepared_splits(
                horizon_hours=horizon,
            )

            # Train candidates, log experiments and register winner.
            trainer = MultiModelTrainer()

            (
                winner_name,
                winner_rmse,
                best_artifact,
            ) = trainer.train_and_evaluate_all(
                splits=splits,
                horizon_hours=horizon,
            )

            # Calculate R² for final reporting.
            test_predictions = best_artifact.predict(
                splits.X_test
            )

            winner_r2 = r2_score(
                splits.y_test,
                test_predictions,
            )

            results[horizon] = {
                "winner": winner_name,
                "rmse": float(winner_rmse),
                "r2": float(winner_r2),
            }

            horizon_duration = (
                time.time()
                - horizon_start
            )

            timings[
                f"{horizon}h Training"
            ] = horizon_duration

            logger.info(
                "-> %dh training completed in %.0f seconds.",
                horizon,
                horizon_duration,
            )

        # ----------------------------------------------------------
        # STEP 3: Final leaderboard
        # ----------------------------------------------------------

        total_pipeline_time = (
            time.time()
            - pipeline_start_time
        )

        timings["Total Pipeline"] = (
            total_pipeline_time
        )

        _print_leaderboard(
            results=results,
            timings=timings,
        )

        return results

    except Exception as err:
        logger.exception(
            "Pipeline execution failed due to error: %s",
            err,
        )
        raise


def _print_leaderboard(
    results: Dict[int, Dict[str, Any]],
    timings: Dict[str, float],
) -> None:
    """
    Print the winning model and metrics for each forecast horizon.
    """

    print("\n======================================")
    print("          TRAINING COMPLETE           ")
    print("======================================")

    for horizon, metrics in results.items():
        print(f"\n{horizon}h")
        print(f"Winner:\t{metrics['winner']}")
        print(f"RMSE:\t{metrics['rmse']:.4f}")
        print(f"R²:\t{metrics['r2']:.4f}")
        print("\n--------------------------------------")

    print("\n======================================")
    print("    Pipeline Finished Successfully    ")
    print("======================================")

    print("\n[ Execution Timings ]")

    for phase, duration in timings.items():
        print(
            f"{phase:<20}: "
            f"{duration:>5.0f} sec"
        )

    print()


if __name__ == "__main__":
    try:
        final_results = (
            run_end_to_end_pipeline()
        )

    except KeyboardInterrupt:
        logger.warning(
            "Pipeline execution forcefully aborted by user."
        )
        sys.exit(1)

    except Exception:
        sys.exit(1)
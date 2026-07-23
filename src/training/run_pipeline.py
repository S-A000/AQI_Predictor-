"""
Suggested path: src/training/run_pipeline.py

SINGLE RESPONSIBILITY: Master orchestrator script to run the end-to-end 
Multi-Horizon AQI Forecasting MLOps pipeline sequentially: 
Feature Engineering -> Model Training (24h, 48h, 72h) -> Model Evaluation -> MLflow Registry.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict

from sklearn.metrics import r2_score

# Ensure root directory is in python path to resolve src imports
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# Enterprise Pipeline Dependencies
from src.feature_engineering.build_features import build_features_pipeline
from src.training.dataset import load_prepared_splits
from src.training.train_multi_models import MultiModelTrainer
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_end_to_end_pipeline() -> Dict[int, Dict[str, Any]]:
    """
    Executes the full Direct Multi-Horizon AQI Forecasting MLOps pipeline.
    
    Workflow:
        1. Build Features (Generates 24h, 48h, 72h targets & applies scaling).
        2. Iterate over horizons [24, 48, 72].
        3. Load horizon-specific dataset splits.
        4. Train multiple algorithms, evaluate, and register the best one.
        5. Aggregate and report final leaderboard metrics.
        
    Returns:
        Dict mapping horizon hours to their respective best model metrics.
    """
    logger.info("==================================================")
    logger.info("🚀 STARTING MULTI-HORIZON MLOPS PIPELINE")
    logger.info("==================================================")

    pipeline_start_time = time.time()
    timings: Dict[str, float] = {}
    results: Dict[int, Dict[str, Any]] = {}
    target_horizons = [24, 48, 72]

    try:
        # -------------------------------------------------------------
        # STEP 1 & 2: Execute Central Feature Engineering Pipeline
        # -------------------------------------------------------------
        logger.info("Step 1: Starting Feature Engineering & Target Generation...")
        fe_start = time.time()
        
        # Generates features_train.parquet, features_val.parquet, features_test.parquet
        build_features_pipeline()
        
        fe_duration = time.time() - fe_start
        timings["Feature Engineering"] = fe_duration
        logger.info("-> Feature engineering completed in %.0f seconds.", fe_duration)

        # -------------------------------------------------------------
        # STEP 3: Iterate and Train for Each Forecast Horizon
        # -------------------------------------------------------------
        for horizon in target_horizons:
            logger.info("--------------------------------------------------")
            logger.info("Step 3: Training Direct Forecast Model for %dh Horizon", horizon)
            logger.info("--------------------------------------------------")
            
            horizon_start = time.time()

            # Load strictly aligned dataset for the specific horizon
            splits = load_prepared_splits(horizon_hours=horizon)
            
            # Train candidate models, log to MLflow, and auto-register winner
            trainer = MultiModelTrainer()
            winner_name, winner_rmse, best_artifact = trainer.train_and_evaluate_all(
                splits=splits, 
                horizon_hours=horizon
            )

            # Calculate R² strictly for reporting purposes
            test_preds = best_artifact.predict(splits.X_test)
            winner_r2 = r2_score(splits.y_test, test_preds)

            # Record metrics
            results[horizon] = {
                "winner": winner_name,
                "rmse": float(winner_rmse),
                "r2": float(winner_r2)
            }

            horizon_duration = time.time() - horizon_start
            timings[f"{horizon}h Training"] = horizon_duration
            logger.info("-> %dh Training completed in %.0f seconds.", horizon, horizon_duration)

        # -------------------------------------------------------------
        # STEP 4: Output Final Leaderboard & Timings
        # -------------------------------------------------------------
        total_pipeline_time = time.time() - pipeline_start_time
        timings["Total Pipeline"] = total_pipeline_time

        _print_leaderboard(results, timings)

        return results

    except Exception as err:
        logger.exception("❌ Pipeline execution failed due to error: %s", err)
        raise


def _print_leaderboard(results: Dict[int, Dict[str, Any]], timings: Dict[str, float]) -> None:
    """Formats and outputs the final execution leaderboard and pipeline timings to standard output."""
    
    print("\n======================================")
    print("          TRAINING COMPLETE           ")
    print("======================================")
    
    # Iterate and print per-horizon results
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
        print(f"{phase:<20}: {duration:>5.0f} sec")
    print("\n")


if __name__ == "__main__":
    try:
        final_results = run_end_to_end_pipeline()
    except KeyboardInterrupt:
        logger.warning("Pipeline execution forcefully aborted by user.")
        sys.exit(1)
    except Exception:
        sys.exit(1)
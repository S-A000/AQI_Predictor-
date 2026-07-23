"""
SINGLE RESPONSIBILITY: Master Entry Point / CLI Orchestrator for the AQI Forecasting MLOps Project.
Parses user commands, maps them to underlying modules, handles global exceptions, 
and provides execution timing and logging. Fully compatible with project architecture.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Main is located at the project root
PROJECT_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------
# Enterprise Module Imports
# ---------------------------------------------------------
from src.prediction.forecast import AQIForecaster
from src.prediction.predictor import AQIPredictor
from src.prediction.validator import PredictionPayload
from src.training.dataset import load_prepared_splits
from src.training.evaluate import ModelEvaluator
from src.training.run_pipeline import run_end_to_end_pipeline
from src.training.train_multi_models import MultiModelTrainer
from src.utils.constants import BASE_DIR, MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_HORIZONS = [24, 48, 72]


# =========================================================
# STARTUP VALIDATION & DISPLAY
# =========================================================

def _validate_startup_environment() -> None:
    """Validates presence of required project paths using central constants."""
    for required_path in [BASE_DIR, MODELS_DIR]:
        if not required_path.exists():
            raise FileNotFoundError(
                f"Required project directory does not exist: {required_path}. "
                "Please verify environment setup."
            )


def _print_banner(command: str) -> None:
    """Prints a lightweight CLI startup banner."""
    python_version = sys.version.split()[0]
    horizons_str = ", ".join(f"{h}h" for h in SUPPORTED_HORIZONS)

    print("=" * 60)
    print("  🏢 ENTERPRISE AQI FORECASTING MLOPS PLATFORM")
    print("=" * 60)
    print(f"  Python Version   : {python_version}")
    print(f"  Project Root     : {PROJECT_ROOT}")
    print(f"  Active Command   : {command.upper()}")
    print(f"  Supported Targets: {horizons_str}")
    print("=" * 60 + "\n")


def _print_execution_summary(
    status: str, 
    execution_time: float, 
    details: List[Dict[str, Any]]
) -> None:
    """Prints a standardized execution summary upon command completion."""
    print("\n" + "=" * 60)
    print("  📊 EXECUTION SUMMARY")
    print("=" * 60)
    print(f"  Status         : {status}")
    print(f"  Total Duration : {execution_time:.2f} seconds")
    print("-" * 60)

    if details:
        for item in details:
            horizon = item.get("horizon", "N/A")
            model = item.get("model", "N/A")
            msg = item.get("message", "Completed")
            print(f"  -> Horizon: {horizon:>3}h | Model: {model:<18} | Status: {msg}")
    else:
        print("  -> No detailed horizon metrics available.")
    print("=" * 60 + "\n")


def _validate_horizons(args: argparse.Namespace) -> List[int]:
    """Validates and resolves requested horizons from CLI arguments."""
    if getattr(args, "all", False):
        return SUPPORTED_HORIZONS

    if getattr(args, "horizon", None) is not None:
        if args.horizon not in SUPPORTED_HORIZONS:
            raise ValueError(f"Unsupported horizon: {args.horizon}. Must be one of {SUPPORTED_HORIZONS}.")
        return [args.horizon]

    return [24]


# =========================================================
# COMMAND HANDLERS
# =========================================================

def handle_train(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Orchestrates model training via MultiModelTrainer."""
    horizons = _validate_horizons(args)
    logger.info("Initializing MultiModelTrainer for horizons: %s", horizons)

    trainer = MultiModelTrainer()
    summary_details = []

    for horizon in horizons:
        logger.info(">>> Executing Training Pipeline for %dh Horizon <<<", horizon)

        splits = load_prepared_splits(horizon_hours=horizon)

        winner_name, winner_rmse, best_artifact = trainer.train_and_evaluate_all(
            splits=splits, 
            horizon_hours=horizon
        )

        model_version = getattr(best_artifact, "model_version", getattr(best_artifact, "version", "Latest"))

        print(f"\n✅ [ HORIZON {horizon}h TRAINING COMPLETE ]")
        print(f"   Winning Model  : {winner_name}")
        print(f"   Validation RMSE: {winner_rmse:.4f}")
        print(f"   Model Version  : {model_version}\n")

        summary_details.append({
            "horizon": horizon,
            "model": winner_name,
            "message": f"RMSE: {winner_rmse:.4f}"
        })

    return summary_details


def handle_evaluate(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Orchestrates model evaluation via ModelEvaluator."""
    horizons = _validate_horizons(args)
    logger.info("Initializing ModelEvaluator for horizons: %s", horizons)

    evaluator = ModelEvaluator()
    summary_details = []

    for horizon in horizons:
        logger.info(">>> Executing Evaluation for %dh Horizon <<<", horizon)

        splits = load_prepared_splits(horizon_hours=horizon)
        # Compatible call: evaluate only takes splits
        evaluator.evaluate(splits=splits)

        summary_details.append({
            "horizon": horizon,
            "model": "All Registered",
            "message": "Evaluation Successful"
        })

    return summary_details


def handle_predict(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Orchestrates single-payload inference via AQIPredictor."""
    payload_path = Path(args.payload)
    if not payload_path.exists():
        raise FileNotFoundError(f"Prediction payload file not found: {payload_path}")

    logger.info("Reading prediction payload from %s", payload_path)
    with open(payload_path, "r", encoding="utf-8") as f:
        raw_payload = json.load(f)

    try:
        validated_payload = PredictionPayload(**raw_payload)
        logger.info("Payload successfully validated against schema.")
    except Exception as e:
        raise ValueError(f"Schema Validation Error: {e}")

    horizon = getattr(args, "horizon", 24)
    if horizon not in SUPPORTED_HORIZONS:
        raise ValueError(f"Unsupported prediction horizon: {horizon}")

    predictor = AQIPredictor()

    # Pydantic v2 and v1 backward compatibility
    if hasattr(validated_payload, "model_dump"):
        payload_dict = validated_payload.model_dump()
    else:
        payload_dict = validated_payload.dict()

    result = predictor.predict_single(payload=payload_dict, horizon_hours=horizon)

    print("\n" + "=" * 40)
    print("🎯 INFERENCE RESULT")
    print("=" * 40)
    print(json.dumps(result, indent=4))
    print("=" * 40 + "\n")

    model_used = result.get("model_version", "Production")
    return [{"horizon": horizon, "model": model_used, "message": f"Predicted AQI: {result.get('predicted_aqi', 'N/A')}"}]


def handle_forecast(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Orchestrates direct multi-step forecasting via AQIForecaster."""
    hours = args.hours
    if hours < 1 or hours > 72:
        raise ValueError(f"Forecast hours must be between 1 and 72. Received: {hours}")

    logger.info("Initializing AQIForecaster for %d-hour forecast...", hours)
    forecaster = AQIForecaster()

    # Dynamic method resolution for forecaster API compatibility
    if hasattr(forecaster, "forecast"):
        forecasts = forecaster.forecast(horizon_hours=hours)
    elif hasattr(forecaster, "predict"):
        forecasts = forecaster.predict(horizon_hours=hours)
    elif hasattr(forecaster, "generate_forecast"):
        forecasts = forecaster.generate_forecast(horizon_hours=hours)
    else:
        raise AttributeError("AQIForecaster has no supported forecasting method ('forecast', 'predict', or 'generate_forecast').")

    print("\n" + "=" * 50)
    print(f"📈 DIRECT FORECAST HORIZON: {hours} HOURS")
    print("=" * 50)
    if isinstance(forecasts, list):
        for f_point in forecasts:
            if isinstance(f_point, dict):
                step = f_point.get("horizon_step", "N/A")
                ts = f_point.get("timestamp", "N/A")
                aqi = f_point.get("predicted_aqi", 0.0)
                print(f"Step {step:>2} | {ts} | Predicted AQI: {aqi:.2f}" if isinstance(aqi, (int, float)) else f"Step {step:>2} | {ts} | Output: {aqi}")
            else:
                print(f"Forecast output: {f_point}")
    print("=" * 50 + "\n")

    count = len(forecasts) if isinstance(forecasts, list) else 1
    return [{"horizon": hours, "model": "Auto-selected Direct", "message": f"Generated {count} step(s)"}]


def handle_pipeline(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Orchestrates the complete end-to-end MLOps pipeline."""
    logger.info("Triggering Full MLOps Pipeline Execution...")

    pipeline_results = run_end_to_end_pipeline()

    summary_details = []
    if isinstance(pipeline_results, dict):
        for hrz, metrics in pipeline_results.items():
            if isinstance(metrics, dict):
                summary_details.append({
                    "horizon": hrz,
                    "model": metrics.get("winner", metrics.get("model", "Unknown")),
                    "message": f"RMSE: {metrics.get('rmse', 0.0):.4f}"
                })
            else:
                summary_details.append({"horizon": hrz, "model": "Pipeline", "message": str(metrics)})
    elif isinstance(pipeline_results, list):
        for item in pipeline_results:
            summary_details.append({"horizon": "All", "model": "Pipeline", "message": str(item)})
    else:
        summary_details.append({"horizon": "All", "model": "Pipeline", "message": "Pipeline Executed Successfully"})

    return summary_details


# =========================================================
# CLI SETUP & ROUTING
# =========================================================

def create_parser() -> argparse.ArgumentParser:
    """Constructs the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="aqi_mlops",
        description="Enterprise AQI Forecasting MLOps Platform - Unified CLI",
        epilog="Examples:\n"
               "  python main.py train --all\n"
               "  python main.py forecast --hours 72\n"
               "  python main.py predict payload.json --horizon 24\n"
               "  python main.py pipeline\n",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(title="Commands", dest="command", required=True)

    # TRAIN
    train_parser = subparsers.add_parser("train", help="Train models for specific or all horizons")
    train_parser.add_argument("--horizon", type=int, choices=SUPPORTED_HORIZONS, help="Target horizon (24, 48, or 72)")
    train_parser.add_argument("--all", action="store_true", help="Train models for all horizons sequentially")
    train_parser.set_defaults(func=handle_train)

    # EVALUATE
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate existing models")
    eval_parser.add_argument("--horizon", type=int, choices=SUPPORTED_HORIZONS, help="Target horizon (24, 48, or 72)")
    eval_parser.add_argument("--all", action="store_true", help="Evaluate models for all horizons")
    eval_parser.set_defaults(func=handle_evaluate)

    # PREDICT
    predict_parser = subparsers.add_parser("predict", help="Run single inference from a JSON payload")
    predict_parser.add_argument("payload", type=str, help="Path to the JSON payload file")
    predict_parser.add_argument("--horizon", type=int, choices=SUPPORTED_HORIZONS, default=24, help="Model horizon to use (default: 24)")
    predict_parser.set_defaults(func=handle_predict)

    # FORECAST
    forecast_parser = subparsers.add_parser("forecast", help="Generate a multi-step forward time-series forecast")
    forecast_parser.add_argument("--hours", type=int, required=True, help="Number of hours to forecast (1 to 72)")
    forecast_parser.set_defaults(func=handle_forecast)

    # PIPELINE
    pipeline_parser = subparsers.add_parser("pipeline", help="Run the complete end-to-end training pipeline")
    pipeline_parser.add_argument("--horizon", type=int, choices=SUPPORTED_HORIZONS, help="Run pipeline for a specific horizon")
    pipeline_parser.add_argument("--all", action="store_true", help="Run pipeline for all horizons")
    pipeline_parser.set_defaults(func=handle_pipeline)

    return parser


def main() -> None:
    """Main CLI entry point with exception handling."""
    parser = create_parser()
    args = parser.parse_args()

    start_time = time.time()
    logger.info(">>> AQI MLOps CLI Started | Command: %s <<<", args.command.upper())

    try:
        _validate_startup_environment()
        _print_banner(args.command)

        summary_details = args.func(args)

        execution_time = time.time() - start_time
        _print_execution_summary("SUCCESS", execution_time, summary_details)
        sys.exit(0)

    except ValueError as ve:
        logger.error("Validation Error: %s", ve)
        _print_execution_summary("FAILED (Validation)", time.time() - start_time, [{"message": str(ve)}])
        sys.exit(1)

    except FileNotFoundError as fnfe:
        logger.error("File/Directory Not Found Error: %s", fnfe)
        _print_execution_summary("FAILED (Not Found)", time.time() - start_time, [{"message": str(fnfe)}])
        sys.exit(1)

    except json.JSONDecodeError as jde:
        logger.error("Payload Parsing Error - Invalid JSON format: %s", jde)
        _print_execution_summary("FAILED (JSON Parsing)", time.time() - start_time, [{"message": str(jde)}])
        sys.exit(1)

    except KeyboardInterrupt:
        logger.warning("Execution interrupted by user.")
        _print_execution_summary("ABORTED", time.time() - start_time, [{"message": "Interrupted by User"}])
        sys.exit(130)

    except Exception as e:
        logger.exception("Fatal error occurred during '%s': %s", args.command, e)
        _print_execution_summary("FATAL ERROR", time.time() - start_time, [{"message": str(e)}])
        sys.exit(1)


if __name__ == "__main__":
    main()
"""
build_training_dataset.py
===========================
Suggested path: src/dataset_pipeline/build_training_dataset.py

Orchestration script (Phase 5, dataset building for training). Ties
together the three single-responsibility modules — exactly the way
run_pipeline.py orchestrates ingestion and historical_backfill.py
orchestrates the historical path:

    HistoricalDatasetBuilder  -> merges all engineered sources
    QualityChecker            -> reports + cleans
    DatasetStatisticsGenerator -> summary stats, saved for the record

Usage:
    python -m src.dataset_pipeline.build_training_dataset
    python -m src.dataset_pipeline.build_training_dataset --missing-strategy zero
    python -m src.dataset_pipeline.build_training_dataset --fail-on-issues

Output (data/training/):
    training_dataset.parquet   <- final, cleaned dataset for Phase 7 training
    quality_report.json        <- pre-cleaning quality findings
    dataset_statistics.json    <- post-cleaning summary statistics
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.dataset_pipeline.dataset_statistics import DatasetStatisticsGenerator
from src.dataset_pipeline.historical_dataset_builder import HistoricalDatasetBuilder
from src.dataset_pipeline.quality_checker import QualityChecker
from src.utils.logger import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = PROJECT_ROOT / "data" / "training"


def build_training_dataset(
    *,
    missing_value_strategy: str = "mean",
    fail_on_quality_issues: bool = False,
    output_dir: Path = TRAINING_DIR,
) -> Path:
    """
    Runs the full Phase 5 sequence and writes the final training
    dataset + its accompanying quality/statistics reports.

    Returns the path to the written training_dataset.parquet.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # 1. Merge every engineered source (live + historical, all versions)
    # --------------------------------------------------
    builder = HistoricalDatasetBuilder()
    build_result = builder.build()
    df = build_result.dataframe
    logger.info(
        "Built merged dataset: %d row(s) from %d source file(s).",
        len(df), len(build_result.sources),
    )

    # --------------------------------------------------
    # 2. Quality check BEFORE cleaning (report reflects the raw merge)
    # --------------------------------------------------
    checker = QualityChecker()
    report = checker.check(df)

    report_path = output_dir / "quality_report.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")
    logger.info("Saved quality report -> %s", report_path)

    if fail_on_quality_issues and not report.is_clean:
        raise ValueError(
            f"Quality check failed with {len(report.issues)} issue(s); "
            f"see {report_path} for details. Aborting training dataset build "
            "(pass --fail-on-issues=False, or fix the underlying data, to proceed)."
        )

    # --------------------------------------------------
    # 3. Clean (dedup + missing-value handling) — same rules as ingestion
    # --------------------------------------------------
    clean_df = checker.clean(df, missing_value_strategy=missing_value_strategy)

    # --------------------------------------------------
    # 4. Statistics on the CLEANED dataset — what training will actually see
    # --------------------------------------------------
    stats_generator = DatasetStatisticsGenerator()
    stats = stats_generator.generate(clean_df)
    stats.save(output_dir / "dataset_statistics.json")

    # --------------------------------------------------
    # 5. Write final training dataset
    # --------------------------------------------------
    output_path = output_dir / "training_dataset.parquet"
    clean_df.to_parquet(output_path, index=False)
    logger.info("Saved final training dataset (%d row(s)) -> %s", len(clean_df), output_path)

    return output_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the final training dataset (Phase 5)")
    parser.add_argument(
        "--missing-strategy", default="mean", choices=["mean", "zero", "drop"],
        help="How to fill missing numeric values during cleaning.",
    )
    parser.add_argument(
        "--fail-on-issues", action="store_true",
        help="Abort the build if the quality report finds issues, instead of just warning.",
    )
    return parser


def _main() -> int:
    args = _build_arg_parser().parse_args()

    try:
        build_training_dataset(
            missing_value_strategy=args.missing_strategy,
            fail_on_quality_issues=args.fail_on_issues,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Failed to build training dataset: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(_main())
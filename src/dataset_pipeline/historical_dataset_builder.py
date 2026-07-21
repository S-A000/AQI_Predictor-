"""
historical_dataset_builder.py
==============================
Suggested path: src/dataset_pipeline/historical_dataset_builder.py

SINGLE RESPONSIBILITY: discover every engineered-feature parquet file
produced by the pipeline so far (both per-run versioned snapshots
under data/processed/v<N>/ and the consolidated Feast-ready file
under data/feast_ready/) and merge them into one raw-merged
DataFrame, de-duplicated on (city, timestamp).

Does NOT validate, clean, or compute statistics — see
quality_checker.py and dataset_statistics.py for those.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FEAST_READY_FILE = PROJECT_ROOT / "data" / "feast_ready" / "aqi_features.parquet"


@dataclass
class BuildSource:
    """One parquet file that contributed rows to the merged dataset."""
    path: Path
    row_count: int


@dataclass
class BuildResult:
    dataframe: pd.DataFrame
    sources: list[BuildSource] = field(default_factory=list)

    @property
    def total_rows_before_dedup(self) -> int:
        return sum(source.row_count for source in self.sources)


class HistoricalDatasetBuilder:
    """
    Discovers and merges every engineered feature dataset the
    pipeline has produced. Engineered files are recognized by name:
        - exactly "features.parquet" (live pipeline's per-version output)
        - anything ending in "_features.parquet" (historical backfill's
          per-version output, e.g. "historical_karachi_features.parquet")
        - the consolidated Feast-ready file (data/feast_ready/aqi_features.parquet)

    Raw (non-engineered) parquet files — e.g. "historical_karachi.parquet"
    or the raw MergedFeature batch — are intentionally NOT included;
    training should consume engineered features only.
    """

    ENGINEERED_SUFFIX = "_features.parquet"
    ENGINEERED_EXACT_NAME = "features.parquet"

    def __init__(
        self,
        *,
        processed_dir: Path = PROCESSED_DIR,
        feast_ready_file: Path = FEAST_READY_FILE,
        include_feast_ready: bool = True,
    ):
        self.processed_dir = processed_dir
        self.feast_ready_file = feast_ready_file
        self.include_feast_ready = include_feast_ready

    def _discover_engineered_files(self) -> list[Path]:
        found: list[Path] = []

        if self.processed_dir.exists():
            for version_dir in sorted(self.processed_dir.iterdir()):
                if not version_dir.is_dir() or not version_dir.name.startswith("v"):
                    continue
                for file in version_dir.iterdir():
                    if file.suffix != ".parquet":
                        continue
                    if file.name == self.ENGINEERED_EXACT_NAME or file.name.endswith(self.ENGINEERED_SUFFIX):
                        found.append(file)
        else:
            logger.warning("Processed data directory not found: %s", self.processed_dir)

        if self.include_feast_ready and self.feast_ready_file.exists():
            found.append(self.feast_ready_file)

        return found

    def build(self, *, key_cols: tuple[str, ...] = ("city", "timestamp")) -> BuildResult:
        """
        Merge all discovered engineered files into one DataFrame.
        Rows are de-duplicated on `key_cols` — the LAST occurrence
        wins, so if the same (city, timestamp) exists in both a
        versioned snapshot and the Feast-ready file, the Feast-ready
        copy (loaded last) takes precedence, since it reflects the
        most recent dedup/consolidation logic.
        """
        files = self._discover_engineered_files()

        if not files:
            raise FileNotFoundError(
                f"No engineered feature files found under {self.processed_dir} "
                f"or at {self.feast_ready_file}. Run the live pipeline "
                "(run_pipeline.py) and/or the historical backfill "
                "(historical_backfill.py) at least once first."
            )

        frames: list[pd.DataFrame] = []
        sources: list[BuildSource] = []

        for file in files:
            df = pd.read_parquet(file)
            if "timestamp" not in df.columns:
                logger.warning("Skipping %s: no 'timestamp' column.", file)
                continue
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            frames.append(df)
            sources.append(BuildSource(path=file, row_count=len(df)))
            logger.info("Loaded %d row(s) from %s", len(df), file)

        if not frames:
            raise ValueError("All discovered files were unusable (missing 'timestamp' column).")

        merged = pd.concat(frames, ignore_index=True)
        before = len(merged)
        merged = merged.drop_duplicates(subset=list(key_cols), keep="last")
        merged = merged.sort_values(list(key_cols)).reset_index(drop=True)
        after = len(merged)

        logger.info(
            "Merged %d file(s) -> %d row(s) (%d duplicate row(s) resolved).",
            len(files), after, before - after,
        )

        return BuildResult(dataframe=merged, sources=sources)
"""
quality_checker.py
===================
Suggested path: src/dataset_pipeline/quality_checker.py

SINGLE RESPONSIBILITY: quality assessment (and optional cleaning) of
an already-merged dataset. Does NOT merge sources
(historical_dataset_builder.py's job) and does NOT compute
descriptive statistics (dataset_statistics.py's job).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TimestampGap:
    """A contiguous run of missing expected timestamps for one city."""
    city: str
    gap_start: pd.Timestamp
    gap_end: pd.Timestamp
    missing_hours: int


@dataclass
class QualityReport:
    row_count: int
    duplicate_count: int
    missing_value_counts: dict[str, int]
    missing_value_pct: dict[str, float]
    timestamp_gaps: list[TimestampGap] = field(default_factory=list)
    invalid_value_counts: dict[str, int] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return self.duplicate_count == 0 and not self.timestamp_gaps and not self.issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_count": self.row_count,
            "duplicate_count": self.duplicate_count,
            "missing_value_counts": self.missing_value_counts,
            "missing_value_pct": self.missing_value_pct,
            "timestamp_gap_count": len(self.timestamp_gaps),
            "timestamp_gaps": [
                {
                    "city": gap.city,
                    "gap_start": gap.gap_start.isoformat(),
                    "gap_end": gap.gap_end.isoformat(),
                    "missing_hours": gap.missing_hours,
                }
                for gap in self.timestamp_gaps
            ],
            "invalid_value_counts": self.invalid_value_counts,
            "issues": self.issues,
            "is_clean": self.is_clean,
        }


class QualityChecker:
    """
    Runs duplicate / missing-value / timestamp-continuity checks over
    a merged dataset (typically HistoricalDatasetBuilder's output),
    and can clean it afterwards.

    `clean()` resolves duplicate keys and clear domain violations, while
    deliberately leaving statistical imputation to train-fitted model
    preprocessing after the chronological split.
    """

    def __init__(
        self,
        *,
        key_cols: tuple[str, ...] = ("city", "timestamp"),
        expected_frequency: str = "1h",
        max_missing_pct_warning: float = 5.0,
    ):
        self.key_cols = key_cols
        self.expected_frequency = expected_frequency
        self.max_missing_pct_warning = max_missing_pct_warning

    # --------------------------------------------------
    # Individual checks
    # --------------------------------------------------

    def _check_duplicates(self, df: pd.DataFrame) -> int:
        return int(df.duplicated(subset=list(self.key_cols)).sum())

    def _check_missing_values(self, df: pd.DataFrame) -> tuple[dict[str, int], dict[str, float]]:
        counts = df.isnull().sum()
        counts = counts[counts > 0]
        pct = (counts / len(df) * 100).round(2) if len(df) else counts
        return counts.to_dict(), pct.to_dict()

    def _check_timestamp_continuity(
        self,
        df: pd.DataFrame,
        *,
        timestamp_col: str = "timestamp",
        city_col: str = "city",
    ) -> list[TimestampGap]:
        """
        For each city, builds the expected hourly timestamp range
        (min -> max) and reports any missing hours as contiguous
        gap ranges (rather than one entry per missing hour, which
        would be unreadable for long gaps).
        """
        gaps: list[TimestampGap] = []

        for city, group in df.groupby(city_col):
            ts = (
                pd.to_datetime(
                    group[timestamp_col], utc=True, errors="coerce"
                )
                .dropna()
                .drop_duplicates()
                .sort_values()
            )
            if len(ts) < 2:
                continue

            expected = pd.date_range(ts.min(), ts.max(), freq=self.expected_frequency)
            missing = expected.difference(ts)
            if missing.empty:
                continue

            missing_sorted = missing.sort_values()
            step = pd.Timedelta(self.expected_frequency)

            gap_start = missing_sorted[0]
            prev = missing_sorted[0]
            run_length = 1

            for current in missing_sorted[1:]:
                if (current - prev) <= step:
                    run_length += 1
                else:
                    gaps.append(TimestampGap(city=str(city), gap_start=gap_start, gap_end=prev, missing_hours=run_length))
                    gap_start = current
                    run_length = 1
                prev = current

            gaps.append(TimestampGap(city=str(city), gap_start=gap_start, gap_end=prev, missing_hours=run_length))

        return gaps

    @staticmethod
    def _check_invalid_values(df: pd.DataFrame) -> dict[str, int]:
        """Report only clear project-domain violations, not rare high events."""

        invalid_counts: dict[str, int] = {}
        if "timestamp" in df.columns:
            invalid_timestamp_count = int(
                pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
                .isna()
                .sum()
            )
            if invalid_timestamp_count:
                invalid_counts["timestamp"] = invalid_timestamp_count

        for column in ("pm25", "pm10", "no2", "so2", "co", "o3"):
            if column not in df.columns:
                continue
            numeric = pd.to_numeric(df[column], errors="coerce")
            count = int((numeric < 0).sum())
            if count:
                invalid_counts[column] = count

        if "aqi" in df.columns:
            aqi = pd.to_numeric(df["aqi"], errors="coerce")
            count = int(((aqi < 0) | (aqi > 500)).sum())
            if count:
                invalid_counts["aqi"] = count

        return invalid_counts

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def check(self, df: pd.DataFrame) -> QualityReport:
        missing_key_columns = [
            column for column in self.key_cols if column not in df.columns
        ]
        duplicate_count = (
            self._check_duplicates(df) if not missing_key_columns else 0
        )
        missing_counts, missing_pct = self._check_missing_values(df)
        gaps = (
            self._check_timestamp_continuity(df)
            if {"city", "timestamp"}.issubset(df.columns)
            else []
        )
        invalid_counts = self._check_invalid_values(df)

        issues: list[str] = []

        if missing_key_columns:
            issues.append(
                f"Missing required schema column(s): {missing_key_columns}"
            )

        if duplicate_count:
            issues.append(f"{duplicate_count} duplicate row(s) on {self.key_cols}")

        for col, pct in missing_pct.items():
            if pct > self.max_missing_pct_warning:
                issues.append(
                    f"Column '{col}' has {pct}% missing values "
                    f"(exceeds {self.max_missing_pct_warning}% threshold)"
                )

        if gaps:
            total_missing_hours = sum(gap.missing_hours for gap in gaps)
            issues.append(
                f"{len(gaps)} timestamp gap(s) across cities, "
                f"totaling {total_missing_hours} missing hour(s)"
            )

        for column, count in invalid_counts.items():
            issues.append(
                f"Column '{column}' has {count} clearly invalid value(s)"
            )

        report = QualityReport(
            row_count=len(df),
            duplicate_count=duplicate_count,
            missing_value_counts=missing_counts,
            missing_value_pct=missing_pct,
            timestamp_gaps=gaps,
            invalid_value_counts=invalid_counts,
            issues=issues,
        )

        if report.is_clean:
            logger.info("Quality check passed: %d row(s), no issues found.", report.row_count)
        else:
            logger.warning("Quality check found %d issue(s): %s", len(issues), issues)

        return report

    def clean(self, df: pd.DataFrame, *, missing_value_strategy: str = "mean") -> pd.DataFrame:
        """
        Deduplicates on key_cols (last occurrence wins) and normalizes
        clearly invalid domain values. Missing-value handling is deferred
        to train-fitted preprocessing. `missing_value_strategy` remains
        accepted for backward-compatible callers.
        Supported legacy values are "mean", "zero", and "drop". Timestamp gaps are NOT filled
        here — a missing hour of real-world data should not be
        silently invented; handle gaps explicitly upstream
        (re-run ingestion for that window) if they matter for your
        model.
        """
        if missing_value_strategy not in ("mean", "zero", "drop"):
            raise ValueError(f"Unknown missing_value_strategy: {missing_value_strategy}")

        missing_key_columns = [
            column for column in self.key_cols if column not in df.columns
        ]
        if missing_key_columns:
            raise ValueError(
                f"Cannot clean dataset; missing key columns: {missing_key_columns}"
            )

        df = df.copy()
        before = len(df)
        df = df.drop_duplicates(subset=list(self.key_cols), keep="last")

        logger.info(
            "Deferring '%s' missing-value handling to train-fitted preprocessing.",
            missing_value_strategy,
        )

        for column in ("pm25", "pm10", "no2", "so2", "co", "o3"):
            if column in df.columns:
                numeric = pd.to_numeric(df[column], errors="coerce")
                df.loc[numeric < 0, column] = pd.NA

        if "aqi" in df.columns:
            aqi = pd.to_numeric(df["aqi"], errors="coerce")
            df.loc[(aqi < 0) | (aqi > 500), "aqi"] = pd.NA

        df = df.sort_values(list(self.key_cols)).reset_index(drop=True)

        logger.info(
            "Cleaned dataset: %d -> %d row(s) (dedup + invalid-domain normalization; imputation deferred).",
            before,
            len(df),
        )

        return df

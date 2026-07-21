"""
dataset_statistics.py
======================
Suggested path: src/dataset_pipeline/dataset_statistics.py

SINGLE RESPONSIBILITY: compute descriptive summary statistics over an
already-built, already-quality-checked dataset. Does NOT merge
sources (historical_dataset_builder.py) and does NOT perform quality
checks or cleaning (quality_checker.py).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DatasetStatistics:
    row_count: int
    city_counts: dict[str, int]
    date_range: dict[str, str]
    numeric_summary: dict[str, dict[str, float]]
    dominant_pollutant_distribution: dict[str, int]
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "row_count": self.row_count,
            "city_counts": self.city_counts,
            "date_range": self.date_range,
            "numeric_summary": self.numeric_summary,
            "dominant_pollutant_distribution": self.dominant_pollutant_distribution,
        }

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        logger.info("Saved dataset statistics -> %s", path)
        return path


class DatasetStatisticsGenerator:
    """
    Computes descriptive statistics used for EDA sanity-checks
    (Phase 5) and kept as a permanent record alongside every built
    training_dataset.parquet: row/city coverage, date range, numeric
    summary (mean/min/max/std) per feature, and dominant-pollutant
    distribution (useful to confirm Open-Meteo's dominant_pollutant
    calculation isn't collapsing to a single value across the whole
    dataset — flagged as worth checking once more historical data
    is pulled in).
    """

    NUMERIC_COLUMNS = (
        "temperature", "feels_like", "humidity", "pressure", "wind_speed",
        "cloudiness", "aqi", "pm25", "pm10", "no2", "so2", "co", "o3",
        "aqi_change_rate", "aqi_rolling_mean_3h",
    )

    def generate(
        self,
        df: pd.DataFrame,
        *,
        timestamp_col: str = "timestamp",
        city_col: str = "city",
        dominant_pollutant_col: str = "dominant_pollutant",
    ) -> DatasetStatistics:
        if df.empty:
            raise ValueError("Cannot generate statistics for an empty dataset.")

        city_counts = df[city_col].value_counts().to_dict()

        ts = pd.to_datetime(df[timestamp_col])
        date_range = {
            "start": ts.min().isoformat(),
            "end": ts.max().isoformat(),
        }

        numeric_summary: dict[str, dict[str, float]] = {}
        for col in self.NUMERIC_COLUMNS:
            if col not in df.columns:
                continue
            series = df[col].dropna()
            if series.empty:
                continue
            numeric_summary[col] = {
                "mean": round(float(series.mean()), 3),
                "min": round(float(series.min()), 3),
                "max": round(float(series.max()), 3),
                "std": round(float(series.std()), 3) if len(series) > 1 else 0.0,
            }

        dominant_pollutant_distribution = (
            df[dominant_pollutant_col].value_counts().to_dict()
            if dominant_pollutant_col in df.columns else {}
        )

        stats = DatasetStatistics(
            row_count=len(df),
            city_counts={str(k): int(v) for k, v in city_counts.items()},
            date_range=date_range,
            numeric_summary=numeric_summary,
            dominant_pollutant_distribution={
                str(k): int(v) for k, v in dominant_pollutant_distribution.items()
            },
        )

        logger.info(
            "Generated statistics: %d row(s), %d city/cities, range %s -> %s",
            stats.row_count, len(stats.city_counts), date_range["start"], date_range["end"],
        )

        return stats
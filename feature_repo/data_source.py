"""
data_source.py
==============
Feast Batch Data Source

Author:
    Syed Abdullah

IMPORTANT: This points to a STABLE, append-only file
(data/feast_ready/aqi_features.parquet) that BOTH the live pipeline
and the historical backfill write into via
`src.feature_pipeline.feast_writer.write_to_feast_source()`.

Do NOT point this at the versioned `data/processed/v<N>/` folders —
those are per-run audit-trail snapshots (one version = one pipeline
run's raw output) and picking "the latest version" would silently
drop either the historical backfill's data or the live data,
depending on which ran most recently. See feast_writer.py for why.
"""

from pathlib import Path

from feast import FileSource

# ----------------------------------------------------------
# Locate project root
# ----------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ----------------------------------------------------------
# Stable, consolidated Feast-ready parquet
# (written/appended to by feast_writer.write_to_feast_source)
# ----------------------------------------------------------

FEAST_READY_DIR = PROJECT_ROOT / "data" / "feast_ready"
PARQUET_FILE = FEAST_READY_DIR / "aqi_features.parquet"

if not PARQUET_FILE.exists():
    raise FileNotFoundError(
        f"No Feast-ready dataset found at {PARQUET_FILE}. "
        "Run the live pipeline (run_pipeline.py) and/or the historical "
        "backfill (historical_backfill.py) at least once first — both "
        "write into this file via feast_writer.write_to_feast_source()."
    )

# ----------------------------------------------------------
# Feast File Source
# ----------------------------------------------------------

aqi_source = FileSource(
    name="aqi_weather_source",
    path=str(PARQUET_FILE),
    timestamp_field="timestamp",
)
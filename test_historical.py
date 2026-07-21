import asyncio
from datetime import date
from src.ingestion.historical_backfill import run_backfill

asyncio.run(run_backfill(
    city="Karachi", country="PK",
    latitude=24.8607, longitude=67.0011,
    start_date=date(2026, 7, 1),
    end_date=date(2026, 7, 7),
))
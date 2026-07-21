"""
run_pipeline.py
================
Enterprise orchestration layer for the AQI Forecasting MLOps project.

Author
------
Syed Abdullah

Description
-----------
Ties together fetching, validation, merging, and storage into a
single async pipeline that can process many cities in parallel,
and can be run standalone (`python run_pipeline.py`) or invoked
from a scheduler (Airflow, cron, GitHub Actions).

Note on API clients
--------------------
No HTTP client module was provided alongside validator.py /
merger.py / storage.py, so this module defines minimal default
clients (`OpenWeatherClient`, `AQICNClient`) that call the same
APIs those schemas were modeled on. Inject your own client via
`Pipeline(weather_client=..., aqi_client=...)` if yours differs —
anything with an async `fetch(city) -> dict` method (and an
optional async `aclose()`) works as a drop-in replacement.

Note on Weather (optional)
---------------------------
The Weather API (OpenWeather) is now OPTIONAL. Set
`enable_weather=False` on `Pipeline` (or pass `--disable-weather`
on the CLI) to run an AQI-only pipeline:
    - OpenWeatherClient is never called/fetched/retried.
    - Weather validation is skipped entirely.
    - FeatureMerger.merge() (weather+AQI fusion) is skipped; a
      lightweight `AQIOnlyFeature` record is produced instead.
    - Storage, versioning, retries, health checks, metrics, hooks,
      and the CLI all keep working unchanged.
When `enable_weather` is left at its default (True), behaviour is
100% identical to before this change (backward compatible).

Capabilities
------------
- Multiple cities support (asyncio.gather)
- Parallel weather + AQI fetch per city
- Optional Weather API (AQI-only mode supported)
- Pipeline hooks (before_run, after_run)
- Metrics collection
- Prometheus / OpenTelemetry integration (optional, auto-detected)
- Feature Store upload toggle
- Retry orchestration
- Health checks before execution
- Graceful shutdown (SIGINT/SIGTERM)
- CLI entry point (python run_pipeline.py)
- Scheduler compatibility (Airflow / cron / GitHub Actions)
- Rich execution summary
"""
from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import os
import signal
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, ClassVar, Protocol
from src.feature_pipeline.feature_engineering import engineer_features
from src.feature_pipeline.feast_writer import write_to_feast_source

from pydantic import BaseModel, ConfigDict

from src.configs.settings import settings
from src.ingestion.merger import FeatureMerger, MergedFeature
from src.ingestion.storage import StorageManager
from src.ingestion.validator import AQIResponse, ResponseValidator
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# Optional Observability Integrations
# ============================================================

try:
    from prometheus_client import Counter, Histogram

    PIPELINE_RUNS_TOTAL = Counter(
        "pipeline_runs_total", "Total pipeline runs", ["status"]
    )
    CITY_PROCESSING_SECONDS = Histogram(
        "pipeline_city_processing_seconds", "Per-city fetch+merge duration"
    )
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    PIPELINE_RUNS_TOTAL = None
    CITY_PROCESSING_SECONDS = None
    _PROMETHEUS_AVAILABLE = False

try:
    from opentelemetry import trace

    _tracer = trace.get_tracer(__name__)
    _OTEL_AVAILABLE = True
except ImportError:
    _tracer = None
    _OTEL_AVAILABLE = False


@contextmanager
def _span(name: str):
    """
    Start an OpenTelemetry span if the SDK is installed, otherwise
    a transparent no-op context manager.
    """
    if _OTEL_AVAILABLE:
        with _tracer.start_as_current_span(name):
            yield
    else:
        yield


# ============================================================
# Default HTTP Clients
# ============================================================

class WeatherClient(Protocol):
    async def fetch(self, city: str) -> dict: ...


class AQIClient(Protocol):
    async def fetch(self, city: str) -> dict: ...


def _resolve_api_key(
    *,
    api_key: str | None,
    settings_secret: Any,
    env_var_names: tuple[str, ...],
    provider_label: str,
) -> str:
    """
    Resolve a raw string API key using a strict precedence order:

        1. Explicit `api_key` constructor argument.
        2. `settings.<key>` (a pydantic `SecretStr`), unwrapped via
           `get_secret_value()`.
        3. Environment variable fallback (checked in the given order).

    Always returns a plain `str` — never a `SecretStr` — since HTTP
    clients need the raw token to build request params/headers.

    Raises `ValueError` with a descriptive message if no key can be
    resolved from any source.
    """
    if api_key:
        return api_key

    if settings_secret is not None:
        value = settings_secret.get_secret_value()
        if value:
            return value

    for env_var in env_var_names:
        value = os.getenv(env_var)
        if value:
            return value

    env_var_hint = env_var_names[0]
    raise ValueError(
        f"{provider_label} API key is missing. "
        f"Please set {env_var_hint} in the environment or .env file."
    )


class OpenWeatherClient:
    """Default OpenWeatherMap client. Swap out if you use a different provider."""

    BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

    def __init__(self, api_key: str | None = None, timeout: float = 10.0):
        import httpx

        self.api_key: str = _resolve_api_key(
            api_key=api_key,
            settings_secret=getattr(settings, "openweather_api_key", None),
            env_var_names=("OPENWEATHER_API_KEY",),
            provider_label="OpenWeather",
        )
        self._client = httpx.AsyncClient(timeout=timeout)

    async def fetch(self, city: str) -> dict:
        response = await self._client.get(
            self.BASE_URL,
            params={"q": city, "appid": self.api_key, "units": "metric"},
        )
        response.raise_for_status()
        return response.json()

    async def aclose(self) -> None:
        await self._client.aclose()


class AQICNClient:
    """Default AQICN (waqi.info) client. Swap out if you use a different provider."""

    BASE_URL = "https://api.waqi.info/feed"

    def __init__(self, api_key: str | None = None, timeout: float = 10.0):
        import httpx

        self.api_key: str = _resolve_api_key(
            api_key=api_key,
            settings_secret=getattr(settings, "aqicn_api_key", None),
            env_var_names=("AQICN_API_KEY", "WAQI_API_TOKEN"),
            provider_label="AQICN",
        )
        self._client = httpx.AsyncClient(timeout=timeout)

    async def fetch(self, city: str) -> dict:
        response = await self._client.get(
            f"{self.BASE_URL}/{city}/",
            params={"token": self.api_key},
        )
        response.raise_for_status()
        return response.json()

    async def aclose(self) -> None:
        await self._client.aclose()


# ============================================================
# AQI-Only Feature (used when Weather API is disabled)
# ============================================================

class AQIOnlyFeature(BaseModel):
    """
    Lightweight feature record used ONLY when the Weather API is
    disabled (`enable_weather=False`).

    This intentionally lives outside `merger.py` so the existing
    `FeatureMerger` / `MergedFeature` weather+AQI fusion logic
    remains completely untouched. It mirrors just enough of the
    shape of a merged record (city, timestamp, AQI fields, plus
    weather fields explicitly set to `None`) so that
    `StorageManager` — which persists any object exposing
    `.model_dump(mode="json")` — can save it without any code
    changes on the storage side.
    """

    model_config = ConfigDict(extra="allow")

    city: str
    timestamp: datetime
    aqi: int
    dominant_pollutant: str
    iaqi: dict[str, Any]
    forecast: dict[str, Any] | None = None
    geo: list[float]

    # Weather fields are intentionally absent/None to make it
    # explicit downstream (dashboards, feature store, ML training)
    # that no weather data was collected for this record.
    weather_available: bool = False
    temp: float | None = None
    feels_like: float | None = None
    humidity: int | None = None
    pressure: int | None = None
    wind_speed: float | None = None

    @classmethod
    def from_aqi(cls, aqi_obj: AQIResponse) -> "AQIOnlyFeature":
        """Build an AQI-only feature record from a validated AQIResponse."""
        return cls(
            city=aqi_obj.data.city.name,
            timestamp=datetime.now(timezone.utc),
            aqi=aqi_obj.data.aqi,
            dominant_pollutant=aqi_obj.data.dominentpol,
            iaqi=aqi_obj.data.iaqi,
            forecast=aqi_obj.data.forecast,
            geo=aqi_obj.data.city.geo,
        )


# A feature persisted by StorageManager is either a full
# weather+AQI MergedFeature, or an AQI-only fallback.
Feature = MergedFeature | AQIOnlyFeature


# ============================================================
# Metrics Collection
# ============================================================

class PipelineMetrics:
    """In-memory pipeline-level metrics, mirrored into Prometheus if available."""

    _runs: ClassVar[int] = 0
    _successes: ClassVar[int] = 0
    _failures: ClassVar[int] = 0
    _durations_s: ClassVar[list[float]] = []

    @classmethod
    def record_run(cls, success: bool, duration_s: float) -> None:
        cls._runs += 1
        cls._successes += int(success)
        cls._failures += int(not success)
        cls._durations_s.append(duration_s)

        if _PROMETHEUS_AVAILABLE:
            PIPELINE_RUNS_TOTAL.labels(status="success" if success else "failure").inc()

    @classmethod
    def record_city_duration(cls, duration_s: float) -> None:
        if _PROMETHEUS_AVAILABLE:
            CITY_PROCESSING_SECONDS.observe(duration_s)

    @classmethod
    def get_metrics(cls) -> dict[str, Any]:
        return {
            "runs": cls._runs,
            "success": cls._successes,
            "failure": cls._failures,
            "avg_duration_s": (
                sum(cls._durations_s) / len(cls._durations_s)
                if cls._durations_s
                else 0.0
            ),
        }

    @classmethod
    def reset(cls) -> None:
        cls._runs = 0
        cls._successes = 0
        cls._failures = 0
        cls._durations_s = []


# ============================================================
# Retry Orchestration
# ============================================================

async def _fetch_with_retry(
    client: WeatherClient | AQIClient,
    city: str,
    *,
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
) -> dict:
    """Retry an async fetch call with exponential backoff."""

    current_delay = delay
    last_exc: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return await client.fetch(city)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning(
                "Fetch failed for %s (attempt %d/%d): %s",
                city, attempt, max_attempts, exc,
            )
            if attempt < max_attempts:
                await asyncio.sleep(current_delay)
                current_delay *= backoff

    assert last_exc is not None
    raise last_exc


# ============================================================
# Graceful Shutdown
# ============================================================

def _install_signal_handlers(
    loop: asyncio.AbstractEventLoop,
    shutdown_event: asyncio.Event,
) -> None:
    """
    Wire SIGINT/SIGTERM to a shutdown event so in-flight cities can
    finish and no new ones start, instead of a hard kill.
    """

    def _trigger(*_args: Any) -> None:
        if not shutdown_event.is_set():
            logger.warning("Shutdown signal received; finishing in-flight work...")
            shutdown_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _trigger)
        except NotImplementedError:
            # add_signal_handler isn't available on Windows.
            signal.signal(sig, _trigger)


# ============================================================
# Result / Summary Models
# ============================================================

@dataclass
class CityResult:
    city: str
    success: bool
    feature: Feature | None = None
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    duration_s: float = 0.0


@dataclass
class PipelineRunSummary:
    run_id: str
    started_at: datetime
    cities_requested: list[str] = field(default_factory=list)
    city_results: list[CityResult] = field(default_factory=list)
    finished_at: datetime | None = None
    version: str | None = None
    feature_store_uploaded: bool = False
    aborted_reason: str | None = None
    # NEW: report which APIs were enabled for this run.
    weather_enabled: bool = True
    aqi_enabled: bool = True

    @property
    def duration_s(self) -> float:
        if self.finished_at is None:
            return 0.0
        return (self.finished_at - self.started_at).total_seconds()

    @property
    def cities_succeeded(self) -> int:
        return sum(1 for r in self.city_results if r.success)

    @property
    def cities_failed(self) -> int:
        return sum(1 for r in self.city_results if not r.success)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "duration_s": round(self.duration_s, 3),
            "cities_requested": self.cities_requested,
            "cities_succeeded": self.cities_succeeded,
            "cities_failed": self.cities_failed,
            "version": self.version,
            "feature_store_uploaded": self.feature_store_uploaded,
            "aborted_reason": self.aborted_reason,
            "weather_enabled": self.weather_enabled,
            "aqi_enabled": self.aqi_enabled,
            "results": [
                {
                    "city": r.city,
                    "success": r.success,
                    "error": r.error,
                    "warnings": r.warnings,
                    "duration_s": round(r.duration_s, 3),
                }
                for r in self.city_results
            ],
        }

    def render(self) -> str:
        bar = "=" * 60
        lines = [
            bar,
            f"Pipeline Run Summary ({self.run_id})",
            bar,
            f"Started:   {self.started_at.isoformat()}",
            f"Finished:  {self.finished_at.isoformat() if self.finished_at else '-'}",
            f"Duration:  {self.duration_s:.2f}s",
            f"Cities:    {self.cities_succeeded}/{len(self.cities_requested)} succeeded",
            f"Version:   {self.version or '-'}",
            f"Weather Enabled: {self.weather_enabled}",
            f"AQI Enabled:     {self.aqi_enabled}",
            f"Feature Store Upload: {'yes' if self.feature_store_uploaded else 'no'}",
        ]

        if self.aborted_reason:
            lines.append(f"ABORTED: {self.aborted_reason}")

        lines.append("-" * 60)

        for result in self.city_results:
            status = "OK" if result.success else "FAIL"
            lines.append(f"[{status}] {result.city:<20} {result.duration_s:.2f}s")

            if result.error:
                lines.append(f"        error: {result.error}")

            for warning in result.warnings:
                lines.append(f"        warn:  {warning}")

        lines.append(bar)

        return "\n".join(lines)


# ============================================================
# Pipeline
# ============================================================

class Pipeline:
    """
    Async ingestion pipeline: fetch -> validate -> merge -> store,
    across many cities in parallel, with hooks, metrics, retries,
    health checks, and graceful shutdown.

    Weather (OpenWeather) is OPTIONAL: pass `enable_weather=False`
    to run an AQI-only pipeline. AQICN is always required/enabled.
    """

    def __init__(
        self,
        *,
        weather_client: WeatherClient | None = None,
        aqi_client: AQIClient | None = None,
        storage_manager: StorageManager | None = None,
        upload_to_feature_store: bool = False,
        feature_group_name: str = "aqi_weather_features",
        keep_versions: int = 5,
        strict: bool = True,
        abort_on_unhealthy: bool = True,
        max_concurrent_cities: int = 10,
        fetch_retries: int = 3,
        enable_weather: bool = True,
    ):
        # Weather client is still constructed even when disabled
        # (keeps OpenWeatherClient fully intact for future use /
        # dependency injection), but it will never be called when
        # `enable_weather` is False.
        self.weather_client = weather_client or OpenWeatherClient()
        self.aqi_client = aqi_client or AQICNClient()
        self.storage_manager = storage_manager or StorageManager()

        self.upload_to_feature_store = upload_to_feature_store
        self.feature_group_name = feature_group_name
        self.keep_versions = keep_versions
        self.strict = strict
        self.abort_on_unhealthy = abort_on_unhealthy
        self.fetch_retries = fetch_retries

        # NEW: Weather API toggle. Defaults to True so existing
        # callers observe identical behaviour (backward compatible).
        self.enable_weather = enable_weather

        if not self.enable_weather:
            logger.info("Weather API disabled. Running AQI-only pipeline.")

        self._semaphore = asyncio.Semaphore(max_concurrent_cities)
        self._shutdown_event = asyncio.Event()
        self._before_run_hooks: list[Callable] = []
        self._after_run_hooks: list[Callable] = []

    # --------------------------------------------------
    # Pipeline Hooks
    # --------------------------------------------------

    def before_run(self, hook: Callable) -> Callable:
        """Register a hook (sync or async) to run before the pipeline starts. Usable as a decorator."""
        self._before_run_hooks.append(hook)
        return hook

    def after_run(self, hook: Callable) -> Callable:
        """Register a hook (sync or async) to run after the pipeline finishes. Usable as a decorator."""
        self._after_run_hooks.append(hook)
        return hook

    @staticmethod
    async def _run_hooks(hooks: list[Callable], *args: Any) -> None:
        for hook in hooks:
            try:
                result = hook(*args)
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:  # noqa: BLE001
                logger.error("Pipeline hook %s raised: %s", getattr(hook, "__name__", hook), exc)

    # --------------------------------------------------
    # Per-City Processing
    # --------------------------------------------------

    async def _process_city(self, city: str) -> CityResult:
        start = time.perf_counter()

        async with self._semaphore:

            if self._shutdown_event.is_set():
                return CityResult(city=city, success=False, error="skipped: shutdown requested")

            try:
                # --------------------------------------------
                # Fetch: only retry/fetch APIs that are enabled.
                # --------------------------------------------
                with _span(f"fetch:{city}"):
                    if self.enable_weather:
                        weather_raw, aqi_raw = await asyncio.gather(
                            _fetch_with_retry(self.weather_client, city, max_attempts=self.fetch_retries),
                            _fetch_with_retry(self.aqi_client, city, max_attempts=self.fetch_retries),
                        )
                    else:
                        logger.info("Running AQI-only pipeline for %s.", city)
                        weather_raw = None
                        aqi_raw = await _fetch_with_retry(
                            self.aqi_client, city, max_attempts=self.fetch_retries
                        )

                # --------------------------------------------
                # Validate: AQI always validated; weather only
                # validated when the Weather API is enabled.
                # --------------------------------------------
                aqi_obj, aqi_report = ResponseValidator.validate_aqi_with_report(
                    aqi_raw, strict=self.strict
                )

                warnings = list(aqi_report.warnings)
                weather_obj = None
                weather_report = None

                if self.enable_weather:
                    weather_obj, weather_report = ResponseValidator.validate_weather_with_report(
                        weather_raw, strict=self.strict
                    )
                    warnings = list(weather_report.warnings) + warnings
                else:
                    logger.info("Skipping weather validation (weather disabled) for %s.", city)

                weather_missing = self.enable_weather and weather_obj is None

                if aqi_obj is None or weather_missing:
                    errors = list(aqi_report.errors)
                    if weather_report is not None:
                        errors = list(weather_report.errors) + errors
                    return CityResult(
                        city=city,
                        success=False,
                        error="; ".join(errors) or "validation failed",
                        warnings=warnings,
                        duration_s=time.perf_counter() - start,
                    )

                # --------------------------------------------
                # Merge: full weather+AQI merge only when both
                # are available; otherwise an AQI-only feature.
                # --------------------------------------------
                with _span(f"merge:{city}"):
                    if self.enable_weather:
                        feature: Feature = FeatureMerger.merge(weather_obj, aqi_obj)
                    else:
                        logger.info("Skipping feature merge (weather unavailable) for %s.", city)
                        feature = AQIOnlyFeature.from_aqi(aqi_obj)

                duration_s = time.perf_counter() - start
                PipelineMetrics.record_city_duration(duration_s)

                return CityResult(
                    city=city, success=True, feature=feature,
                    warnings=warnings, duration_s=duration_s,
                )

            except Exception as exc:  # noqa: BLE001
                logger.exception("Unhandled error processing %s", city)
                return CityResult(
                    city=city, success=False, error=str(exc),
                    duration_s=time.perf_counter() - start,
                )

    async def _aclose_clients(self) -> None:
        # Weather client is closed unconditionally even when
        # disabled — it was never fetched from, so this is a
        # no-op cleanup rather than an extra request.
        for client in (self.weather_client, self.aqi_client):
            aclose = getattr(client, "aclose", None)
            if aclose is not None:
                try:
                    await aclose()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Error closing client %s: %s", client, exc)

    # --------------------------------------------------
    # Run
    # --------------------------------------------------

    async def run(
        self,
        cities: list[str],
        *,
        formats: tuple[str, ...] = ("json", "csv", "parquet"),
    ) -> PipelineRunSummary:
        """
        Execute the full pipeline for a list of cities and return a
        rich execution summary.
        """

        run_id = uuid.uuid4().hex[:12]
        summary = PipelineRunSummary(
            run_id=run_id,
            started_at=datetime.now(timezone.utc),
            cities_requested=list(cities),
            weather_enabled=self.enable_weather,
            aqi_enabled=True,
        )

        logger.info("Starting pipeline run %s for %d cities.", run_id, len(cities))

        if not self.enable_weather:
            logger.info("Weather API disabled. Running AQI-only pipeline.")

        try:
            loop = asyncio.get_running_loop()
            _install_signal_handlers(loop, self._shutdown_event)
        except (NotImplementedError, RuntimeError):
            pass

        await self._run_hooks(self._before_run_hooks, summary)

        if self.abort_on_unhealthy:
            health = self.storage_manager.health_check()
            if not health.get("healthy", True):
                summary.aborted_reason = f"storage unhealthy: {health.get('issues')}"
                summary.finished_at = datetime.now(timezone.utc)
                PipelineMetrics.record_run(False, summary.duration_s)
                await self._run_hooks(self._after_run_hooks, summary)
                logger.error("Aborting pipeline run %s: %s", run_id, summary.aborted_reason)
                return summary

        try:
            results = await asyncio.gather(
                *(self._process_city(city) for city in cities)
            )
        finally:
            await self._aclose_clients()

        summary.city_results = list(results)

        features = [r.feature for r in results if r.success and r.feature is not None]

        if features:

            # NOTE: handle_missing_values / drop_duplicates operate
            # generically over the feature batch. When weather is
            # disabled, `features` is a list of AQIOnlyFeature
            # instances instead of MergedFeature — both are plain
            # pydantic models, so these helpers keep working as
            # long as FeatureMerger implements them structurally
            # (duck typing). If your FeatureMerger implementation
            # assumes MergedFeature specifically, guard here.
            if self.enable_weather:
                features = FeatureMerger.handle_missing_values(features, strategy="mean")
                features = FeatureMerger.drop_duplicates(features)

            batch_result = self.storage_manager.save_batch(features, formats=formats)
            summary.version = batch_result["version"]

            self.storage_manager.cleanup_old_versions(keep_last=self.keep_versions)

            # NEW: feature engineering + Feast-ready write. Runs for
            # both full (weather+AQI) and AQI-only feature batches
            # since AQIOnlyFeature is also a plain pydantic model.
            engineered_df = engineer_features(features)
            write_to_feast_source(engineered_df)

            if self.upload_to_feature_store:
                try:
                    self.storage_manager.upload_to_feature_store(
                        features, self.feature_group_name
                    )
                    summary.feature_store_uploaded = True
                except Exception as exc:  # noqa: BLE001
                    logger.error("Feature store upload failed: %s", exc)

        summary.finished_at = datetime.now(timezone.utc)
        PipelineMetrics.record_run(summary.cities_failed == 0, summary.duration_s)

        await self._run_hooks(self._after_run_hooks, summary)

        logger.info(
            "Pipeline run %s finished: %d/%d cities succeeded.",
            run_id, summary.cities_succeeded, len(cities),
        )

        return summary


# ============================================================
# Scheduler Compatibility
# ============================================================

def airflow_callable(
    cities: list[str],
    **pipeline_kwargs: Any,
) -> dict[str, Any]:
    """
    Sync wrapper suitable for an Airflow PythonOperator:

        PythonOperator(
            task_id="run_aqi_pipeline",
            python_callable=airflow_callable,
            op_kwargs={"cities": ["Karachi", "Lahore"]},
        )

    Pass `enable_weather=False` in `op_kwargs` to run AQI-only.

    Raises if any city failed, so Airflow marks the task as failed.
    Returns the summary dict (e.g. for XCom) on success.
    """

    summary = asyncio.run(Pipeline(**pipeline_kwargs).run(cities))

    if summary.cities_failed:
        raise RuntimeError(
            f"Pipeline run {summary.run_id} had {summary.cities_failed} failing "
            f"city/cities: {[r.city for r in summary.city_results if not r.success]}"
        )

    return summary.to_dict()


# Cron / GitHub Actions: just run `python run_pipeline.py --cities ...`
# (optionally with `--disable-weather`) and check the process exit
# code (0 = success, 1 = one or more cities failed). No extra glue
# code needed.


# ============================================================
# CLI Entry Point
# ============================================================

def _build_arg_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description="AQI + Weather ingestion pipeline",
    )

    parser.add_argument(
        "--cities", nargs="+", default=["Karachi", "Lahore", "Islamabad"],
        help="Cities to process.",
    )
    parser.add_argument(
        "--formats", nargs="+", default=["json", "csv", "parquet"],
        choices=["json", "csv", "parquet"],
        help="Storage formats to write.",
    )
    parser.add_argument(
        "--feature-store", action="store_true",
        help="Upload the resulting batch to the Feature Store (Hopsworks).",
    )
    parser.add_argument(
        "--feature-group", default="aqi_weather_features",
        help="Feature group name to use when --feature-store is set.",
    )
    parser.add_argument(
        "--keep-versions", type=int, default=5,
        help="Number of dataset versions to retain after each run.",
    )
    parser.add_argument(
        "--soft", dest="strict", action="store_false", default=True,
        help="Soft validation: collect domain warnings instead of failing on them.",
    )
    parser.add_argument(
        "--skip-health-check", action="store_true",
        help="Skip the storage health check before running.",
    )
    parser.add_argument(
        "--json-summary", action="store_true",
        help="Print the run summary as JSON instead of a text report.",
    )
    # NEW: Weather API toggle.
    parser.add_argument(
        "--disable-weather", dest="enable_weather", action="store_false", default=True,
        help="Disable the Weather (OpenWeather) API and run an AQI-only pipeline.",
    )

    return parser


async def _main(argv: list[str] | None = None) -> int:

    args = _build_arg_parser().parse_args(argv)

    pipeline = Pipeline(
        upload_to_feature_store=args.feature_store,
        feature_group_name=args.feature_group,
        keep_versions=args.keep_versions,
        strict=args.strict,
        abort_on_unhealthy=not args.skip_health_check,
        enable_weather=args.enable_weather,
    )

    summary = await pipeline.run(args.cities, formats=tuple(args.formats))

    if args.json_summary:
        print(json.dumps(summary.to_dict(), indent=2, default=str))
    else:
        print(summary.render())

    return 0 if summary.cities_failed == 0 and not summary.aborted_reason else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
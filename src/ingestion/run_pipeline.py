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

Note on Telemetry / Degraded-Mode Tracking (NEW)
--------------------------------------------------
Previously, a Weather or AQI fetch failure was only visible as a
plain log line inside `_fetch_with_retry` — there was no counter
for how often each provider was failing, and no tracking of how
many city-results ended up without weather data. If the Weather
API went down for weeks, the only symptom would be scattered log
lines nobody was watching.

`PipelineMetrics` now tracks, in addition to run-level success/
failure:
    - `_weather_failures` / `_aqi_failures`: provider-specific
      failure counters, incremented by `_fetch_with_retry` itself
      (via `record_api_failure`) the moment retries are exhausted.
    - `_full_mode_results` / `_degraded_mode_results`: how many
      city-results had weather data vs. didn't (AQI-only), whether
      that was because of the static `enable_weather=False` toggle
      or a runtime auto-degrade (see below).
    - A rolling window (`_degradation_window`) used by
      `check_degradation_alert()` to log a loud, actionable alert
      once the degraded-mode ratio crosses a threshold over the
      last N city-results — instead of this staying invisible
      until someone notices the training data looks "off".

Note on Auto-Degrade (NEW, opt-in, default OFF)
--------------------------------------------------
Before this change, if `enable_weather=True` but the Weather API
specifically failed for a city (AQI still succeeded), the WHOLE
city-result failed — there was no dynamic fallback to AQI-only.
`Pipeline(auto_degrade_on_weather_failure=True)` opts into that
fallback: if weather fails after retries but AQI succeeds, the
city still produces an `AQIOnlyFeature` (with a telemetry event)
instead of failing outright. This is entirely opt-in — the
default (`False`) preserves the exact prior behaviour, so nothing
that depends on today's failure semantics breaks.

Capabilities
------------
- Multiple cities support (asyncio.gather)
- Parallel weather + AQI fetch per city
- Optional Weather API (AQI-only mode supported)
- Structured API-failure & degraded-mode telemetry with alerting
- Opt-in dynamic weather-fallback (auto-degrade)
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
    # NEW: provider-specific failure counter + degraded-mode gauge-like
    # counter, mirrored alongside the existing metrics when prometheus
    # is installed. Purely additive — no existing metric is touched.
    API_FAILURES_TOTAL = Counter(
        "pipeline_api_failures_total", "Total provider fetch failures", ["provider"]
    )
    DEGRADED_CITY_RESULTS_TOTAL = Counter(
        "pipeline_degraded_city_results_total",
        "City results produced without weather data", ["reason"],
    )
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    PIPELINE_RUNS_TOTAL = None
    CITY_PROCESSING_SECONDS = None
    API_FAILURES_TOTAL = None
    DEGRADED_CITY_RESULTS_TOTAL = None
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
    Lightweight feature record used when weather data is unavailable —
    either because the pipeline is statically configured with
    `enable_weather=False`, or (NEW) because of an opt-in runtime
    auto-degrade after a weather fetch failure. See module docstring.

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

    # ------------------------------------------------------------
    # NEW — API-health / degraded-mode telemetry (Missing Telemetry fix).
    # See module docstring "Note on Telemetry / Degraded-Mode Tracking".
    # ------------------------------------------------------------
    _weather_failures: ClassVar[int] = 0
    _aqi_failures: ClassVar[int] = 0
    _full_mode_results: ClassVar[int] = 0       # weather+AQI succeeded
    _degraded_mode_results: ClassVar[int] = 0   # AQI-only (any reason)
    # Rolling window of (timestamp, was_degraded), used to compute a
    # live "% degraded" ratio for alerting without keeping unbounded history.
    _degradation_window: ClassVar[list[tuple[datetime, bool]]] = []
    _DEGRADATION_WINDOW_MAXLEN: ClassVar[int] = 500

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
    def record_api_failure(cls, *, provider: str, city: str, error: str) -> None:
        """
        NEW: structured record of a provider-specific fetch failure,
        called by `_fetch_with_retry` the moment it exhausts all
        retry attempts. This is the piece that was previously
        invisible: before, the exception was logged once and then
        only surfaced as a generic city-level failure — there was no
        running count of "how many times has the Weather API failed
        this week" to look at.
        """
        if provider == "weather":
            cls._weather_failures += 1
        elif provider == "aqi":
            cls._aqi_failures += 1

        if _PROMETHEUS_AVAILABLE:
            API_FAILURES_TOTAL.labels(provider=provider).inc()

        logger.error(
            "api_failure_event provider=%s city=%s error=%s "
            "weather_failures_total=%d aqi_failures_total=%d",
            provider, city, error, cls._weather_failures, cls._aqi_failures,
        )

    @classmethod
    def record_degraded_mode(cls, *, city: str, reason: str) -> None:
        """
        NEW: structured record every time a city-result ends up WITHOUT
        weather data — whether because the pipeline was statically
        configured with `enable_weather=False`, or because of the
        opt-in auto-degrade fallback after a weather fetch failure.
        Feeds the rolling window used by `check_degradation_alert()`.
        """
        cls._degraded_mode_results += 1
        cls._degradation_window.append((datetime.now(timezone.utc), True))
        cls._trim_window()

        if _PROMETHEUS_AVAILABLE:
            DEGRADED_CITY_RESULTS_TOTAL.labels(reason=reason).inc()

        logger.warning(
            "degraded_mode_event city=%s reason=%s degraded_total=%d full_total=%d",
            city, reason, cls._degraded_mode_results, cls._full_mode_results,
        )

    @classmethod
    def record_full_mode(cls, *, city: str) -> None:
        """NEW: structured record every time a city-result HAS weather data."""
        cls._full_mode_results += 1
        cls._degradation_window.append((datetime.now(timezone.utc), False))
        cls._trim_window()

    @classmethod
    def _trim_window(cls) -> None:
        if len(cls._degradation_window) > cls._DEGRADATION_WINDOW_MAXLEN:
            cls._degradation_window = cls._degradation_window[-cls._DEGRADATION_WINDOW_MAXLEN:]

    @classmethod
    def get_degradation_ratio(cls, *, last_n: int = 100) -> float:
        """NEW: fraction of the last `last_n` city-results that were degraded (AQI-only)."""
        window = cls._degradation_window[-last_n:]
        if not window:
            return 0.0
        return sum(1 for _, degraded in window if degraded) / len(window)

    @classmethod
    def check_degradation_alert(cls, *, threshold: float = 0.20, last_n: int = 100) -> bool:
        """
        NEW: returns True (and logs a loud, actionable error) if the
        degraded-mode ratio over the last `last_n` city-results exceeds
        `threshold` (default 20%). Called automatically at the end of
        `Pipeline.run()` — this is the piece that turns "weeks of
        silently degraded training data" into something that shows up
        in logs/alerting immediately instead of at model-evaluation time.
        """
        ratio = cls.get_degradation_ratio(last_n=last_n)
        if ratio > threshold:
            logger.error(
                "DEGRADATION_ALERT: %.1f%% of the last %d city-result(s) were "
                "AQI-only (no weather data) — threshold is %.0f%%. The Weather "
                "API may be down, rate-limited, or misconfigured.",
                ratio * 100, min(last_n, len(cls._degradation_window)), threshold * 100,
            )
            return True
        return False

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
            # NEW
            "weather_failures": cls._weather_failures,
            "aqi_failures": cls._aqi_failures,
            "full_mode_results": cls._full_mode_results,
            "degraded_mode_results": cls._degraded_mode_results,
            "degradation_ratio_last_100": round(cls.get_degradation_ratio(last_n=100), 4),
        }

    @classmethod
    def reset(cls) -> None:
        cls._runs = 0
        cls._successes = 0
        cls._failures = 0
        cls._durations_s = []
        cls._weather_failures = 0
        cls._aqi_failures = 0
        cls._full_mode_results = 0
        cls._degraded_mode_results = 0
        cls._degradation_window = []


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
    provider: str | None = None,
) -> dict:
    """
    Retry an async fetch call with exponential backoff.

    CHANGED: new optional `provider` kwarg ("weather" / "aqi"). When
    given, a final failure (after all retries are exhausted) is
    reported to `PipelineMetrics.record_api_failure` before the
    exception is re-raised — see that method's docstring. Passing no
    `provider` preserves the exact prior behavior (log-only, no counter).
    """

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
    if provider:
        PipelineMetrics.record_api_failure(provider=provider, city=city, error=str(last_exc))
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
    # NEW: visibility into whether this result came back without
    # weather data, and why — None means "full weather+AQI result".
    degraded_reason: str | None = None


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
    # NEW: telemetry snapshot — degraded-mode ratio over the trailing
    # window at the time this run finished (see PipelineMetrics).
    degradation_ratio_last_100: float = 0.0
    degradation_alert_triggered: bool = False

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

    @property
    def cities_degraded(self) -> int:
        """NEW: how many successful city-results were AQI-only (no weather)."""
        return sum(1 for r in self.city_results if r.success and r.degraded_reason)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "duration_s": round(self.duration_s, 3),
            "cities_requested": self.cities_requested,
            "cities_succeeded": self.cities_succeeded,
            "cities_failed": self.cities_failed,
            "cities_degraded": self.cities_degraded,
            "version": self.version,
            "feature_store_uploaded": self.feature_store_uploaded,
            "aborted_reason": self.aborted_reason,
            "weather_enabled": self.weather_enabled,
            "aqi_enabled": self.aqi_enabled,
            "degradation_ratio_last_100": round(self.degradation_ratio_last_100, 4),
            "degradation_alert_triggered": self.degradation_alert_triggered,
            "results": [
                {
                    "city": r.city,
                    "success": r.success,
                    "error": r.error,
                    "warnings": r.warnings,
                    "duration_s": round(r.duration_s, 3),
                    "degraded_reason": r.degraded_reason,
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
            f"Cities:    {self.cities_succeeded}/{len(self.cities_requested)} succeeded"
            f" ({self.cities_degraded} degraded/AQI-only)",
            f"Version:   {self.version or '-'}",
            f"Weather Enabled: {self.weather_enabled}",
            f"AQI Enabled:     {self.aqi_enabled}",
            f"Feature Store Upload: {'yes' if self.feature_store_uploaded else 'no'}",
            f"Degradation Ratio (last 100): {self.degradation_ratio_last_100:.1%}"
            + (" ⚠ ALERT" if self.degradation_alert_triggered else ""),
        ]

        if self.aborted_reason:
            lines.append(f"ABORTED: {self.aborted_reason}")

        lines.append("-" * 60)

        for result in self.city_results:
            status = "OK" if result.success else "FAIL"
            degraded_tag = f" [DEGRADED: {result.degraded_reason}]" if result.degraded_reason else ""
            lines.append(f"[{status}] {result.city:<20} {result.duration_s:.2f}s{degraded_tag}")

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
        auto_degrade_on_weather_failure: bool = False,
        degradation_alert_threshold: float = 0.20,
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

        # NEW — opt-in, default OFF. See module docstring "Note on
        # Auto-Degrade". False = 100% identical behaviour to before:
        # a weather-fetch failure still fails the whole city-result.
        self.auto_degrade_on_weather_failure = auto_degrade_on_weather_failure
        self.degradation_alert_threshold = degradation_alert_threshold

        if not self.enable_weather:
            logger.info("Weather API disabled. Running AQI-only pipeline.")
        if self.auto_degrade_on_weather_failure:
            logger.info(
                "Auto-degrade enabled: cities will fall back to AQI-only "
                "if the Weather API fails after retries (AQI still required)."
            )

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
                #
                # CHANGED: when `auto_degrade_on_weather_failure` is
                # True, weather and AQI are fetched with
                # `return_exceptions=True` so a weather failure doesn't
                # cancel/kill the AQI fetch — AQI is still required,
                # weather becomes optional-at-runtime. Default
                # behaviour (flag off) is byte-for-byte the same as
                # before: a weather failure propagates and fails the
                # whole city-result.
                # --------------------------------------------
                weather_degraded = False
                weather_degraded_reason: str | None = None

                with _span(f"fetch:{city}"):
                    if self.enable_weather and self.auto_degrade_on_weather_failure:
                        weather_result, aqi_result = await asyncio.gather(
                            _fetch_with_retry(
                                self.weather_client, city,
                                max_attempts=self.fetch_retries, provider="weather",
                            ),
                            _fetch_with_retry(
                                self.aqi_client, city,
                                max_attempts=self.fetch_retries, provider="aqi",
                            ),
                            return_exceptions=True,
                        )
                        if isinstance(aqi_result, Exception):
                            # AQI is always required — no fallback for it.
                            raise aqi_result
                        aqi_raw = aqi_result

                        if isinstance(weather_result, Exception):
                            logger.warning(
                                "Weather fetch failed for %s after retries (%s) — "
                                "auto-degrading to AQI-only for this city.",
                                city, weather_result,
                            )
                            weather_raw = None
                            weather_degraded = True
                            weather_degraded_reason = "weather_fetch_failed"
                        else:
                            weather_raw = weather_result

                    elif self.enable_weather:
                        weather_raw, aqi_raw = await asyncio.gather(
                            _fetch_with_retry(
                                self.weather_client, city,
                                max_attempts=self.fetch_retries, provider="weather",
                            ),
                            _fetch_with_retry(
                                self.aqi_client, city,
                                max_attempts=self.fetch_retries, provider="aqi",
                            ),
                        )
                    else:
                        logger.info("Running AQI-only pipeline for %s.", city)
                        weather_raw = None
                        aqi_raw = await _fetch_with_retry(
                            self.aqi_client, city,
                            max_attempts=self.fetch_retries, provider="aqi",
                        )
                        weather_degraded_reason = "weather_disabled"

                # --------------------------------------------
                # Validate: AQI always validated; weather only
                # validated when it was actually fetched.
                # --------------------------------------------
                aqi_obj, aqi_report = ResponseValidator.validate_aqi_with_report(
                    aqi_raw, strict=self.strict
                )

                warnings = list(aqi_report.warnings)
                weather_obj = None
                weather_report = None

                if self.enable_weather and weather_raw is not None:
                    weather_obj, weather_report = ResponseValidator.validate_weather_with_report(
                        weather_raw, strict=self.strict
                    )
                    warnings = list(weather_report.warnings) + warnings
                elif self.enable_weather and weather_degraded:
                    logger.info(
                        "Weather data unavailable for %s (auto-degraded); "
                        "skipping weather validation.", city,
                    )
                else:
                    logger.info("Skipping weather validation (weather disabled) for %s.", city)

                # A city only fails on missing weather if weather was
                # REQUIRED (enabled) AND we're not in an auto-degrade
                # fallback — auto-degrade intentionally tolerates
                # missing weather without failing the city.
                weather_missing = self.enable_weather and not weather_degraded and weather_obj is None

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
                # Merge: full weather+AQI merge only when weather
                # data actually made it through; otherwise an
                # AQI-only feature. Every branch now records
                # structured telemetry (Missing Telemetry fix).
                # --------------------------------------------
                with _span(f"merge:{city}"):
                    if self.enable_weather and not weather_degraded:
                        feature: Feature = FeatureMerger.merge(weather_obj, aqi_obj)
                        PipelineMetrics.record_full_mode(city=city)
                    else:
                        logger.info("Skipping feature merge (weather unavailable) for %s.", city)
                        feature = AQIOnlyFeature.from_aqi(aqi_obj)
                        reason = weather_degraded_reason or "weather_disabled"
                        PipelineMetrics.record_degraded_mode(city=city, reason=reason)
                        weather_degraded_reason = reason

                duration_s = time.perf_counter() - start
                PipelineMetrics.record_city_duration(duration_s)

                return CityResult(
                    city=city, success=True, feature=feature,
                    warnings=warnings, duration_s=duration_s,
                    degraded_reason=weather_degraded_reason if (not self.enable_weather or weather_degraded) else None,
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

        # NEW: snapshot the trailing degradation ratio onto the summary
        # and run the threshold check — this is what turns "degraded
        # data has been accumulating silently" into a logged alert on
        # every single run, without requiring anyone to go dig through
        # PipelineMetrics.get_metrics() manually.
        summary.degradation_ratio_last_100 = PipelineMetrics.get_degradation_ratio(last_n=100)
        summary.degradation_alert_triggered = PipelineMetrics.check_degradation_alert(
            threshold=self.degradation_alert_threshold, last_n=100,
        )

        await self._run_hooks(self._after_run_hooks, summary)

        logger.info(
            "Pipeline run %s finished: %d/%d cities succeeded (%d degraded).",
            run_id, summary.cities_succeeded, len(cities), summary.cities_degraded,
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
    # NEW: opt-in dynamic auto-degrade + alert threshold.
    parser.add_argument(
        "--auto-degrade-weather", dest="auto_degrade_on_weather_failure",
        action="store_true", default=False,
        help="If the Weather API fails after retries, fall back to an "
             "AQI-only result for that city instead of failing it outright.",
    )
    parser.add_argument(
        "--degradation-alert-threshold", type=float, default=0.20,
        help="Log a DEGRADATION_ALERT if more than this fraction of the "
             "last 100 city-results were AQI-only (default: 0.20).",
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
        auto_degrade_on_weather_failure=args.auto_degrade_on_weather_failure,
        degradation_alert_threshold=args.degradation_alert_threshold,
    )

    summary = await pipeline.run(args.cities, formats=tuple(args.formats))

    if args.json_summary:
        print(json.dumps(summary.to_dict(), indent=2, default=str))
    else:
        print(summary.render())

    return 0 if summary.cities_failed == 0 and not summary.aborted_reason else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
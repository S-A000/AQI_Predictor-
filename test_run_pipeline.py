"""
test_run_pipeline.py
=====================
Tests for src/ingestion/run_pipeline.py

Pipeline is exercised with fully injected fake weather/aqi/storage
clients so no real HTTP, filesystem, or Prometheus/OTel machinery is
touched. `asyncio.sleep` is patched to a no-op to keep retry-backoff
tests instant.
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from src.ingestion.run_pipeline import (
    CityResult,
    Pipeline,
    PipelineMetrics,
    PipelineRunSummary,
    _build_arg_parser,
    _fetch_with_retry,
    airflow_callable,
)


@pytest.fixture(autouse=True)
def _fast_sleep(monkeypatch):
    async def _noop(*_a, **_kw):
        return None
    monkeypatch.setattr(asyncio, "sleep", _noop)


# ============================================================
# Fake clients
# ============================================================

class FakeWeatherClient:
    def __init__(self, responses=None, fail_cities=None):
        self.responses = responses or {}
        self.fail_cities = fail_cities or set()

    async def fetch(self, city: str) -> dict:
        if city in self.fail_cities:
            raise RuntimeError(f"weather fetch failed for {city}")
        return self.responses.get(city, _default_weather_raw(city))

    async def aclose(self):
        pass


class FakeAQIClient:
    def __init__(self, responses=None, fail_cities=None):
        self.responses = responses or {}
        self.fail_cities = fail_cities or set()

    async def fetch(self, city: str) -> dict:
        if city in self.fail_cities:
            raise RuntimeError(f"aqi fetch failed for {city}")
        return self.responses.get(city, _default_aqi_raw(city))

    async def aclose(self):
        pass


def _default_weather_raw(city: str) -> dict:
    return {
        "coord": {"lon": 67.0, "lat": 24.8},
        "weather": [{"id": 800, "main": "Clear", "description": "clear", "icon": "01d"}],
        "main": {"temp": 30.0, "feels_like": 31.0, "pressure": 1010, "humidity": 50,
                  "temp_min": 28.0, "temp_max": 32.0},
        "wind": {"speed": 3.0, "deg": 180},
        "clouds": {"all": 10},
        "visibility": 9000,
        "dt": 1_721_000_000,
        "timezone": 18000,
        "name": city,
        "sys": {"sunrise": 1, "sunset": 2, "country": "PK"},
    }


def _default_aqi_raw(city: str) -> dict:
    return {
        "status": "ok",
        "data": {
            "aqi": 100, "idx": 1, "dominentpol": "pm25",
            "city": {"name": city, "geo": [24.8, 67.0]},
            "time": {"s": "now"},
            "iaqi": {"pm25": {"v": 100}},
            "forecast": None,
        },
    }


@pytest.fixture
def fake_storage():
    storage = MagicMock()
    storage.health_check.return_value = {"healthy": True, "issues": []}
    storage.save_batch.return_value = {"version": "v1", "paths": {}, "metadata": {}}
    storage.cleanup_old_versions.return_value = []
    return storage


# ============================================================
# _fetch_with_retry
# ============================================================

class TestFetchWithRetry:

    @pytest.mark.asyncio
    async def test_succeeds_first_try(self):
        client = FakeWeatherClient()
        result = await _fetch_with_retry(client, "Karachi")
        assert result["name"] == "Karachi"

    @pytest.mark.asyncio
    async def test_retries_then_succeeds(self):
        calls = {"n": 0}

        class FlakyClient:
            async def fetch(self, city):
                calls["n"] += 1
                if calls["n"] < 3:
                    raise RuntimeError("transient")
                return {"ok": True}

        result = await _fetch_with_retry(FlakyClient(), "Karachi", max_attempts=5, delay=0.01)
        assert result == {"ok": True}
        assert calls["n"] == 3

    @pytest.mark.asyncio
    async def test_exhausts_attempts_and_raises(self):
        class AlwaysFails:
            async def fetch(self, city):
                raise RuntimeError("permanent failure")

        with pytest.raises(RuntimeError):
            await _fetch_with_retry(AlwaysFails(), "Karachi", max_attempts=2, delay=0.01)


# ============================================================
# Pipeline._process_city
# ============================================================

class TestProcessCity:

    @pytest.mark.asyncio
    async def test_successful_city(self, fake_storage):
        pipeline = Pipeline(
            weather_client=FakeWeatherClient(),
            aqi_client=FakeAQIClient(),
            storage_manager=fake_storage,
        )
        result = await pipeline._process_city("Karachi")
        assert result.success is True
        assert result.feature is not None
        assert result.feature.city == "Karachi"

    @pytest.mark.asyncio
    async def test_fetch_failure_marks_city_failed(self, fake_storage):
        pipeline = Pipeline(
            weather_client=FakeWeatherClient(fail_cities={"Karachi"}),
            aqi_client=FakeAQIClient(),
            storage_manager=fake_storage,
            fetch_retries=1,
        )
        result = await pipeline._process_city("Karachi")
        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_validation_failure_marks_city_failed(self, fake_storage):
        bad_weather = {"Karachi": {"not": "a valid payload"}}
        pipeline = Pipeline(
            weather_client=FakeWeatherClient(responses=bad_weather),
            aqi_client=FakeAQIClient(),
            storage_manager=fake_storage,
        )
        result = await pipeline._process_city("Karachi")
        assert result.success is False
        assert result.error

    @pytest.mark.asyncio
    async def test_shutdown_event_skips_city(self, fake_storage):
        pipeline = Pipeline(
            weather_client=FakeWeatherClient(),
            aqi_client=FakeAQIClient(),
            storage_manager=fake_storage,
        )
        pipeline._shutdown_event.set()
        result = await pipeline._process_city("Karachi")
        assert result.success is False
        assert "shutdown" in result.error


# ============================================================
# Pipeline.run
# ============================================================

class TestPipelineRun:

    @pytest.mark.asyncio
    async def test_run_all_cities_succeed(self, fake_storage):
        pipeline = Pipeline(
            weather_client=FakeWeatherClient(),
            aqi_client=FakeAQIClient(),
            storage_manager=fake_storage,
        )
        summary = await pipeline.run(["Karachi", "Lahore"])
        assert summary.cities_succeeded == 2
        assert summary.cities_failed == 0
        fake_storage.save_batch.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_partial_failure(self, fake_storage):
        pipeline = Pipeline(
            weather_client=FakeWeatherClient(fail_cities={"BadCity"}),
            aqi_client=FakeAQIClient(),
            storage_manager=fake_storage,
            fetch_retries=1,
        )
        summary = await pipeline.run(["Karachi", "BadCity"])
        assert summary.cities_succeeded == 1
        assert summary.cities_failed == 1

    @pytest.mark.asyncio
    async def test_run_aborts_when_storage_unhealthy(self, fake_storage):
        fake_storage.health_check.return_value = {"healthy": False, "issues": ["disk full"]}
        pipeline = Pipeline(
            weather_client=FakeWeatherClient(),
            aqi_client=FakeAQIClient(),
            storage_manager=fake_storage,
            abort_on_unhealthy=True,
        )
        summary = await pipeline.run(["Karachi"])
        assert summary.aborted_reason is not None
        fake_storage.save_batch.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_skips_health_check_when_disabled(self, fake_storage):
        pipeline = Pipeline(
            weather_client=FakeWeatherClient(),
            aqi_client=FakeAQIClient(),
            storage_manager=fake_storage,
            abort_on_unhealthy=False,
        )
        await pipeline.run(["Karachi"])
        fake_storage.health_check.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_uploads_to_feature_store_when_enabled(self, fake_storage):
        pipeline = Pipeline(
            weather_client=FakeWeatherClient(),
            aqi_client=FakeAQIClient(),
            storage_manager=fake_storage,
            upload_to_feature_store=True,
        )
        summary = await pipeline.run(["Karachi"])
        fake_storage.upload_to_feature_store.assert_called_once()
        assert summary.feature_store_uploaded is True

    @pytest.mark.asyncio
    async def test_feature_store_upload_failure_does_not_fail_run(self, fake_storage):
        fake_storage.upload_to_feature_store.side_effect = RuntimeError("boom")
        pipeline = Pipeline(
            weather_client=FakeWeatherClient(),
            aqi_client=FakeAQIClient(),
            storage_manager=fake_storage,
            upload_to_feature_store=True,
        )
        summary = await pipeline.run(["Karachi"])
        assert summary.feature_store_uploaded is False
        assert summary.cities_failed == 0

    @pytest.mark.asyncio
    async def test_before_and_after_hooks_are_called(self, fake_storage):
        pipeline = Pipeline(
            weather_client=FakeWeatherClient(),
            aqi_client=FakeAQIClient(),
            storage_manager=fake_storage,
        )
        calls = []

        @pipeline.before_run
        def before(summary):
            calls.append(("before", summary.run_id))

        @pipeline.after_run
        async def after(summary):
            calls.append(("after", summary.run_id))

        summary = await pipeline.run(["Karachi"])
        assert calls == [("before", summary.run_id), ("after", summary.run_id)]

    @pytest.mark.asyncio
    async def test_hook_exception_does_not_abort_run(self, fake_storage):
        pipeline = Pipeline(
            weather_client=FakeWeatherClient(),
            aqi_client=FakeAQIClient(),
            storage_manager=fake_storage,
        )

        @pipeline.before_run
        def bad_hook(summary):
            raise RuntimeError("hook broke")

        summary = await pipeline.run(["Karachi"])
        assert summary.cities_succeeded == 1

    @pytest.mark.asyncio
    async def test_clients_closed_after_run(self, fake_storage):
        weather = FakeWeatherClient()
        aqi = FakeAQIClient()
        weather.aclose = MagicMock(side_effect=lambda: asyncio.sleep(0))
        pipeline = Pipeline(weather_client=weather, aqi_client=aqi, storage_manager=fake_storage)
        await pipeline.run(["Karachi"])
        weather.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_features_skips_storage_save(self, fake_storage):
        pipeline = Pipeline(
            weather_client=FakeWeatherClient(fail_cities={"OnlyCity"}),
            aqi_client=FakeAQIClient(),
            storage_manager=fake_storage,
            fetch_retries=1,
        )
        await pipeline.run(["OnlyCity"])
        fake_storage.save_batch.assert_not_called()


# ============================================================
# CityResult / PipelineRunSummary
# ============================================================

class TestSummaryModels:

    def test_city_result_defaults(self):
        result = CityResult(city="Karachi", success=True)
        assert result.warnings == []
        assert result.duration_s == 0.0

    def test_summary_counts_success_and_failure(self):
        summary = PipelineRunSummary(
            run_id="abc",
            started_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
            cities_requested=["A", "B"],
            city_results=[
                CityResult(city="A", success=True),
                CityResult(city="B", success=False, error="oops"),
            ],
        )
        assert summary.cities_succeeded == 1
        assert summary.cities_failed == 1

    def test_summary_to_dict_and_render(self):
        import datetime as dt
        started = dt.datetime.now(dt.timezone.utc)
        summary = PipelineRunSummary(
            run_id="abc",
            started_at=started,
            finished_at=started + dt.timedelta(seconds=2),
            cities_requested=["A"],
            city_results=[CityResult(city="A", success=True, duration_s=1.0)],
        )
        data = summary.to_dict()
        assert data["cities_succeeded"] == 1
        rendered = summary.render()
        assert "Pipeline Run Summary" in rendered
        assert "[OK] A" in rendered


# ============================================================
# PipelineMetrics
# ============================================================

class TestPipelineMetrics:

    def setup_method(self):
        PipelineMetrics.reset()

    def teardown_method(self):
        PipelineMetrics.reset()

    def test_records_runs(self):
        PipelineMetrics.record_run(True, 1.5)
        PipelineMetrics.record_run(False, 2.5)
        metrics = PipelineMetrics.get_metrics()
        assert metrics["runs"] == 2
        assert metrics["success"] == 1
        assert metrics["failure"] == 1
        assert metrics["avg_duration_s"] == 2.0


# ============================================================
# airflow_callable
# ============================================================

class TestAirflowCallable:

    def test_raises_when_cities_fail(self, fake_storage, monkeypatch):
        def fake_pipeline_init(self, **kwargs):
            self._pipeline = Pipeline(
                weather_client=FakeWeatherClient(fail_cities={"Bad"}),
                aqi_client=FakeAQIClient(),
                storage_manager=fake_storage,
                fetch_retries=1,
            )

        # Monkeypatch Pipeline construction inside airflow_callable by
        # patching the Pipeline class used in run_pipeline module.
        import src.ingestion.run_pipeline as rp

        def fake_pipeline_factory(**kwargs):
            return Pipeline(
                weather_client=FakeWeatherClient(fail_cities={"Bad"}),
                aqi_client=FakeAQIClient(),
                storage_manager=fake_storage,
                fetch_retries=1,
            )

        monkeypatch.setattr(rp, "Pipeline", fake_pipeline_factory)

        with pytest.raises(RuntimeError):
            airflow_callable(["Bad"])

    def test_returns_summary_dict_on_success(self, fake_storage, monkeypatch):
        import src.ingestion.run_pipeline as rp

        def fake_pipeline_factory(**kwargs):
            return Pipeline(
                weather_client=FakeWeatherClient(),
                aqi_client=FakeAQIClient(),
                storage_manager=fake_storage,
            )

        monkeypatch.setattr(rp, "Pipeline", fake_pipeline_factory)

        result = airflow_callable(["Karachi"])
        assert result["cities_succeeded"] == 1


# ============================================================
# CLI arg parser
# ============================================================

class TestArgParser:

    def test_defaults(self):
        args = _build_arg_parser().parse_args([])
        assert args.cities == ["Karachi", "Lahore", "Islamabad"]
        assert args.strict is True
        assert args.feature_store is False

    def test_custom_cities_and_formats(self):
        args = _build_arg_parser().parse_args(
            ["--cities", "Multan", "Quetta", "--formats", "json"]
        )
        assert args.cities == ["Multan", "Quetta"]
        assert args.formats == ["json"]

    def test_soft_flag_disables_strict(self):
        args = _build_arg_parser().parse_args(["--soft"])
        assert args.strict is False

    def test_feature_store_flag(self):
        args = _build_arg_parser().parse_args(["--feature-store", "--feature-group", "custom"])
        assert args.feature_store is True
        assert args.feature_group == "custom"

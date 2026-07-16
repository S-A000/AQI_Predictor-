"""
test_settings.py
=================
Tests for src/configs/settings.py

Isolation strategy
-------------------
`Settings.load_yaml_config` reads `configs/config.yaml` and
`configs/{env}.yaml` from disk (relative to the real project root) as
part of a `model_validator(mode="before")`. To keep these tests
independent of whatever config files happen to exist in the repo
running them, every test that constructs a `Settings()` instance
directly:

1. Passes `project` / `location` / `api` / `paths` explicitly as init
   kwargs — these take precedence over anything YAML would supply
   (`load_yaml_config` only fills in keys that are *missing*).
2. Patches `yaml.safe_load` to return `{}` and `pathlib.Path.exists`
   to return `False`, so no real config file is read even if one
   exists on disk.
3. Points every `PathConfig` field at a `tmp_path` subdirectory, so
   `model_post_init`'s directory creation never touches the real
   filesystem.
"""
from __future__ import annotations

import pathlib

import pytest
import yaml
from pydantic import ValidationError

from src.configs.settings import (
    APIConfig,
    ConfigurationError,
    Environment,
    LocationConfig,
    PathConfig,
    ProjectConfig,
    Settings,
    YAMLLoadError,
    get_settings,
    settings,
)


# ============================================================
# Isolation fixtures
# ============================================================

@pytest.fixture
def no_yaml_on_disk(monkeypatch):
    """Prevent load_yaml_config from picking up real config files."""
    monkeypatch.setattr(yaml, "safe_load", lambda *_a, **_kw: {})
    monkeypatch.setattr(pathlib.Path, "exists", lambda self: False)


@pytest.fixture
def settings_kwargs(tmp_path):
    """A complete, valid set of init kwargs for Settings()."""
    return {
        "AQICN_API_KEY": "aqicn-secret-value",
        "OPENWEATHER_API_KEY": "openweather-secret-value",
        "project": {"name": "aqi-forecasting-test"},
        "location": {"city": "Karachi", "country": "PK"},
        "api": {"timeout": 15, "retries": 2, "batch_size": 50},
        "paths": {
            "raw_data": str(tmp_path / "raw"),
            "processed_data": str(tmp_path / "processed"),
            "models": str(tmp_path / "models"),
            "reports": str(tmp_path / "reports"),
            "logs": str(tmp_path / "logs"),
        },
    }


@pytest.fixture
def built_settings(no_yaml_on_disk, settings_kwargs):
    return Settings(**settings_kwargs)


# ============================================================
# Custom exceptions
# ============================================================

class TestExceptions:

    def test_configuration_error_is_an_exception(self):
        assert issubclass(ConfigurationError, Exception)

    def test_yaml_load_error_is_a_configuration_error(self):
        assert issubclass(YAMLLoadError, ConfigurationError)

    def test_exceptions_carry_message(self):
        with pytest.raises(ConfigurationError, match="boom"):
            raise ConfigurationError("boom")


# ============================================================
# Environment enum
# ============================================================

class TestEnvironmentEnum:

    def test_values(self):
        assert Environment.DEV == "dev"
        assert Environment.STAGING == "staging"
        assert Environment.PROD == "prod"

    def test_is_str_enum(self):
        assert isinstance(Environment.DEV, str)


# ============================================================
# Nested config models (no filesystem involved)
# ============================================================

class TestProjectConfig:

    def test_defaults(self):
        project = ProjectConfig()
        assert project.name == "aqi-forecasting"
        assert project.version == "1.0.0"
        assert project.description

    def test_overrides(self):
        project = ProjectConfig(name="custom", version="2.0.0")
        assert project.name == "custom"
        assert project.version == "2.0.0"


class TestLocationConfig:

    def test_requires_city_and_country(self):
        with pytest.raises(ValidationError):
            LocationConfig()

    def test_valid_construction(self):
        loc = LocationConfig(city="Karachi", country="PK")
        assert loc.city == "Karachi"
        assert loc.coordinates is None

    def test_optional_coordinates(self):
        loc = LocationConfig(city="Karachi", country="PK", coordinates=[24.86, 67.0])
        assert loc.coordinates == [24.86, 67.0]


class TestAPIConfig:

    def test_defaults(self):
        api = APIConfig()
        assert api.timeout == 30
        assert api.retries == 3
        assert api.batch_size == 100

    @pytest.mark.parametrize("timeout", [0, -1, 121, 500])
    def test_timeout_out_of_range_raises(self, timeout):
        with pytest.raises(ValidationError):
            APIConfig(timeout=timeout)

    @pytest.mark.parametrize("timeout", [1, 30, 120])
    def test_timeout_boundaries_are_valid(self, timeout):
        assert APIConfig(timeout=timeout).timeout == timeout

    def test_negative_retries_raises(self):
        with pytest.raises(ValidationError):
            APIConfig(retries=-1)


class TestPathConfig:

    def test_requires_all_fields(self):
        with pytest.raises(ValidationError):
            PathConfig(raw_data="a", processed_data="b")  # missing models/reports/logs

    def test_relative_paths_are_resolved_to_absolute(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        paths = PathConfig(
            raw_data="raw",
            processed_data="processed",
            models="models",
            reports="reports",
            logs="logs",
        )
        for field_name, value in paths:
            assert isinstance(value, pathlib.Path)
            assert value.is_absolute()

    def test_already_absolute_path_stays_equivalent(self, tmp_path):
        paths = PathConfig(
            raw_data=tmp_path / "raw",
            processed_data=tmp_path / "processed",
            models=tmp_path / "models",
            reports=tmp_path / "reports",
            logs=tmp_path / "logs",
        )
        assert paths.raw_data == (tmp_path / "raw").resolve()


# ============================================================
# Settings — construction & secret handling
# ============================================================

class TestSettingsConstruction:

    def test_builds_with_valid_kwargs(self, built_settings):
        assert built_settings.location.city == "Karachi"
        assert built_settings.api.timeout == 15

    def test_missing_required_secret_raises(self, no_yaml_on_disk, settings_kwargs, monkeypatch):
        monkeypatch.delenv("AQICN_API_KEY", raising=False)
        monkeypatch.delenv("OPENWEATHER_API_KEY", raising=False)
        kwargs = dict(settings_kwargs)
        del kwargs["AQICN_API_KEY"]
        with pytest.raises(ValidationError):
            Settings(**kwargs)

    def test_secret_repr_does_not_leak_raw_value(self, built_settings):
        assert "aqicn-secret-value" not in repr(built_settings.aqicn_api_key)
        assert "openweather-secret-value" not in repr(built_settings.openweather_api_key)

    def test_revealed_properties_return_raw_secret(self, built_settings):
        assert built_settings.aqicn_api_key_revealed == "aqicn-secret-value"
        assert built_settings.openweather_api_key_revealed == "openweather-secret-value"

    def test_default_environment_is_dev(self, no_yaml_on_disk, settings_kwargs):
        s = Settings(**settings_kwargs)
        assert s.app_env == Environment.DEV
        assert s.is_production is False

    def test_prod_environment_flag(self, no_yaml_on_disk, settings_kwargs):
        kwargs = {**settings_kwargs, "APP_ENV": "prod"}
        s = Settings(**kwargs)
        assert s.app_env == Environment.PROD
        assert s.is_production is True

    def test_extra_unknown_fields_are_ignored(self, no_yaml_on_disk, settings_kwargs):
        kwargs = {**settings_kwargs, "some_totally_unknown_field": "whatever"}
        Settings(**kwargs)  # should not raise, extra="ignore"


# ============================================================
# Settings — directory creation (model_post_init)
# ============================================================

class TestDirectoryCreation:

    def test_creates_all_configured_directories(self, built_settings):
        for _field_name, path in built_settings.paths:
            assert path.exists()
            assert path.is_dir()

    def test_idempotent_directory_creation(self, no_yaml_on_disk, settings_kwargs):
        # Constructing twice with the same paths should not raise
        # (exist_ok=True), even though the directories already exist
        # after the first construction.
        Settings(**settings_kwargs)
        Settings(**settings_kwargs)  # should not raise

    def test_directory_creation_failure_raises_configuration_error(
        self, no_yaml_on_disk, settings_kwargs, monkeypatch
    ):
        def _boom(self, parents=True, exist_ok=True):
            raise OSError("disk full")

        monkeypatch.setattr(pathlib.Path, "mkdir", _boom)
        with pytest.raises(ConfigurationError):
            Settings(**settings_kwargs)


# ============================================================
# Singleton behaviour (get_settings / module-level `settings`)
# ============================================================

class TestSingleton:

    def test_get_settings_is_cached(self):
        assert get_settings() is get_settings()

    def test_module_level_settings_matches_get_settings(self):
        assert settings is get_settings()

    def test_module_level_settings_is_a_settings_instance(self):
        assert isinstance(settings, Settings)


# ============================================================
# YAML merge behaviour (load_yaml_config), isolated from disk
# ============================================================

class TestYamlMergeLogic:
    """
    `load_yaml_config` is a `@model_validator(mode="before")` wrapping a
    `@classmethod`. We don't assert exact merge semantics against real
    files (that depends on the project's actual `configs/*.yaml`
    layout, which wasn't provided), but we do verify the documented
    precedence rule: explicit init kwargs always win over YAML values.
    """

    def test_settings_construction_does_not_touch_disk_when_isolated(
        self, no_yaml_on_disk, settings_kwargs
    ):
        # If load_yaml_config tried to actually open a real file despite
        # Path.exists() being patched to False, this would raise —
        # constructing successfully is itself the assertion.
        s = Settings(**settings_kwargs)
        assert s.project.name == "aqi-forecasting-test"

    def test_init_kwargs_take_precedence_over_yaml(self, monkeypatch, settings_kwargs):
        # Simulate a base config.yaml that *would* set a different city,
        # to confirm explicit init kwargs win (per load_yaml_config's
        # `if key not in values` merge rule).
        monkeypatch.setattr(pathlib.Path, "exists", lambda self: True)
        monkeypatch.setattr(
            yaml, "safe_load", lambda *_a, **_kw: {"location": {"city": "Lahore", "country": "PK"}}
        )
        monkeypatch.setattr(pathlib.Path, "mkdir", lambda self, parents=True, exist_ok=True: None)

        s = Settings(**settings_kwargs)
        assert s.location.city == "Karachi"  # explicit kwarg wins over YAML

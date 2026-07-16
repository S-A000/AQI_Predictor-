"""
test_validator.py
==================
Tests for src/ingestion/validator.py
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.ingestion.validator import (
    AQIResponse,
    AQIResponseValidator,
    AQIValidationError,
    ResponseValidator,
    ValidationMetrics,
    WeatherResponse,
    WeatherResponseValidator,
    WeatherValidationError,
    _looks_like_kelvin,
    _normalize_value,
    celsius_to_kelvin,
    compute_payload_checksum,
    kelvin_to_celsius,
)


# ============================================================
# Normalization helpers
# ============================================================

class TestNormalizeValue:

    def test_dash_becomes_none(self):
        assert _normalize_value("-") is None

    def test_empty_string_becomes_none(self):
        assert _normalize_value("") is None

    def test_regular_string_untouched(self):
        assert _normalize_value("Karachi") == "Karachi"

    def test_recurses_into_dict(self):
        assert _normalize_value({"a": "-", "b": "x"}) == {"a": None, "b": "x"}

    def test_recurses_into_list(self):
        assert _normalize_value(["-", "", "x"]) == [None, None, "x"]

    def test_non_string_passthrough(self):
        assert _normalize_value(42) == 42
        assert _normalize_value(3.14) == 3.14
        assert _normalize_value(None) is None


# ============================================================
# Unit conversion
# ============================================================

class TestUnitConversion:

    def test_kelvin_to_celsius(self):
        assert kelvin_to_celsius(273.15) == 0.0

    def test_celsius_to_kelvin(self):
        assert celsius_to_kelvin(0.0) == 273.15

    def test_round_trip(self):
        original = 25.37
        assert kelvin_to_celsius(celsius_to_kelvin(original)) == original

    @pytest.mark.parametrize("value,expected", [(131, True), (130, False), (25, False), (305, True)])
    def test_looks_like_kelvin(self, value, expected):
        assert _looks_like_kelvin(value) is expected


# ============================================================
# Checksum
# ============================================================

class TestChecksum:

    def test_checksum_is_stable(self):
        data = {"b": 2, "a": 1}
        assert compute_payload_checksum(data) == compute_payload_checksum({"a": 1, "b": 2})

    def test_checksum_changes_with_data(self):
        assert compute_payload_checksum({"a": 1}) != compute_payload_checksum({"a": 2})


# ============================================================
# WeatherResponse
# ============================================================

class TestWeatherResponse:

    def test_valid_payload_parses(self, sample_weather_raw):
        weather = WeatherResponse.model_validate(sample_weather_raw)
        assert weather.name == "Karachi"
        assert weather.sys.country == "PK"
        assert weather.main.temp == pytest.approx(32.5)

    def test_kelvin_auto_converted_to_celsius(self, sample_weather_raw_kelvin):
        weather = WeatherResponse.model_validate(sample_weather_raw_kelvin)
        # 305.65 K -> ~32.5 C
        assert 30 <= weather.main.temp <= 35

    def test_missing_required_field_raises(self, sample_weather_raw):
        bad = {k: v for k, v in sample_weather_raw.items() if k != "coord"}
        with pytest.raises(ValidationError):
            WeatherResponse.model_validate(bad)

    def test_humidity_out_of_range_raises(self, sample_weather_raw):
        bad = {**sample_weather_raw, "main": {**sample_weather_raw["main"], "humidity": 150}}
        with pytest.raises(ValidationError):
            WeatherResponse.model_validate(bad)

    def test_model_is_frozen(self, weather_response):
        with pytest.raises(ValidationError):
            weather_response.name = "Lahore"

    def test_dash_placeholder_normalized(self, sample_weather_raw):
        raw = {**sample_weather_raw, "wind": {**sample_weather_raw["wind"], "gust": "-"}}
        weather = WeatherResponse.model_validate(raw)
        assert weather.wind.gust is None


# ============================================================
# AQIResponse
# ============================================================

class TestAQIResponse:

    def test_valid_payload_parses(self, sample_aqi_raw):
        aqi = AQIResponse.model_validate(sample_aqi_raw)
        assert aqi.data.aqi == 152
        assert aqi.data.dominentpol == "pm25"

    def test_dash_pollutant_normalized_to_none(self, sample_aqi_raw):
        aqi = AQIResponse.model_validate(sample_aqi_raw)
        assert aqi.data.iaqi["so2"]["v"] is None

    def test_status_not_ok_raises(self, sample_aqi_raw):
        bad = {**sample_aqi_raw, "status": "error"}
        with pytest.raises(ValidationError):
            AQIResponse.model_validate(bad)

    def test_aqi_out_of_range_raises(self, sample_aqi_raw):
        bad = {**sample_aqi_raw, "data": {**sample_aqi_raw["data"], "aqi": 999}}
        with pytest.raises(ValidationError):
            AQIResponse.model_validate(bad)

    def test_empty_dominant_pollutant_raises(self, sample_aqi_raw):
        bad = {**sample_aqi_raw, "data": {**sample_aqi_raw["data"], "dominentpol": ""}}
        with pytest.raises(ValidationError):
            AQIResponse.model_validate(bad)


# ============================================================
# Domain warnings (soft validation)
# ============================================================

class TestDomainWarnings:

    def test_hazardous_aqi_produces_warning(self, sample_aqi_raw):
        raw = {**sample_aqi_raw, "data": {**sample_aqi_raw["data"], "aqi": 420}}
        instance, report = AQIResponseValidator.validate_with_report(raw, strict=False)
        assert instance is not None
        assert any("hazardous" in w for w in report.warnings)

    def test_missing_iaqi_produces_warning(self, sample_aqi_raw):
        raw = {**sample_aqi_raw, "data": {**sample_aqi_raw["data"], "iaqi": {}}}
        instance, report = AQIResponseValidator.validate_with_report(raw, strict=False)
        assert instance is not None
        assert any("iaqi" in w for w in report.warnings)

    def test_extreme_temperature_produces_warning(self, sample_weather_raw):
        raw = {**sample_weather_raw, "main": {**sample_weather_raw["main"], "temp": 90.0}}
        instance, report = WeatherResponseValidator.validate_with_report(raw, strict=False)
        assert instance is not None
        assert any("Earth's recorded range" in w for w in report.warnings)

    def test_unrealistic_wind_speed_produces_warning(self, sample_weather_raw):
        raw = {**sample_weather_raw, "wind": {**sample_weather_raw["wind"], "speed": 150.0}}
        instance, report = WeatherResponseValidator.validate_with_report(raw, strict=False)
        assert instance is not None
        assert any("realistic surface winds" in w for w in report.warnings)


# ============================================================
# strict vs soft validation behaviour
# ============================================================

class TestStrictVsSoft:

    def test_strict_raises_on_structural_error(self, sample_aqi_raw):
        bad = {k: v for k, v in sample_aqi_raw.items() if k != "data"}
        with pytest.raises(AQIValidationError):
            AQIResponseValidator.validate_with_report(bad, strict=True)

    def test_soft_returns_none_and_errors_on_structural_error(self, sample_aqi_raw):
        bad = {k: v for k, v in sample_aqi_raw.items() if k != "data"}
        instance, report = AQIResponseValidator.validate_with_report(bad, strict=False)
        assert instance is None
        assert report.success is False
        assert report.errors

    def test_report_checksum_and_timing_present(self, sample_weather_raw):
        _, report = WeatherResponseValidator.validate_with_report(sample_weather_raw, strict=False)
        assert report.checksum is not None
        assert report.duration_ms >= 0


# ============================================================
# ResponseValidator facade
# ============================================================

class TestResponseValidatorFacade:

    def test_validate_weather_returns_model(self, sample_weather_raw):
        weather = ResponseValidator.validate_weather(sample_weather_raw)
        assert isinstance(weather, WeatherResponse)

    def test_validate_aqi_returns_model(self, sample_aqi_raw):
        aqi = ResponseValidator.validate_aqi(sample_aqi_raw)
        assert isinstance(aqi, AQIResponse)

    def test_validate_weather_raises_domain_exception(self, sample_weather_raw):
        bad = {k: v for k, v in sample_weather_raw.items() if k != "main"}
        with pytest.raises(WeatherValidationError):
            ResponseValidator.validate_weather(bad)

    def test_validate_weather_with_report_soft(self, sample_weather_raw):
        instance, report = ResponseValidator.validate_weather_with_report(sample_weather_raw, strict=False)
        assert instance is not None
        assert report.success is True

    def test_validate_aqi_with_report_soft(self, sample_aqi_raw):
        instance, report = ResponseValidator.validate_aqi_with_report(sample_aqi_raw, strict=False)
        assert instance is not None
        assert report.success is True


# ============================================================
# ValidationMetrics
# ============================================================

class TestValidationMetrics:

    def setup_method(self):
        ValidationMetrics.reset()

    def teardown_method(self):
        ValidationMetrics.reset()

    def test_records_success_and_failure(self, sample_weather_raw):
        ResponseValidator.validate_weather_with_report(sample_weather_raw, strict=False)
        bad = {k: v for k, v in sample_weather_raw.items() if k != "coord"}
        ResponseValidator.validate_weather_with_report(bad, strict=False)

        metrics = ResponseValidator.get_metrics()
        assert metrics["WeatherResponse"]["success"] == 1
        assert metrics["WeatherResponse"]["failure"] == 1
        assert metrics["WeatherResponse"]["total"] == 2

    def test_get_metrics_empty_after_reset(self):
        assert ValidationMetrics.get_metrics() == {}

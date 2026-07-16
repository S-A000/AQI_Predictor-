"""
test_merger.py
===============
Tests for src/ingestion/merger.py
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from src.ingestion.merger import FeatureMerger, MergedFeature
from src.ingestion.validator import AQIResponse, WeatherResponse


# ============================================================
# merge()
# ============================================================

class TestMerge:

    def test_merge_produces_expected_fields(self, weather_response, aqi_response):
        feature = FeatureMerger.merge(weather_response, aqi_response)

        assert feature.city == "Karachi"
        assert feature.country == "PK"
        assert feature.aqi == 152
        assert feature.dominant_pollutant == "pm25"
        assert feature.station_id == 1437
        assert feature.pm25 == 152
        assert feature.pm10 == 88
        assert feature.so2 is None  # normalized "-" -> None upstream

    def test_merge_timestamp_is_utc(self, weather_response, aqi_response):
        feature = FeatureMerger.merge(weather_response, aqi_response)
        assert feature.timestamp.tzinfo is not None

    def test_merge_sets_default_source(self, merged_feature):
        assert merged_feature.source == "AQICN + OpenWeather"

    def test_merged_feature_is_frozen(self, merged_feature):
        with pytest.raises(Exception):
            merged_feature.city = "Lahore"


# ============================================================
# _extract_pollutant()
# ============================================================

class TestExtractPollutant:

    def test_extracts_value(self):
        iaqi = {"pm25": {"v": 42}}
        assert FeatureMerger._extract_pollutant(iaqi, "pm25") == 42

    def test_missing_key_returns_none(self):
        assert FeatureMerger._extract_pollutant({}, "pm25") is None

    def test_non_dict_value_returns_none(self):
        assert FeatureMerger._extract_pollutant({"pm25": None}, "pm25") is None


# ============================================================
# merge_batch()
# ============================================================

class TestMergeBatch:

    def test_batch_merges_pairs_index_by_index(self, weather_response, aqi_response):
        features = FeatureMerger.merge_batch(
            [weather_response, weather_response], [aqi_response, aqi_response]
        )
        assert len(features) == 2

    def test_mismatched_lengths_raise(self, weather_response, aqi_response):
        with pytest.raises(ValueError):
            FeatureMerger.merge_batch([weather_response], [aqi_response, aqi_response])

    def test_skip_errors_true_continues_batch(self, weather_response, aqi_response, monkeypatch):
        call_count = {"n": 0}
        original_merge = FeatureMerger.merge.__func__

        def flaky_merge(cls, weather, aqi):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("boom")
            return original_merge(cls, weather, aqi)

        monkeypatch.setattr(FeatureMerger, "merge", classmethod(flaky_merge))

        features = FeatureMerger.merge_batch(
            [weather_response, weather_response], [aqi_response, aqi_response], skip_errors=True
        )
        assert len(features) == 1

    def test_skip_errors_false_raises(self, weather_response, aqi_response, monkeypatch):
        def broken_merge(cls, weather, aqi):
            raise RuntimeError("boom")

        monkeypatch.setattr(FeatureMerger, "merge", classmethod(broken_merge))

        with pytest.raises(RuntimeError):
            FeatureMerger.merge_batch(
                [weather_response], [aqi_response], skip_errors=False
            )


# ============================================================
# Serialization: to_dict / from_dict
# ============================================================

class TestSerialization:

    def test_to_dict_isoformats_datetimes(self, merged_feature):
        data = FeatureMerger.to_dict(merged_feature)
        assert isinstance(data["timestamp"], str)
        assert isinstance(data["created_at"], str)

    def test_round_trip_from_dict(self, merged_feature):
        data = FeatureMerger.to_dict(merged_feature)
        rebuilt = FeatureMerger.from_dict(data)
        assert rebuilt.city == merged_feature.city
        assert rebuilt.aqi == merged_feature.aqi

    def test_to_dict_is_json_serializable(self, merged_feature):
        data = FeatureMerger.to_dict(merged_feature)
        json.dumps(data)  # should not raise


# ============================================================
# to_dataframe
# ============================================================

class TestToDataframe:

    def test_returns_dataframe_with_expected_columns(self, merged_feature):
        pd = pytest.importorskip("pandas")
        df = FeatureMerger.to_dataframe([merged_feature])
        assert isinstance(df, pd.DataFrame)
        assert "aqi" in df.columns
        assert len(df) == 1


# ============================================================
# handle_missing_values
# ============================================================

class TestHandleMissingValues:

    def _feature_with_missing_pm25(self, merged_feature):
        return merged_feature.model_copy(update={"pm25": None})

    def test_mean_strategy_fills_with_mean(self, merged_feature):
        complete = merged_feature
        missing = self._feature_with_missing_pm25(merged_feature)
        filled = FeatureMerger.handle_missing_values([complete, missing], strategy="mean", fields=("pm25",))
        assert filled[1].pm25 == complete.pm25  # only observed value -> mean equals it itself

    def test_zero_strategy_fills_with_zero(self, merged_feature):
        missing = self._feature_with_missing_pm25(merged_feature)
        filled = FeatureMerger.handle_missing_values([missing], strategy="zero", fields=("pm25",))
        assert filled[0].pm25 == 0.0

    def test_drop_strategy_removes_incomplete(self, merged_feature):
        missing = self._feature_with_missing_pm25(merged_feature)
        filled = FeatureMerger.handle_missing_values([merged_feature, missing], strategy="drop", fields=("pm25",))
        assert len(filled) == 1
        assert filled[0].pm25 is not None

    def test_unknown_strategy_raises(self, merged_feature):
        with pytest.raises(ValueError):
            FeatureMerger.handle_missing_values([merged_feature], strategy="bogus")

    def test_no_missing_values_returns_unchanged(self, merged_feature):
        filled = FeatureMerger.handle_missing_values([merged_feature], strategy="mean", fields=("pm25",))
        assert filled[0].pm25 == merged_feature.pm25


# ============================================================
# compute_statistics
# ============================================================

class TestComputeStatistics:

    def test_stats_for_single_feature(self, merged_feature):
        stats = FeatureMerger.compute_statistics([merged_feature], fields=("aqi",))
        assert stats["aqi"]["count"] == 1
        assert stats["aqi"]["mean"] == merged_feature.aqi
        assert stats["aqi"]["stdev"] == 0.0

    def test_field_omitted_when_no_observed_values(self, merged_feature):
        missing = merged_feature.model_copy(update={"pm25": None})
        stats = FeatureMerger.compute_statistics([missing], fields=("pm25",))
        assert "pm25" not in stats


# ============================================================
# Duplicate detection
# ============================================================

class TestDuplicates:

    def test_detect_duplicates_finds_repeat(self, merged_feature):
        dups = FeatureMerger.detect_duplicates([merged_feature, merged_feature])
        assert len(dups) == 1

    def test_detect_duplicates_empty_when_unique(self, merged_feature, weather_response, aqi_response):
        other = merged_feature.model_copy(update={"city": "Lahore"})
        dups = FeatureMerger.detect_duplicates([merged_feature, other])
        assert dups == []

    def test_drop_duplicates_keeps_first(self, merged_feature):
        unique = FeatureMerger.drop_duplicates([merged_feature, merged_feature])
        assert len(unique) == 1


# ============================================================
# Export helpers
# ============================================================

class TestExport:

    def test_export_to_json(self, merged_feature, tmp_path):
        path = FeatureMerger.export_to_json([merged_feature], tmp_path / "out.json")
        assert path.exists()
        rows = json.loads(path.read_text())
        assert len(rows) == 1
        assert rows[0]["city"] == "Karachi"

    def test_export_to_csv(self, merged_feature, tmp_path):
        pytest.importorskip("pandas")
        path = FeatureMerger.export_to_csv([merged_feature], tmp_path / "out.csv")
        assert path.exists()
        content = path.read_text()
        assert "city" in content.splitlines()[0]

    def test_export_to_csv_empty_list_writes_empty_file(self, tmp_path):
        path = FeatureMerger.export_to_csv([], tmp_path / "empty.csv")
        assert path.exists()
        assert path.read_text() == ""

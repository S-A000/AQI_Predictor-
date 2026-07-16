"""
test_storage.py
================
Tests for src/ingestion/storage.py

All tests redirect ``settings.processed_data_directory`` to a pytest
``tmp_path`` so nothing touches the real filesystem location.
"""
from __future__ import annotations

import json

import pytest

from src.ingestion.storage import StorageManager


@pytest.fixture
def storage_manager(tmp_path, monkeypatch):
    from src.configs.settings import settings
    # `processed_data_directory` is a read-only property derived from
    # `paths.processed_data` — patch the underlying nested field instead.
    monkeypatch.setattr(settings.paths, "processed_data", tmp_path / "processed")
    return StorageManager()


# ============================================================
# Versioning
# ============================================================

class TestVersioning:

    def test_no_versions_initially(self, storage_manager):
        assert storage_manager.list_versions() == []
        assert storage_manager.latest_version() is None

    def test_create_new_version_starts_at_v1(self, storage_manager):
        assert storage_manager.create_new_version() == "v1"

    def test_create_new_version_increments(self, storage_manager):
        storage_manager.create_new_version()
        assert storage_manager.create_new_version() == "v2"

    def test_list_versions_sorted_numerically(self, storage_manager):
        for _ in range(11):
            storage_manager.create_new_version()
        versions = storage_manager.list_versions()
        assert versions[-1] == "v11"
        assert versions == sorted(versions, key=lambda v: int(v[1:]))


# ============================================================
# save_json / save_csv / save_parquet
# ============================================================

class TestSaveFormats:

    def test_save_json_writes_file(self, storage_manager, merged_feature):
        path = storage_manager.save_json([merged_feature], "features.json")
        assert path.exists()
        rows = json.loads(path.read_text())
        assert len(rows) == 1

    def test_save_json_gzip_compression(self, storage_manager, merged_feature):
        path = storage_manager.save_json([merged_feature], "features.json", compression="gzip")
        assert path.name.endswith(".json.gz")
        assert path.exists()

    def test_save_csv_writes_file(self, storage_manager, merged_feature):
        pytest.importorskip("pandas")
        path = storage_manager.save_csv([merged_feature], "features.csv")
        assert path.exists()

    def test_save_parquet_writes_file(self, storage_manager, merged_feature):
        pytest.importorskip("pyarrow")
        path = storage_manager.save_parquet([merged_feature], "features.parquet")
        assert path.exists()


# ============================================================
# save_batch (versioned, multi-format, metadata)
# ============================================================

class TestSaveBatch:

    def test_save_batch_creates_version_and_metadata(self, storage_manager, merged_feature):
        pytest.importorskip("pyarrow")
        result = storage_manager.save_batch([merged_feature])
        assert result["version"] == "v1"
        assert set(result["paths"].keys()) == {"json", "csv", "parquet"}
        assert result["metadata"]["row_count"] == 1

    def test_save_batch_respects_requested_formats(self, storage_manager, merged_feature):
        result = storage_manager.save_batch([merged_feature], formats=("json",))
        assert set(result["paths"].keys()) == {"json"}

    def test_metadata_contains_checksums(self, storage_manager, merged_feature):
        result = storage_manager.save_batch([merged_feature], formats=("json",))
        json_info = result["metadata"]["files"]["json"]
        assert "sha256" in json_info
        assert json_info["size_bytes"] > 0

    def test_read_metadata_round_trips(self, storage_manager, merged_feature):
        result = storage_manager.save_batch([merged_feature], formats=("json",))
        metadata = storage_manager.read_metadata(result["version"])
        assert metadata["version"] == result["version"]


# ============================================================
# Load
# ============================================================

class TestLoad:

    def test_load_json_round_trips(self, storage_manager, merged_feature):
        storage_manager.save_json([merged_feature], "features.json")
        loaded = storage_manager.load_json("features.json")
        assert len(loaded) == 1
        assert loaded[0].city == merged_feature.city

    def test_load_csv_returns_dataframe(self, storage_manager, merged_feature):
        pytest.importorskip("pandas")
        storage_manager.save_csv([merged_feature], "features.csv")
        df = storage_manager.load_csv("features.csv")
        assert len(df) == 1


# ============================================================
# Checksum / integrity
# ============================================================

class TestChecksum:

    def test_verify_checksum_true_for_matching(self, storage_manager, merged_feature):
        path = storage_manager.save_json([merged_feature], "features.json")
        checksum = storage_manager._compute_checksum(path)
        assert storage_manager.verify_checksum(path, checksum) is True

    def test_verify_checksum_false_for_mismatch(self, storage_manager, merged_feature):
        path = storage_manager.save_json([merged_feature], "features.json")
        assert storage_manager.verify_checksum(path, "deadbeef") is False


# ============================================================
# Health check
# ============================================================

class TestHealthCheck:

    def test_healthy_when_no_versions_and_writable(self, storage_manager):
        report = storage_manager.health_check()
        assert report["healthy"] is True
        assert report["base_directory_writable"] is True

    def test_unhealthy_on_checksum_mismatch(self, storage_manager, merged_feature):
        result = storage_manager.save_batch([merged_feature], formats=("json",))
        version_dir = storage_manager._version_directory(result["version"])
        json_path = version_dir / "features.json"
        json_path.write_text('[{"tampered": true}]', encoding="utf-8")

        report = storage_manager.health_check()
        assert report["healthy"] is False
        assert any("checksum mismatch" in issue for issue in report["issues"])

    def test_unhealthy_when_metadata_missing(self, storage_manager):
        storage_manager._version_directory("v1")  # dir with no metadata.json
        report = storage_manager.health_check()
        assert report["healthy"] is False


# ============================================================
# Cleanup
# ============================================================

class TestCleanup:

    def test_cleanup_keeps_last_n(self, storage_manager):
        for _ in range(5):
            storage_manager.create_new_version()
        removed = storage_manager.cleanup_old_versions(keep_last=2)
        assert removed == ["v1", "v2", "v3"]
        assert storage_manager.list_versions() == ["v4", "v5"]

    def test_cleanup_keep_zero_removes_all(self, storage_manager):
        for _ in range(3):
            storage_manager.create_new_version()
        storage_manager.cleanup_old_versions(keep_last=0)
        assert storage_manager.list_versions() == []


# ============================================================
# Backup / restore
# ============================================================

class TestBackupRestore:

    def test_backup_raises_when_no_versions(self, storage_manager):
        with pytest.raises(FileNotFoundError):
            storage_manager.backup()

    def test_backup_and_restore_round_trip(self, storage_manager, merged_feature):
        storage_manager.save_batch([merged_feature], formats=("json",))
        backup_path = storage_manager.backup()
        assert backup_path.exists()

        restored = storage_manager.restore(backup_path)
        assert restored.exists()
        assert (restored / "metadata.json").exists()

    def test_list_backups(self, storage_manager, merged_feature):
        storage_manager.save_batch([merged_feature], formats=("json",))
        storage_manager.backup()
        assert len(storage_manager.list_backups()) == 1


# ============================================================
# Storage statistics
# ============================================================

class TestStorageStatistics:

    def test_stats_reflect_saved_versions(self, storage_manager, merged_feature):
        storage_manager.save_batch([merged_feature], formats=("json",))
        stats = storage_manager.get_storage_statistics()
        assert stats["version_count"] == 1
        assert stats["total_size_bytes"] > 0


# ============================================================
# Async wrappers
# ============================================================

class TestAsyncWrappers:

    @pytest.mark.asyncio
    async def test_save_json_async(self, storage_manager, merged_feature):
        path = await storage_manager.save_json_async([merged_feature], "features.json")
        assert path.exists()

    @pytest.mark.asyncio
    async def test_save_batch_async(self, storage_manager, merged_feature):
        result = await storage_manager.save_batch_async([merged_feature], formats=("json",))
        assert result["version"] == "v1"

    @pytest.mark.asyncio
    async def test_load_json_async_round_trips(self, storage_manager, merged_feature):
        await storage_manager.save_json_async([merged_feature], "features.json")
        loaded = await storage_manager.load_json_async("features.json")
        assert len(loaded) == 1


# ============================================================
# Retry decorator behaviour
# ============================================================

class TestRetryDecorator:

    def test_save_json_retries_then_succeeds(self, storage_manager, merged_feature, monkeypatch):
        original_open = storage_manager.base_directory.__class__.open
        attempts = {"n": 0}

        def flaky_open(self, *args, **kwargs):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise OSError("transient failure")
            return original_open(self, *args, **kwargs)

        monkeypatch.setattr("pathlib.Path.open", flaky_open)
        monkeypatch.setattr("time.sleep", lambda *_: None)  # skip real backoff delay

        path = storage_manager.save_json([merged_feature], "features.json")
        assert path.exists()
        assert attempts["n"] >= 2

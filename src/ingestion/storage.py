"""
storage.py
==========

Enterprise Storage Manager

Author:
    Syed Abdullah

Description
-----------
Responsible for persisting merged features into multiple storage
formats for downstream ML pipelines and Feature Store ingestion.

Supported Formats
-----------------
- JSON
- CSV
- Parquet

Additional Capabilities
-----------------------
- Dataset versioning (v1, v2, v3, ...)
- Batch save across multiple formats
- Metadata files per version
- Feature Store (Hopsworks) upload
- Compression options
- Backup & restore
- Storage statistics
- Cleanup of old versions
- Storage health check
- Retry logic
- Async save/load
- File integrity via SHA256 checksums
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd

from src.configs.settings import settings
from src.ingestion.merger import MergedFeature
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ==========================================================
# Retry Decorator
# ==========================================================

def _retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable:
    """
    Retry a storage operation with exponential backoff.

    Parameters
    ----------
    max_attempts:
        Total number of attempts before giving up.
    delay:
        Initial delay (seconds) between attempts.
    backoff:
        Multiplier applied to the delay after each failure.
    exceptions:
        Exception types that should trigger a retry.
    """

    def decorator(func: Callable) -> Callable:

        @functools.wraps(func)
        def wrapper(*args, **kwargs):

            current_delay = delay

            for attempt in range(1, max_attempts + 1):

                try:
                    return func(*args, **kwargs)

                except exceptions as exc:  # noqa: BLE001

                    if attempt == max_attempts:

                        logger.error(
                            "%s failed after %d attempt(s): %s",
                            func.__name__,
                            attempt,
                            exc,
                        )

                        raise

                    logger.warning(
                        "%s failed on attempt %d/%d (%s); "
                        "retrying in %.1fs",
                        func.__name__,
                        attempt,
                        max_attempts,
                        exc,
                        current_delay,
                    )

                    time.sleep(current_delay)

                    current_delay *= backoff

        return wrapper

    return decorator


class StorageManager:
    """
    Enterprise Storage Manager.

    Responsibilities
    ----------------
    - Persist merged features
    - Read merged features
    - Dataset versioning
    - Feature Store upload
    - Backup / restore / health checks
    """

    VERSION_PATTERN = re.compile(r"^v(\d+)$")

    def __init__(self):

        self.base_directory = settings.processed_data_directory

        self.base_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.backup_directory = self.base_directory / "_backups"

        self.backup_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    # =====================================================
    # Helpers
    # =====================================================

    @staticmethod
    def _to_dict(feature: MergedFeature) -> dict[str, Any]:

        return feature.model_dump(mode="json")

    @classmethod
    def _to_dataframe(
        cls,
        features: Iterable[MergedFeature],
    ) -> pd.DataFrame:

        rows = [

            cls._to_dict(feature)

            for feature in features

        ]

        return pd.DataFrame(rows)

    def _resolve_path(
        self,
        filename: str,
    ) -> Path:

        return self.base_directory / filename

    # =====================================================
    # Dataset Versioning
    # =====================================================

    def _version_directory(self, version: str) -> Path:

        path = self.base_directory / version

        path.mkdir(parents=True, exist_ok=True)

        return path

    def list_versions(self) -> list[str]:
        """
        Return all existing version labels (e.g. ["v1", "v2"]),
        sorted numerically.
        """

        versions = [

            entry.name

            for entry in self.base_directory.iterdir()

            if entry.is_dir() and self.VERSION_PATTERN.match(entry.name)

        ]

        versions.sort(
            key=lambda name: int(self.VERSION_PATTERN.match(name).group(1))
        )

        return versions

    def latest_version(self) -> str | None:
        """
        Return the most recent version label, or None if no
        versions exist yet.
        """

        versions = self.list_versions()

        return versions[-1] if versions else None

    def create_new_version(self) -> str:
        """
        Allocate and create the next version directory
        (e.g. "v1" -> "v2" -> "v3").
        """

        latest = self.latest_version()

        next_number = (
            int(self.VERSION_PATTERN.match(latest).group(1)) + 1
            if latest
            else 1
        )

        version = f"v{next_number}"

        self._version_directory(version)

        logger.info("Created new dataset version: %s", version)

        return version

    # =====================================================
    # JSON
    # =====================================================

    @_retry()
    def save_json(
        self,
        features: Iterable[MergedFeature],
        filename: str = "features.json",
        *,
        compression: str | None = None,
    ) -> Path:

        path = self._resolve_path(filename)

        rows = [

            self._to_dict(feature)

            for feature in features

        ]

        if compression == "gzip":

            import gzip

            with gzip.open(path.with_suffix(path.suffix + ".gz"), "wt", encoding="utf-8") as file:

                json.dump(rows, file, indent=4, ensure_ascii=False)

            path = path.with_suffix(path.suffix + ".gz")

        else:

            with path.open(
                "w",
                encoding="utf-8",
            ) as file:

                json.dump(

                    rows,

                    file,

                    indent=4,

                    ensure_ascii=False,

                )

        logger.info(

            "Saved %d feature(s) to %s",

            len(rows),

            path,

        )

        return path

    # =====================================================
    # CSV
    # =====================================================

    @_retry()
    def save_csv(
        self,
        features: Iterable[MergedFeature],
        filename: str = "features.csv",
        *,
        compression: str | None = None,
    ) -> Path:

        path = self._resolve_path(filename)

        dataframe = self._to_dataframe(features)

        dataframe.to_csv(

            path,

            index=False,

            compression=compression,

        )

        logger.info(

            "Saved %d feature(s) to %s",

            len(dataframe),

            path,

        )

        return path

    # =====================================================
    # Parquet
    # =====================================================

    @_retry()
    def save_parquet(
        self,
        features: Iterable[MergedFeature],
        filename: str = "features.parquet",
        compression: str = "snappy",
    ) -> Path:

        path = self._resolve_path(filename)

        dataframe = self._to_dataframe(features)

        dataframe.to_parquet(

            path,

            index=False,

            compression=compression,

        )

        logger.info(

            "Saved %d feature(s) to %s",

            len(dataframe),

            path,

        )

        return path

    # =====================================================
    # Batch Save (all formats at once, versioned)
    # =====================================================

    def save_batch(
        self,
        features: list[MergedFeature],
        *,
        formats: tuple[str, ...] = ("json", "csv", "parquet"),
        filename_stem: str = "features",
        version: str | None = None,
        compression: str | None = None,
    ) -> dict[str, Any]:
        """
        Save a batch of features to one or more formats inside
        a single versioned directory, and write a metadata file
        alongside the data.

        Returns a dict with keys "version", "paths", "metadata".
        """

        version = version or self.create_new_version()

        version_dir = self._version_directory(version)

        paths: dict[str, Path] = {}

        if "json" in formats:

            paths["json"] = self._save_to_dir(
                features, version_dir / f"{filename_stem}.json", "json",
                compression=compression,
            )

        if "csv" in formats:

            paths["csv"] = self._save_to_dir(
                features, version_dir / f"{filename_stem}.csv", "csv",
                compression=compression,
            )

        if "parquet" in formats:

            paths["parquet"] = self._save_to_dir(
                features, version_dir / f"{filename_stem}.parquet", "parquet",
                compression=compression or "snappy",
            )

        metadata = self._write_metadata(version_dir, features, paths)

        logger.info(
            "Batch save complete for %s: %d feature(s), formats=%s",
            version,
            len(features),
            formats,
        )

        return {
            "version": version,
            "paths": paths,
            "metadata": metadata,
        }

    def _save_to_dir(
        self,
        features: Iterable[MergedFeature],
        path: Path,
        fmt: str,
        *,
        compression: str | None,
    ) -> Path:
        """
        Internal helper: write features to an arbitrary path
        (used by save_batch to write into versioned folders,
        bypassing base_directory-relative save_* methods).
        """

        dataframe = self._to_dataframe(features)

        if fmt == "json":

            rows = [self._to_dict(feature) for feature in features]

            path.write_text(
                json.dumps(rows, indent=4, ensure_ascii=False),
                encoding="utf-8",
            )

        elif fmt == "csv":

            dataframe.to_csv(path, index=False, compression=compression)

        elif fmt == "parquet":

            dataframe.to_parquet(
                path, index=False, compression=compression or "snappy"
            )

        else:

            raise ValueError(f"Unsupported format: {fmt}")

        return path

    # =====================================================
    # Metadata Files
    # =====================================================

    def _write_metadata(
        self,
        version_dir: Path,
        features: list[MergedFeature],
        paths: dict[str, Path],
    ) -> dict[str, Any]:
        """
        Write a metadata.json alongside the versioned data,
        capturing row count, columns, checksums, and timestamps.
        """

        dataframe = self._to_dataframe(features)

        metadata = {
            "version": version_dir.name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "row_count": len(dataframe),
            "columns": list(dataframe.columns),
            "files": {
                fmt: {
                    "filename": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": self._compute_checksum(path),
                }
                for fmt, path in paths.items()
            },
        }

        metadata_path = version_dir / "metadata.json"

        metadata_path.write_text(
            json.dumps(metadata, indent=4),
            encoding="utf-8",
        )

        return metadata

    def read_metadata(self, version: str) -> dict[str, Any]:
        """
        Read the metadata.json for a given version.
        """

        metadata_path = self._version_directory(version) / "metadata.json"

        return json.loads(metadata_path.read_text(encoding="utf-8"))

    # =====================================================
    # Load JSON
    # =====================================================

    @_retry()
    def load_json(
        self,
        filename: str = "features.json",
    ) -> list[MergedFeature]:

        path = self._resolve_path(filename)

        with path.open(

            "r",

            encoding="utf-8",

        ) as file:

            rows = json.load(file)

        logger.info(

            "Loaded %d feature(s) from %s",

            len(rows),

            path,

        )

        return [

            MergedFeature.model_validate(row)

            for row in rows

        ]

    # =====================================================
    # Load CSV
    # =====================================================

    @_retry()
    def load_csv(
        self,
        filename: str = "features.csv",
    ) -> pd.DataFrame:

        path = self._resolve_path(filename)

        dataframe = pd.read_csv(path)

        logger.info(

            "Loaded CSV from %s",

            path,

        )

        return dataframe

    # =====================================================
    # Load Parquet
    # =====================================================

    @_retry()
    def load_parquet(
        self,
        filename: str = "features.parquet",
    ) -> pd.DataFrame:

        path = self._resolve_path(filename)

        dataframe = pd.read_parquet(path)

        logger.info(

            "Loaded Parquet from %s",

            path,

        )

        return dataframe

    # =====================================================
    # Feature Store (Hopsworks) Upload
    # =====================================================

    def upload_to_feature_store(
        self,
        features: list[MergedFeature],
        feature_group_name: str,
        *,
        version: int = 1,
        description: str = "",
        primary_key: tuple[str, ...] = ("city", "timestamp"),
        online_enabled: bool = False,
    ) -> None:
        """
        Upload a batch of features to Hopsworks Feature Store.

        Requires the `hopsworks` package and the following
        settings to be configured:
            - settings.hopsworks_api_key
            - settings.hopsworks_project

        Raises
        ------
        ImportError
            If the `hopsworks` package is not installed.
        """

        try:
            import hopsworks
        except ImportError as exc:

            raise ImportError(
                "The 'hopsworks' package is required for "
                "upload_to_feature_store(); install it with "
                "`pip install hopsworks`."
            ) from exc

        logger.info(
            "Connecting to Hopsworks project '%s'...",
            settings.hopsworks_project,
        )

        project = hopsworks.login(
            api_key_value=settings.hopsworks_api_key,
            project=settings.hopsworks_project,
        )

        feature_store = project.get_feature_store()

        dataframe = self._to_dataframe(features)

        feature_group = feature_store.get_or_create_feature_group(
            name=feature_group_name,
            version=version,
            description=description,
            primary_key=list(primary_key),
            online_enabled=online_enabled,
        )

        feature_group.insert(dataframe)

        logger.info(
            "Uploaded %d feature(s) to Hopsworks feature group '%s' (v%d)",
            len(dataframe),
            feature_group_name,
            version,
        )

    # =====================================================
    # Backup & Restore
    # =====================================================

    def backup(self, version: str | None = None) -> Path:
        """
        Back up a versioned directory (defaults to the latest
        version) into the backup directory, timestamped.
        """

        version = version or self.latest_version()

        if version is None:
            raise FileNotFoundError("No versions available to back up.")

        source_dir = self._version_directory(version)

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

        destination = self.backup_directory / f"{version}_{stamp}"

        shutil.copytree(source_dir, destination)

        logger.info("Backed up %s to %s", version, destination)

        return destination

    def restore(
        self,
        backup_path: str | Path,
        *,
        target_version: str | None = None,
    ) -> Path:
        """
        Restore a backup into the main dataset directory. If
        `target_version` is not given, restores into the version
        implied by the backup's own name (before the timestamp
        suffix).
        """

        backup_path = Path(backup_path)

        if not backup_path.exists():
            raise FileNotFoundError(f"Backup not found: {backup_path}")

        if target_version is None:

            match = re.match(r"^(v\d+)_", backup_path.name)

            if not match:
                raise ValueError(
                    "Could not infer target version from backup name; "
                    "pass target_version explicitly."
                )

            target_version = match.group(1)

        destination = self.base_directory / target_version

        if destination.exists():
            shutil.rmtree(destination)

        shutil.copytree(backup_path, destination)

        logger.info("Restored %s from %s", target_version, backup_path)

        return destination

    def list_backups(self) -> list[Path]:
        """
        List all available backup directories.
        """

        return sorted(
            entry for entry in self.backup_directory.iterdir()
            if entry.is_dir()
        )

    # =====================================================
    # Storage Statistics
    # =====================================================

    def get_storage_statistics(self) -> dict[str, Any]:
        """
        Return aggregate statistics about the storage directory:
        total size, file count, and a per-version breakdown.
        """

        def _dir_size(path: Path) -> int:

            return sum(
                file.stat().st_size
                for file in path.rglob("*")
                if file.is_file()
            )

        per_version = {}

        for version in self.list_versions():

            version_dir = self._version_directory(version)

            files = [f for f in version_dir.iterdir() if f.is_file()]

            per_version[version] = {
                "file_count": len(files),
                "size_bytes": _dir_size(version_dir),
            }

        return {
            "total_size_bytes": _dir_size(self.base_directory),
            "version_count": len(per_version),
            "versions": per_version,
            "backup_count": len(self.list_backups()),
        }

    # =====================================================
    # Cleanup Old Versions
    # =====================================================

    def cleanup_old_versions(self, keep_last: int = 3) -> list[str]:
        """
        Delete all but the most recent `keep_last` versions.

        Returns the list of version labels that were removed.
        """

        versions = self.list_versions()

        to_remove = versions[:-keep_last] if keep_last > 0 else versions

        removed: list[str] = []

        for version in to_remove:

            shutil.rmtree(self._version_directory(version))

            removed.append(version)

        if removed:
            logger.info(
                "Cleaned up %d old version(s): %s",
                len(removed),
                removed,
            )

        return removed

    # =====================================================
    # Storage Health Check
    # =====================================================

    def health_check(self) -> dict[str, Any]:
        """
        Verify that the storage directory is healthy:
        writable, has free disk space, and every version's
        files match their recorded checksums.
        """

        report: dict[str, Any] = {
            "base_directory_exists": self.base_directory.exists(),
            "base_directory_writable": True,
            "disk_free_bytes": shutil.disk_usage(self.base_directory).free,
            "versions_checked": [],
            "issues": [],
        }

        probe = self.base_directory / ".health_check_probe"

        try:
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except OSError as exc:
            report["base_directory_writable"] = False
            report["issues"].append(f"Directory not writable: {exc}")

        for version in self.list_versions():

            version_dir = self._version_directory(version)
            metadata_path = version_dir / "metadata.json"

            entry = {"version": version, "ok": True}

            if not metadata_path.exists():
                entry["ok"] = False
                entry["reason"] = "metadata.json missing"
                report["issues"].append(f"{version}: metadata.json missing")
                report["versions_checked"].append(entry)
                continue

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

            for fmt, info in metadata.get("files", {}).items():

                file_path = version_dir / info["filename"]

                if not file_path.exists():
                    entry["ok"] = False
                    report["issues"].append(
                        f"{version}: missing file {info['filename']}"
                    )
                    continue

                actual_checksum = self._compute_checksum(file_path)

                if actual_checksum != info.get("sha256"):
                    entry["ok"] = False
                    report["issues"].append(
                        f"{version}: checksum mismatch for {info['filename']}"
                    )

            report["versions_checked"].append(entry)

        report["healthy"] = (
            report["base_directory_writable"] and not report["issues"]
        )

        return report

    # =====================================================
    # File Integrity (SHA256 Checksums)
    # =====================================================

    @staticmethod
    def _compute_checksum(path: Path, chunk_size: int = 65536) -> str:
        """
        Compute the SHA256 checksum of a file.
        """

        digest = hashlib.sha256()

        with path.open("rb") as file:

            for chunk in iter(lambda: file.read(chunk_size), b""):
                digest.update(chunk)

        return digest.hexdigest()

    def verify_checksum(self, path: str | Path, expected: str) -> bool:
        """
        Verify a file's SHA256 checksum against an expected value.
        """

        actual = self._compute_checksum(Path(path))

        matches = actual == expected

        if not matches:
            logger.warning(
                "Checksum mismatch for %s (expected %s, got %s)",
                path,
                expected,
                actual,
            )

        return matches

    # =====================================================
    # Async Save / Load
    # =====================================================

    async def save_json_async(
        self,
        features: Iterable[MergedFeature],
        filename: str = "features.json",
        *,
        compression: str | None = None,
    ) -> Path:

        return await asyncio.to_thread(
            self.save_json, features, filename, compression=compression
        )

    async def save_csv_async(
        self,
        features: Iterable[MergedFeature],
        filename: str = "features.csv",
        *,
        compression: str | None = None,
    ) -> Path:

        return await asyncio.to_thread(
            self.save_csv, features, filename, compression=compression
        )

    async def save_parquet_async(
        self,
        features: Iterable[MergedFeature],
        filename: str = "features.parquet",
        compression: str = "snappy",
    ) -> Path:

        return await asyncio.to_thread(
            self.save_parquet, features, filename, compression
        )

    async def save_batch_async(
        self,
        features: list[MergedFeature],
        **kwargs,
    ) -> dict[str, Any]:

        return await asyncio.to_thread(self.save_batch, features, **kwargs)

    async def load_json_async(
        self,
        filename: str = "features.json",
    ) -> list[MergedFeature]:

        return await asyncio.to_thread(self.load_json, filename)

    async def load_csv_async(
        self,
        filename: str = "features.csv",
    ) -> pd.DataFrame:

        return await asyncio.to_thread(self.load_csv, filename)

    async def load_parquet_async(
        self,
        filename: str = "features.parquet",
    ) -> pd.DataFrame:

        return await asyncio.to_thread(self.load_parquet, filename)
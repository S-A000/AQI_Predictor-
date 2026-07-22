"""
Suggested path: src/training/register.py

SINGLE RESPONSIBILITY: Manage model versioning, collect evaluation metadata, 
update the central model registry catalog, and promote validated model artifacts.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from src.training.train import BaselineMetrics
from src.utils.constants import MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Registry Paths
REGISTRY_DIR = MODELS_DIR / "registry"
CATALOG_FILE = REGISTRY_DIR / "model_catalog.json"


@dataclass
class ModelRecord:
    """Dataclass defining structure of a registered model entry."""
    model_name: str
    version: str
    status: str  # e.g., "Staging", "Production", "Archived"
    registered_at: str
    artifact_path: str
    metrics: Dict[str, float]
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    num_features: int = 603


class ModelRegistry:
    """
    Enterprise Model Registry Engine.
    Handles artifact versioning, metadata logging, and model promotion.
    """

    def __init__(self, registry_dir: Path | str = REGISTRY_DIR) -> None:
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.catalog_path = self.registry_dir / "model_catalog.json"
        self._init_catalog()

    def _init_catalog(self) -> None:
        """Ensure catalog JSON file exists."""
        if not self.catalog_path.exists():
            logger.info("Initializing new model registry catalog at %s", self.catalog_path)
            with open(self.catalog_path, "w", encoding="utf-8") as f:
                json.dump({"registered_models": []}, f, indent=4)

    def _get_catalog_data(self) -> Dict[str, Any]:
        """Read existing catalog data."""
        with open(self.catalog_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _generate_next_version(self, model_name: str) -> str:
        """Generate auto-incrementing semantic version string for given model name."""
        catalog = self._get_catalog_data()
        existing_versions = [
            m["version"] for m in catalog["registered_models"] if m["model_name"] == model_name
        ]

        if not existing_versions:
            return "v1.0.0"

        # Increment patch version (e.g., v1.0.0 -> v1.0.1)
        latest_version = sorted(existing_versions)[-1]
        major, minor, patch = map(int, latest_version.lstrip("v").split("."))
        return f"v{major}.{minor}.{patch + 1}"

    def register(
        self,
        model_name: str,
        source_artifact_path: Path | str,
        metrics: BaselineMetrics | Dict[str, float],
        hyperparameters: Optional[Dict[str, Any]] = None,
        status: str = "Staging",
    ) -> ModelRecord:
        """
        Registers a model artifact into the registry catalog and copies artifact to target directory.
        """
        source_path = Path(source_artifact_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Source model artifact not found at: {source_path}")

        version = self._generate_next_version(model_name)
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

        # Destination file path in registry
        dest_filename = f"{model_name.lower()}_{version}.joblib"
        dest_path = self.registry_dir / dest_filename

        # Copy artifact
        shutil.copy2(source_path, dest_path)
        logger.info("Copied model artifact to Registry: %s", dest_path)

        # Format metrics
        formatted_metrics = metrics.to_dict() if isinstance(metrics, BaselineMetrics) else metrics

        record = ModelRecord(
            model_name=model_name,
            version=version,
            status=status,
            registered_at=timestamp,
            artifact_path=str(dest_path),
            metrics=formatted_metrics,
            hyperparameters=hyperparameters or {},
        )

        # Update Catalog JSON
        catalog = self._get_catalog_data()
        catalog["registered_models"].append(asdict(record))

        with open(self.catalog_path, "w", encoding="utf-8") as f:
            json.dump(catalog, f, indent=4)

        logger.info(
            "Successfully registered model '%s' | Version: %s | Status: %s",
            model_name, version, status
        )
        return record

    def promote_to_production(self, model_name: str, version: str) -> None:
        """Promote a specific model version to Production status and archive previous versions."""
        catalog = self._get_catalog_data()
        updated = False

        for entry in catalog["registered_models"]:
            if entry["model_name"] == model_name:
                if entry["version"] == version:
                    entry["status"] = "Production"
                    updated = True
                elif entry["status"] == "Production":
                    entry["status"] = "Archived"

        if not updated:
            raise ValueError(f"Model '{model_name}' with version '{version}' not found in registry.")

        with open(self.catalog_path, "w", encoding="utf-8") as f:
            json.dump(catalog, f, indent=4)

        logger.info("Promoted '%s' version '%s' to Production status.", model_name, version)


def register_ridge_baseline() -> ModelRecord:
    """
    Convenience function to register the Ridge baseline model trained in train.py.
    """
    from src.training.train import run_ridge_baseline

    # Run training to get model & metrics
    trainer, metrics = run_ridge_baseline()

    registry = ModelRegistry()
    record = registry.register(
        model_name="Ridge_Baseline",
        source_artifact_path=MODELS_DIR / "ridge_baseline.joblib",
        metrics=metrics,
        hyperparameters={"alpha": trainer.alpha},
        status="Staging",
    )

    # Automatically promote baseline to Production if it's the first model
    registry.promote_to_production(model_name="Ridge_Baseline", version=record.version)

    return record


if __name__ == "__main__":
    try:
        registered_entry = register_ridge_baseline()
        print("\n=== Model Registration Successful ===")
        print(f"Model Name : {registered_entry.model_name}")
        print(f"Version    : {registered_entry.version}")
        print(f"Status     : {registered_entry.status}")
        print(f"Metrics    : {registered_entry.metrics}")
    except Exception as err:
        logger.exception("Model registration failed: %s", err)
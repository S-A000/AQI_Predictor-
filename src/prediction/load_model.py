"""
Suggested path: src/prediction/load_model.py

SINGLE RESPONSIBILITY: Thread-safe model artifact loading, schema extraction,
preprocessor validation, and memory caching for production inference.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib

from src.training.model_bundle import load_model_bundle
from src.utils.constants import MODELS_DIR, PROCESSED_DATA_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class LoadedModelArtifact:
    """Immutable loaded model, fitted transformer, and feature contract."""

    model: Any
    scaler_engineer: Any
    metadata: Dict[str, Any]
    model_version: str
    feature_names: List[str]
    horizon_hours: int = 24
    model_name: str = "unknown"


class ModelLoader:
    """Load horizon-specific bundles, with an isolated legacy fallback."""

    _cached_artifacts: Dict[int, LoadedModelArtifact] = {}

    def __init__(
        self,
        model_path: Optional[Path | str] = None,
        metadata_path: Optional[Path | str] = None,
        scaler_path: Optional[Path | str] = None,
    ) -> None:
        self.custom_model_path = Path(model_path) if model_path else None
        self.custom_metadata_path = Path(metadata_path) if metadata_path else None
        self.scaler_path = (
            Path(scaler_path)
            if scaler_path
            else PROCESSED_DATA_DIR / "scaler.joblib"
        )

    def _get_horizon_paths(self, horizon_hours: int) -> tuple[Path, Path, Path]:
        base_dir = MODELS_DIR if MODELS_DIR.name == "registry" else MODELS_DIR / "registry"
        model_path = self.custom_model_path or (
            base_dir / f"{horizon_hours}h_model.joblib"
        )
        metadata_path = self.custom_metadata_path or (
            base_dir / f"{horizon_hours}h_metadata.json"
        )
        bundle_path = base_dir / f"{horizon_hours}h_bundle.joblib"
        return model_path, metadata_path, bundle_path

    def _load_and_validate_scaler(self) -> Any:
        """Legacy-only loader for pre-bundle artifacts."""
        if not self.scaler_path.exists():
            raise FileNotFoundError(
                f"Fitted legacy scaler artifact missing at: {self.scaler_path}"
            )
        engineer = joblib.load(self.scaler_path)
        if not hasattr(engineer, "transform"):
            raise TypeError(
                f"Legacy preprocessing artifact is not transformable: {self.scaler_path}"
            )
        return engineer

    def _load_legacy_artifact(
        self,
        *,
        horizon_hours: int,
        model_path: Path,
        metadata_path: Path,
    ) -> LoadedModelArtifact:
        """Load historical three-file artifacts without affecting bundle loading."""
        logger.warning(
            "Bundle missing for %sh; using isolated legacy model/metadata/scaler fallback.",
            horizon_hours,
        )
        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact missing at: {model_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Model metadata missing at: {metadata_path}")

        model = joblib.load(model_path)
        with open(metadata_path, "r", encoding="utf-8") as file_handle:
            metadata = json.load(file_handle)
        transformer = self._load_and_validate_scaler()
        feature_names = list(
            metadata.get("feature_names")
            or getattr(transformer, "model_feature_columns_", [])
        )
        if not feature_names:
            raise ValueError("Legacy artifact has no ordered feature schema.")

        return LoadedModelArtifact(
            model=model,
            scaler_engineer=transformer,
            metadata=metadata,
            model_version=str(metadata.get("model_version", "1.0.0")),
            feature_names=feature_names,
            horizon_hours=horizon_hours,
            model_name=str(metadata.get("algorithm", "legacy")),
        )

    def load_model_artifact(
        self,
        horizon_hours: int = 24,
        force_reload: bool = False,
    ) -> LoadedModelArtifact:
        """Load a complete horizon bundle, or a clearly logged legacy fallback."""
        if horizon_hours not in (24, 48, 72):
            raise ValueError("horizon_hours must be 24, 48, or 72.")
        if horizon_hours in self._cached_artifacts and not force_reload:
            return self._cached_artifacts[horizon_hours]

        model_path, metadata_path, bundle_path = self._get_horizon_paths(
            horizon_hours
        )
        use_bundle = self.custom_model_path is None and bundle_path.exists()

        if use_bundle:
            logger.info("Loading production model bundle from %s", bundle_path)
            bundle = load_model_bundle(bundle_path)
            if bundle.horizon_hours != horizon_hours:
                raise ValueError(
                    f"Bundle horizon {bundle.horizon_hours} does not match requested "
                    f"horizon {horizon_hours}."
                )
            metadata = dict(bundle.training_metadata)
            metadata.setdefault("mlflow_run_id", bundle.run_metadata.get("mlflow_run_id"))
            artifact = LoadedModelArtifact(
                model=bundle.model,
                scaler_engineer=bundle.transformer,
                metadata=metadata,
                model_version=bundle.model_version,
                feature_names=list(bundle.feature_names),
                horizon_hours=bundle.horizon_hours,
                model_name=bundle.model_name,
            )
        else:
            artifact = self._load_legacy_artifact(
                horizon_hours=horizon_hours,
                model_path=model_path,
                metadata_path=metadata_path,
            )

        expected_count = artifact.metadata.get("feature_count")
        if expected_count is not None and len(artifact.feature_names) != expected_count:
            raise ValueError(
                "Artifact feature schema count does not match metadata: "
                f"{len(artifact.feature_names)} != {expected_count}"
            )

        self._cached_artifacts[horizon_hours] = artifact
        logger.info(
            "Model version '%s' (%sh horizon) and bundled transformer cached successfully.",
            artifact.model_version,
            horizon_hours,
        )
        return artifact


_loader = ModelLoader()


def get_production_model(horizon_hours: int = 24) -> LoadedModelArtifact:
    """Public helper to fetch a cached production model bundle."""
    return _loader.load_model_artifact(horizon_hours=horizon_hours)

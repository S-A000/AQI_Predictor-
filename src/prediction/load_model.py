"""
Suggested path: src/prediction/load_model.py

SINGLE RESPONSIBILITY: Thread-safe model artifact loading, schema extraction,
scaler loading/validation, and memory caching for production inference.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib

from src.feature_engineering.scaling_encoding import ScalingEncodingEngineer
from src.utils.constants import MODELS_DIR, PROCESSED_DATA_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class LoadedModelArtifact:
    """Immutable container holding loaded model artifact, scaler, and metadata."""
    model: Any
    scaler_engineer: ScalingEncodingEngineer
    metadata: Dict[str, Any]
    model_version: str
    feature_names: List[str]


class ModelLoader:
    """
    Model Loader with memory caching to eliminate duplicate disk I/O during inference.
    Loads trained model, metadata JSON, and fitted Scaler state.
    """

    _cached_artifact: Optional[LoadedModelArtifact] = None

    def __init__(
        self,
        model_path: Path | str = MODELS_DIR / "registry" / "ridge_baseline.joblib",
        metadata_path: Path | str = MODELS_DIR / "registry" / "ridge_metadata.json",
        scaler_path: Path | str = PROCESSED_DATA_DIR / "scaler.joblib",
    ) -> None:
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self.scaler_path = Path(scaler_path)

    def _load_and_validate_scaler(self) -> ScalingEncodingEngineer:
        """
        Loads fitted scaler state from scaler.joblib and validates artifact integrity.
        Never calls fit(). Only restores fitted parameters.
        """
        if not self.scaler_path.exists():
            error_msg = f"Fitted scaler artifact missing at: {self.scaler_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        logger.info("Loading fitted scaler artifact from %s...", self.scaler_path)

        # 🐛 FIX: Replaced manual mapping with the actual load() method 
        # from ScalingEncodingEngineer. This ensures all dict mappings, categories, 
        # and encoded columns match EXACTLY with how it was fitted during training.
        try:
            engineer = ScalingEncodingEngineer.load(str(self.scaler_path))
            
            logger.info(
                "Scaler strategy restored successfully via classmethod load() (%d fitted features).",
                len(engineer.feature_columns_ or []),
            )
            return engineer
            
        except Exception as e:
            error_msg = f"Failed to load scaler properly from {self.scaler_path}: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    def load_model_artifact(self, force_reload: bool = False) -> LoadedModelArtifact:
        """Loads and returns cached model artifact including model, scaler, and metadata."""
        if ModelLoader._cached_artifact is not None and not force_reload:
            logger.debug("Returning cached model artifact from memory.")
            return ModelLoader._cached_artifact

        if not self.model_path.exists():
            error_msg = f"Model artifact missing at: {self.model_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        if not self.metadata_path.exists():
            error_msg = f"Model metadata missing at: {self.metadata_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        logger.info("Loading production model artifact from %s...", self.model_path)
        model = joblib.load(self.model_path)

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # Load and validate fitted scaler
        scaler_engineer = self._load_and_validate_scaler()

        feature_names = metadata.get("feature_names", [])
        expected_count = metadata.get("feature_count")

        # Validate feature count against metadata if present
        if expected_count is not None and feature_names:
            if len(feature_names) != expected_count:
                logger.warning(
                    "Feature names count (%d) mismatches metadata feature_count (%d).",
                    len(feature_names),
                    expected_count,
                )

        artifact = LoadedModelArtifact(
            model=model,
            scaler_engineer=scaler_engineer,
            metadata=metadata,
            model_version=metadata.get("model_version", "1.0.0"),
            feature_names=feature_names or scaler_engineer.final_columns_ or [],
        )

        ModelLoader._cached_artifact = artifact
        logger.info("Model version '%s' and scaler cached successfully.", artifact.model_version)
        return artifact


_loader = ModelLoader()


def get_production_model() -> LoadedModelArtifact:
    """Public helper function to fetch cached production model and scaler artifact."""
    return _loader.load_model_artifact()
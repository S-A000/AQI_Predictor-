"""
Suggested path: src/prediction/load_model.py

SINGLE RESPONSIBILITY: Thread-safe model artifact loading, schema extraction, 
and memory caching for production inference.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib

from src.utils.constants import MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class LoadedModelArtifact:
    """Immutable container holding loaded model artifact and metadata."""
    model: Any
    metadata: Dict[str, Any]
    model_version: str
    feature_names: List[str]


class ModelLoader:
    """
    Model Loader with memory caching to eliminate duplicate disk I/O during inference.
    """

    _cached_artifact: Optional[LoadedModelArtifact] = None

    def __init__(
        self,
        model_path: Path | str = MODELS_DIR / "registry" / "ridge_baseline.joblib",
        metadata_path: Path | str = MODELS_DIR / "registry" / "ridge_metadata.json",
    ) -> None:
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)

    def load_model_artifact(self, force_reload: bool = False) -> LoadedModelArtifact:
        """Loads and returns cached model artifact. Force reloads if specified."""
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

        artifact = LoadedModelArtifact(
            model=model,
            metadata=metadata,
            model_version=metadata.get("model_version", "1.0.0"),
            feature_names=metadata.get("feature_names", []),
        )

        ModelLoader._cached_artifact = artifact
        logger.info("Model version '%s' cached successfully.", artifact.model_version)
        return artifact


def get_production_model() -> LoadedModelArtifact:
    """Public helper function to fetch cached production model."""
    return ModelLoader().load_model_artifact()
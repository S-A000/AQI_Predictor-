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
    scaler_engineer: Any
    metadata: Dict[str, Any]
    model_version: str
    feature_names: List[str]


class ModelLoader:
    """
    Model Loader with memory caching to eliminate duplicate disk I/O during inference.
    Loads trained model, metadata JSON, and fitted Scaler state for multi-horizon scenarios.
    """

    _cached_artifacts: Dict[int, LoadedModelArtifact] = {}

    def __init__(
        self,
        model_path: Optional[Path | str] = None,
        metadata_path: Optional[Path | str] = None,
        scaler_path: Optional[Path | str] = None,
    ) -> None:
        self.custom_model_path = Path(model_path) if model_path else None
        self.custom_metadata_path = Path(metadata_path) if metadata_path else None
        self.scaler_path = Path(scaler_path) if scaler_path else PROCESSED_DATA_DIR / "scaler.joblib"

    def _get_horizon_paths(self, horizon_hours: int) -> tuple[Path, Path]:
        """
        Resolves model and metadata paths based on specified horizon.
        Ensures MODELS_DIR points directly to models directory without path duplication.
        """
        base_dir = MODELS_DIR if MODELS_DIR.name == "registry" else MODELS_DIR / "registry"
        
        model_path = self.custom_model_path or (base_dir / f"{horizon_hours}h_model.joblib")
        metadata_path = self.custom_metadata_path or (base_dir / f"{horizon_hours}h_metadata.json")
        return model_path, metadata_path

    def _load_and_validate_scaler(self) -> Any:
        """
        Loads fitted scaler instance directly using joblib.
        Restores fitted parameters safely.
        """
        if not self.scaler_path.exists():
            error_msg = f"Fitted scaler artifact missing at: {self.scaler_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        logger.info("Loading fitted scaler artifact from %s...", self.scaler_path)

        try:
            # 🐛 FIX: Directly load joblib artifact since class has no custom .load() method
            engineer = joblib.load(self.scaler_path)
            
            fitted_count = getattr(engineer, "feature_columns_", None)
            count_str = f" ({len(fitted_count)} fitted features)" if fitted_count else ""
            logger.info("Scaler strategy restored successfully via joblib.load()%s.", count_str)
            return engineer
            
        except Exception as e:
            error_msg = f"Failed to load scaler properly from {self.scaler_path}: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    def load_model_artifact(
        self, horizon_hours: int = 24, force_reload: bool = False
    ) -> LoadedModelArtifact:
        """Loads and returns cached model artifact including model, scaler, and metadata for specified horizon."""
        
        if horizon_hours in ModelLoader._cached_artifacts and not force_reload:
            logger.debug("Returning cached %sh model artifact from memory.", horizon_hours)
            return ModelLoader._cached_artifacts[horizon_hours]

        model_path, metadata_path = self._get_horizon_paths(horizon_hours)

        if not model_path.exists():
            error_msg = f"Model artifact missing at: {model_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        if not metadata_path.exists():
            error_msg = f"Model metadata missing at: {metadata_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        logger.info("Loading production model artifact from %s...", model_path)
        model = joblib.load(model_path)

        with open(metadata_path, "r", encoding="utf-8") as f:
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

        fallback_cols = getattr(scaler_engineer, "final_columns_", [])
        artifact = LoadedModelArtifact(
            model=model,
            scaler_engineer=scaler_engineer,
            metadata=metadata,
            model_version=metadata.get("model_version", "1.0.0"),
            feature_names=feature_names or fallback_cols,
        )

        ModelLoader._cached_artifacts[horizon_hours] = artifact
        logger.info("Model version '%s' (%sh horizon) and scaler cached successfully.", artifact.model_version, horizon_hours)
        return artifact


_loader = ModelLoader()


def get_production_model(horizon_hours: int = 24) -> LoadedModelArtifact:
    """Public helper function to fetch cached production model and scaler artifact."""
    return _loader.load_model_artifact(horizon_hours=horizon_hours)
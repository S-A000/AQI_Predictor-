"""Consistent, horizon-specific model artifacts for production inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Sequence

import joblib


@dataclass(frozen=True)
class ModelBundle:
    """A trained model and the exact preprocessing contract it requires."""

    model: Any
    transformer: Any
    feature_names: tuple[str, ...]
    horizon_hours: int
    model_name: str
    model_version: str = "1.0.0"
    run_metadata: Dict[str, Any] = field(default_factory=dict)
    training_metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        model: Any,
        transformer: Any,
        feature_names: Sequence[str],
        horizon_hours: int,
        model_name: str,
        model_version: str = "1.0.0",
        run_metadata: Dict[str, Any] | None = None,
        training_metadata: Dict[str, Any] | None = None,
    ) -> "ModelBundle":
        bundle = cls(
            model=model,
            transformer=transformer,
            feature_names=tuple(feature_names),
            horizon_hours=horizon_hours,
            model_name=model_name,
            model_version=model_version,
            run_metadata=dict(run_metadata or {}),
            training_metadata=dict(training_metadata or {}),
        )
        bundle.validate()
        return bundle

    def validate(self) -> None:
        if self.model is None or not hasattr(self.model, "predict"):
            raise ValueError("Model bundle does not contain a prediction model.")
        if self.transformer is None or not hasattr(self.transformer, "transform"):
            raise ValueError("Model bundle does not contain a fitted transformer.")
        if not self.feature_names:
            raise ValueError("Model bundle feature schema is empty.")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("Model bundle feature schema contains duplicates.")
        if self.horizon_hours not in (24, 48, 72):
            raise ValueError(
                f"Unsupported model bundle horizon: {self.horizon_hours}."
            )
        fitted_columns = getattr(self.transformer, "model_feature_columns_", None)
        if fitted_columns and tuple(fitted_columns) != self.feature_names:
            raise ValueError(
                "Model bundle feature schema does not match the fitted transformer."
            )


def save_model_bundle(bundle: ModelBundle, path: Path | str) -> Path:
    """Validate and persist a bundle with the project's existing Joblib stack."""

    bundle.validate()
    bundle_path = Path(path)
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, bundle_path)
    return bundle_path


def load_model_bundle(path: Path | str) -> ModelBundle:
    """Load a bundle and reject incomplete or incompatible artifacts."""

    bundle_path = Path(path)
    bundle = joblib.load(bundle_path)
    if not isinstance(bundle, ModelBundle):
        raise TypeError(f"Artifact at {bundle_path} is not a ModelBundle.")
    bundle.validate()
    return bundle

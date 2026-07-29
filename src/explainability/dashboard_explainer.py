from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.prediction.predictor import AQIPredictor
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ExplanationResult:
    method: str
    note: str
    top_factors: List[Dict[str, Any]]


class DashboardExplainer:
    """
    Production-safe SHAP explainer for dashboard predictions.

    This class:
    - Reuses AQIPredictor model resources.
    - Rebuilds the exact model-ready feature row.
    - Aligns features with model metadata.
    - Computes SHAP values where possible.
    - Falls back to model feature importance if SHAP fails.
    """

    def __init__(self, predictor: Optional[AQIPredictor] = None) -> None:
        self.predictor = predictor or AQIPredictor()
        self._explainer_cache: Dict[int, Any] = {}

    @staticmethod
    def _clean_feature_name(name: str) -> str:
        replacements = {
            "_": " ",
            "pm25": "PM2.5",
            "pm10": "PM10",
            "no2": "NO₂",
            "so2": "SO₂",
            "co": "CO",
            "o3": "O₃",
            "aqi": "AQI",
            "humidity": "Humidity",
            "temperature": "Temperature",
            "pressure": "Pressure",
            "wind speed": "Wind Speed",
            "wind degree": "Wind Direction",
        }

        cleaned = name

        for old, new in replacements.items():
            cleaned = cleaned.replace(old, new)

        return cleaned.title()

    @staticmethod
    def _impact_label(value: float) -> str:
        abs_value = abs(value)

        if abs_value >= 10:
            return "High"
        if abs_value >= 3:
            return "Medium"
        return "Low"

    @staticmethod
    def _direction(value: float) -> str:
        if value > 0:
            return "increased prediction"
        if value < 0:
            return "reduced prediction"
        return "neutral effect"

    def _get_model_ready_features(
        self,
        payload: Dict[str, Any],
        context_df: pd.DataFrame,
        horizon_hours: int,
    ) -> tuple[pd.DataFrame, Dict[str, Any]]:
        resources = self.predictor._get_horizon_resources(horizon_hours)

        features_df = resources["pipeline"].build_features(
            payload,
            context_df=context_df,
        )

        aligned_df = self.predictor._align_and_validate(
            features_df,
            resources["expected_features"],
            strict=False,
            min_completeness=0.90,
        )

        return aligned_df, resources

    def _get_shap_explainer(self, horizon_hours: int, model: Any) -> Any:
        if horizon_hours in self._explainer_cache:
            return self._explainer_cache[horizon_hours]

        import shap

        explainer = shap.Explainer(model)
        self._explainer_cache[horizon_hours] = explainer
        return explainer

    def _explain_with_shap(
        self,
        aligned_df: pd.DataFrame,
        resources: Dict[str, Any],
        horizon_hours: int,
        top_k: int,
    ) -> ExplanationResult:
        model = resources["model"]
        explainer = self._get_shap_explainer(horizon_hours, model)

        shap_values = explainer(aligned_df)

        values = shap_values.values

        if isinstance(values, list):
            values = values[0]

        values = np.asarray(values)

        if values.ndim == 2:
            row_values = values[0]
        else:
            row_values = values

        feature_names = list(aligned_df.columns)

        ranking = sorted(
            zip(feature_names, row_values),
            key=lambda item: abs(float(item[1])),
            reverse=True,
        )[:top_k]

        top_factors = []

        for feature, contribution in ranking:
            contribution_float = round(float(contribution), 4)

            top_factors.append(
                {
                    "feature": self._clean_feature_name(feature),
                    "raw_feature": feature,
                    "impact": self._impact_label(contribution_float),
                    "contribution": contribution_float,
                    "direction": self._direction(contribution_float),
                    "reason": (
                        f"{self._clean_feature_name(feature)} {self._direction(contribution_float)} "
                        f"for the {horizon_hours}h AQI forecast."
                    ),
                }
            )

        return ExplanationResult(
            method="shap",
            note="SHAP explanation based on the active trained model and aligned feature vector.",
            top_factors=top_factors,
        )

    def _fallback_feature_importance(
        self,
        aligned_df: pd.DataFrame,
        resources: Dict[str, Any],
        horizon_hours: int,
        top_k: int,
    ) -> ExplanationResult:
        model = resources["model"]

        importances = getattr(model, "feature_importances_", None)

        if importances is None:
            coefficients = getattr(model, "coef_", None)

            if coefficients is None:
                return ExplanationResult(
                    method="fallback_unavailable",
                    note="Model does not expose SHAP-compatible output, feature_importances_, or coef_.",
                    top_factors=[],
                )

            importances = np.abs(np.asarray(coefficients)).ravel()

        importances = np.asarray(importances).ravel()
        feature_names = list(aligned_df.columns)

        if len(importances) != len(feature_names):
            return ExplanationResult(
                method="fallback_unavailable",
                note="Feature importance length does not match model feature count.",
                top_factors=[],
            )

        ranking = sorted(
            zip(feature_names, importances),
            key=lambda item: abs(float(item[1])),
            reverse=True,
        )[:top_k]

        top_factors = []

        for feature, importance in ranking:
            importance_float = round(float(importance), 4)

            top_factors.append(
                {
                    "feature": self._clean_feature_name(feature),
                    "raw_feature": feature,
                    "impact": self._impact_label(importance_float),
                    "contribution": importance_float,
                    "direction": "important feature",
                    "reason": (
                        f"{self._clean_feature_name(feature)} is one of the most important "
                        f"features for the {horizon_hours}h model."
                    ),
                }
            )

        return ExplanationResult(
            method="model_feature_importance",
            note="SHAP failed, so explanation uses model feature importance or coefficients.",
            top_factors=top_factors,
        )

    def explain_prediction(
        self,
        payload: Dict[str, Any],
        context_df: pd.DataFrame,
        horizon_hours: int = 24,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Public method used by dashboard service/API.

        Returns:
        {
            "method": "shap",
            "note": "...",
            "top_factors": [...]
        }
        """
        try:
            aligned_df, resources = self._get_model_ready_features(
                payload=payload,
                context_df=context_df,
                horizon_hours=horizon_hours,
            )

            try:
                result = self._explain_with_shap(
                    aligned_df=aligned_df,
                    resources=resources,
                    horizon_hours=horizon_hours,
                    top_k=top_k,
                )
            except Exception as shap_err:
                logger.warning(
                    "SHAP explanation failed for %sh horizon. Falling back. Error: %s",
                    horizon_hours,
                    shap_err,
                )

                result = self._fallback_feature_importance(
                    aligned_df=aligned_df,
                    resources=resources,
                    horizon_hours=horizon_hours,
                    top_k=top_k,
                )

            return {
                "method": result.method,
                "note": result.note,
                "top_factors": result.top_factors,
            }

        except Exception as err:
            logger.exception("Explainability failed for %sh horizon: %s", horizon_hours, err)

            return {
                "method": "error",
                "note": str(err),
                "top_factors": [],
            }
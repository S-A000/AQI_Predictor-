

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.prediction.predictor import AQIPredictor
from src.utils.logger import get_logger


logger = get_logger(__name__)


# ------------------------------------------------------------------
# Result contract
# ------------------------------------------------------------------

@dataclass(frozen=True)
class ExplanationResult:
    method: str
    note: str
    top_factors: List[Dict[str, Any]]


# ------------------------------------------------------------------
# Dashboard explainer
# ------------------------------------------------------------------

class DashboardExplainer:
    """
    Production-safe dashboard explainer.

    Strategy
    --------

    Linear models:
        local contribution = transformed_feature_value * coefficient

    Tree models:
        TreeSHAP when SHAP is available.

    Tree SHAP unavailable:
        feature_importances_ fallback.

    Unsupported model:
        safe "unavailable" explanation.

    SHAP is intentionally imported lazily so an optional dependency /
    binary ABI problem cannot break normal application startup.
    """

    def __init__(
        self,
        predictor: Optional[AQIPredictor] = None,
    ) -> None:

        self.predictor = (
            predictor
            or AQIPredictor()
        )

        self._explainer_cache: Dict[
            int,
            Any,
        ] = {}

    # ==================================================================
    # Display helpers
    # ==================================================================

    @staticmethod
    def _clean_feature_name(
        name: str,
    ) -> str:
        """
        Convert internal feature names into dashboard-friendly labels.
        """

        cleaned = (
            str(name)
            .replace("_", " ")
        )

        replacements = {
            "pm25": "PM2.5",
            "pm10": "PM10",
            "no2": "NO₂",
            "so2": "SO₂",
            "co": "CO",
            "o3": "O₃",
            "aqi": "AQI",
        }

        lowered = cleaned.lower()

        for old, new in replacements.items():
            lowered = (
                lowered
                .replace(
                    old,
                    new,
                )
            )

        # Preserve pollutant abbreviations.
        words = []

        protected = {
            "PM2.5",
            "PM10",
            "NO₂",
            "SO₂",
            "CO",
            "O₃",
            "AQI",
        }

        for word in lowered.split():
            if word in protected:
                words.append(word)
            else:
                words.append(
                    word.title()
                )

        return " ".join(
            words
        )

    @staticmethod
    def _impact_label(
        magnitude: float,
        *,
        max_magnitude: float,
    ) -> str:
        """
        Relative local importance label.

        Avoids hard-coded contribution thresholds because contribution
        scales differ between models.
        """

        magnitude = abs(
            float(magnitude)
        )

        if max_magnitude <= 0:
            return "Low"

        ratio = (
            magnitude
            / max_magnitude
        )

        if ratio >= 0.66:
            return "High"

        if ratio >= 0.33:
            return "Medium"

        return "Low"

    @staticmethod
    def _direction(
        contribution: float,
    ) -> str:

        if contribution > 0:
            return "increased prediction"

        if contribution < 0:
            return "reduced prediction"

        return "neutral effect"

    # ==================================================================
    # Model-ready features
    # ==================================================================

    def _get_model_ready_features(
        self,
        payload: Dict[str, Any],
        context_df: pd.DataFrame,
        horizon_hours: int,
    ) -> tuple[
        pd.DataFrame,
        Dict[str, Any],
    ]:
        """
        Rebuild exactly the same feature row used by production inference.
        """

        resources = (
            self.predictor
            ._get_horizon_resources(
                horizon_hours
            )
        )

        features_df = (
            resources[
                "pipeline"
            ]
            .build_features(
                payload,
                context_df=context_df,
            )
        )

        self.predictor._reject_preprocessing_schema_gap(
            resources[
                "pipeline"
            ]
        )

        aligned_df = (
            self.predictor
            ._align_and_validate(
                features_df,
                resources[
                    "expected_features"
                ],
                strict=True,
                min_completeness=1.0,
            )
        )

        if len(aligned_df) != 1:
            raise ValueError(
                "Dashboard explainability requires exactly one "
                f"model-ready row; received={len(aligned_df)}"
            )

        return (
            aligned_df,
            resources,
        )

    # ==================================================================
    # Linear models
    # ==================================================================

    def _explain_linear_model(
        self,
        aligned_df: pd.DataFrame,
        resources: Dict[str, Any],
        horizon_hours: int,
        top_k: int,
    ) -> ExplanationResult:
        """
        Local explanation for linear models.

        For a transformed model input x and model coefficient beta:

            local contribution_i = x_i * beta_i

        This is substantially more meaningful for an individual
        prediction than ranking absolute coefficients alone.
        """

        model = resources[
            "model"
        ]

        coefficients = getattr(
            model,
            "coef_",
            None,
        )

        if coefficients is None:
            raise ValueError(
                "Model does not expose coef_."
            )

        coefficients = (
            np.asarray(
                coefficients,
                dtype=float,
            )
            .reshape(-1)
        )

        feature_values = (
            aligned_df
            .iloc[0]
            .to_numpy(
                dtype=float
            )
            .reshape(-1)
        )

        feature_names = list(
            aligned_df.columns
        )

        if (
            len(coefficients)
            != len(feature_names)
        ):
            raise ValueError(
                "Linear coefficient count does not match "
                "model feature schema."
            )

        contributions = (
            coefficients
            * feature_values
        )

        contributions = np.nan_to_num(
            contributions,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        ranking = sorted(
            zip(
                feature_names,
                contributions,
                feature_values,
                coefficients,
            ),
            key=lambda item: abs(
                float(
                    item[1]
                )
            ),
            reverse=True,
        )[
            :top_k
        ]

        max_magnitude = max(
            (
                abs(float(item[1]))
                for item in ranking
            ),
            default=0.0,
        )

        top_factors: List[
            Dict[str, Any]
        ] = []

        for (
            feature,
            contribution,
            feature_value,
            coefficient,
        ) in ranking:

            contribution_float = round(
                float(contribution),
                4,
            )

            display_name = (
                self._clean_feature_name(
                    feature
                )
            )

            top_factors.append(
                {
                    "feature": display_name,
                    "raw_feature": feature,

                    "impact": (
                        self._impact_label(
                            contribution_float,
                            max_magnitude=max_magnitude,
                        )
                    ),

                    "contribution": (
                        contribution_float
                    ),

                    "direction": (
                        self._direction(
                            contribution_float
                        )
                    ),

                    "reason": (
                        f"{display_name} "
                        f"{self._direction(contribution_float)} "
                        f"for the {horizon_hours}h AQI forecast."
                    ),

                    # Useful internally if API schema later exposes them.
                    "feature_value": round(
                        float(feature_value),
                        4,
                    ),

                    "coefficient": round(
                        float(coefficient),
                        6,
                    ),
                }
            )

        return ExplanationResult(
            method="linear_local_contribution",

            note=(
                "Local explanation calculated directly from the "
                "active linear model using transformed feature values "
                "multiplied by fitted coefficients."
            ),

            top_factors=top_factors,
        )

    # ==================================================================
    # Tree SHAP
    # ==================================================================

    def _get_tree_shap_explainer(
        self,
        horizon_hours: int,
        model: Any,
    ) -> Any:
        """
        Lazily import SHAP and cache TreeExplainer.

        Importing SHAP is intentionally delayed because SHAP may pull in
        optional compiled packages such as OpenCV.
        """

        if (
            horizon_hours
            in self._explainer_cache
        ):
            return self._explainer_cache[
                horizon_hours
            ]

        try:
            import shap

        except Exception as exc:
            raise RuntimeError(
                "SHAP dependency is unavailable or incompatible "
                "with the current Python/NumPy environment."
            ) from exc

        explainer = (
            shap.TreeExplainer(
                model
            )
        )

        self._explainer_cache[
            horizon_hours
        ] = explainer

        return explainer

    def _explain_tree_with_shap(
        self,
        aligned_df: pd.DataFrame,
        resources: Dict[str, Any],
        horizon_hours: int,
        top_k: int,
    ) -> ExplanationResult:

        model = resources[
            "model"
        ]

        explainer = (
            self._get_tree_shap_explainer(
                horizon_hours,
                model,
            )
        )

        shap_values = (
            explainer(
                aligned_df
            )
        )

        values = np.asarray(
            shap_values.values
        )

        if values.ndim == 2:
            row_values = values[0]

        elif values.ndim == 1:
            row_values = values

        else:
            raise ValueError(
                "Unsupported SHAP value shape: "
                f"{values.shape}"
            )

        feature_names = list(
            aligned_df.columns
        )

        if (
            len(row_values)
            != len(feature_names)
        ):
            raise ValueError(
                "SHAP feature count does not match model schema."
            )

        row_values = np.nan_to_num(
            row_values,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        ranking = sorted(
            zip(
                feature_names,
                row_values,
            ),
            key=lambda item: abs(
                float(
                    item[1]
                )
            ),
            reverse=True,
        )[
            :top_k
        ]

        max_magnitude = max(
            (
                abs(float(item[1]))
                for item in ranking
            ),
            default=0.0,
        )

        top_factors: List[
            Dict[str, Any]
        ] = []

        for (
            feature,
            contribution,
        ) in ranking:

            contribution_float = round(
                float(contribution),
                4,
            )

            display_name = (
                self._clean_feature_name(
                    feature
                )
            )

            top_factors.append(
                {
                    "feature": display_name,

                    "raw_feature": feature,

                    "impact": (
                        self._impact_label(
                            contribution_float,
                            max_magnitude=max_magnitude,
                        )
                    ),

                    "contribution": (
                        contribution_float
                    ),

                    "direction": (
                        self._direction(
                            contribution_float
                        )
                    ),

                    "reason": (
                        f"{display_name} "
                        f"{self._direction(contribution_float)} "
                        f"for the {horizon_hours}h AQI forecast."
                    ),
                }
            )

        return ExplanationResult(
            method="tree_shap",

            note=(
                "Local TreeSHAP explanation generated using the "
                "active trained tree model."
            ),

            top_factors=top_factors,
        )

    # ==================================================================
    # Model-native tree fallback
    # ==================================================================

    def _tree_importance_fallback(
        self,
        aligned_df: pd.DataFrame,
        resources: Dict[str, Any],
        horizon_hours: int,
        top_k: int,
    ) -> ExplanationResult:
        """
        Global model-native tree importance fallback.

        IMPORTANT:
        feature_importances_ is global rather than local, therefore we do
        not label it as a per-prediction contribution.
        """

        model = resources[
            "model"
        ]

        importances = getattr(
            model,
            "feature_importances_",
            None,
        )

        if importances is None:
            return ExplanationResult(
                method="unavailable",
                note=(
                    "The active model does not expose a supported "
                    "native explanation interface."
                ),
                top_factors=[],
            )

        importances = (
            np.asarray(
                importances,
                dtype=float,
            )
            .reshape(-1)
        )

        feature_names = list(
            aligned_df.columns
        )

        if (
            len(importances)
            != len(feature_names)
        ):
            return ExplanationResult(
                method="unavailable",
                note=(
                    "Model feature importance length does not match "
                    "the persisted feature schema."
                ),
                top_factors=[],
            )

        importances = np.nan_to_num(
            importances,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        ranking = sorted(
            zip(
                feature_names,
                importances,
            ),
            key=lambda item: abs(
                float(
                    item[1]
                )
            ),
            reverse=True,
        )[
            :top_k
        ]

        max_magnitude = max(
            (
                abs(float(item[1]))
                for item in ranking
            ),
            default=0.0,
        )

        top_factors: List[
            Dict[str, Any]
        ] = []

        for feature, importance in ranking:

            importance_float = round(
                float(importance),
                6,
            )

            display_name = (
                self._clean_feature_name(
                    feature
                )
            )

            top_factors.append(
                {
                    "feature": display_name,

                    "raw_feature": feature,

                    "impact": (
                        self._impact_label(
                            importance_float,
                            max_magnitude=max_magnitude,
                        )
                    ),

                    "contribution": None,

                    "direction": (
                        "globally important feature"
                    ),

                    "reason": (
                        f"{display_name} is globally important "
                        f"to the trained {horizon_hours}h model."
                    ),
                }
            )

        return ExplanationResult(
            method="model_feature_importance",

            note=(
                "SHAP was unavailable, so the dashboard is showing "
                "global model-native feature importance instead of "
                "local SHAP contributions."
            ),

            top_factors=top_factors,
        )

    # ==================================================================
    # Public API
    # ==================================================================

    def explain_prediction(
        self,
        payload: Dict[str, Any],
        context_df: pd.DataFrame,
        horizon_hours: int = 24,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Generate a production-safe explanation for one prediction.
        """

        top_k = max(
            1,
            min(
                int(top_k),
                20,
            ),
        )

        try:
            aligned_df, resources = (
                self._get_model_ready_features(
                    payload=payload,
                    context_df=context_df,
                    horizon_hours=horizon_hours,
                )
            )

            model = resources[
                "model"
            ]

            # ----------------------------------------------------------
            # Linear models
            # ----------------------------------------------------------
            #
            # Do NOT import SHAP for a linear model unnecessarily.
            # Native local contributions are deterministic, fast,
            # dependency-safe, and directly interpretable.
            # ----------------------------------------------------------

            if getattr(
                model,
                "coef_",
                None,
            ) is not None:

                result = (
                    self._explain_linear_model(
                        aligned_df=aligned_df,
                        resources=resources,
                        horizon_hours=horizon_hours,
                        top_k=top_k,
                    )
                )

            # ----------------------------------------------------------
            # Tree models
            # ----------------------------------------------------------

            elif getattr(
                model,
                "feature_importances_",
                None,
            ) is not None:

                try:
                    result = (
                        self._explain_tree_with_shap(
                            aligned_df=aligned_df,
                            resources=resources,
                            horizon_hours=horizon_hours,
                            top_k=top_k,
                        )
                    )

                except Exception as shap_error:
                    logger.warning(
                        "TreeSHAP unavailable for horizon=%sh. "
                        "Using model-native importance fallback. "
                        "Reason: %s",
                        horizon_hours,
                        shap_error,
                    )

                    result = (
                        self._tree_importance_fallback(
                            aligned_df=aligned_df,
                            resources=resources,
                            horizon_hours=horizon_hours,
                            top_k=top_k,
                        )
                    )

            else:
                result = ExplanationResult(
                    method="unavailable",

                    note=(
                        "The active model does not expose a supported "
                        "production explainability interface."
                    ),

                    top_factors=[],
                )

            return {
                "method": result.method,
                "note": result.note,
                "top_factors": (
                    result.top_factors
                ),
            }

        except Exception as err:
            # Explainability is secondary. Never leak internal traceback
            # or filesystem/model information into the API response.

            logger.warning(
                "Dashboard explainability unavailable "
                "for horizon=%sh: %s",
                horizon_hours,
                err,
            )

            return {
                "method": "unavailable",
                "note": (
                    "Explainability is temporarily unavailable."
                ),
                "top_factors": [],
            }
"""
src/explainability/shap_analysis.py
===================================

Offline SHAP / model-native explanation analysis for registered AQI models.

Features
--------
- Loads production model bundles instead of hard-coded artifacts.
- Uses exact persisted feature schema.
- Supports linear and tree models.
- Lazily imports SHAP.
- Gracefully handles SHAP/NumPy/OpenCV binary incompatibility.
- Provides model-native fallback importance when SHAP is unavailable.
- Produces structured top-feature output.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.prediction.load_model import get_production_model
from src.utils.logger import get_logger


logger = get_logger(__name__)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

DEFAULT_TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "training"
    / "features_test.parquet"
)

DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "models"
    / "registry"
    / "evaluation"
)


class AQIShapAnalyzer:
    """
    Offline global/local explainability for one registered horizon model.

    SHAP is optional at runtime. If its binary dependencies are broken,
    model-native feature importance is still available where supported.
    """

    def __init__(
        self,
        *,
        horizon_hours: int = 24,
        data_path: Path | str = DEFAULT_TEST_PATH,
    ) -> None:

        if horizon_hours not in (
            24,
            48,
            72,
        ):
            raise ValueError(
                "horizon_hours must be one of: 24, 48, 72"
            )

        self.horizon_hours = (
            horizon_hours
        )

        self.data_path = Path(
            data_path
        )

        self.artifact = None
        self.model = None

        self.feature_names: List[
            str
        ] = []

        self.X: Optional[
            pd.DataFrame
        ] = None

        self.explainer = None
        self.shap_values = None

        self.method: Optional[
            str
        ] = None

    # ==================================================================
    # Load production assets
    # ==================================================================

    def load_assets(
        self,
        *,
        sample_size: int = 300,
        random_state: int = 42,
    ) -> None:

        if sample_size <= 0:
            raise ValueError(
                "sample_size must be > 0"
            )

        logger.info(
            "Loading production model bundle | horizon=%sh",
            self.horizon_hours,
        )

        self.artifact = (
            get_production_model(
                horizon_hours=self.horizon_hours
            )
        )

        self.model = (
            self.artifact.model
        )

        self.feature_names = list(
            self.artifact.feature_names
        )

        if not self.feature_names:
            raise ValueError(
                "Registered model has no persisted feature schema."
            )

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Explanation dataset not found: {self.data_path}"
            )

        df = pd.read_parquet(
            self.data_path
        )

        missing = [
            feature
            for feature in self.feature_names
            if feature not in df.columns
        ]

        if missing:
            raise ValueError(
                "Explanation dataset is missing model feature(s): "
                f"{missing}"
            )

        X = (
            df
            .loc[
                :,
                self.feature_names,
            ]
            .copy()
        )

        X = X.apply(
            pd.to_numeric,
            errors="coerce",
        )

        if X.isna().any().any():

            bad_columns = (
                X.columns[
                    X.isna().any()
                ]
                .tolist()
            )

            raise ValueError(
                "Explanation dataset contains unresolved NaNs: "
                f"{bad_columns}"
            )

        sample_size = min(
            sample_size,
            len(X),
        )

        if sample_size == 0:
            raise ValueError(
                "Explanation dataset is empty."
            )

        self.X = (
            X
            .sample(
                n=sample_size,
                random_state=random_state,
            )
            .reset_index(
                drop=True
            )
        )

        logger.info(
            "Explanation assets loaded | horizon=%sh | "
            "rows=%d | features=%d | model=%s",
            self.horizon_hours,
            len(self.X),
            len(self.feature_names),
            type(self.model).__name__,
        )

    # ==================================================================
    # SHAP
    # ==================================================================

    @staticmethod
    def _load_shap() -> Any:
        """
        Lazy import isolates SHAP binary dependency failures from normal
        project imports.
        """

        try:
            import shap

            return shap

        except Exception as exc:
            raise RuntimeError(
                "SHAP could not be imported. This may indicate an "
                "optional binary dependency incompatibility with the "
                "installed NumPy version."
            ) from exc

    def compute_shap_values(
        self,
    ) -> Any:
        """
        Calculate SHAP values using an explainer appropriate to the model.
        """

        if (
            self.model is None
            or self.X is None
        ):
            raise RuntimeError(
                "load_assets() must be called before "
                "compute_shap_values()."
            )

        shap = (
            self._load_shap()
        )

        # --------------------------------------------------------------
        # Linear model
        # --------------------------------------------------------------

        if getattr(
            self.model,
            "coef_",
            None,
        ) is not None:

            logger.info(
                "Using SHAP LinearExplainer | horizon=%sh",
                self.horizon_hours,
            )

            self.explainer = (
                shap.LinearExplainer(
                    self.model,
                    self.X,
                )
            )

            self.shap_values = (
                self.explainer(
                    self.X
                )
            )

            self.method = (
                "linear_shap"
            )

        # --------------------------------------------------------------
        # Tree model
        # --------------------------------------------------------------

        elif getattr(
            self.model,
            "feature_importances_",
            None,
        ) is not None:

            logger.info(
                "Using SHAP TreeExplainer | horizon=%sh",
                self.horizon_hours,
            )

            self.explainer = (
                shap.TreeExplainer(
                    self.model
                )
            )

            self.shap_values = (
                self.explainer(
                    self.X
                )
            )

            self.method = (
                "tree_shap"
            )

        else:
            raise TypeError(
                "Active model type does not have a supported "
                "production SHAP strategy: "
                f"{type(self.model).__name__}"
            )

        logger.info(
            "SHAP calculation complete | horizon=%sh | method=%s",
            self.horizon_hours,
            self.method,
        )

        return (
            self.shap_values
        )

    # ==================================================================
    # Native fallback
    # ==================================================================

    def get_native_feature_importance(
        self,
        *,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """
        Return model-native global importance without SHAP.

        Linear:
            absolute fitted coefficient

        Tree:
            feature_importances_
        """

        if self.model is None:
            raise RuntimeError(
                "Model is not loaded."
            )

        if getattr(
            self.model,
            "coef_",
            None,
        ) is not None:

            values = np.abs(
                np.asarray(
                    self.model.coef_,
                    dtype=float,
                )
                .reshape(-1)
            )

            method = (
                "absolute_coefficient"
            )

        elif getattr(
            self.model,
            "feature_importances_",
            None,
        ) is not None:

            values = (
                np.asarray(
                    self.model.feature_importances_,
                    dtype=float,
                )
                .reshape(-1)
            )

            method = (
                "model_feature_importance"
            )

        else:
            raise TypeError(
                "Model does not expose coef_ or feature_importances_."
            )

        if (
            len(values)
            != len(self.feature_names)
        ):
            raise ValueError(
                "Native importance count does not match "
                "persisted feature schema."
            )

        result = pd.DataFrame(
            {
                "feature": (
                    self.feature_names
                ),

                "importance": values,

                "method": method,
            }
        )

        result = (
            result
            .sort_values(
                "importance",
                ascending=False,
            )
            .head(
                top_n
            )
            .reset_index(
                drop=True
            )
        )

        return result

    # ==================================================================
    # SHAP top features
    # ==================================================================

    def get_top_features(
        self,
        *,
        top_n: int = 10,
    ) -> pd.DataFrame:

        if (
            self.shap_values is None
            or self.X is None
        ):
            raise RuntimeError(
                "compute_shap_values() must be called first."
            )

        values = np.asarray(
            self.shap_values.values
        )

        if values.ndim != 2:
            raise ValueError(
                "Expected a 2D SHAP matrix; "
                f"received shape={values.shape}"
            )

        if (
            values.shape[1]
            != len(self.feature_names)
        ):
            raise ValueError(
                "SHAP feature count does not match persisted model schema."
            )

        mean_abs_shap = (
            np.abs(
                values
            )
            .mean(
                axis=0
            )
        )

        result = pd.DataFrame(
            {
                "feature": (
                    self.feature_names
                ),

                "mean_abs_shap": (
                    mean_abs_shap
                ),
            }
        )

        return (
            result
            .sort_values(
                "mean_abs_shap",
                ascending=False,
            )
            .head(
                top_n
            )
            .reset_index(
                drop=True
            )
        )

    # ==================================================================
    # Plot
    # ==================================================================

    def plot_summary(
        self,
        *,
        save_path: Path | str | None = None,
        max_display: int = 20,
    ) -> Path:

        if (
            self.shap_values is None
            or self.X is None
        ):
            raise RuntimeError(
                "compute_shap_values() must be called before plotting."
            )

        shap = (
            self._load_shap()
        )

        import matplotlib.pyplot as plt

        if save_path is None:
            save_path = (
                DEFAULT_OUTPUT_ROOT
                / f"{self.horizon_hours}h"
                / "shap_summary.png"
            )

        output_path = Path(
            save_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        plt.figure(
            figsize=(
                12,
                8,
            )
        )

        shap.summary_plot(
            self.shap_values,
            self.X,
            show=False,
            max_display=max_display,
        )

        plt.title(
            (
                f"SHAP Feature Importance — "
                f"{self.horizon_hours}h AQI Forecast"
            )
        )

        plt.tight_layout()

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        logger.info(
            "SHAP summary saved | path=%s",
            output_path,
        )

        return output_path

    # ==================================================================
    # Robust execution
    # ==================================================================

    def run_analysis(
        self,
        *,
        sample_size: int = 300,
        top_n: int = 10,
        create_plot: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute SHAP if possible, otherwise return model-native
        importance instead of crashing.
        """

        self.load_assets(
            sample_size=sample_size
        )

        try:
            self.compute_shap_values()

            top_features = (
                self.get_top_features(
                    top_n=top_n
                )
            )

            plot_path = None

            if create_plot:
                plot_path = (
                    self.plot_summary()
                )

            return {
                "horizon_hours": (
                    self.horizon_hours
                ),

                "method": (
                    self.method
                ),

                "shap_available": True,

                "top_features": (
                    top_features
                    .to_dict(
                        orient="records"
                    )
                ),

                "plot_path": (
                    str(plot_path)
                    if plot_path
                    else None
                ),
            }

        except Exception as shap_error:

            logger.warning(
                "SHAP unavailable for horizon=%sh. "
                "Using model-native global importance. Error: %s",
                self.horizon_hours,
                shap_error,
            )

            native = (
                self.get_native_feature_importance(
                    top_n=top_n
                )
            )

            return {
                "horizon_hours": (
                    self.horizon_hours
                ),

                "method": (
                    native[
                        "method"
                    ]
                    .iloc[0]
                    if not native.empty
                    else "unavailable"
                ),

                "shap_available": False,

                "top_features": (
                    native
                    .to_dict(
                        orient="records"
                    )
                ),

                "plot_path": None,

                "note": (
                    "SHAP was unavailable; model-native "
                    "importance was returned instead."
                ),
            }


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Run explainability analysis for registered AQI models."
        )
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=24,
        choices=[
            24,
            48,
            72,
        ],
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--no-plot",
        action="store_true",
    )

    args = (
        parser.parse_args()
    )

    analyzer = AQIShapAnalyzer(
        horizon_hours=args.horizon
    )

    result = (
        analyzer.run_analysis(
            sample_size=args.sample_size,
            top_n=args.top_n,
            create_plot=not args.no_plot,
        )
    )

    logger.info(
        "Explainability analysis result: %s",
        result,
    )


if __name__ == "__main__":
    main()
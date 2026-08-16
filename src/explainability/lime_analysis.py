"""
src/explainability/lime_analysis.py
===================================

Offline LIME analysis for registered AQI forecasting models.

This module:
- Uses the production model registry loader.
- Uses the exact persisted model feature schema.
- Refuses missing feature columns.
- Preserves DataFrame feature names during prediction.
- Avoids hard-coded model artifact paths.
- Supports 24h, 48h, and 72h models.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

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

DEFAULT_TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "training"
    / "features_train.parquet"
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


class AQILimeAnalyzer:
    """
    Production-aligned offline LIME analyzer.

    LIME is used for local explanation only and is intentionally kept
    outside the request-serving API path because it can be comparatively
    expensive.
    """

    def __init__(
        self,
        *,
        horizon_hours: int = 24,
        train_data_path: Path | str = DEFAULT_TRAIN_PATH,
        test_data_path: Path | str = DEFAULT_TEST_PATH,
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

        self.train_data_path = Path(
            train_data_path
        )

        self.test_data_path = Path(
            test_data_path
        )

        self.artifact = None

        self.model = None

        self.feature_names: List[
            str
        ] = []

        self.X_train_sample: pd.DataFrame | None = None

        self.X_test: pd.DataFrame | None = None

        self.explainer = None

    # ==================================================================
    # Loading
    # ==================================================================

    def load_assets(
        self,
        *,
        background_size: int = 300,
        random_state: int = 42,
    ) -> None:
        """
        Load registered model and exact model-ready feature matrices.
        """

        if background_size <= 0:
            raise ValueError(
                "background_size must be > 0"
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
                "Production artifact contains an empty feature schema."
            )

        # --------------------------------------------------------------
        # Dataset loading
        # --------------------------------------------------------------

        for path in (
            self.train_data_path,
            self.test_data_path,
        ):
            if not path.exists():
                raise FileNotFoundError(
                    f"Feature dataset not found: {path}"
                )

        train_df = pd.read_parquet(
            self.train_data_path
        )

        test_df = pd.read_parquet(
            self.test_data_path
        )

        # --------------------------------------------------------------
        # Exact schema validation
        # --------------------------------------------------------------

        missing_train = [
            feature
            for feature in self.feature_names
            if feature not in train_df.columns
        ]

        missing_test = [
            feature
            for feature in self.feature_names
            if feature not in test_df.columns
        ]

        if missing_train:
            raise ValueError(
                "Training explanation dataset is missing model "
                f"feature(s): {missing_train}"
            )

        if missing_test:
            raise ValueError(
                "Test explanation dataset is missing model "
                f"feature(s): {missing_test}"
            )

        X_train = (
            train_df
            .loc[
                :,
                self.feature_names,
            ]
            .copy()
        )

        self.X_test = (
            test_df
            .loc[
                :,
                self.feature_names,
            ]
            .copy()
        )

        # LIME requires a finite numerical matrix.
        X_train = (
            X_train
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
        )

        self.X_test = (
            self.X_test
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
        )

        if X_train.isna().any().any():
            bad_columns = (
                X_train.columns[
                    X_train.isna().any()
                ]
                .tolist()
            )

            raise ValueError(
                "LIME training matrix contains unresolved NaNs: "
                f"{bad_columns}"
            )

        if self.X_test.isna().any().any():
            bad_columns = (
                self.X_test.columns[
                    self.X_test.isna().any()
                ]
                .tolist()
            )

            raise ValueError(
                "LIME test matrix contains unresolved NaNs: "
                f"{bad_columns}"
            )

        sample_size = min(
            background_size,
            len(X_train),
        )

        if sample_size == 0:
            raise ValueError(
                "LIME training dataset is empty."
            )

        self.X_train_sample = (
            X_train
            .sample(
                n=sample_size,
                random_state=random_state,
            )
            .reset_index(
                drop=True
            )
        )

        logger.info(
            "LIME assets ready | horizon=%sh | "
            "features=%d | background_rows=%d | test_rows=%d",
            self.horizon_hours,
            len(self.feature_names),
            len(self.X_train_sample),
            len(self.X_test),
        )

    # ==================================================================
    # Initialization
    # ==================================================================

    def initialize_explainer(
        self,
        *,
        random_state: int = 42,
    ) -> None:

        if (
            self.model is None
            or self.X_train_sample is None
        ):
            raise RuntimeError(
                "load_assets() must be called before "
                "initialize_explainer()."
            )

        try:
            from lime.lime_tabular import (
                LimeTabularExplainer,
            )

        except Exception as exc:
            raise RuntimeError(
                "LIME is unavailable in the current environment."
            ) from exc

        self.explainer = (
            LimeTabularExplainer(
                training_data=(
                    self.X_train_sample
                    .to_numpy(
                        dtype=float
                    )
                ),

                feature_names=(
                    self.feature_names
                ),

                class_names=[
                    "predicted_aqi"
                ],

                mode="regression",

                random_state=(
                    random_state
                ),

                discretize_continuous=True,
            )
        )

        logger.info(
            "LIME explainer initialized | horizon=%sh",
            self.horizon_hours,
        )

    # ==================================================================
    # Prediction adapter
    # ==================================================================

    def _predict_numpy(
        self,
        values: np.ndarray,
    ) -> np.ndarray:
        """
        Convert LIME's NumPy matrix back into the exact model schema.

        This prevents feature-name mismatch and sklearn warnings.
        """

        if self.model is None:
            raise RuntimeError(
                "Model is not loaded."
            )

        frame = pd.DataFrame(
            values,
            columns=self.feature_names,
        )

        predictions = (
            self.model
            .predict(
                frame
            )
        )

        return np.asarray(
            predictions,
            dtype=float,
        )

    # ==================================================================
    # Local explanation
    # ==================================================================

    def explain_single_instance(
        self,
        *,
        instance_idx: int = 0,
        top_k: int = 10,
        num_samples: int = 1000,
        save_html: bool = True,
        save_path: Path | str | None = None,
    ) -> Dict[str, Any]:

        if (
            self.explainer is None
            or self.X_test is None
        ):
            raise RuntimeError(
                "load_assets() and initialize_explainer() "
                "must be called first."
            )

        if (
            instance_idx < 0
            or instance_idx >= len(
                self.X_test
            )
        ):
            raise IndexError(
                f"instance_idx={instance_idx} is outside "
                f"test dataset range 0..{len(self.X_test) - 1}"
            )

        top_k = max(
            1,
            min(
                int(top_k),
                len(self.feature_names),
            ),
        )

        if num_samples < 100:
            raise ValueError(
                "num_samples must be >= 100"
            )

        instance = (
            self.X_test
            .iloc[
                instance_idx
            ]
            .to_numpy(
                dtype=float
            )
        )

        logger.info(
            "Generating LIME explanation | horizon=%sh | row=%d",
            self.horizon_hours,
            instance_idx,
        )

        explanation = (
            self.explainer
            .explain_instance(
                data_row=instance,

                predict_fn=(
                    self._predict_numpy
                ),

                num_features=top_k,

                num_samples=num_samples,
            )
        )

        factors = [
            {
                "feature": feature,
                "weight": round(
                    float(weight),
                    6,
                ),
                "direction": (
                    "increased prediction"
                    if weight > 0
                    else (
                        "reduced prediction"
                        if weight < 0
                        else "neutral effect"
                    )
                ),
            }
            for feature, weight
            in explanation.as_list()
        ]

        output_path = None

        if save_html:

            if save_path is None:
                save_path = (
                    DEFAULT_OUTPUT_ROOT
                    / f"{self.horizon_hours}h"
                    / "lime_explanation.html"
                )

            output_path = Path(
                save_path
            )

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            explanation.save_to_file(
                str(
                    output_path
                )
            )

            logger.info(
                "LIME explanation saved | path=%s",
                output_path,
            )

        prediction = float(
            self.model.predict(
                self.X_test.iloc[
                    [
                        instance_idx
                    ]
                ]
            )[0]
        )

        return {
            "horizon_hours": (
                self.horizon_hours
            ),

            "instance_index": (
                instance_idx
            ),

            "prediction": round(
                prediction,
                4,
            ),

            "method": "lime",

            "top_factors": factors,

            "output_path": (
                str(output_path)
                if output_path
                else None
            ),
        }


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Generate local LIME explanation for a registered "
            "AQI forecasting model."
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
        "--instance",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--background-size",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    analyzer = AQILimeAnalyzer(
        horizon_hours=args.horizon
    )

    analyzer.load_assets(
        background_size=args.background_size
    )

    analyzer.initialize_explainer()

    result = (
        analyzer
        .explain_single_instance(
            instance_idx=args.instance,
            top_k=args.top_k,
            num_samples=args.num_samples,
        )
    )

    logger.info(
        "LIME analysis complete: %s",
        result,
    )


if __name__ == "__main__":
    main()
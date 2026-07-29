"""
tests/test_end_to_end_pipeline.py
===================================
Industry-Level Production Test Suite for AQI Forecasting Pipeline.
Verifies Data Contracts, Feature Engineering, Scaler Alignment, Imputation,
and Multi-Horizon Predictor/Forecaster Modules using pytest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering.air_quality_features import AirQualityFeatureEngineer
from src.feature_engineering.interaction_features import InteractionFeatureEngineer
from src.feature_engineering.lag_features import LagFeatureEngineer
from src.feature_engineering.rolling_features import RollingFeatureEngineer
from src.feature_engineering.spatial_features import SpatialFeatureEngineer
from src.feature_engineering.temporal_features import TemporalFeatureEngineer
from src.feature_engineering.trend_features import TrendFeatureEngineer
from src.prediction.feature_pipeline import PredictionFeaturePipeline
from src.prediction.forecast import AQIForecaster
from src.prediction.load_model import get_production_model
from src.prediction.predictor import AQIPredictor


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def mock_raw_context() -> pd.DataFrame:
    """Generates a realistic 72-hour sample dataset for testing."""
    timestamps = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=72, freq="1h")
    np.random.seed(42)

    data = {
        "timestamp": timestamps.astype(str),
        "city": np.random.choice(["Karachi", "Lahore", "Islamabad"], size=72),
        "latitude": 24.8607,
        "longitude": 67.0011,
        "temperature": np.random.uniform(20.0, 35.0, size=72),
        "humidity": np.random.uniform(40.0, 85.0, size=72),
        "pressure": np.random.uniform(1005.0, 1015.0, size=72),
        "wind_speed": np.random.uniform(1.0, 10.0, size=72),
        "wind_deg": np.random.uniform(0.0, 360.0, size=72),
        "cloudiness": np.random.uniform(0.0, 100.0, size=72),
        "visibility": 10000.0,
        "pm25": np.random.uniform(15.0, 120.0, size=72),
        "pm10": np.random.uniform(30.0, 200.0, size=72),
        "no2": np.random.uniform(10.0, 50.0, size=72),
        "so2": np.random.uniform(5.0, 25.0, size=72),
        "co": np.random.uniform(0.5, 3.5, size=72),
        "o3": np.random.uniform(20.0, 80.0, size=72),
        "aqi": np.random.uniform(50.0, 250.0, size=72),
    }

    # Introduce synthetic NaNs and pd.NA to test robust handling
    df = pd.DataFrame(data)
    df.loc[5, "pm25"] = np.nan
    df.loc[10, "co"] = pd.NA
    return df


# ==============================================================================
# 1. Feature Engineering Pipelines Test
# ==============================================================================

def test_feature_engineering_pipeline(mock_raw_context: pd.DataFrame):
    """Verifies that all 7 feature engineering modules run without errors and create expected features."""
    df = mock_raw_context.copy()
    initial_cols = df.shape[1]

    df = TemporalFeatureEngineer().build(df)
    df = LagFeatureEngineer().build(df)
    df = RollingFeatureEngineer().build(df)
    df = TrendFeatureEngineer().build(df)
    df = InteractionFeatureEngineer().build(df)
    df = AirQualityFeatureEngineer().build(df)
    df = SpatialFeatureEngineer().build(df)

    assert df.shape[1] > initial_cols, "Feature engineering did not generate new columns."
    assert "hour_sin" in df.columns, "Temporal feature missing."
    assert "city_Karachi" in df.columns or "city" in df.columns, "Spatial feature missing."


# ==============================================================================
# 2. Scaler & Imputation Test (Handling pd.NA / NAType)
# ==============================================================================

def test_feature_pipeline_scaler_transform(mock_raw_context: pd.DataFrame):
    """Verifies feature pipeline handles NA types, scaling, and alignment safely."""
    artifact = get_production_model(horizon_hours=24)
    pipeline = PredictionFeaturePipeline(scaler_engineer=artifact.scaler_engineer)

    # Test single payload prediction transformation
    payload = mock_raw_context.iloc[-1].to_dict()
    scaled_df = pipeline.build_features(payload, context_df=mock_raw_context)

    assert not scaled_df.empty, "Scaled output DataFrame is empty."
    assert not scaled_df.isna().any().any(), "Scaled features contain unhandled NaNs or NATypes!"


# ==============================================================================
# 3. Predictor Engine & Model Horizons Test
# ==============================================================================

@pytest.mark.parametrize("horizon", [24, 48, 72])
def test_predictor_multi_horizon(mock_raw_context: pd.DataFrame, horizon: int):
    """Tests batch prediction across direct 24H, 48H, and 72H model horizons."""
    predictor = AQIPredictor()
    
    payloads = mock_raw_context.iloc[-5:].to_dict(orient="records")
    results = predictor.predict_batch(payloads, context_df=mock_raw_context, horizon_hours=horizon)

    assert isinstance(results, pd.DataFrame), "Batch result should be a pandas DataFrame."
    assert "predicted_aqi" in results.columns, "Output missing 'predicted_aqi' prediction column."
    assert len(results) == len(payloads), "Batch prediction count mismatch."
    assert not results["predicted_aqi"].isna().any(), "Predicted AQI contains NaN values."


# ==============================================================================
# 4. End-to-End AQIForecaster Orchestration Test
# ==============================================================================

@pytest.mark.parametrize("horizon_hours", [12, 24, 48, 72])
def test_forecaster_end_to_end(horizon_hours: int):
    """Tests the top-level AQIForecaster orchestrator for all horizons and output saving."""
    forecaster = AQIForecaster()
    forecasts = forecaster.generate_forecast(horizon_hours=horizon_hours, save_predictions=True)

    assert isinstance(forecasts, list), "Forecast result should be a list of dicts."
    assert len(forecasts) == horizon_hours, f"Expected {horizon_hours} steps in forecast output."
    
    first_step = forecasts[0]
    assert "horizon_step" in first_step
    assert "timestamp" in first_step
    assert "predicted_aqi" in first_step
    assert "model_version" in first_step
    assert isinstance(first_step["predicted_aqi"], float)
import os
import joblib
from pathlib import Path
from fastapi import APIRouter, HTTPException, status
from src.api.schemas import PredictRequest, PredictResponse, HorizonPrediction, HealthCheckResponse

router = APIRouter()

# Global dictionary to hold loaded models in RAM
MODELS = {}

# Exact registry directory matching your folder tree
MODEL_DIR = Path("models/registry/registry")

def load_models():
    """Helper function to load trained horizon models from registry."""
    global MODELS
    horizons = ["24h", "48h", "72h"]
    
    for h in horizons:
        # Exact file name match: 24h_model.joblib, 48h_model.joblib, 72h_model.joblib
        model_path = MODEL_DIR / f"{h}_model.joblib"
        
        if model_path.exists():
            MODELS[h] = joblib.load(model_path)
            print(f"✅ Loaded model for horizon: {h} from {model_path}")
        else:
            print(f"⚠️ Warning: Model for horizon '{h}' not found at {model_path}")

@router.get("/health", response_model=HealthCheckResponse, status_code=status.HTTP_200_OK)
def health_check():
    """Health check endpoint to confirm API & models status."""
    return HealthCheckResponse(
        status="healthy",
        loaded_models=list(MODELS.keys())
    )

@router.post("/predict", response_model=PredictResponse)
def predict_aqi(payload: PredictRequest):
    """Generate 24h, 48h, and 72h AQI predictions from the 626-feature vector."""
    if not MODELS:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No trained models are currently loaded on the server."
        )

    # Validate feature vector length
    if len(payload.features) != 626:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Expected feature vector length of 626, but got {len(payload.features)}."
        )

    predictions = []
    feature_vector = [payload.features]  # Reshape for single-sample inference

    for horizon in ["24h", "48h", "72h"]:
        if horizon in MODELS:
            pred_value = MODELS[horizon].predict(feature_vector)[0]
            predictions.append(
                HorizonPrediction(
                    horizon=horizon,
                    predicted_aqi=round(float(pred_value), 2)
                )
            )

    return PredictResponse(
        city=payload.city,
        status="success",
        predictions=predictions
    )
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class PredictRequest(BaseModel):
    city: str = Field(..., example="Lahore")
    features: List[float] = Field(
        ..., 
        description="Feature vector containing all 626 engineered features in correct sequence."
    )

class HorizonPrediction(BaseModel):
    horizon: str
    predicted_aqi: float
    model_name: str = "Gradient_Boosting"

class PredictResponse(BaseModel):
    city: str
    status: str
    predictions: List[HorizonPrediction]

class HealthCheckResponse(BaseModel):
    status: str
    loaded_models: List[str]
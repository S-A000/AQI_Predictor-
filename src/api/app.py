from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router

app = FastAPI(
    title="AQI Forecasting API",
    version="1.0.0",
    description="FastAPI backend for AQI forecasting dashboard.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/")
def root() -> dict:
    return {
        "message": "AQI Forecasting API is running.",
        "docs": "/docs",
        "health": "/api/v1/health",
        "dashboard": "/api/v1/dashboard",
    }
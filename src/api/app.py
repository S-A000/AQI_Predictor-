from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router


app = FastAPI(
    title="AQI Forecasting API",
    version="1.0.0",
    description=(
        "Production API for Islamabad, Karachi, and Lahore "
        "AQI monitoring and direct 24h/48h/72h forecasting."
    ),
)


# ------------------------------------------------------------------
# CORS
# ------------------------------------------------------------------

cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ALLOWED_ORIGINS",
        (
            "http://127.0.0.1:3000,"
            "http://localhost:3000,"
            "http://127.0.0.1:8501,"
            "http://localhost:8501"
        ),
    ).split(",")
    if origin.strip()
]


cors_allow_credentials = (
    os.getenv(
        "CORS_ALLOW_CREDENTIALS",
        "false",
    )
    .strip()
    .lower()
    in {
        "1",
        "true",
        "yes",
        "on",
    }
)


# Browsers do not allow wildcard credentials safely.
if (
    "*"
    in cors_origins
    and cors_allow_credentials
):
    cors_allow_credentials = False


app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=cors_allow_credentials,
    allow_methods=[
        "GET",
        "OPTIONS",
    ],
    allow_headers=[
        "Content-Type",
        "Authorization",
    ],
)


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

app.include_router(
    router,
    prefix="/api/v1",
)


@app.get("/")
def root() -> dict:
    return {
        "message": "AQI Forecasting API is running.",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
        "dashboard": "/api/v1/dashboard",
    }
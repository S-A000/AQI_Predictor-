from fastapi import FastAPI
from contextlib import asynccontextmanager
from src.api.routes import router, load_models

@asynccontextmanager
async def lifespan(app: FastAPI):
    # App start hotay hi models RAM mein load honge
    load_models()
    yield
    # Cleanup logic (if needed on shutdown)

app = FastAPI(
    title="Enterprise AQI Forecasting Platform API",
    description="Multi-Horizon (24h, 48h, 72h) AQI Forecasting Service powered by MLOps Pipeline.",
    version="1.0.0",
    lifespan=lifespan
)

# Attach routes
app.include_router(router, prefix="/api/v1")
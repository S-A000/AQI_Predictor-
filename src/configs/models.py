from pathlib import Path
from pydantic import BaseModel, Field


class ProjectConfig(BaseModel):
    name: str
    version: str
    environment: str


class LocationConfig(BaseModel):
    city: str
    country: str


class APIConfig(BaseModel):
    timeout: int = Field(default=30, ge=1)
    retries: int = Field(default=3, ge=0)


class SecretConfig(BaseModel):
    aqicn_api_key: str
    openweather_api_key: str


class PathConfig(BaseModel):
    raw_data: Path
    processed_data: Path
    models: Path
    reports: Path
    logs: Path
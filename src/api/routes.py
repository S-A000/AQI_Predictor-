from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.api.schemas import CityDashboardResponse, DashboardResponse, HealthResponse
from src.prediction.dashboard_service import DashboardForecastService

router = APIRouter()

dashboard_service = DashboardForecastService()


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service="AQI Forecasting API",
    )


@router.get("/dashboard", response_model=DashboardResponse)
def get_dashboard_data() -> DashboardResponse:
    try:
        results = dashboard_service.get_all_cities_dashboard_data()
        return DashboardResponse(cities=results)

    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err)) from err


@router.get("/dashboard/{city}", response_model=CityDashboardResponse)
def get_city_dashboard_data(city: str) -> CityDashboardResponse:
    allowed_cities = {"islamabad", "karachi", "lahore"}

    if city.lower() not in allowed_cities:
        raise HTTPException(
            status_code=400,
            detail="Unsupported city. Use Islamabad, Karachi, or Lahore.",
        )

    try:
        result = dashboard_service.get_city_dashboard_data(city=city)
        return CityDashboardResponse(**result)

    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err)) from err


@router.get("/dashboard/{city}/explain")
def get_city_explainability(city: str) -> dict:
    allowed_cities = {"islamabad", "karachi", "lahore"}

    if city.lower() not in allowed_cities:
        raise HTTPException(
            status_code=400,
            detail="Unsupported city. Use Islamabad, Karachi, or Lahore.",
        )

    try:
        result = dashboard_service.get_city_dashboard_data(city=city)

        return {
            "city": result["city"],
            "explainability": result.get("explainability"),
        }

    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err)) from err
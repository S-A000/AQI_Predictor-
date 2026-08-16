from __future__ import annotations

from functools import lru_cache

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)

from src.api.schemas import (
    CityDashboardResponse,
    DashboardResponse,
    HealthResponse,
)
from src.prediction.dashboard_service import (
    DashboardForecastService,
)
from src.utils.logger import get_logger


router = APIRouter()

logger = get_logger(__name__)


SUPPORTED_CITIES = {
    "islamabad",
    "karachi",
    "lahore",
}


# ------------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_dashboard_service() -> DashboardForecastService:
    """
    Lazily initialize heavy prediction/dashboard resources.

    The service is created only when a dashboard endpoint is first used,
    rather than during FastAPI module import.
    """

    return DashboardForecastService()


# ------------------------------------------------------------------
# Health
# ------------------------------------------------------------------

@router.get(
    "/health",
    response_model=HealthResponse,
)
def health_check() -> HealthResponse:
    """
    Lightweight application health endpoint.

    Does not perform model inference or external API calls.
    """

    return HealthResponse(
        status="ok",
        service="AQI Forecasting API",
    )


# ------------------------------------------------------------------
# Complete dashboard
# ------------------------------------------------------------------

@router.get(
    "/dashboard",
    response_model=DashboardResponse,
    response_model_exclude_none=True,
)
def get_dashboard_data(
    dashboard_service: DashboardForecastService = Depends(
        get_dashboard_service
    ),
) -> DashboardResponse:

    try:
        results = (
            dashboard_service
            .get_all_cities_dashboard_data()
        )

        return DashboardResponse(
            cities=results
        )

    except Exception as err:
        logger.exception(
            "Dashboard request failed: %s",
            err,
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Dashboard data is temporarily unavailable."
            ),
        ) from err


# ------------------------------------------------------------------
# Explainability
# ------------------------------------------------------------------

@router.get(
    "/dashboard/{city}/explain",
    response_model_exclude_none=True,
)
def get_city_explainability(
    city: str,
    dashboard_service: DashboardForecastService = Depends(
        get_dashboard_service
    ),
) -> dict:

    normalized_city = (
        city
        .strip()
        .lower()
    )

    if normalized_city not in SUPPORTED_CITIES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Unsupported city. Use Islamabad, Karachi, or Lahore."
            ),
        )

    try:
        result = (
            dashboard_service
            .get_city_dashboard_data(
                city=city
            )
        )

        return {
            "city": result["city"],
            "explainability": result.get(
                "explainability"
            ),
        }

    except Exception as err:
        logger.exception(
            "City explainability request failed "
            "for city=%s: %s",
            city,
            err,
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Explainability data is temporarily unavailable."
            ),
        ) from err


# ------------------------------------------------------------------
# Individual city
# ------------------------------------------------------------------

@router.get(
    "/dashboard/{city}",
    response_model=CityDashboardResponse,
    response_model_exclude_none=True,
)
def get_city_dashboard_data(
    city: str,
    dashboard_service: DashboardForecastService = Depends(
        get_dashboard_service
    ),
) -> CityDashboardResponse:

    normalized_city = (
        city
        .strip()
        .lower()
    )

    if normalized_city not in SUPPORTED_CITIES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Unsupported city. Use Islamabad, Karachi, or Lahore."
            ),
        )

    try:
        result = (
            dashboard_service
            .get_city_dashboard_data(
                city=city
            )
        )

        return CityDashboardResponse(
            **result
        )

    except Exception as err:
        logger.exception(
            "City dashboard request failed "
            "for city=%s: %s",
            city,
            err,
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "City dashboard data is temporarily unavailable."
            ),
        ) from err
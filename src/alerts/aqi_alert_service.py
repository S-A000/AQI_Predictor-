from __future__ import annotations

from src.alerts.email_alert_service import EmailAlertService


class AQIAlertService:
    """
    Sends hazardous forecast alerts through EmailAlertService.

    Threshold checking is performed by the calling prediction/dashboard service.
    """

    def __init__(self) -> None:
        self.alert_service = EmailAlertService()

    def send_hazardous_alert(
        self,
        city: str,
        horizon: str,
        predicted_aqi: float,
    ) -> bool:
        return self.alert_service.send_alert(
            title=f"AQI Alert — {city}",
            message=(
                f"Hazardous AQI is predicted for {city} "
                f"within {horizon}. Predicted AQI: {predicted_aqi:.1f}"
            ),
            severity="critical",
        )
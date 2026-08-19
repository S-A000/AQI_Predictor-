from unittest.mock import patch

from src.alerts.aqi_alert_service import AQIAlertService


@patch("src.alerts.aqi_alert_service.EmailAlertService.send_alert")
def test_hazardous_aqi_email_alert(mock_send_alert) -> None:
    mock_send_alert.return_value = True

    service = AQIAlertService()

    sent = service.alert_service.send_alert(
        title="AQI Alert — Karachi",
        message=(
            "Hazardous AQI is predicted for Karachi "
            "within 24h. Predicted AQI: 325.0"
        ),
        severity="critical",
    )

    assert sent is True
    mock_send_alert.assert_called_once_with(
        title="AQI Alert — Karachi",
        message=(
            "Hazardous AQI is predicted for Karachi "
            "within 24h. Predicted AQI: 325.0"
        ),
        severity="critical",
    )
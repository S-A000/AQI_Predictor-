from src.alerts.email_alert_service import EmailAlertService


def test_email_alert() -> None:
    service = EmailAlertService()

    sent = service.send_alert(
        title="Email Alert Test",
        message=(
            "This is a test alert from the AQI Forecasting "
            "MLOps platform."
        ),
        severity="info",
    )

    assert sent is True, "Email alert could not be sent."
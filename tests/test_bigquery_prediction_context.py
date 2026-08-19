from src.prediction.dashboard_service import (
    DashboardForecastService,
)


def test_bigquery_prediction_context() -> None:
    service = DashboardForecastService()

    context_df = service._load_prediction_context()

    print(f"Total context rows: {len(context_df)}")
    print(context_df["city"].value_counts())

    assert not context_df.empty

    for city in ["Karachi", "Lahore", "Islamabad"]:
        city_rows = context_df[
            context_df["city"].astype(str).str.lower()
            == city.lower()
        ]

        assert not city_rows.empty
        assert len(city_rows) <= 72
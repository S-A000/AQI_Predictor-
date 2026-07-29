from src.feature_store.bigquery_feature_store import (
    BigQueryFeatureStore,
)


def test_bigquery_connection() -> None:
    store = BigQueryFeatureStore()

    total_rows = store.test_connection()

    print(f"BigQuery table rows: {total_rows}")

    assert total_rows > 0
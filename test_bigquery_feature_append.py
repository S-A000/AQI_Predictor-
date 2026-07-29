from datetime import datetime, timezone

from src.feature_store.bigquery_feature_store import (
    BigQueryFeatureStore,
)


def test_bigquery_feature_append() -> None:
    store = BigQueryFeatureStore()

    source_df = store.get_latest_context(
        city="Karachi",
        rows=1,
    )

    test_df = source_df.copy()

    # Unique timestamp so test reruns do not intentionally create
    # the same city/timestamp combination.
    test_df["timestamp"] = datetime.now(
        timezone.utc
    )

    test_df["city"] = "Karachi"

    uploaded_rows = store.append_features(
        dataframe=test_df,
    )

    print(f"BigQuery appended rows: {uploaded_rows}")

    assert uploaded_rows == 1
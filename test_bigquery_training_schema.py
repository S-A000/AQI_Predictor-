from pathlib import Path

import pandas as pd

from src.feature_store.bigquery_feature_store import (
    BigQueryFeatureStore,
)


LOCAL_PATH = Path(
    "data/training/training_dataset.parquet"
)


def test_bigquery_training_schema() -> None:
    store = BigQueryFeatureStore()

    local_df = pd.read_parquet(LOCAL_PATH)
    bigquery_df = store.get_training_features()

    local_columns = set(local_df.columns)
    bigquery_columns = set(bigquery_df.columns)

    missing_in_bigquery = sorted(
        local_columns - bigquery_columns
    )

    extra_in_bigquery = sorted(
        bigquery_columns - local_columns
    )

    print(f"Local shape: {local_df.shape}")
    print(f"BigQuery shape: {bigquery_df.shape}")

    print(
        f"Missing in BigQuery: "
        f"{len(missing_in_bigquery)}"
    )
    print(missing_in_bigquery[:30])

    print(
        f"Extra in BigQuery: "
        f"{len(extra_in_bigquery)}"
    )
    print(extra_in_bigquery[:30])

    assert not local_df.empty
    assert not bigquery_df.empty
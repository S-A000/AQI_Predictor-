from feast import FeatureStore

store = FeatureStore(repo_path=".")

response = store.get_online_features(
    features=[
        "aqi_features:temperature",
        "aqi_features:humidity",
        "aqi_features:aqi",
        "aqi_features:pm25",
        "aqi_features:pm10",
    ],
    entity_rows=[
        {
            "city": "Karachi"
        },
        {"city": "Lahore"},
        {"city": "Islamabad"}
    ]
)

print(response.to_dict())
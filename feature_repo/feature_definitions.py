from datetime import timedelta

from feast import FeatureView
from feast import Field
from feast import FileSource
from feast.types import Float32, Int64, String

from entities import city
from data_source import aqi_source


aqi_feature_view = FeatureView(
    name="aqi_features",
    entities=[city],
    ttl=timedelta(days=365),
    schema=[
        # -----------------------------
        # Location
        # -----------------------------
        Field(name="country", dtype=String),
        Field(name="latitude", dtype=Float32),
        Field(name="longitude", dtype=Float32),

        # -----------------------------
        # Weather
        # -----------------------------
        Field(name="temperature", dtype=Float32),
        Field(name="feels_like", dtype=Float32),
        Field(name="humidity", dtype=Int64),
        Field(name="pressure", dtype=Int64),
        Field(name="visibility", dtype=Int64),
        Field(name="wind_speed", dtype=Float32),
        Field(name="wind_degree", dtype=Int64),
        Field(name="cloudiness", dtype=Int64),

        # -----------------------------
        # AQI
        # -----------------------------
        Field(name="aqi", dtype=Int64),
        Field(name="dominant_pollutant", dtype=String),
        Field(name="station_id", dtype=Int64),

        # -----------------------------
        # Pollutants
        # -----------------------------
        Field(name="pm25", dtype=Float32),
        Field(name="pm10", dtype=Float32),
        Field(name="no2", dtype=Float32),
        Field(name="so2", dtype=Float32),
        Field(name="co", dtype=Float32),
        Field(name="o3", dtype=Float32),

        # -----------------------------
        # Time-based features (engineered)
        # -----------------------------
        Field(name="hour", dtype=Int64),
        Field(name="day", dtype=Int64),
        Field(name="month", dtype=Int64),
        Field(name="day_of_week", dtype=Int64),
        Field(name="is_weekend", dtype=Int64),

        # -----------------------------
        # AQI-derived features (engineered)
        # -----------------------------
        Field(name="aqi_change_rate", dtype=Float32),
        Field(name="aqi_rolling_mean_3h", dtype=Float32),
    ],

    source=aqi_source,
)
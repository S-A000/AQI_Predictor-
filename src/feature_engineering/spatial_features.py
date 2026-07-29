"""
spatial_features.py
=====================
Suggested path: src/feature_engineering/spatial_features.py

Phase 3, Part 7 — Spatial Features.

SINGLE RESPONSIBILITY: encode location identity (city, station,
coordinates) into model-usable numeric form. Does not touch
temporal/lag/rolling/trend/interaction/air-quality features — see
sibling modules.

No groupby-by-city needed: all row-wise, same reasoning as
interaction_features.py / air_quality_features.py.

Note on distance features: marked optional in the Phase 3 plan.
With only 3 cities and no reference/monitoring-station coordinates
beyond the 3 cities themselves, "distance to X" isn't meaningful yet
(distance to what?) — implemented here as distance-between-cities
only, ready to extend if a reference point (e.g. nearest industrial
zone, largest traffic hub) is defined later.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class SpatialFeatureEngineer:
    """
    Adds city one-hot encoding, station encoding, and normalized
    lat/lon features. Distance features are opt-in via
    `add_distance_features()` since they need a reference point.
    """

    def __init__(
        self,
        *,
        city_col: str = "city",
        station_col: str = "station_id",
        latitude_col: str = "latitude",
        longitude_col: str = "longitude",
    ):
        self.city_col = city_col
        self.station_col = station_col
        self.latitude_col = latitude_col
        self.longitude_col = longitude_col

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------

    def _has_columns(self, df: pd.DataFrame, *columns: str) -> bool:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            logger.warning("Column(s) not found, skipping dependent feature(s): %s", missing)
            return False
        return True

    # --------------------------------------------------
    # City encoding (one-hot — only 3 cities, so this stays compact;
    # ordinal/target encoding would be needed if this scales to
    # dozens of cities later)
    # --------------------------------------------------

    def add_city_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._has_columns(df, self.city_col):
            return df
        df = df.copy()
        dummies = pd.get_dummies(df[self.city_col], prefix="city", dtype=int)
        df = pd.concat([df, dummies], axis=1)
        logger.info("City one-hot encoded: %s", list(dummies.columns))
        return df

    # --------------------------------------------------
    # Station encoding
    # --------------------------------------------------

    def add_station_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        One-hot encodes `station_id`. NOTE: historical (Open-Meteo)
        rows all carry the sentinel station_id=-1 (no physical
        station in a reanalysis model — see historical_client.py),
        so this column mixes real AQICN station IDs with that
        sentinel. Check `dataset_statistics.json` / VIF results in
        Phase 4 before trusting this feature — if -1 dominates the
        distribution, it may carry more "is this historical or live
        data" signal than genuine spatial signal, which would be a
        subtle source-leakage risk worth dropping in Part 8.
        """
        if not self._has_columns(df, self.station_col):
            return df
        df = df.copy()
        dummies = pd.get_dummies(df[self.station_col], prefix="station", dtype=int)
        df = pd.concat([df, dummies], axis=1)
        return df

    # --------------------------------------------------
    # Lat/Lon encoding
    # --------------------------------------------------

    def add_lat_lon_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Raw lat/lon are kept as-is (they're already small, bounded
        numbers — no cyclical encoding needed since Pakistan doesn't
        span the antimeridian/poles where lat/lon wrap around). Also
        adds a single combined "coordinate hash" proxy via lat*lon,
        which — combined with the one-hot city columns — gives
        tree-based models an easy numeric handle on location without
        relying solely on one-hot splits.
        """
        if not self._has_columns(df, self.latitude_col, self.longitude_col):
            return df
        df = df.copy()
        df["lat_lon_product"] = df[self.latitude_col] * df[self.longitude_col]
        return df

    # --------------------------------------------------
    # Distance features (optional — distance between the 3 known cities)
    # --------------------------------------------------

    @staticmethod
    def _haversine_km(lat1, lon1, lat2, lon2) -> float:
        """Great-circle distance between two lat/lon points, in km."""
        r = 6371.0  # Earth's radius in km
        lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        return r * 2 * np.arcsin(np.sqrt(a))

    def add_distance_features(
        self,
        df: pd.DataFrame,
        *,
        reference_point: tuple[float, float] | None = None,
        reference_name: str = "reference",
    ) -> pd.DataFrame:
        """
        Optional (per the Phase 3 plan). Adds distance-in-km from
        each row's lat/lon to a given `reference_point` (lat, lon) —
        e.g. distance to a known industrial zone or the national
        capital. Not called by build() automatically since there is
        no default reference point defined for this project yet;
        call it explicitly if/when one is decided.
        """
        if reference_point is None:
            logger.info("No reference_point given; skipping distance features.")
            return df
        if not self._has_columns(df, self.latitude_col, self.longitude_col):
            return df

        df = df.copy()
        ref_lat, ref_lon = reference_point
        df[f"distance_to_{reference_name}_km"] = self._haversine_km(
            df[self.latitude_col], df[self.longitude_col], ref_lat, ref_lon,
        )
        return df

    # --------------------------------------------------
    # Full Part 7 pipeline (distance features excluded — opt-in only)
    # --------------------------------------------------

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        before_cols = df.shape[1]

        df = self.add_city_encoding(df)
        df = self.add_station_encoding(df)
        df = self.add_lat_lon_features(df)

        after_cols = df.shape[1]
        logger.info(
            "Spatial features added: %d new column(s) (%d -> %d).",
            after_cols - before_cols, before_cols, after_cols,
        )
        return df
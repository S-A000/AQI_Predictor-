from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logger import get_logger


logger = get_logger(__name__)


SUPPORTED_CITIES = (
    "Islamabad",
    "Karachi",
    "Lahore",
)


class SpatialFeatureEngineer:
    """
    Adds stable spatial features for AQI forecasting.

    Production features:
    - Fixed-schema city one-hot encoding
    - Latitude / longitude interaction features

    station_id is intentionally NOT used as a model feature.

    Reasons:
    - station IDs can have high cardinality
    - new station IDs can appear at inference time
    - historical rows may use station_id=-1
    - live AQICN rows may contain real station IDs
    - station identity can accidentally become a source-indicator

    Distance features remain opt-in.
    """

    def __init__(
        self,
        *,
        city_col: str = "city",
        station_col: str = "station_id",
        latitude_col: str = "latitude",
        longitude_col: str = "longitude",
    ) -> None:
        self.city_col = city_col
        self.station_col = station_col
        self.latitude_col = latitude_col
        self.longitude_col = longitude_col

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------

    def _has_columns(
        self,
        df: pd.DataFrame,
        *columns: str,
    ) -> bool:
        missing = [
            column
            for column in columns
            if column not in df.columns
        ]

        if missing:
            logger.warning(
                "Column(s) not found, skipping dependent feature(s): %s",
                missing,
            )
            return False

        return True

    # --------------------------------------------------
    # City encoding
    # --------------------------------------------------

    def add_city_encoding(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        One-hot encode the project's fixed supported city set.

        The same three columns are ALWAYS generated:

            city_Islamabad
            city_Karachi
            city_Lahore

        This prevents train/validation/test/inference schema drift.
        """

        if not self._has_columns(
            df,
            self.city_col,
        ):
            return df

        result = df.copy()

        normalized_city = (
            result[self.city_col]
            .astype(str)
            .str.strip()
            .str.title()
        )

        unsupported = sorted(
            set(normalized_city.dropna().unique())
            - set(SUPPORTED_CITIES)
        )

        if unsupported:
            logger.warning(
                "Unsupported city value(s) encountered during spatial "
                "encoding: %s",
                unsupported,
            )

        city_series = pd.Series(
            pd.Categorical(
                normalized_city,
                categories=list(SUPPORTED_CITIES),
            ),
            index=result.index,
            name=self.city_col,
        )

        dummies = pd.get_dummies(
            city_series,
            prefix="city",
            dtype=int,
        )

        expected_dummy_columns = [
            f"city_{city}"
            for city in SUPPORTED_CITIES
        ]

        dummies = dummies.reindex(
            columns=expected_dummy_columns,
            fill_value=0,
        )

        result = pd.concat(
            [
                result,
                dummies,
            ],
            axis=1,
        )

        logger.info(
            "City one-hot encoded with fixed schema: %s",
            expected_dummy_columns,
        )

        return result

    # --------------------------------------------------
    # Station encoding
    # --------------------------------------------------

    def add_station_encoding(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Optional/manual station_id one-hot encoding.

        IMPORTANT:
        This method is intentionally NOT called by build().

        It remains only for backwards compatibility or experiments.
        """

        if not self._has_columns(
            df,
            self.station_col,
        ):
            return df

        result = df.copy()

        dummies = pd.get_dummies(
            result[self.station_col],
            prefix="station",
            dtype=int,
        )

        result = pd.concat(
            [
                result,
                dummies,
            ],
            axis=1,
        )

        logger.info(
            "Station IDs manually one-hot encoded: %s",
            list(dummies.columns),
        )

        return result

    # --------------------------------------------------
    # Latitude / longitude features
    # --------------------------------------------------

    def add_lat_lon_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Keep latitude/longitude and add a simple geographic interaction.
        """

        if not self._has_columns(
            df,
            self.latitude_col,
            self.longitude_col,
        ):
            return df

        result = df.copy()

        result[self.latitude_col] = pd.to_numeric(
            result[self.latitude_col],
            errors="coerce",
        )

        result[self.longitude_col] = pd.to_numeric(
            result[self.longitude_col],
            errors="coerce",
        )

        result["lat_lon_product"] = (
            result[self.latitude_col]
            * result[self.longitude_col]
        )

        return result

    # --------------------------------------------------
    # Distance features
    # --------------------------------------------------

    @staticmethod
    def _haversine_km(
        lat1,
        lon1,
        lat2,
        lon2,
    ):
        """
        Calculate great-circle distance in kilometres.
        """

        earth_radius_km = 6371.0

        lat1, lon1, lat2, lon2 = map(
            np.radians,
            (
                lat1,
                lon1,
                lat2,
                lon2,
            ),
        )

        delta_latitude = lat2 - lat1
        delta_longitude = lon2 - lon1

        a = (
            np.sin(delta_latitude / 2) ** 2
            + np.cos(lat1)
            * np.cos(lat2)
            * np.sin(delta_longitude / 2) ** 2
        )

        return (
            earth_radius_km
            * 2
            * np.arcsin(np.sqrt(a))
        )

    def add_distance_features(
        self,
        df: pd.DataFrame,
        *,
        reference_point: tuple[float, float] | None = None,
        reference_name: str = "reference",
    ) -> pd.DataFrame:
        """
        Add distance to an optional reference coordinate.

        This is not part of the default production feature pipeline.
        """

        if reference_point is None:
            logger.info(
                "No reference_point given; skipping distance features."
            )
            return df

        if not self._has_columns(
            df,
            self.latitude_col,
            self.longitude_col,
        ):
            return df

        result = df.copy()

        reference_latitude, reference_longitude = reference_point

        result[f"distance_to_{reference_name}_km"] = (
            self._haversine_km(
                result[self.latitude_col],
                result[self.longitude_col],
                reference_latitude,
                reference_longitude,
            )
        )

        return result

    # --------------------------------------------------
    # Full spatial pipeline
    # --------------------------------------------------

    def build(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Run production spatial feature engineering.

        station_id is deliberately excluded.
        """

        before_columns = df.shape[1]

        # Stable fixed-schema city features.
        df = self.add_city_encoding(df)

        # DO NOT:
        #
        # df = self.add_station_encoding(df)
        #
        # station_id remains metadata only.

        df = self.add_lat_lon_features(df)

        after_columns = df.shape[1]

        logger.info(
            "Spatial features added: %d new column(s) (%d -> %d).",
            after_columns - before_columns,
            before_columns,
            after_columns,
        )

        return df
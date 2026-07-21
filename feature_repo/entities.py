"""
entities.py
===========

Feast Entity Definitions

Author:
    Syed Abdullah

Description
-----------
Defines the primary entities used by the AQI Forecasting
Feature Store.

For this project, every feature belongs to a city.
"""

from feast import Entity

# ==========================================================
# City Entity
# ==========================================================

city = Entity(
    name="city",
    join_keys=["city"],
    description="City for which AQI and weather features are collected.",
)
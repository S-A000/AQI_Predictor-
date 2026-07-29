"""AQI Forecasting MLOps premium Streamlit dashboard.

Run from the repository root:
    streamlit run src/dashboard/streamlit_app.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import requests
import streamlit as st

try:
    from .components import CITY_META, inject_styles, render_city_selector, render_dashboard, render_topbar
except ImportError:  # supports `streamlit run src/dashboard/streamlit_app.py`
    from components import CITY_META, inject_styles, render_city_selector, render_dashboard, render_topbar


API_URL = "http://127.0.0.1:8000/api/v1/dashboard"
ASSET_DIR = Path(__file__).resolve().parent / "assets"


def _fetch_dashboard() -> tuple[Mapping[str, Any] | None, str | None]:
    try:
        response = requests.get(API_URL, timeout=180)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, Mapping):
            return None, "FastAPI returned an unexpected response shape."
        return payload, None
    except requests.exceptions.ConnectionError:
        return None, "FastAPI is not running. Start the backend and refresh this page."
    except requests.exceptions.Timeout:
        return None, "The AQI API took too long to respond. Please try again."
    except requests.exceptions.RequestException as exc:
        return None, f"The AQI API could not be reached: {exc}"
    except ValueError:
        return None, "FastAPI returned invalid JSON. Check the dashboard endpoint response."


def _city_payload(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    cities = payload.get("cities")
    if not isinstance(cities, list):
        return []
    return [city for city in cities if isinstance(city, Mapping) and city.get("city")]


def main() -> None:
    st.set_page_config(page_title="Atmos · AQI forecast", page_icon="◌", layout="wide", initial_sidebar_state="collapsed")
    inject_styles()
    render_topbar()
    payload, error = _fetch_dashboard()
    if error:
        st.error(error)
        st.info(f"Expected endpoint: {API_URL}")
        return
    cities = _city_payload(payload or {})
    available_names = [str(city["city"]) for city in cities]
    if not available_names:
        st.warning("No city readings were returned yet. The dashboard is ready once the API provides cities.")
        return
    selected_default = st.session_state.get("selected_city", available_names[0])
    selected = render_city_selector(available_names, selected_default)
    st.session_state["selected_city"] = selected
    chosen = next((city for city in cities if str(city.get("city")) == selected), cities[0])
    render_dashboard(chosen, ASSET_DIR)


if __name__ == "__main__":
    main()
"""Premium AQI dashboard presentation components.

This module intentionally contains presentation only. It does not import the
FastAPI application or prediction code; the Streamlit entrypoint supplies the
already-fetched dashboard payload.
"""

from __future__ import annotations

import base64
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Mapping, Sequence

import streamlit as st


CITY_META = {
    "Karachi": {"code": "KHI", "landmark": "karachi_landmark.png", "accent": "#ef9b55"},
    "Lahore": {"code": "LHE", "landmark": "lahore_landmark.png", "accent": "#db7c63"},
    "Islamabad": {"code": "ISB", "landmark": "islamabad_landmark.png", "accent": "#75a88b"},
}

AQI_RANGES = (
    ("0–50", "Good", "#70b68f"),
    ("51–100", "Moderate", "#e6bd65"),
    ("101–150", "Sensitive", "#e28b55"),
    ("151–200", "Unhealthy", "#d86d62"),
    ("201–300", "Very unhealthy", "#ae6b91"),
    ("300+", "Hazardous", "#744d68"),
)


def _text(value: Any, fallback: str = "—") -> str:
    return fallback if value is None or value == "" else str(value)


def _number(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value) if value is not None else fallback
    except (TypeError, ValueError):
        return fallback


def _whole(value: Any, fallback: int = 0) -> int:
    return round(_number(value, fallback))


def _fmt_time(value: Any) -> str:
    if not value:
        return "Live reading"
    raw = str(value)
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return parsed.strftime("%d %b %Y · %H:%M")
    except ValueError:
        return raw[:28]


def category_color(category: Any, aqi: Any = 0) -> str:
    label = str(category or "").lower()
    if "good" in label:
        return "#70b68f"
    if "moderate" in label:
        return "#e6bd65"
    if "sensitive" in label:
        return "#e28b55"
    if "unhealthy" in label and "very" not in label:
        return "#d86d62"
    if "very" in label:
        return "#ae6b91"
    if _number(aqi) > 300:
        return "#744d68"
    if _number(aqi) > 200:
        return "#ae6b91"
    if _number(aqi) > 150:
        return "#d86d62"
    if _number(aqi) > 100:
        return "#e28b55"
    if _number(aqi) > 50:
        return "#e6bd65"
    return "#70b68f"


def inject_styles() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --ink:#17252d; --muted:#819099; --cream:#f7f4ed; --line:#e4e5df; }
.stApp { background:#e9ece7; color:var(--ink); font-family:'Manrope',sans-serif; animation:pageIn .7s ease both; }
header[data-testid="stHeader"], #MainMenu, footer { display:none !important; }
[data-testid="stAppViewContainer"] > .main { padding:0 2.5rem 3rem; }
[data-testid="stAppViewContainer"] .block-container { max-width:1440px; padding-top:1.5rem; }
.topbar { display:flex; align-items:center; justify-content:space-between; margin:0 0 1.2rem; }
.brand { display:flex; align-items:center; gap:.75rem; font-size:.75rem; letter-spacing:.18em; font-weight:800; text-transform:uppercase; }
.brand-mark { width:30px; height:30px; border:1px solid #98a5a3; border-radius:50%; display:grid; place-items:center; font-size:.65rem; color:#4d686c; }
.live-pill { color:#56766c; font:500 .67rem 'DM Mono',monospace; letter-spacing:.08em; text-transform:uppercase; }
.live-dot { display:inline-block; width:7px; height:7px; border-radius:50%; background:#75b192; box-shadow:0 0 0 4px #d6e8db; margin-right:7px; }
.hero { min-height:548px; position:relative; overflow:hidden; border-radius:30px; color:#f8f5ed; background:#1b3037; box-shadow:0 26px 55px rgba(22,39,42,.18); }
.hero:after { content:''; position:absolute; inset:auto -8% -18% -8%; height:230px; background:var(--cream); border-radius:50% 50% 0 0 / 42% 42% 0 0; z-index:2; }
.hero-content { position:relative; z-index:3; display:grid; grid-template-columns:1.05fr .9fr 1.1fr; padding:3rem 3.1rem 12.5rem; min-height:365px; }
.eyebrow { margin:0 0 .75rem; color:#a7b9b6; font:500 .68rem 'DM Mono',monospace; letter-spacing:.2em; text-transform:uppercase; }
.city-name { margin:0; font-size:clamp(3rem,6vw,6.8rem); line-height:.88; letter-spacing:-.08em; font-weight:800; text-transform:uppercase; }
.city-code { margin-top:1.35rem; color:#b5c5c1; font:500 .72rem 'DM Mono',monospace; letter-spacing:.13em; }
.weather { align-self:center; padding:2.5rem 0 0 1rem; }
.temp { font-size:3.2rem; line-height:1; font-weight:700; letter-spacing:-.08em; }
.weather-copy { color:#c4d0c9; margin-top:.8rem; font-size:.78rem; line-height:1.8; }
.category-side { text-align:right; padding-top:1rem; }
.category-label { color:#b8c7c2; font:500 .66rem 'DM Mono',monospace; letter-spacing:.2em; text-transform:uppercase; }
.category { max-width:430px; margin:.75rem 0 0 auto; color:#fff; font-size:clamp(2.1rem,4vw,4.65rem); line-height:.91; letter-spacing:-.07em; font-weight:800; text-transform:uppercase; }
.landmark-wrap { position:absolute; z-index:4; right:5%; bottom:13%; width:min(39%,530px); height:360px; display:flex; justify-content:center; align-items:flex-end; animation:landmarkRise 1.1s cubic-bezier(.2,.85,.25,1) 2s both; }
.landmark { max-width:100%; max-height:100%; width:auto; height:auto; object-fit:contain; filter:drop-shadow(0 22px 25px rgba(0,0,0,.38)) brightness(.98); }
.landmark-missing { width:270px; height:210px; border:1px dashed #849694; border-radius:18px; display:grid; place-items:center; color:#bfd0ca; font:500 .72rem 'DM Mono',monospace; text-transform:uppercase; letter-spacing:.12em; text-align:center; padding:1rem; }
.cloud { position:absolute; z-index:5; width:92px; height:24px; border-radius:30px; background:rgba(255,255,255,.82); filter:blur(.2px); opacity:.82; }
.cloud:before,.cloud:after { content:''; position:absolute; border-radius:50%; background:inherit; }
.cloud:before { width:38px; height:38px; left:17px; top:-19px; }.cloud:after { width:52px; height:52px; right:11px; top:-28px; }
.cloud.one { right:25%; bottom:35%; animation:floatA 8s ease-in-out infinite; }.cloud.two { right:7%; bottom:49%; transform:scale(.66); animation:floatB 11s ease-in-out 1s infinite; }.cloud.three { right:38%; bottom:52%; transform:scale(.43); opacity:.55; animation:floatA 13s ease-in-out 3s infinite; }
.hero-bottom { position:absolute; z-index:6; left:3.1rem; right:3.1rem; bottom:1.55rem; display:flex; justify-content:space-between; align-items:end; color:var(--ink); }
.hero-stat-label { color:#98a0a0; font:500 .66rem 'DM Mono',monospace; letter-spacing:.15em; text-transform:uppercase; }.hero-stat { margin-top:.3rem; font-size:2.1rem; font-weight:800; letter-spacing:-.07em; }.hero-stat small { font-size:.75rem; color:#9aa3a0; letter-spacing:0; font-weight:500; }
.scale { position:absolute; z-index:8; left:1.25rem; top:11.5rem; width:12px; display:flex; flex-direction:column; gap:2px; }
.scale-segment { position:relative; width:12px; height:46px; border-radius:7px; }.scale-segment span { position:absolute; left:21px; top:0; width:95px; color:#aab9b5; font:500 .55rem 'DM Mono',monospace; line-height:1.15; opacity:.85; }
.dashboard-section { background:var(--cream); padding:2.7rem 2.2rem 1rem; margin-top:-1px; }
.section-head { display:flex; justify-content:space-between; align-items:end; gap:1rem; margin:0 0 1.3rem; }.section-kicker { color:#a2aaa6; font:500 .65rem 'DM Mono',monospace; letter-spacing:.2em; text-transform:uppercase; }.section-title { margin:.3rem 0 0; font-size:1.75rem; letter-spacing:-.055em; font-weight:800; }
.aqi-overview { display:grid; grid-template-columns:minmax(230px,.85fr) 1.15fr; gap:1rem; margin-bottom:2.3rem; }.current-reading { min-height:245px; padding:1.8rem; border-radius:21px; background:#17282e; color:#f9f5ec; position:relative; overflow:hidden; }.current-reading:after { content:''; position:absolute; width:170px; height:170px; right:-50px; bottom:-80px; border:1px solid rgba(255,255,255,.1); border-radius:50%; box-shadow:0 0 0 28px rgba(255,255,255,.025),0 0 0 56px rgba(255,255,255,.025); }.reading-number { font-size:5.5rem; line-height:.9; letter-spacing:-.1em; font-weight:800; margin-top:1.2rem; }.reading-meta { margin-top:1.3rem; display:flex; flex-wrap:wrap; align-items:center; gap:.65rem; font-size:.72rem; color:#c6d1c9; }.badge { display:inline-block; padding:.42rem .65rem; border-radius:999px; color:#15262c; font-size:.63rem; font-weight:800; text-transform:uppercase; letter-spacing:.07em; }.trend { font:500 .68rem 'DM Mono',monospace; }.trend.up { color:#e39579; }.trend.down { color:#83c198; }
.forecast-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:.75rem; align-items:stretch; }.forecast-card { min-height:245px; padding:1.5rem; border:1px solid var(--line); border-radius:20px; background:#fffefa; animation:cardUp .7s ease both; transition:transform .25s,box-shadow .25s; }.forecast-card:nth-child(2){animation-delay:.1s}.forecast-card:nth-child(3){animation-delay:.2s}.forecast-card:hover { transform:translateY(-5px); box-shadow:0 15px 28px rgba(31,51,50,.1); }.forecast-time { color:#a0aaa7; font:500 .65rem 'DM Mono',monospace; text-transform:uppercase; letter-spacing:.13em; }.forecast-number { margin:1.65rem 0 .5rem; font-size:3rem; font-weight:800; letter-spacing:-.09em; }.forecast-category { font-size:.72rem; font-weight:800; text-transform:uppercase; }.forecast-detail { margin-top:1.1rem; color:#8b9691; font:500 .64rem 'DM Mono',monospace; line-height:1.8; }
.metric-grid { display:grid; grid-template-columns:repeat(5,1fr); gap:.65rem; }.metric-card { padding:1rem; border:1px solid var(--line); border-radius:16px; background:#fffefa; transition:transform .2s, border-color .2s; }.metric-card:hover { transform:translateY(-3px); border-color:#9ab4ac; }.metric-icon { display:grid; place-items:center; width:31px; height:31px; border-radius:10px; background:#e7eee8; color:#527c70; font:700 .62rem 'DM Mono',monospace; }.metric-name { margin-top:1rem; color:#929c98; font:500 .63rem 'DM Mono',monospace; text-transform:uppercase; letter-spacing:.06em; }.metric-value { margin-top:.3rem; font-size:1.35rem; font-weight:800; letter-spacing:-.06em; }.metric-unit { color:#a4aca8; font-size:.62rem; font-weight:500; letter-spacing:0; }
.lower-grid { display:grid; grid-template-columns:1.2fr .8fr; gap:1rem; margin-top:1.1rem; }.panel { padding:1.55rem; border:1px solid var(--line); border-radius:20px; background:#fffefa; }.panel-title { margin:0 0 1rem; font-size:.9rem; font-weight:800; letter-spacing:-.03em; }.dominant { display:flex; align-items:center; gap:.9rem; padding:.9rem; background:#f0f2ec; border-radius:14px; }.dominant strong { display:block; font-size:1rem; }.dominant span { display:block; margin-top:.22rem; color:#88938e; font-size:.68rem; line-height:1.45; }.explain-row { display:grid; grid-template-columns:1.1fr .7fr .7fr; gap:.5rem; padding:.8rem 0; border-top:1px solid var(--line); font-size:.68rem; }.explain-row:first-of-type{border-top:0}.explain-label{color:#9ba5a0;font:500 .58rem 'DM Mono',monospace;text-transform:uppercase}.explain-value{margin-top:.25rem;font-weight:700}.direction{color:#cd765e}
@keyframes pageIn{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}} @keyframes landmarkRise{from{opacity:0;transform:translateY(65px)}to{opacity:1;transform:none}} @keyframes cardUp{from{opacity:0;transform:translateY(24px)}to{opacity:1;transform:none}} @keyframes floatA{0%,100%{transform:translate3d(0,0,0)}50%{transform:translate3d(-18px,-9px,0)}} @keyframes floatB{0%,100%{margin-right:0}50%{margin-right:22px;margin-bottom:10px}}
@media(max-width:980px){[data-testid="stAppViewContainer"] > .main{padding:0 1rem 2rem}.hero-content{grid-template-columns:1fr 1fr;padding:2rem 2rem 11rem}.weather{padding:1rem 0 0}.category-side{grid-column:2}.landmark-wrap{width:46%;right:3%;height:270px}.hero-bottom{left:2rem;right:2rem}.dashboard-section{padding:2rem 1.2rem 1rem}.metric-grid{grid-template-columns:repeat(3,1fr)}.lower-grid{grid-template-columns:1fr}}
@media(max-width:620px){.topbar{margin-bottom:.7rem}.hero{border-radius:20px;min-height:720px}.hero-content{display:block;padding:2rem 1.4rem 13rem}.city-name{font-size:3.6rem}.weather{padding:3.5rem 0 0}.category-side{text-align:left;padding-top:3rem}.category{margin:.6rem 0 0;font-size:2.55rem}.landmark-wrap{right:3%;bottom:16%;width:75%;height:245px}.scale{left:.65rem;top:9.5rem}.hero-bottom{left:1.4rem;right:1.4rem;bottom:1.3rem}.hero-stat{font-size:1.5rem}.aqi-overview{grid-template-columns:1fr}.forecast-grid{grid-template-columns:1fr 1fr}.forecast-card:last-child{grid-column:span 2}.metric-grid{grid-template-columns:repeat(2,1fr)}}
</style>
""",
        unsafe_allow_html=True,
    )


def render_topbar() -> None:
    st.markdown(
        '<div class="topbar"><div class="brand"><span class="brand-mark">AQ</span><span>Atmos / forecast lab</span></div><div class="live-pill"><span class="live-dot"></span>Model telemetry · live</div></div>',
        unsafe_allow_html=True,
    )


def render_city_selector(cities: Sequence[str], selected: str) -> str:
    options = [city for city in ("Karachi", "Lahore", "Islamabad") if city in cities] or ["Karachi", "Lahore", "Islamabad"]
    return st.selectbox("City", options, index=options.index(selected) if selected in options else 0, label_visibility="collapsed")


def render_aqi_scale() -> str:
    segments = "".join(f'<div class="scale-segment" style="background:{color}"><span>{rng}<br>{label}</span></div>' for rng, label, color in AQI_RANGES)
    return f'<div class="scale" aria-label="AQI scale">{segments}</div>'


def render_hero(city: Mapping[str, Any], asset_dir: Path) -> None:
    name = _text(city.get("city"), "Karachi")
    current = city.get("current") or {}
    aqi = _whole(current.get("aqi"))
    category = _text(current.get("category"), "Awaiting reading")
    color = category_color(category, aqi)
    meta = CITY_META.get(name, CITY_META["Karachi"])
    image_path = asset_dir / meta["landmark"]
    if image_path.exists():
        try:
            encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
            image_html = f'<img class="landmark" src="data:image/png;base64,{encoded}" alt="{escape(name)} landmark">'
        except OSError:
            image_html = '<div class="landmark-missing">Landmark image unavailable</div>'
    else:
        image_html = '<div class="landmark-missing">Landmark image missing</div>'
    # The image is emitted by render_hero_with_image so binary encoding stays out of this renderer.
    st.markdown(
        f'''<section class="hero">{render_aqi_scale()}<div class="hero-content"><div><p class="eyebrow">Air quality index · {escape(meta["code"])}</p><h1 class="city-name">{escape(name)}</h1><p class="city-code">{escape(_fmt_time(city.get("last_updated")))}</p></div><div class="weather"><div class="temp">{_whole(current.get("temperature"))}°</div><div class="weather-copy">Local conditions<br>Humidity {_whole(current.get("humidity"))}% · Wind {_number(current.get("wind_speed")):.1f} m/s</div></div><div class="category-side"><div class="category-label">Current air quality</div><div class="category" style="color:{color}">{escape(category)}</div></div></div><div class="cloud one"></div><div class="cloud two"></div><div class="cloud three"></div><div class="landmark-wrap">{image_html}</div><div class="hero-bottom"><div><div class="hero-stat-label">Now / {escape(_text(city.get("trend", {}).get("label"), "steady trend"))}</div><div class="hero-stat">{aqi} <small>AQI</small></div></div><div><div class="hero-stat-label">Dominant pollutant</div><div class="hero-stat">{escape(_text((city.get("dominant_pollutant") or {}).get("name"), "—"))}</div></div></div></section>''',
        unsafe_allow_html=True,
    )


def render_hero_with_image(city: Mapping[str, Any], asset_dir: Path) -> None:
    # Kept as a compatibility helper for callers that used the earlier renderer.
    render_hero(city, asset_dir)


def render_forecasts(city: Mapping[str, Any]) -> None:
    forecast = city.get("forecast") or {}
    categories = city.get("forecast_categories") or {}
    confidence = city.get("forecast_confidence") or {}
    cards = []
    for key, title in (("24h", "Next 24 hours"), ("48h", "Next 48 hours"), ("72h", "Next 72 hours")):
        value = _whole(forecast.get(key))
        cat = _text(categories.get(key), "No category")
        details = confidence.get(key) or {}
        rmse = details.get("rmse")
        rmse_text = f"RMSE { _number(rmse):.1f}" if rmse is not None else "RMSE n/a"
        cards.append(f'<div class="forecast-card"><div class="forecast-time">{title}</div><div class="forecast-number">{value}</div><div class="forecast-category" style="color:{category_color(cat,value)}">{escape(cat)}</div><div class="forecast-detail">Confidence · {escape(_text(details.get("label"), "n/a"))}<br>{rmse_text}</div></div>')
    st.markdown('<div class="section-head"><div><div class="section-kicker">Projection window</div><h2 class="section-title">3-day forecast</h2></div></div><div class="forecast-grid">' + "".join(cards) + "</div>", unsafe_allow_html=True)


def render_overview(city: Mapping[str, Any]) -> None:
    current = city.get("current") or {}
    aqi = _whole(current.get("aqi"))
    category = _text(current.get("category"), "Unknown")
    trend = city.get("trend") or {}
    direction = str(trend.get("direction") or "steady").lower()
    pollutant = city.get("dominant_pollutant") or {}
    st.markdown(f'<div class="aqi-overview"><div class="current-reading"><div class="section-kicker">Current AQI</div><div class="reading-number">{aqi}</div><div class="reading-meta"><span class="badge" style="background:{category_color(category,aqi)}">{escape(category)}</span><span class="trend {"up" if direction == "up" else "down"}">{escape(_text(trend.get("label"), "Trend unavailable"))}</span></div></div><div class="panel"><h3 class="panel-title">Signal summary</h3><div class="dominant"><div class="metric-icon">{escape(_text(pollutant.get("name"), "AQI")[:5])}</div><div><strong>{escape(_text(pollutant.get("name"), "Dominant pollutant unavailable"))} · {_number(pollutant.get("value")):.1f}</strong><span>{escape(_text(pollutant.get("reason"), "No dominant-pollutant explanation was returned."))}</span></div></div><div style="margin-top:1rem;color:#8b9691;font-size:.7rem;line-height:1.6">{escape(_text(current.get("message"), "Your latest AQI reading is ready."))}</div></div></div>', unsafe_allow_html=True)
    render_forecasts(city)


def render_metrics(city: Mapping[str, Any]) -> None:
    current = city.get("current") or {}
    metrics = (("PM2.5", "pm25", "µg/m³", "PM"), ("PM10", "pm10", "µg/m³", "PM"), ("NO₂", "no2", "ppb", "NO"), ("SO₂", "so2", "ppb", "SO"), ("CO", "co", "ppm", "CO"), ("O₃", "o3", "ppb", "O₃"), ("Temperature", "temperature", "°C", "°"), ("Humidity", "humidity", "%", "RH"), ("Pressure", "pressure", "hPa", "P"), ("Wind speed", "wind_speed", "m/s", "W") )
    html = []
    for label, key, unit, icon in metrics:
        value = current.get(key)
        formatted = "—" if value is None else (f"{_number(value):.1f}" if isinstance(value, float) else str(value))
        html.append(f'<div class="metric-card"><div class="metric-icon">{escape(icon)}</div><div class="metric-name">{label}</div><div class="metric-value">{escape(formatted)} <span class="metric-unit">{unit}</span></div></div>')
    st.markdown('<div class="section-head"><div><div class="section-kicker">Atmospheric signals</div><h2 class="section-title">Pollutant profile</h2></div></div><div class="metric-grid">' + "".join(html) + "</div>", unsafe_allow_html=True)


def render_explainability(city: Mapping[str, Any]) -> None:
    explanation = city.get("explainability") or {}
    factors = explanation.get("top_factors") or []
    with st.expander("Why this forecast?", expanded=bool(factors)):
        st.caption(f"Method: {_text(explanation.get('method'), 'not available')} · {_text(explanation.get('note'), 'The API did not include a note.')}")
        if not factors:
            st.info("No SHAP/LIME factors were returned for this reading.")
            return
        rows = []
        for factor in factors:
            rows.append(f'<div class="explain-row"><div><div class="explain-label">Feature</div><div class="explain-value">{escape(_text(factor.get("feature")))}</div></div><div><div class="explain-label">Impact</div><div class="explain-value">{escape(_text(factor.get("impact")))}</div></div><div><div class="explain-label">Contribution</div><div class="explain-value">{_number(factor.get("contribution")):.1f}</div></div><div><div class="explain-label">Direction</div><div class="explain-value direction">{escape(_text(factor.get("direction")))}</div></div><div><div class="explain-label">Reason</div><div class="explain-value">{escape(_text(factor.get("reason")))}</div></div></div>')
        st.markdown("".join(rows), unsafe_allow_html=True)


def render_history(city: Mapping[str, Any]) -> None:
    history = city.get("history") or {}
    st.markdown('<div class="section-head"><div><div class="section-kicker">Observed movement</div><h2 class="section-title">AQI history</h2></div></div>', unsafe_allow_html=True)
    tabs = st.tabs(["Last 24h", "Last 7d", "Last 30d"])
    for tab, key in zip(tabs, ("last_24h", "last_7d", "last_30d")):
        with tab:
            points = history.get(key) or []
            clean = [{"timestamp": p.get("timestamp"), "AQI": _number(p.get("aqi"))} for p in points if isinstance(p, Mapping) and p.get("aqi") is not None]
            if not clean:
                st.info("No historical readings are available for this window.")
                continue
            try:
                import pandas as pd
                import plotly.express as px
                frame = pd.DataFrame(clean)
                fig = px.area(frame, x="timestamp", y="AQI", markers=True)
                fig.update_traces(line_color="#5d8f82", fillcolor="rgba(93,143,130,.16)", marker_color="#17282e")
                fig.update_layout(height=260, margin=dict(l=0,r=0,t=10,b=0), paper_bgcolor="#fffefa", plot_bgcolor="#fffefa", font=dict(family="DM Mono",size=10,color="#89938f"), xaxis_title=None, yaxis_title=None, hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
            except ImportError:
                st.line_chart({"AQI": [point["AQI"] for point in clean]})


def render_dashboard(city: Mapping[str, Any], asset_dir: Path) -> None:
    render_hero(city, asset_dir)
    st.markdown('<main class="dashboard-section">', unsafe_allow_html=True)
    render_overview(city)
    render_metrics(city)
    render_history(city)
    render_explainability(city)
    st.markdown('</main>', unsafe_allow_html=True)
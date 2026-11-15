import datetime as dt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="ELBTAL-T2M", page_icon="🌡️", layout="wide")
st.title("🌡️ ELBTAL-T2M")
st.caption("Prototyp 0.1 · echte Vorhersagedaten · noch kein 100-m-Downscaling")

SOUTH, NORTH = 51.28, 51.50
WEST, EAST = 13.10, 13.37
# Sparsames API-Abtastraster; kein physikalisches 100-m-Modell.
LATS = np.linspace(SOUTH, NORTH, 5)
LONS = np.linspace(WEST, EAST, 5)

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_forecast():
    coords = [(float(lat), float(lon)) for lat in LATS for lon in LONS]
    params = {
        "latitude": ",".join(f"{lat:.5f}" for lat, _ in coords),
        "longitude": ",".join(f"{lon:.5f}" for _, lon in coords),
        "hourly": "temperature_2m",
        "forecast_days": 2,
        "timezone": "Europe/Berlin",
    }
    response = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=45)
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict):
        payload = [payload]
    if len(payload) != len(coords):
        raise ValueError("Die API lieferte nicht die erwartete Anzahl an Rasterpunkten.")
    return coords, payload

if st.button("🌤️ Prognose jetzt abrufen", type="primary"):
    try:
        with st.spinner("Temperaturprognosen werden abgerufen …"):
            coords, payload = fetch_forecast()
        st.session_state["forecast"] = (coords, payload)
    except (requests.RequestException, ValueError, KeyError) as exc:
        st.error(f"Abruf fehlgeschlagen: {exc}")

if "forecast" not in st.session_state:
    st.info("Starte den Datenabruf über die Schaltfläche oben.")
    st.stop()

coords, payload = st.session_state["forecast"]
times = pd.to_datetime(payload[0]["hourly"]["time"])
now = pd.Timestamp.now(tz="Europe/Berlin").tz_localize(None)
valid = [i for i, t in enumerate(times) if now <= t <= now + pd.Timedelta(hours=6, minutes=59)]
if not valid:
    st.warning("Für die nächsten sechs Stunden sind derzeit keine Stundenwerte vorhanden.")
    st.stop()

step = st.select_slider("Prognosezeitpunkt (stündliche Originaldaten)", options=valid, value=valid[0], format_func=lambda i: times[i].strftime("%d.%m. %H:%M"))
values = np.array([item["hourly"]["temperature_2m"][step] for item in payload], dtype=float)
if not np.all(np.isfinite(values)):
    st.error("Einige Temperaturwerte fehlen. Bitte später erneut versuchen.")
    st.stop()
z = values.reshape(len(LATS), len(LONS))
fig = go.Figure(go.Heatmap(x=LONS, y=LATS, z=z, colorscale="RdYlBu_r", colorbar=dict(title="°C"), hovertemplate="Länge: %{x:.3f}°<br>Breite: %{y:.3f}°<br>T2M: %{z:.1f} °C<extra></extra>"))
fig.update_layout(title=f"2-m-Temperatur · {times[step]:%d.%m.%Y %H:%M} (Ortszeit)", xaxis_title="Längengrad (°O)", yaxis_title="Breitengrad (°N)", height=520, margin=dict(l=20,r=20,t=60,b=20))
st.plotly_chart(fig, use_container_width=True)
a, b, c = st.columns(3)
a.metric("Minimum", f"{values.min():.1f} °C")
b.metric("Maximum", f"{values.max():.1f} °C")
c.metric("Abtastpunkte", str(len(values)))
st.warning("Wichtig: Die API kann benachbarte Koordinaten derselben Modellzelle zuordnen. Die Karte zeigt nur eine Visualisierung der gelieferten Werte, keine unabhängige hochauflösende Prognose. Die 100-m-Korrekturen, Geländedaten und 15-Minuten-Zeitschritte folgen in späteren Versionen.")
st.caption("Quelle: Open-Meteo Forecast API. Lizenz-/Attributionsbedingungen des jeweiligen Datenanbieters beachten.")

"""DWD CDC hourly 2-m temperature observations for KRUMET verification."""
import io
import re
import zipfile
import pandas as pd
import requests
import streamlit as st

BASE = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/air_temperature/recent/"
HEADERS = {"User-Agent": "KRUMET weather research / station verification"}

@st.cache_data(ttl=86400, show_spinner=False)
def stations():
    response = requests.get(BASE + "TU_Stundenwerte_Beschreibung_Stationen.txt", headers=HEADERS, timeout=25)
    response.raise_for_status()
    result = []
    for line in response.content.decode("latin-1").splitlines()[2:]:
        # DWD fixed-width station metadata: ID, dates, height, latitude, longitude, name.
        try:
            fields = line.split(maxsplit=6)
            if len(fields) < 7:
                continue
            sid, _, _, h, la, lo, tail = fields
            height, lat, lon = float(h), float(la), float(lo)
            name = tail.strip()
            if sid.isdigit() and 50.0 <= lat <= 52.2 and 11.5 <= lon <= 15.4:
                result.append((sid, name, lat, lon, height))
        except (ValueError, IndexError):
            continue
    return sorted(result, key=lambda row: row[1])

@st.cache_data(ttl=3600, show_spinner=False)
def hourly_temperature(station_id):
    """Read DWD CDC recent hourly temperature ZIP, discovering its dated filename."""
    listing = requests.get(BASE, headers=HEADERS, timeout=25)
    listing.raise_for_status()
    # DWD CDC ZIPs have a date range in their filenames, not an '_akt' suffix.
    matches = re.findall(r'href=["\\\']([^"\\\']*stundenwerte_TU_' + re.escape(str(station_id).zfill(5)) + r'_[^"\\\']*?\\.zip)["\\\']', listing.text, re.I)
    if not matches:
        raise FileNotFoundError(f'Keine aktuelle DWD-Temperaturdatei für Station {station_id}')
    from urllib.parse import urljoin
    url = urljoin(BASE, sorted(matches)[-1])
    response = requests.get(url, headers=HEADERS, timeout=35)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        names = [n for n in archive.namelist() if n.startswith("produkt_tu_stunde_") and n.endswith(".txt")]
        if not names:
            raise ValueError("Keine stündlichen Temperaturdaten im DWD-Archiv")
        with archive.open(names[0]) as f:
            df = pd.read_csv(f, sep=";", skipinitialspace=True)
    df.columns = df.columns.str.strip()
    df["time"] = pd.to_datetime(df["MESS_DATUM"].astype(str).str.strip(), format="%Y%m%d%H", utc=True, errors="coerce")
    df["temperature"] = pd.to_numeric(df["TT_TU"], errors="coerce").replace(-999, float("nan"))
    return df[["time","temperature"]].dropna()


def station_grid_value(X, Y, Z, lat, lon):
    import numpy as np
    i = int(np.abs(Y[:,0]-lat).argmin())
    j = int(np.abs(X[0,:]-lon).argmin())
    return float(Z[i,j])


@st.cache_data(ttl=86400, show_spinner=False)
def archived_icon_d2(lat, lon, start_date, end_date, lead_days=1):
    """Archived fixed-lead ICON-D2 forecasts; never current forecasts or reanalysis."""
    params = {
        "latitude":lat, "longitude":lon, "start_date":str(start_date),
        "end_date":str(end_date), "models":"icon_d2",
        "hourly":",".join(f"{v}_previous_day{lead_days}" for v in
                           ("temperature_2m","wind_speed_10m","cloud_cover")),
        "timezone":"GMT", "temperature_unit":"celsius", "wind_speed_unit":"kmh",
    }
    response = requests.get("https://previous-runs-api.open-meteo.com/v1/forecast",
                            params=params, headers=HEADERS, timeout=40)
    response.raise_for_status()
    data = response.json().get("hourly",{})
    result = pd.DataFrame({"time":pd.to_datetime(data["time"],utc=True)})
    for field in ("temperature_2m","wind_speed_10m","cloud_cover"):
        key = f"{field}_previous_day{lead_days}"
        result[field] = pd.to_numeric(data.get(key,[]),errors="coerce")
    return result

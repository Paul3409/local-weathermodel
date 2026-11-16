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
            sid = line[0:5].strip()
            height = float(line[32:38].strip())
            lat = float(line[38:50].strip())
            lon = float(line[50:60].strip())
            name = line[60:101].strip()
            if sid.isdigit() and 50.0 <= lat <= 52.2 and 11.5 <= lon <= 15.4:
                result.append((sid, name, lat, lon, height))
        except (ValueError, IndexError):
            continue
    return sorted(result, key=lambda row: row[1])

@st.cache_data(ttl=3600, show_spinner=False)
def hourly_temperature(station_id):
    url = BASE + f"stundenwerte_TU_{station_id}_akt.zip"
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

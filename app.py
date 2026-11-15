import io
import time
import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm

st.set_page_config(page_title='ELBTAL-T2M', page_icon='🌡️', layout='wide')
st.title('🌡️ ELBTAL-T2M')
st.caption('Version 0.2 · statische PNG-Modellkarten · noch kein 100-m-Downscaling')
SOUTH, NORTH = 51.28, 51.50
WEST, EAST = 13.10, 13.37
LATS = np.linspace(SOUTH, NORTH, 5)
LONS = np.linspace(WEST, EAST, 5)

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_forecast():
    coords = [(float(lat), float(lon)) for lat in LATS for lon in LONS]
    payload = []
    for start in range(0, len(coords), 5):
        batch = coords[start:start + 5]
        params = {
            'latitude': ','.join(f'{lat:.5f}' for lat, _ in batch),
            'longitude': ','.join(f'{lon:.5f}' for _, lon in batch),
            'hourly': 'temperature_2m',
            'forecast_hours': 8,
            'timezone': 'Europe/Berlin',
        }
        for attempt in range(3):
            try:
                response = requests.get('https://api.open-meteo.com/v1/forecast', params=params, timeout=(10, 25))
                response.raise_for_status()
                result = response.json()
                if isinstance(result, dict):
                    result = [result]
                if len(result) != len(batch):
                    raise ValueError('Unerwartete Anzahl an Wetterdatenpunkten.')
                payload.extend(result)
                break
            except (requests.Timeout, requests.ConnectionError) as exc:
                if attempt == 2:
                    raise requests.RequestException(f'Datenabruf nach 3 Versuchen fehlgeschlagen: {exc}') from exc
                time.sleep(1.5 * (attempt + 1))
    return coords, payload

def make_png(z, valid_time, issued_time):
    # Statische Grafik mit echten 5x5-Daten; keine erfundenen Zwischenwerte.
    fig, ax = plt.subplots(figsize=(10, 10), dpi=160)
    fig.patch.set_facecolor('#f5f7fa')
    ax.set_facecolor('#eef2f5')
    finite = z[np.isfinite(z)]
    low = np.floor(finite.min()) - 1
    high = np.ceil(finite.max()) + 1
    if high <= low:
        high = low + 2
    levels = np.linspace(low, high, 17)
    norm = BoundaryNorm(levels, ncolors=256, clip=True)
    # pcolormesh zeigt die originalen Abtastzellen statt scheinbarer 100-m-Details.
    lon_edges = np.linspace(WEST-(LONS[1]-LONS[0])/2, EAST+(LONS[1]-LONS[0])/2, len(LONS)+1)
    lat_edges = np.linspace(SOUTH-(LATS[1]-LATS[0])/2, NORTH+(LATS[1]-LATS[0])/2, len(LATS)+1)
    mesh = ax.pcolormesh(lon_edges, lat_edges, z, cmap='coolwarm', norm=norm, shading='flat')
    # Isothermen nur soweit die sehr wenigen Stützpunkte sie hergeben.
    if finite.max()-finite.min() >= 1:
        contour_levels = np.arange(np.ceil(finite.min()), np.floor(finite.max())+0.01, 1.0)
        if len(contour_levels):
            cs = ax.contour(LONS, LATS, z, levels=contour_levels, colors='#283744', linewidths=0.7, alpha=0.8)
            ax.clabel(cs, fmt='%.0f°', fontsize=8, inline=True)
    places = [('Belgern',13.1246,51.4828),('Nünchritz',13.386,51.30),('Glaubitz',13.379,51.325)]
    for name, lon, lat in places:
        if WEST <= lon <= EAST and SOUTH <= lat <= NORTH:
            ax.plot(lon,lat,marker='o',markersize=3.5,color='#1b2632',markeredgecolor='white')
            ax.annotate(name,(lon,lat),xytext=(5,5),textcoords='offset points',fontsize=9,fontweight='bold',color='#1b2632',bbox={'facecolor':'white','alpha':0.7,'edgecolor':'none','pad':1.5})
    ax.set_xlim(WEST,EAST)
    ax.set_ylim(SOUTH,NORTH)
    ax.set_aspect(1/np.cos(np.deg2rad((NORTH+SOUTH)/2)))
    ax.set_xlabel('Längengrad (°O)')
    ax.set_ylabel('Breitengrad (°N)')
    ax.grid(alpha=.16, linewidth=.5)
    fig.suptitle('ELBTAL-T2M  |  2-m-Temperatur',fontsize=17,fontweight='bold',y=.98)
    ax.set_title(f'Gültig: {valid_time:%d.%m.%Y %H:%M} Uhr (Europe/Berlin)\nDatenabruf: {issued_time:%d.%m.%Y %H:%M} Uhr',fontsize=10,pad=14)
    cbar=fig.colorbar(mesh,ax=ax,orientation='horizontal',fraction=.045,pad=.11,aspect=35,ticks=np.linspace(low,high,9))
    cbar.set_label('2-m-Lufttemperatur (°C)',fontsize=11)
    fig.text(.5,.025,'Quelle: Open-Meteo · 25 Abtastpunkte · kein 100-m-Modell · statische Rasterkarte',ha='center',fontsize=8,color='#465568')
    fig.subplots_adjust(left=.11,right=.93,top=.87,bottom=.13)
    buffer=io.BytesIO()
    fig.savefig(buffer,format='png',dpi=160,facecolor=fig.get_facecolor())
    plt.close(fig)
    return buffer.getvalue()

if st.button('🌤️ Prognose jetzt abrufen',type='primary'):
    try:
        with st.spinner('Temperaturprognosen werden abgerufen …'):
            coords,payload=fetch_forecast()
        st.session_state['forecast']=(coords,payload,pd.Timestamp.now(tz='Europe/Berlin'))
    except (requests.RequestException,ValueError,KeyError) as exc:
        st.error(f'Abruf fehlgeschlagen: {exc}')

if 'forecast' not in st.session_state:
    st.info('Starte den Datenabruf über die Schaltfläche oben.')
    st.stop()
coords,payload,issued=st.session_state['forecast']
times=pd.to_datetime(payload[0]['hourly']['time'])
now=pd.Timestamp.now(tz='Europe/Berlin').tz_localize(None)
valid=[i for i,t in enumerate(times) if now <= t <= now+pd.Timedelta(hours=6,minutes=59)]
if not valid:
    st.warning('Keine passenden Stundenwerte verfügbar. Bitte erneut abrufen.')
    st.stop()
step=st.select_slider('Prognosezeitpunkt (stündliche Originaldaten)',options=valid,value=valid[0],format_func=lambda i:times[i].strftime('%d.%m. %H:%M'))
values=np.array([item['hourly']['temperature_2m'][step] for item in payload],dtype=float)
if not np.all(np.isfinite(values)):
    st.error('Einige Temperaturwerte fehlen.')
    st.stop()
z=values.reshape(len(LATS),len(LONS))
png=make_png(z,times[step],issued)
st.image(png,caption='Statische Modellkarte als PNG – kein Zoomen oder Klicken',use_container_width=True)
st.download_button('⬇️ Diese Karte als PNG speichern',data=png,file_name=f'ELBTAL_T2M_{times[step]:%Y%m%d_%H%M}.png',mime='image/png')
a,b,c=st.columns(3)
a.metric('Minimum',f'{values.min():.1f} °C')
b.metric('Maximum',f'{values.max():.1f} °C')
c.metric('Abtastpunkte',str(len(values)))
st.warning('Die Karte zeigt nur 25 abgefragte Modellwerte. Isothermen sind eine grafische Interpolation; es gibt noch keine unabhängigen 100-m-Prognosen. Gelände- und Landnutzungskorrekturen folgen später.')
st.caption('Quelle: Open-Meteo Forecast API. Lizenz- und Attributionsbedingungen der Datenanbieter beachten.')

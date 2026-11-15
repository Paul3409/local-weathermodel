"""ELBTAL-T2M v0.3 – manually triggered DWD ICON-D2 PNG maps."""
import bz2
import io
import tempfile
import os
import re
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from urllib.parse import urljoin, unquote

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import requests
import streamlit as st
from scipy.interpolate import griddata, RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from PIL import Image
import math
from eccodes import codes_grib_new_from_file, codes_get, codes_get_array, codes_release

BASE = 'https://opendata.dwd.de/weather/nwp/icon-d2/grib'
LAT_MIN, LAT_MAX = 51.28, 51.50
LON_MIN, LON_MAX = 13.10, 13.37
PLACES = {
    'Belgern': (51.483, 13.125),
    'Mühlberg': (51.434, 13.222),
    'Nünchritz': (51.299, 13.388),
    'Glaubitz': (51.325, 13.379),
    'Zeithain': (51.334, 13.338),
}
# Slightly wider longitude extent so the eastern towns remain visible.
LON_MAX = 13.40
TZ = ZoneInfo('Europe/Berlin')
HEADERS = {'User-Agent': 'ELBTAL-T2M/0.3 (educational weather visualization)'}


def fetch(url, timeout=35):
    r = requests.get(url, timeout=timeout, headers=HEADERS)
    r.raise_for_status()
    return r.content


def find_runs():
    """Use directory contents, not guessed initialization times; fallback across recent cycles."""
    now = datetime.now(timezone.utc)
    candidates = []
    for hours_ago in range(0, 31, 3):
        t = (now - timedelta(hours=hours_ago)).replace(minute=0, second=0, microsecond=0)
        t = t.replace(hour=t.hour // 3 * 3)
        if t not in candidates:
            candidates.append(t)
    return candidates


def list_run(run):
    """Read DWD directory index and find T_2M GRIB files without assuming suffix layout."""
    folder = f'{BASE}/{run:%H}/t_2m/'
    html = fetch(folder).decode('utf-8', 'replace')
    from html.parser import HTMLParser

    class HrefParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.links = []
        def handle_starttag(self, tag, attrs):
            if tag.lower() == 'a':
                self.links.extend(v for k, v in attrs if k.lower() == 'href' and v)

    parser = HrefParser()
    parser.feed(html)
    filenames = [unquote(link.split('?')[0].split('/')[-1]) for link in parser.links]
    expected_stamp = run.strftime('%Y%m%d%H')
    files = {}
    matching = []
    for name in filenames:
        lower = name.lower()
        if not (lower.endswith('.grib2.bz2') and 't_2m' in lower and expected_stamp in lower):
            continue
        matching.append(name)
        # DWD naming includes a 3-digit forecast hour immediately after run timestamp.
        match = re.search(r'_' + expected_stamp + r'_(\d{3})(?:_|\.)', lower)
        if match:
            lead = int(match.group(1))
            files[lead] = urljoin(folder, name)
    return files, filenames[:8], matching[:8]


def decode_grib(url):
    raw = fetch(url, timeout=65)
    if url.endswith('.bz2'):
        raw = bz2.decompress(raw)
    messages = []
    # ecCodes requires a real file descriptor; BytesIO.fileno() is unsupported.
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.grib2', delete=False) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name
        del raw
        with open(tmp_path, 'rb') as fh:
            while True:
                gid = codes_grib_new_from_file(fh)
                if gid is None:
                    break
                try:
                    lats = np.asarray(codes_get_array(gid, 'latitudes'), dtype=float)
                    lons = np.asarray(codes_get_array(gid, 'longitudes'), dtype=float)
                    vals = np.asarray(codes_get_array(gid, 'values'), dtype=float) - 273.15
                    valid_date = int(codes_get(gid, 'validityDate'))
                    valid_time = int(codes_get(gid, 'validityTime'))
                    valid = datetime.strptime(f'{valid_date:08d}{valid_time:04d}', '%Y%m%d%H%M').replace(tzinfo=timezone.utc)
                    messages.append((lats, lons, vals, valid))
                finally:
                    codes_release(gid)
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    if not messages:
        raise ValueError('Keine GRIB-Nachrichten gefunden.')
    return messages


def interpolate(lats, lons, vals):
    # Only a buffered area is needed; keep the download decoding manageable.
    keep = (lats >= LAT_MIN-.09) & (lats <= LAT_MAX+.09) & (lons >= LON_MIN-.09) & (lons <= LON_MAX+.09)
    if keep.sum() < 6:
        raise ValueError('Zu wenige ICON-D2-Rasterpunkte im Modellgebiet.')
    lat_axis = np.linspace(LAT_MIN, LAT_MAX, 245)
    lon_axis = np.linspace(LON_MIN, LON_MAX, 235)
    X, Y = np.meshgrid(lon_axis, lat_axis)
    # Linear spatial interpolation, not physical 100 m downscaling.
    grid = griddata(np.column_stack([lons[keep], lats[keep]]), vals[keep], (X, Y), method='linear')
    if np.isnan(grid).any():
        nearest = griddata(np.column_stack([lons[keep], lats[keep]]), vals[keep], (X, Y), method='nearest')
        grid = np.where(np.isnan(grid), nearest, grid)
    return X, Y, grid


@st.cache_data(ttl=86400, show_spinner=False)
def get_terrain():
    """Load public Terrarium elevation PNG tiles and resample to our 100-m-ish grid.

    Tiles are elevation datasets, not weather observations. If unavailable, raise
    an error rather than silently generating synthetic elevation.
    """
    z = 11
    lat_axis = np.linspace(LAT_MIN, LAT_MAX, 245)
    lon_axis = np.linspace(LON_MIN, LON_MAX, 235)
    X, Y = np.meshgrid(lon_axis, lat_axis)
    scale = 2 ** z
    xpix = (X + 180.0) / 360.0 * scale * 256
    latitude_rad = np.deg2rad(Y)
    ypix = (1 - np.arcsinh(np.tan(latitude_rad)) / np.pi) / 2 * scale * 256
    x0 = int(np.floor(np.min(xpix) / 256))
    x1 = int(np.floor(np.max(xpix) / 256))
    y0 = int(np.floor(np.min(ypix) / 256))
    y1 = int(np.floor(np.max(ypix) / 256))
    mosaic = np.zeros(((y1-y0+1)*256, (x1-x0+1)*256), dtype=np.float32)
    for tx in range(x0, x1+1):
        for ty in range(y0, y1+1):
            url = f'https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{tx}/{ty}.png'
            response = requests.get(url, headers=HEADERS, timeout=30)
            response.raise_for_status()
            rgb = np.asarray(Image.open(io.BytesIO(response.content)).convert('RGB'), dtype=np.float32)
            height = rgb[...,0]*256 + rgb[...,1] + rgb[...,2]/256 - 32768
            mosaic[(ty-y0)*256:(ty-y0+1)*256, (tx-x0)*256:(tx-x0+1)*256] = height
    from scipy.ndimage import map_coordinates
    terrain = map_coordinates(mosaic, [ypix-y0*256, xpix-x0*256], order=1, mode='nearest')
    if not np.isfinite(terrain).all() or terrain.min() < -100 or terrain.max() > 1500:
        raise ValueError('Geländedaten außerhalb des plausiblen Wertebereichs')
    return terrain


def downscale_temperature(Z, terrain, lapse_rate=0.0065):
    """Experimental high-pass terrain correction, NOT a validated T2M forecast.

    A smooth 2.2 km terrain reference avoids adding a domain-wide height offset.
    It does not replace ICON's native model orography or represent cold-air drainage.
    """
    mean_lat = (LAT_MIN + LAT_MAX) / 2
    dy_m = (LAT_MAX-LAT_MIN)*111_200/(terrain.shape[0]-1)
    dx_m = (LON_MAX-LON_MIN)*111_200*math.cos(math.radians(mean_lat))/(terrain.shape[1]-1)
    sigma_y = 2200 / 2.355 / dy_m
    sigma_x = 2200 / 2.355 / dx_m
    coarse_terrain = gaussian_filter(terrain, sigma=(sigma_y,sigma_x), mode='nearest')
    correction = np.clip(-lapse_rate*(terrain-coarse_terrain), -1.5, 1.5)
    return Z+correction, correction


def render_png(X, Y, Z, run, valid, mode="ICON-D2 interpoliert", correction=None):
    fig, ax = plt.subplots(figsize=(9, 9.3), dpi=150)
    fig.patch.set_facecolor('#f6f8fa')
    low = np.floor(np.nanmin(Z) * 2) / 2
    high = np.ceil(np.nanmax(Z) * 2) / 2
    if high - low < 2:
        mid = (low + high) / 2
        low, high = mid - 1, mid + 1
    levels = np.linspace(low, high, 21)
    filled = ax.contourf(X, Y, Z, levels=levels, cmap='coolwarm', extend='both')
    isolines = np.arange(np.floor(low), np.ceil(high)+.01, .5)
    contours = ax.contour(X, Y, Z, levels=isolines, colors='#273445', linewidths=.55, alpha=.75)
    ax.clabel(contours, inline=True, fontsize=7, fmt='%.1f')
    for name, (lat, lon) in PLACES.items():
        if LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX:
            ax.plot(lon, lat, marker='o', color='#101828', markersize=2.5)
            ax.annotate(name, (lon, lat), xytext=(4, 4), textcoords='offset points',
                        fontsize=8, weight='bold', color='#15263a',
                        bbox=dict(facecolor='white', alpha=.72, edgecolor='none', pad=1))
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_aspect(1 / np.cos(np.deg2rad((LAT_MIN+LAT_MAX)/2)))
    ax.set_xlabel('Längengrad (°O)')
    ax.set_ylabel('Breitengrad (°N)')
    ax.grid(color='white', alpha=.25, linewidth=.5)
    fig.suptitle('ELBTAL-T2M  |  2-m-Temperatur', fontsize=16, weight='bold', y=.98)
    ax.set_title(f'{mode}  ·  Lauf: {run:%d.%m.%Y %H:%M} UTC  ·  Gültig: {valid.astimezone(TZ):%d.%m.%Y %H:%M} MEZ/MESZ', fontsize=8, pad=12)
    cb = fig.colorbar(filled, ax=ax, orientation='horizontal', fraction=.055, pad=.09)
    cb.set_label('2-m-Lufttemperatur (°C)')
    fig.text(.5, .025, 'DWD ICON-D2 · Geländedaten: AWS Terrain Tiles (Terrarium) · experimentelle Höhenkorrektur' if correction is not None else 'DWD ICON-D2 · interpolierte Originalwerte',
             ha='center', fontsize=8, color='#48566a')
    out = io.BytesIO()
    fig.savefig(out, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    return out.getvalue()


st.set_page_config(page_title='ELBTAL-T2M', page_icon='🌡️', layout='centered')
st.title('🌡️ ELBTAL-T2M · v0.4')
st.caption('Statische DWD-ICON-D2-Temperaturkarten für Belgern, Mühlberg, Nünchritz, Glaubitz und Zeithain')
st.info('Experimentelles topografisches Downscaling: lokale Höhenabweichungen werden mit einer Temperaturabnahme von 0,65 °C pro 100 m korrigiert. Noch nicht anhand von Messstationen validiert; keine eigenständige physikalische Wettermodellrechnung.')
max_steps = st.select_slider('Anzahl der Prognosestunden', options=[1, 2, 3, 4, 5, 6], value=6)
if st.button('DWD-Prognose manuell berechnen', type='primary'):
    with st.status('Suche verfügbaren DWD-Modelllauf …', expanded=True) as status:
        try:
            selected = None
            diagnostics = []
            for run in find_runs():
                try:
                    files, examples, matching = list_run(run)
                    available = [h for h in range(0, max_steps+1) if h in files]
                    diagnostics.append(f'{run:%Y-%m-%d %H} UTC: {len(files)} Dateien; Stunden {available}; Beispiele: {matching or examples}')
                    if len(available) >= max(2, max_steps-1):
                        selected = (run, files, available)
                        break
                except requests.RequestException as exc:
                    diagnostics.append(f'{run:%Y-%m-%d %H} UTC: {type(exc).__name__}: {exc}')
            if selected is None:
                with st.expander('DWD-Diagnose (zum Kopieren)', expanded=True):
                    st.code('\n'.join(diagnostics))
                raise RuntimeError('Kein passender ICON-D2-Lauf gefunden. Siehe DWD-Diagnose oben.')
            run, files, hours = selected
            st.write(f'Gefundener Modelllauf: {run:%Y-%m-%d %H:%M} UTC')
            images = []
            terrain = None
            try:
                terrain = get_terrain()
                st.write(f'Geländedaten geladen: {terrain.min():.0f} bis {terrain.max():.0f} m')
            except Exception as terrain_error:
                st.warning(f'Geländedaten nicht verfügbar: {terrain_error}. Es werden nur ICON-D2-Karten erstellt.')
            for idx, hour in enumerate(hours):
                st.write(f'Lade und zeichne +{hour} h ({idx+1}/{len(hours)}) …')
                messages = decode_grib(files[hour])
                # Hourly files can contain multiple 15-minute messages: choose exact valid time.
                target = run + timedelta(hours=hour)
                msg = min(messages, key=lambda m: abs((m[3]-target).total_seconds()))
                X, Y, Z = interpolate(*msg[:3])
                original_png = render_png(X, Y, Z, run, msg[3])
                local_png = None
                if terrain is not None:
                    local, correction = downscale_temperature(Z, terrain)
                    local_png = render_png(X, Y, local, run, msg[3], mode='Lokale Höhenkorrektur · experimentell', correction=correction)
                images.append((hour, msg[3], original_png, local_png))
            st.session_state['images'] = images
            status.update(label='Temperaturkarten erstellt', state='complete')
        except Exception as exc:
            status.update(label='Abruf oder Verarbeitung fehlgeschlagen', state='error')
            st.error(f'{type(exc).__name__}: {exc}')

if 'images' in st.session_state:
    images = st.session_state['images']
    labels = [f'+{h} h · {v.astimezone(TZ):%H:%M}' for h, v, _, _ in images]
    choice = st.select_slider('Vorhersagekarte', options=range(len(images)),
                              format_func=lambda i: labels[i])
    h, valid, original_png, local_png = images[choice]
    variants = ['ICON-D2 (interpoliert)'] + (['Lokales Downscaling (experimentell)'] if local_png else [])
    variant = st.radio('Kartentyp', variants, horizontal=True)
    png = local_png if variant.startswith('Lokales') else original_png
    st.image(png, width='stretch')
    st.download_button('PNG herunterladen', data=png,
                       file_name=f'ELBTAL_T2M_{"LOCAL" if variant.startswith("Lokales") else "ICON"}_{valid:%Y%m%d_%H%M}_UTC.png', mime='image/png')

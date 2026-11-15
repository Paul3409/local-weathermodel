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
from scipy.interpolate import RegularGridInterpolator, griddata
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


def render_png(X, Y, Z, run, valid):
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
    fig.suptitle('ELBTAL-T2M  |  ICON-D2 2-m-Temperatur', fontsize=16, weight='bold', y=.98)
    ax.set_title(f'Modelllauf: {run:%d.%m.%Y %H:%M} UTC  ·  Gültig: {valid.astimezone(TZ):%d.%m.%Y %H:%M} MEZ/MESZ', fontsize=9, pad=12)
    cb = fig.colorbar(filled, ax=ax, orientation='horizontal', fraction=.055, pad=.09)
    cb.set_label('2-m-Lufttemperatur (°C)')
    fig.text(.5, .025, 'Quelle: DWD OpenData / ICON-D2 · interpolierte Modellwerte, kein 100-m-Downscaling',
             ha='center', fontsize=8, color='#48566a')
    out = io.BytesIO()
    fig.savefig(out, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    return out.getvalue()


st.set_page_config(page_title='ELBTAL-T2M', page_icon='🌡️', layout='centered')
st.title('🌡️ ELBTAL-T2M · v0.3.2')
st.caption('Statische DWD-ICON-D2-Temperaturkarten für Belgern, Mühlberg, Nünchritz, Glaubitz und Zeithain')
st.info('Diese Version interpoliert ICON-D2 auf ein feines Darstellungsraster. Eigenes lokales Downscaling folgt später.')
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
            for idx, hour in enumerate(hours):
                st.write(f'Lade und zeichne +{hour} h ({idx+1}/{len(hours)}) …')
                messages = decode_grib(files[hour])
                # Hourly files can contain multiple 15-minute messages: choose exact valid time.
                target = run + timedelta(hours=hour)
                msg = min(messages, key=lambda m: abs((m[3]-target).total_seconds()))
                X, Y, Z = interpolate(*msg[:3])
                images.append((hour, msg[3], render_png(X, Y, Z, run, msg[3])))
            st.session_state['images'] = images
            status.update(label='Temperaturkarten erstellt', state='complete')
        except Exception as exc:
            status.update(label='Abruf oder Verarbeitung fehlgeschlagen', state='error')
            st.error(f'{type(exc).__name__}: {exc}')

if 'images' in st.session_state:
    images = st.session_state['images']
    labels = [f'+{h} h · {v.astimezone(TZ):%H:%M}' for h, v, _ in images]
    choice = st.select_slider('Vorhersagekarte', options=range(len(images)),
                              format_func=lambda i: labels[i])
    h, valid, png = images[choice]
    st.image(png, width='stretch')
    st.download_button('PNG herunterladen', data=png,
                       file_name=f'ELBTAL_T2M_{valid:%Y%m%d_%H%M}_UTC.png', mime='image/png')

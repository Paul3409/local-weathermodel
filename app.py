"""ELBTAL-T2M v0.5.3 – ESA WorldCover offline land-use raster."""
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
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import requests
import streamlit as st
from scipy.interpolate import griddata, RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from matplotlib.path import Path as MplPath
from PIL import Image
import math
from pathlib import Path
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


# ESA WorldCover 2021 v200: 10-m land cover, sampled to the forecast grid.
# Local classes: 0 unknown, 1 trees/shrub, 2 crops/grass, 3 bare,
# 4 built-up, 5 water/wetland. These are NOT OSM polygon counts.
WORLDCOVER_URL = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
    "ESA_WorldCover_10m_2021_v200_N51E012_Map.tif"
)
WORLDCOVER_CLASSES = {10: 1, 20: 1, 30: 2, 40: 2, 50: 4,
                      60: 3, 70: 0, 80: 5, 90: 5, 95: 1, 100: 2}


def get_landuse():
    """Fetch only the Elbtal window from ESA's cloud-optimized GeoTIFF.

    Called exclusively by the one-time preparation button. The raster
    remains local to the Streamlit session until the NPZ is downloaded.
    """
    import rasterio
    from rasterio.windows import from_bounds, transform as window_transform
    from rasterio.transform import rowcol
    lat_axis = np.linspace(LAT_MIN, LAT_MAX, LANDUSE_SHAPE[0])
    lon_axis = np.linspace(LON_MIN, LON_MAX, LANDUSE_SHAPE[1])
    X, Y = np.meshgrid(lon_axis, lat_axis)
    try:
        with rasterio.Env(GDAL_HTTP_MAX_RETRY="2", GDAL_HTTP_RETRY_DELAY="2"):
            with rasterio.open(WORLDCOVER_URL) as ds:
                window = from_bounds(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX, ds.transform)
                window = window.round_offsets().round_lengths()
                source = ds.read(1, window=window, boundless=True, fill_value=0)
                transform = window_transform(window, ds.transform)
    except Exception as exc:
        raise RuntimeError(
            "ESA-WorldCover-Download fehlgeschlagen. Prüfe die Verbindung "
            "zum ESA-S3-Server und versuche es später erneut."
        ) from exc
    rows, cols = rowcol(transform, X.ravel(), Y.ravel())
    rows = np.asarray(rows, dtype=int)
    cols = np.asarray(cols, dtype=int)
    valid = (rows >= 0) & (rows < source.shape[0]) & (cols >= 0) & (cols < source.shape[1])
    sampled = np.zeros(X.size, dtype=np.uint8)
    sampled[valid] = source[rows[valid], cols[valid]]
    result = np.zeros(sampled.shape, dtype=np.uint8)
    for esa_class, local_class in WORLDCOVER_CLASSES.items():
        result[sampled == esa_class] = local_class
    raster = result.reshape(X.shape)
    if (raster > 0).mean() < 0.75:
        raise ValueError("WorldCover-Raster unvollständig; Datei wird nicht erstellt.")
    return raster, int((raster > 0).sum())


LANDUSE_FILE = Path(__file__).resolve().parent / 'data' / 'landuse.npz'
LANDUSE_SHAPE = (245, 235)


def load_landuse_file():
    """Load only the committed land-use raster; no Overpass request during forecasts."""
    if not LANDUSE_FILE.is_file():
        raise FileNotFoundError('data/landuse.npz fehlt. Zuerst unter „Landnutzung vorbereiten“ erzeugen und auf GitHub hochladen.')
    with np.load(LANDUSE_FILE, allow_pickle=False) as data:
        raster = data['landuse']
        bounds = data['bounds']
        count = int(data['polygon_count'])
    expected_bounds = np.array([LAT_MIN, LAT_MAX, LON_MIN, LON_MAX])
    if raster.shape != LANDUSE_SHAPE or raster.dtype != np.uint8:
        raise ValueError(f'Landnutzungsraster hat unerwartetes Format: {raster.shape} / {raster.dtype}')
    if not np.allclose(bounds, expected_bounds, rtol=0, atol=1e-8):
        raise ValueError('Landnutzungsdatei passt nicht zu den aktuellen Modellgrenzen.')
    if not np.isin(raster, list(range(6))).all():
        raise ValueError('Ungültige Landnutzungsklassen in der Datei.')
    return raster, count


def encode_landuse_file(raster, count):
    out = io.BytesIO()
    np.savez_compressed(out, landuse=raster,
                        bounds=np.array([LAT_MIN, LAT_MAX, LON_MIN, LON_MAX]),
                        polygon_count=np.array(count))
    return out.getvalue()


def landuse_temperature_correction(landuse, valid, amplitude=1.0):
    """Experimental, heuristic land-use signal, NOT a validated physical model.

    Assumes a generic calm clear night; no wind/cloud input available yet.
    Zero-centered spatial correction prevents artificial domain-wide bias.
    """
    hour = valid.astimezone(TZ).hour
    # Smooth day-night transition, largest magnitude near 03:00 local time.
    night = float(np.clip((np.cos((hour - 3) * 2 * np.pi / 24) + 0.2) / 1.2, 0, 1))
    # Illustrative coefficients in Celsius for each OSM category.
    effects = np.array([0.0, 0.15, -0.45, 0.05, 0.30, 0.25], dtype=float)
    correction = effects[landuse] * night * amplitude
    known = landuse > 0
    if np.any(known):
        correction[known] -= float(correction[known].mean())
    correction = gaussian_filter(correction, sigma=1.0)
    return np.clip(correction, -0.8, 0.8)


TEMP_STOPS = [(-20,'#9851BD'),(-15,'#4C207F'),(-10,'#16346E'),(-5,'#3377C6'),(0,'#B8E5F4'),(.4999,'#B8E5F4'),(.5,'#174C35'),(5,'#438A49'),(10,'#A9C64D'),(15,'#F0D64B'),(20,'#F5B16C')]
TEMP_CMAP = LinearSegmentedColormap.from_list('elbtal', [((t+20)/40,c) for t,c in TEMP_STOPS], N=4096)
TEMP_CMAP.set_under('#9851BD')
TEMP_CMAP.set_over('#F5B16C')
TEMP_NORM = Normalize(-20,20)

def draw_km_temperatures(ax, X, Y, Z):
    """Temperatures interpolated at points roughly 1 km apart; not observations."""
    dy = 1 / 111.2
    dx = 1 / (111.2 * math.cos(math.radians((LAT_MIN+LAT_MAX)/2)))
    latitudes = np.arange(LAT_MIN+dy/2, LAT_MAX, dy)
    longitudes = np.arange(LON_MIN+dx/2, LON_MAX, dx)
    points = np.array([(lat,lon) for lat in latitudes for lon in longitudes])
    values = RegularGridInterpolator((Y[:,0],X[0,:]),Z)(points)
    for (lat,lon),value in zip(points,values):
        # Keep a clear space around each named place.
        if any(abs(lat-p_lat) < .005 and abs(lon-p_lon) < .010 for p_lat,p_lon in PLACES.values()):
            continue
        if np.isfinite(value):
            ax.text(lon,lat,f'{value:.1f}',ha='center',va='center',fontsize=5.2,
                    color='#15263a',zorder=7,
                    bbox=dict(facecolor='white',alpha=.7,edgecolor='none',pad=.4))

def render_png(X, Y, Z, run, valid, mode="ICON-D2 interpoliert", correction=None, show_km_values=False):
    fig, ax = plt.subplots(figsize=(9, 9.3), dpi=150)
    fig.patch.set_facecolor('#f6f8fa')
    filled = ax.contourf(X,Y,Z,levels=np.arange(-20,20.001,.1),cmap=TEMP_CMAP,norm=TEMP_NORM,extend='both')
    zmin,zmax = float(np.nanmin(Z)),float(np.nanmax(Z))
    isolines = np.arange(np.ceil(zmin*2)/2,np.floor(zmax*2)/2+.01,.5)
    if len(isolines):
        ax.contour(X,Y,Z,levels=isolines,colors='#1c2836',linewidths=.65,alpha=.85)
    if show_km_values:
        draw_km_temperatures(ax,X,Y,Z)
    for name, (lat, lon) in PLACES.items():
        if LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX:
            ax.plot(lon, lat, marker='o', color='#101828', markersize=2.5, zorder=12)
            ax.annotate(name, (lon, lat), xytext=(4, 4), textcoords='offset points',
                        fontsize=8, weight='bold', color='#15263a',
                        bbox=dict(facecolor='white', alpha=.88, edgecolor='none', pad=1), zorder=13)
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_aspect(1 / np.cos(np.deg2rad((LAT_MIN+LAT_MAX)/2)))
    ax.set_xlabel('Längengrad (°O)')
    ax.set_ylabel('Breitengrad (°N)')
    ax.grid(color='white', alpha=.25, linewidth=.5)
    fig.suptitle('ELBTAL-T2M  |  2-m-Temperatur', fontsize=16, weight='bold', y=.98)
    ax.set_title(f'{mode}  ·  Lauf: {run:%d.%m.%Y %H:%M} UTC  ·  Gültig: {valid.astimezone(TZ):%d.%m.%Y %H:%M} MEZ/MESZ', fontsize=8, pad=12)
    cb = fig.colorbar(filled, ax=ax, orientation='horizontal', fraction=.055, pad=.09)
    cb.set_ticks([-20,-15,-10,-5,0,.5,5,10,15,20])
    cb.set_ticklabels(['−20','−15','−10','−5','0','0,5','5','10','15','20'])
    cb.ax.tick_params(labelsize=7)
    cb.ax.get_xticklabels()[4].set_ha('right')
    cb.ax.get_xticklabels()[5].set_ha('left')
    cb.set_label('2-m-Lufttemperatur (°C) · feste Skala; Frostnähe < 0,5 °C')
    fig.text(.5, .025, 'DWD ICON-D2 · Geländedaten: AWS Terrain Tiles (Terrarium) · experimentelle Höhenkorrektur' if correction is not None else 'DWD ICON-D2 · interpolierte Originalwerte',
             ha='center', fontsize=8, color='#48566a')
    out = io.BytesIO()
    fig.savefig(out, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    return out.getvalue()


def render_difference(X, Y, difference, run, valid):
    fig, ax = plt.subplots(figsize=(9, 9.3), dpi=150)
    fig.patch.set_facecolor('#f6f8fa')
    bound = max(.2, np.ceil(np.nanmax(np.abs(difference)) * 10) / 10)
    levels = np.linspace(-bound, bound, 21)
    filled = ax.contourf(X, Y, difference, levels=levels, cmap='RdBu_r', extend='both')
    for name, (lat, lon) in PLACES.items():
        if LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX:
            ax.plot(lon, lat, 'ko', markersize=2.5)
            ax.annotate(name, (lon, lat), xytext=(4, 4), textcoords='offset points',
                        fontsize=8, weight='bold', bbox=dict(facecolor='white', alpha=.75, edgecolor='none', pad=1))
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_aspect(1 / np.cos(np.deg2rad((LAT_MIN+LAT_MAX)/2)))
    ax.set_xlabel('Längengrad (°O)')
    ax.set_ylabel('Breitengrad (°N)')
    ax.grid(color='white', alpha=.25, linewidth=.5)
    fig.suptitle('ELBTAL-T2M  |  Differenzkarte', fontsize=16, weight='bold', y=.98)
    ax.set_title(f'Lokales Experiment minus ICON-D2 · Lauf: {run:%d.%m.%Y %H:%M} UTC · Gültig: {valid.astimezone(TZ):%d.%m.%Y %H:%M}', fontsize=8, pad=12)
    cb = fig.colorbar(filled, ax=ax, orientation='horizontal', fraction=.055, pad=.09)
    cb.set_label('Temperaturabweichung (°C)')
    fig.text(.5, .025, 'Heuristische Höhen- und WorldCover-Korrektur · keine validierte Prognose', ha='center', fontsize=8, color='#48566a')
    out = io.BytesIO()
    fig.savefig(out, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    return out.getvalue()


st.set_page_config(page_title='ELBTAL-T2M', page_icon='🌡️', layout='centered')
st.title('🌡️ ELBTAL-T2M · v0.5.5')
st.caption('Statische DWD-ICON-D2-Temperaturkarten für Belgern, Mühlberg, Nünchritz, Glaubitz und Zeithain')
st.info('Experimentelles Downscaling: Höhenkorrektur und optionale gespeicherte WorldCover-Landnutzung. Die Koeffizienten sind heuristisch und nicht validiert.')
with st.expander('Landnutzung einmalig vorbereiten', expanded=not LANDUSE_FILE.exists()):
    st.write('Einmalige ESA-WorldCover-Abfrage; anschließend die erzeugte NPZ-Datei unter `data/landuse.npz` in GitHub hochladen. Bei regulären Prognosen wird kein Landnutzungsserver aufgerufen.')
    if st.button('ESA-WorldCover-Raster einmalig abrufen'):
        try:
            with st.spinner('Lade ESA-WorldCover-Ausschnitt …'):
                raster, polygon_count = get_landuse()
                st.session_state['prepared_landuse'] = encode_landuse_file(raster, polygon_count)
            st.success(f'{polygon_count} klassifizierte Rasterzellen; {(raster > 0).mean():.1%} Rasterabdeckung. Datei jetzt herunterladen.')
        except Exception as exc:
            st.error(f'Einmaliger WorldCover-Abruf fehlgeschlagen: {type(exc).__name__}: {exc}. Bitte später erneut versuchen.')
    if 'prepared_landuse' in st.session_state:
        st.download_button('landuse.npz herunterladen', data=st.session_state['prepared_landuse'],
                           file_name='landuse.npz', mime='application/octet-stream')
if LANDUSE_FILE.exists():
    st.success('Gespeichertes Landnutzungsraster gefunden: data/landuse.npz')
else:
    st.warning('Noch keine gespeicherte Landnutzungsdatei. Prognosen funktionieren vorerst nur mit Höhenkorrektur.')
enable_landuse = st.checkbox('Gespeicherte Landnutzung verwenden (experimentell)', value=True)
start_hour, end_hour = st.select_slider('Prognosezeitraum auswählen (+0 bis +12 h)', options=list(range(13)), value=(0, 6), format_func=lambda h: f'+{h} h')
st.caption(f'Es werden nur die Stunden +{start_hour} bis +{end_hour} (einschließlich) heruntergeladen und gezeichnet.')
if st.button('DWD-Prognose manuell berechnen', type='primary'):
    with st.status('Suche verfügbaren DWD-Modelllauf …', expanded=True) as status:
        try:
            selected = None
            diagnostics = []
            for run in find_runs():
                try:
                    files, examples, matching = list_run(run)
                    requested = list(range(start_hour, end_hour + 1))
                    available = [h for h in requested if h in files]
                    diagnostics.append(f'{run:%Y-%m-%d %H} UTC: {len(files)} Dateien; Stunden {available}; Beispiele: {matching or examples}')
                    if len(available) == len(requested):
                        selected = (run, files, available)
                        break
                except requests.RequestException as exc:
                    diagnostics.append(f'{run:%Y-%m-%d %H} UTC: {type(exc).__name__}: {exc}')
            if selected is None:
                with st.expander('DWD-Diagnose (zum Kopieren)', expanded=True):
                    st.code('\n'.join(diagnostics))
                raise RuntimeError(f'Kein ICON-D2-Lauf mit allen Stunden +{start_hour} bis +{end_hour} gefunden. Siehe DWD-Diagnose oben.')
            run, files, hours = selected
            st.write(f'Gefundener Modelllauf: {run:%Y-%m-%d %H:%M} UTC')
            images = []
            terrain = None
            try:
                terrain = get_terrain()
                st.write(f'Geländedaten geladen: {terrain.min():.0f} bis {terrain.max():.0f} m')
            except Exception as terrain_error:
                st.warning(f'Geländedaten nicht verfügbar: {terrain_error}. Es werden nur ICON-D2-Karten erstellt.')
            landuse = None
            if enable_landuse and terrain is not None:
                try:
                    landuse, polygon_count = load_landuse_file()
                    st.write(f'WorldCover-Landnutzung geladen: {polygon_count} klassifizierte Rasterzellen; {(landuse > 0).mean():.0%} Rasterabdeckung')
                except Exception as land_error:
                    st.warning(f'Gespeicherte Landnutzung nicht verfügbar: {land_error}. Höhenkorrektur bleibt aktiv.')
            for idx, hour in enumerate(hours):
                st.write(f'Lade und zeichne +{hour} h ({idx+1}/{len(hours)}) …')
                messages = decode_grib(files[hour])
                # Hourly files can contain multiple 15-minute messages: choose exact valid time.
                target = run + timedelta(hours=hour)
                msg = min(messages, key=lambda m: abs((m[3]-target).total_seconds()))
                X, Y, Z = interpolate(*msg[:3])
                original_png = (render_png(X,Y,Z,run,msg[3]), render_png(X,Y,Z,run,msg[3],show_km_values=True))
                local_png = None
                combined_png = None
                diff_png = None
                if terrain is not None:
                    local, correction = downscale_temperature(Z, terrain)
                    local_png = tuple(render_png(X,Y,local,run,msg[3],mode='Lokale Höhenkorrektur · experimentell',correction=correction,show_km_values=flag) for flag in (False,True))
                    if landuse is not None:
                        lu_corr = landuse_temperature_correction(landuse, msg[3])
                        combined = local + lu_corr
                        combined_png = tuple(render_png(X,Y,combined,run,msg[3],mode='Höhe + WorldCover-Landnutzung · experimentell',correction=combined-Z,show_km_values=flag) for flag in (False,True))
                        diff_png = render_difference(X, Y, combined-Z, run, msg[3])
                images.append((hour, msg[3], original_png, local_png, combined_png, diff_png))
            st.session_state['images'] = images
            status.update(label='Temperaturkarten erstellt', state='complete')
        except Exception as exc:
            status.update(label='Abruf oder Verarbeitung fehlgeschlagen', state='error')
            st.error(f'{type(exc).__name__}: {exc}')

if 'images' in st.session_state:
    images = st.session_state['images']
    labels = [f'+{item[0]} h · {item[1].astimezone(TZ):%d.%m.%Y %H:%M %Z}' for item in images]
    choice = st.select_slider('Vorhersagekarte', options=range(len(images)),
                              format_func=lambda i: labels[i])
    st.caption(f'Ausgewählt: {labels[choice]}')
    h, valid, original_png, local_png, combined_png, diff_png = images[choice]
    variants = {'ICON-D2 (interpoliert)': original_png}
    if local_png is not None:
        variants['Höhenkorrektur'] = local_png
    if combined_png is not None:
        variants['Höhe + Landnutzung'] = combined_png
        variants['Differenz zu ICON-D2'] = diff_png
    variant = st.radio('Kartentyp', list(variants), horizontal=True)
    show_km_values = st.checkbox('Temperaturzahlen alle ca. 1 km anzeigen', value=False, help='Interpolierte Modelltemperaturen am jeweiligen Punkt, keine Messwerte.')
    selected_map = variants[variant]
    png = selected_map if variant == 'Differenz zu ICON-D2' else selected_map[int(show_km_values)]
    st.image(png, width='stretch')
    st.download_button('PNG herunterladen', data=png,
                       file_name=f'ELBTAL_T2M_{list(variants).index(variant)}_{valid:%Y%m%d_%H%M}_UTC.png', mime='image/png')

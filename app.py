from map_design import style_figure, frame_png
from app_version import VERSION
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
from krumet_verify import stations as dwd_stations, hourly_temperature as dwd_hourly_temperature, station_grid_value, archived_icon_d2

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


@st.cache_data(ttl=600,show_spinner=False)
def list_run(run, parameter="t_2m"):
    """Read DWD directory index and find T_2M GRIB files without assuming suffix layout."""
    folder = f'{BASE}/{run:%H}/{parameter}/'
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
        if not (lower.endswith('.grib2.bz2') and parameter in lower and expected_stamp in lower):
            continue
        matching.append(name)
        # DWD naming includes a 3-digit forecast hour immediately after run timestamp.
        match = re.search(r'_' + expected_stamp + r'_(\d{3})(?:_|\.)', lower)
        if match:
            lead = int(match.group(1))
            files[lead] = urljoin(folder, name)
    return files, filenames[:8], matching[:8]


def decode_grib(url, parameter='t_2m'):
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
                    vals = np.asarray(codes_get_array(gid, 'values'), dtype=float)
                    vals = vals - 273.15 if parameter in ('t_2m','td_2m') else (vals * 3.6 if parameter in ('vmax_10m',) else vals)
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


def interpolate(lats, lons, vals, grid_m=None, categorical=False):
    # Only a buffered area is needed; keep the download decoding manageable.
    keep = (lats >= LAT_MIN-.09) & (lats <= LAT_MAX+.09) & (lons >= LON_MIN-.09) & (lons <= LON_MAX+.09)
    if keep.sum() < 6:
        raise ValueError('Zu wenige ICON-D2-Rasterpunkte im Modellgebiet.')
    nlat = round((LAT_MAX-LAT_MIN)*111200/grid_m)+1 if grid_m else (125 if LAT_MAX-LAT_MIN > 1 else 245)
    nlon = round((LON_MAX-LON_MIN)*111200*math.cos(math.radians((LAT_MIN+LAT_MAX)/2))/grid_m)+1 if grid_m else (160 if LON_MAX-LON_MIN > 1 else 235)
    lat_axis = np.linspace(LAT_MIN, LAT_MAX, nlat)
    lon_axis = np.linspace(LON_MIN, LON_MAX, nlon)
    X, Y = np.meshgrid(lon_axis, lat_axis)
    # Linear spatial interpolation, not physical 100 m downscaling.
    grid = griddata(np.column_stack([lons[keep], lats[keep]]), vals[keep], (X, Y), method='nearest' if categorical else 'linear')
    if np.isnan(grid).any():
        nearest = griddata(np.column_stack([lons[keep], lats[keep]]), vals[keep], (X, Y), method='nearest')
        grid = np.where(np.isnan(grid), nearest, grid)
    return X, Y, grid


@st.cache_data(ttl=86400, show_spinner=False)
def get_terrain(bounds=None):
    """Load public Terrarium elevation PNG tiles and resample to our 100-m-ish grid.

    Tiles are elevation datasets, not weather observations. If unavailable, raise
    an error rather than silently generating synthetic elevation.
    """
    z = 9 if bounds is not None else 11
    lat_axis = np.linspace(LAT_MIN, LAT_MAX, 125 if bounds is not None else 245)
    lon_axis = np.linspace(LON_MIN, LON_MAX, 160 if bounds is not None else 235)
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


def krumet_cold_air_diagnostic(terrain, valid, u, v, cloud):
    """Heuristic nocturnal cold-air potential; not a validated forecast."""
    hour = valid.astimezone(TZ).hour
    night = max(0., np.cos((hour-3)*np.pi/12))
    dy = (LAT_MAX-LAT_MIN)*111200/(terrain.shape[0]-1)
    dx = (LON_MAX-LON_MIN)*111200*math.cos(math.radians((LAT_MIN+LAT_MAX)/2))/(terrain.shape[1]-1)
    valley = np.clip((gaussian_filter(terrain,(10000/dy,10000/dx))-terrain)/100,0,1)
    calm = np.clip((18-np.hypot(u,v)*3.6)/15,0,1)
    clear = np.clip((85-cloud)/70,0,1)
    return -1.5*night*valley*calm*clear


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

def draw_km_temperatures(ax, X, Y, Z, decimals=1):
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
            ax.text(lon,lat,f'{value:.{decimals}f}',ha='center',va='center',fontsize=5.2,
                    color='#15263a',zorder=7,
                    bbox=dict(facecolor='white',alpha=.7,edgecolor='none',pad=.4))

# Administrative overlays are cached and only fetched when KRUMET maps are rendered.
KRUMET_BORDER_BASE = 'https://raw.githubusercontent.com/isellsoap/deutschlandGeoJSON/main'
KRUMET_BORDER_FILES = {
    'Bundesländer': KRUMET_BORDER_BASE + '/2_bundeslaender/3_mittel.geo.json',
    'Landkreise': KRUMET_BORDER_BASE + '/4_kreise/4_niedrig.geo.json',
}

@st.cache_data(ttl=86400, show_spinner=False)
def load_krumet_border(kind):
    response = requests.get(KRUMET_BORDER_FILES[kind], headers=HEADERS, timeout=20)
    response.raise_for_status()
    return response.json()

def draw_krumet_borders(ax):
    """Overlay administrative borders without altering temperature fields."""
    for kind, color, width, alpha, zorder in (
        ('Landkreise', '#f1f5f9', .52, .82, 9),
        ('Bundesländer', '#17273d', 1.4, .95, 10),
    ):
        try:
            features = load_krumet_border(kind).get('features', [])
        except (requests.RequestException, ValueError):
            continue  # Forecast rendering must work even if a border source is offline.
        for feature in features:
            geometry = feature.get('geometry') or {}
            typ = geometry.get('type')
            coords = geometry.get('coordinates', [])
            polygons = [coords] if typ == 'Polygon' else (coords if typ == 'MultiPolygon' else [])
            for polygon in polygons:
                for ring in polygon:
                    if len(ring) < 2:
                        continue
                    arr = np.asarray(ring, dtype=float)
                    if arr.ndim != 2 or arr.shape[1] < 2:
                        continue
                    if (arr[:,0].max() < LON_MIN or arr[:,0].min() > LON_MAX or
                            arr[:,1].max() < LAT_MIN or arr[:,1].min() > LAT_MAX):
                        continue
                    ax.plot(arr[:,0], arr[:,1], color=color, linewidth=width,
                            alpha=alpha, zorder=zorder, solid_joinstyle='round')

def render_png(X, Y, Z, run, valid, mode="ICON-D2 interpoliert", correction=None, show_km_values=False, model_name='ICON-D2'):
    fig, ax = plt.subplots(figsize=(9, 9.3), dpi=150)
    fig.patch.set_facecolor('#f6f8fa')
    filled = ax.contourf(X,Y,Z,levels=np.arange(-20,20.001,.25),cmap=TEMP_CMAP,norm=TEMP_NORM,extend='both')
    zmin,zmax = float(np.nanmin(Z)),float(np.nanmax(Z))
    isolines = np.arange(np.ceil(zmin*2)/2,np.floor(zmax*2)/2+.01,.5)
    if len(isolines):
        ax.contour(X,Y,Z,levels=isolines,colors='#1c2836',linewidths=.65,alpha=.85)
    if show_km_values:
        draw_km_temperatures(ax,X,Y,Z)
    if model_name == 'KRUMET':
        draw_krumet_borders(ax)
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
    fig.suptitle(('KRUMET | ' if model_name=='KRUMET' else 'ELBTAL | ')+('2-m-Taupunkt' if 'Taupunkt' in mode else '2-m-Temperatur'), fontsize=16, weight='bold', y=.98)
    ax.set_title(f'{mode}  ·  Lauf: {run:%d.%m.%Y %H:%M} UTC  ·  Gültig: {valid.astimezone(TZ):%d.%m.%Y %H:%M} MEZ/MESZ', fontsize=8, pad=12)
    cb = fig.colorbar(filled, ax=ax, orientation='horizontal', fraction=.055, pad=.09)
    cb.set_ticks([-20,-15,-10,-5,0,.5,5,10,15,20])
    cb.set_ticklabels(['−20','−15','−10','−5','0','0,5','5','10','15','20'])
    cb.ax.tick_params(labelsize=7)
    cb.ax.get_xticklabels()[4].set_ha('right')
    cb.ax.get_xticklabels()[5].set_ha('left')
    cb.set_label('2-m-Taupunkt (°C) · feste Farbskala' if 'Taupunkt' in mode else '2-m-Lufttemperatur (°C) · feste Skala; Frostnähe < 0,5 °C')
    fig.text(.5, .025, 'DWD ICON-D2 · Geländedaten: AWS Terrain Tiles (Terrarium) · experimentelle Höhenkorrektur' if correction is not None else 'DWD ICON-D2 · interpolierte Originalwerte',
             ha='center', fontsize=8, color='#48566a')
    style_figure(fig, ax)
    out = io.BytesIO()
    fig.savefig(out, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    return frame_png(out.getvalue(), ('Taupunkt' if 'Taupunkt' in mode else 'Temperatur'), model_name, 'Sachsen + Umgebung' if model_name=='KRUMET' else 'Elbtal', run, valid)


def render_difference(X, Y, difference, run, valid, model_name='ICON-D2'):
    fig, ax = plt.subplots(figsize=(9, 9.3), dpi=150)
    fig.patch.set_facecolor('#f6f8fa')
    bound = max(.2, np.ceil(np.nanmax(np.abs(difference)) * 10) / 10)
    levels = np.linspace(-bound, bound, 21)
    filled = ax.contourf(X, Y, difference, levels=levels, cmap='RdBu_r', extend='both')
    if model_name == 'KRUMET':
        draw_krumet_borders(ax)
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
    fig.suptitle(('KRUMET' if model_name=='KRUMET' else 'ELBTAL-T2M')+' | Differenzkarte', fontsize=16, weight='bold', y=.98)
    ax.set_title(f'Lokales Experiment minus ICON-D2 · Lauf: {run:%d.%m.%Y %H:%M} UTC · Gültig: {valid.astimezone(TZ):%d.%m.%Y %H:%M}', fontsize=8, pad=12)
    cb = fig.colorbar(filled, ax=ax, orientation='horizontal', fraction=.055, pad=.09)
    cb.set_label('Temperaturabweichung (°C)')
    fig.text(.5, .025, 'Heuristische Höhen- und WorldCover-Korrektur · keine validierte Prognose', ha='center', fontsize=8, color='#48566a')
    style_figure(fig, ax)
    out = io.BytesIO()
    fig.savefig(out, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    return frame_png(out.getvalue(), 'Temperaturdifferenz', model_name, 'Sachsen + Umgebung' if model_name=='KRUMET' else 'Elbtal', run, valid)


GUST_STOPS = [(0,'#499FE1'),(10,'#204D99'),(20,'#347B4A'),(30,'#A3BE4B'),(40,'#E8D24A'),(50,'#F3A143'),(60,'#E96A38'),(70,'#DB3834'),(80,'#B9252E'),(90,'#811A2C'),(100,'#48205F'),(110,'#853AA8'),(120,'#C52D9A')]
GUST_CMAP = LinearSegmentedColormap.from_list('elbtal_gust', [(v/120,col) for v,col in GUST_STOPS], N=2048)
GUST_CMAP.set_under('#499FE1')
GUST_CMAP.set_over('#C52D9A')
GUST_NORM = Normalize(0,120)


def render_gust(X,Y,Z,run,valid,show_km_values=False):
    fig,ax = plt.subplots(figsize=(9,9.3),dpi=150)
    fig.patch.set_facecolor('#f6f8fa')
    filled = ax.contourf(X,Y,Z,levels=np.arange(0,120.01,1),cmap=GUST_CMAP,norm=GUST_NORM,extend='max')
    lo,hi = float(np.nanmin(Z)),float(np.nanmax(Z))
    contours = np.arange(max(0,np.ceil(lo/10)*10),min(120,np.floor(hi/10)*10)+.01,10)
    if len(contours):
        ax.contour(X,Y,Z,levels=contours,colors='#263345',linewidths=.6,alpha=.8)
    if show_km_values:
        draw_km_temperatures(ax,X,Y,Z,decimals=0)
    for name,(lat,lon) in PLACES.items():
        if LAT_MIN<=lat<=LAT_MAX and LON_MIN<=lon<=LON_MAX:
            ax.plot(lon,lat,'o',color='#101828',markersize=2.5,zorder=12)
            ax.annotate(name,(lon,lat),xytext=(4,4),textcoords='offset points',
                        fontsize=8,weight='bold',color='#15263a',zorder=13,
                        bbox=dict(facecolor='white',alpha=.88,edgecolor='none',pad=1))
    ax.set_xlim(LON_MIN,LON_MAX)
    ax.set_ylim(LAT_MIN,LAT_MAX)
    ax.set_aspect(1/np.cos(np.deg2rad((LAT_MIN+LAT_MAX)/2)))
    ax.set_xlabel('Längengrad (°O)')
    ax.set_ylabel('Breitengrad (°N)')
    ax.grid(color='white',alpha=.25,linewidth=.5)
    fig.suptitle('ELBTAL  |  Windböen',fontsize=16,weight='bold',y=.98)
    ax.set_title(f'ICON-D2 Windböen · Lauf: {run:%d.%m.%Y %H:%M} UTC · Gültig: {valid.astimezone(TZ):%d.%m.%Y %H:%M %Z}',fontsize=8,pad=12)
    cb=fig.colorbar(filled,ax=ax,orientation='horizontal',fraction=.055,pad=.09)
    cb.set_ticks(np.arange(0,121,10))
    cb.ax.tick_params(labelsize=7)
    cb.set_label('Windböen (km/h) · feste Farbskala')
    fig.text(.5,.025,'DWD ICON-D2 · interpolierte Windböen, keine lokale Geländekorrektur',
             ha='center',fontsize=8,color='#48566a')
    style_figure(fig, ax)
    out = io.BytesIO()
    fig.savefig(out,format='png',bbox_inches='tight',facecolor=fig.get_facecolor())
    plt.close(fig)
    return frame_png(out.getvalue(), 'Windböen', 'ICON-D2', 'Elbtal', run, valid)



PREC_RATE = [(0,'#F2F5F7'),(.1,'#C8EAF7'),(.5,'#80C9EC'),(1,'#368DD1'),(2,'#2269A9'),(5,'#3B9A68'),(10,'#C4C948'),(15,'#F1D34E'),(20,'#F3A04A'),(30,'#E85A38'),(50,'#C72F43'),(75,'#83215F'),(100,'#C137A8')]
PREC_SUM = [(0,'#F2F5F7'),(.5,'#D1F2ED'),(1,'#91DCD8'),(2,'#51B9D4'),(5,'#3787C1'),(10,'#2465A0'),(15,'#3D986A'),(20,'#A8C34E'),(30,'#E6D14B'),(40,'#F1A24D'),(50,'#E46840'),(75,'#C43D53'),(100,'#793A94')]
from significant_weather import classify_ww, palette, legend_handles


def classify_precipitation(hourly_total, hourly_rain, hourly_snow, hourly_graupel):
    """Classify hourly increments, not cumulative model-run totals.

    0 dry, 1 rain, 2 mixed rain/snow, 3 snow, 5 uncertain.
    Graupel is not silently counted as rain or snow. Freezing rain requires
    independent WW diagnostics and is not asserted here.
    """
    result = np.full(hourly_total.shape, 5, dtype=int)
    wet = hourly_total >= .05
    result[~wet] = 0
    known = hourly_rain + hourly_snow
    graupel_significant = hourly_graupel > np.maximum(.02, .1 * hourly_total)
    valid = wet & (known > .02) & ~graupel_significant
    # Unaccounted precipitation may include additional hydrometeors.
    valid &= np.abs(hourly_total - known - hourly_graupel) <= np.maximum(.1, .2 * hourly_total)
    snow_fraction = hourly_snow / np.maximum(known, 1e-9)
    result[valid & (snow_fraction < .1)] = 1
    result[valid & (snow_fraction >= .1) & (snow_fraction <= .9)] = 2
    result[valid & (snow_fraction > .9)] = 3
    return result



def render_precip(X,Y,Z,run,valid,parameter_page,show_km_values=False):
    from matplotlib.colors import ListedColormap, BoundaryNorm
    fig,ax=plt.subplots(figsize=(9,9.3),dpi=150)
    fig.patch.set_facecolor('#f6f8fa')
    if parameter_page == '🌦️ Signifikantes Wetter':
        cmap,norm,bounds=palette()
        filled=ax.pcolormesh(X,Y,Z,cmap=cmap,norm=norm,shading="nearest",rasterized=True)
        # The grouped WW legend is drawn by the shared navy frame.
        fig.subplots_adjust(left=.11,right=.97,top=.95,bottom=.10)
    else:
        stops=PREC_RATE if parameter_page=='🌧️ Rate' else PREC_SUM
        vmax=100
        cmap=LinearSegmentedColormap.from_list('precip',[(v/vmax,col) for v,col in stops],N=2048)
        cmap.set_over(stops[-1][1])
        norm=Normalize(0,vmax)
        filled=ax.contourf(X,Y,Z,levels=np.linspace(0,vmax,151),cmap=cmap,norm=norm,extend='max')
        zmin,zmax=float(np.nanmin(Z)),float(np.nanmax(Z))
        boundaries=[v for v,_ in stops[1:] if zmin<v<zmax]
        if boundaries:
            ax.contour(X,Y,Z,levels=boundaries,colors='#293748',linewidths=.5,alpha=.7)
        if show_km_values:
            draw_km_temperatures(ax,X,Y,Z)
        cb=fig.colorbar(filled,ax=ax,orientation='horizontal',fraction=.055,pad=.09)
        cb.set_ticks([0,10,20,30,50,75,100])
        cb.ax.tick_params(labelsize=7)
        cb.set_label('mm/h (stündliches Mittel)' if parameter_page=='🌧️ Rate' else 'mm = l/m² (seit Modellstart)')
    for name,(lat,lon) in PLACES.items():
        if LAT_MIN<=lat<=LAT_MAX and LON_MIN<=lon<=LON_MAX:
            ax.plot(lon,lat,'o',color='#101828',markersize=2.5,zorder=12)
            ax.annotate(name,(lon,lat),xytext=(4,4),textcoords='offset points',
                        fontsize=8,weight='bold',color='#15263a',zorder=13,
                        bbox=dict(facecolor='white',alpha=.9,edgecolor='none',pad=1))
    ax.set_xlim(LON_MIN,LON_MAX)
    ax.set_ylim(LAT_MIN,LAT_MAX)
    ax.set_aspect(1/np.cos(np.deg2rad((LAT_MIN+LAT_MAX)/2)))
    ax.set_xlabel('Längengrad (°O)')
    ax.set_ylabel('Breitengrad (°N)')
    ax.grid(color='white',alpha=.2,linewidth=.5)
    fig.suptitle('ELBTAL  |  '+parameter_page,fontsize=16,weight='bold',y=.98)
    ax.set_title(f'ICON-D2 · Lauf {run:%d.%m.%Y %H:%M} UTC · {valid.astimezone(TZ):%d.%m.%Y %H:%M %Z}',fontsize=8,pad=12)
    fig.text(.5,.025,'DWD ICON-D2 · Modellwerte, keine Messungen',ha='center',fontsize=8,color='#48566a')
    if parameter_page == '🌦️ Signifikantes Wetter':
        fig.suptitle('')
        ax.set_title('')
    style_figure(fig, ax)
    out = io.BytesIO()
    fig.savefig(out,format='png',bbox_inches='tight',facecolor=fig.get_facecolor())
    plt.close(fig)
    return frame_png(out.getvalue(), parameter_page, 'ICON-D2', 'Elbtal', run, valid)



IMAGE_KEYS = {'🌡️ Temperatur':'images_temp','🌬️ Windböen':'images_gust',
              '🌧️ Rate':'images_rate','💧 Summe':'images_sum','🌦️ Signifikantes Wetter':'images_kind',
              '💦 Taupunkt':'images_dew','🌫️ Nebel':'images_fog'}


def render_fog(X,Y,Z,run,valid):
    from matplotlib.colors import ListedColormap, BoundaryNorm
    colors=['#E8EDF1','#C7E6D7','#72B6A6','#346C89','#B39A80']
    cmap=ListedColormap(colors)
    fig,ax=plt.subplots(figsize=(9,9.3),dpi=150)
    fig.patch.set_facecolor('#f6f8fa')
    bounds=np.arange(-.5,5.5,1)
    filled=ax.contourf(X,Y,Z,levels=bounds,cmap=cmap,norm=BoundaryNorm(bounds,5))
    cb=fig.colorbar(filled,ax=ax,orientation='horizontal',fraction=.055,pad=.09)
    cb.set_ticks(range(5))
    cb.set_ticklabels(['Gering','Mäßig','Hoch','Sehr hoch','Regen'])
    cb.ax.tick_params(labelsize=8)
    for name,(lat,lon) in PLACES.items():
        if LAT_MIN<=lat<=LAT_MAX and LON_MIN<=lon<=LON_MAX:
            ax.plot(lon,lat,'o',color='#101828',markersize=2.5,zorder=12)
            ax.annotate(name,(lon,lat),xytext=(4,4),textcoords='offset points',
                        fontsize=8,weight='bold',color='#15263a',zorder=13,
                        bbox=dict(facecolor='white',alpha=.9,edgecolor='none',pad=1))
    ax.set_xlim(LON_MIN,LON_MAX)
    ax.set_ylim(LAT_MIN,LAT_MAX)
    ax.set_aspect(1/np.cos(np.deg2rad((LAT_MIN+LAT_MAX)/2)))
    ax.set_xlabel('Längengrad (°O)')
    ax.set_ylabel('Breitengrad (°N)')
    ax.grid(color='white',alpha=.2,linewidth=.5)
    fig.suptitle('ELBTAL | Nebelpotenzial (experimentell)',fontsize=14,weight='bold',y=.98)
    ax.set_title(f'ICON-D2 · {valid.astimezone(TZ):%d.%m.%Y %H:%M %Z}',fontsize=9)
    fig.text(.5,.025,'Heuristik: Feuchte/Wind/optional tiefe Wolken · Regen separat · keine Nebelwahrscheinlichkeit',ha='center',fontsize=8)
    style_figure(fig, ax)
    out = io.BytesIO()
    fig.savefig(out,format='png',bbox_inches='tight',facecolor=fig.get_facecolor())
    plt.close(fig)
    return frame_png(out.getvalue(), 'Nebelpotenzial', 'ICON-D2', 'Elbtal', run, valid)


st.set_page_config(page_title='ELBTAL Wetterkarten', page_icon='🌦️', layout='wide')
from v2_style import apply as apply_v2_style
apply_v2_style()
st.title(f'🌦️ ELBTAL · Wetterkarten')
st.caption(f'WETTERKARTEN  /  VERSION {VERSION}  /  V2-DESIGNVORSCHAU')
from v2_navigation import show as show_v2_navigation
if show_v2_navigation():
    st.stop()
map_region = st.segmented_control('Lokales Kartengebiet', options=['Elbtal','KRUMET'], default='Elbtal', selection_mode='single')
if map_region == '📊 Ensembles':
    from ensembles import show as show_ensembles
    show_ensembles()
    st.stop()
if map_region == '📡 Radar':
    from radar import show as show_radar
    show_radar()
    st.stop()
if map_region == '🇩🇪 Deutschland Detail':
    st.subheader('🇩🇪 Deutschland · Detailkarte (3:4)')
    detail_model = st.segmented_control(
        'Modell für Deutschland',
        ['ICON-D2', 'ICON-D2-RUC', 'ICON-EU', 'GFS', 'ECMWF IFS', 'AIGFS', 'AIGEFS Mittel'],
        default='ICON-D2', key='germany_detail_model',
    )
    st.caption('Deutschland im 3:4-Hochformat · ICON-D2 als Standard; andere Modelle zum direkten Vergleich.')
    if detail_model in ('AIGFS', 'AIGEFS Mittel'):
        from ai_models import show as show_ai_detail
        show_ai_detail('Deutschland', forced_model=detail_model)
    elif detail_model == 'ICON-D2-RUC':
        from icon_d2_ruc import show as show_ruc_detail
        show_ruc_detail('Deutschland')
    elif detail_model == 'ECMWF IFS':
        from global_models import show as show_germany_global
        show_germany_global('Deutschland', forced_model=detail_model)
    else:
        from icon_eu import show as show_germany_icon
        show_germany_icon('Deutschland', forced_model=detail_model)
    st.stop()
if map_region == '🌍 Globalmodelle':
    from global_models import show as show_global_models
    global_area = st.segmented_control('Globales Kartengebiet', ['Mitteleuropa','Europa & Nordatlantik'], default='Europa & Nordatlantik', key='global_area')
    global_model = st.segmented_control('Globales Modell', ['ECMWF IFS', 'GEFS Mittel', 'AIGFS', 'AIGEFS Mittel'], default='ECMWF IFS', key='global_ai_choice')
    if global_model in ('AIGFS', 'AIGEFS Mittel'):
        from ai_models import show as show_ai_global
        show_ai_global(global_area, forced_model=global_model)
    else:
        show_global_models(global_area, forced_model=global_model)
    st.stop()

if map_region in ('Mitteleuropa','Europa & Nordatlantik'):
    regional_model = st.segmented_control(
        "Wettermodell",
        ["ICON-EU", "ICON-D2-RUC", "GFS", "ECMWF IFS", "GEFS Mittel", "AIGFS", "AIGEFS Mittel"],
        default="ICON-EU",
        key="regional_synoptic_model",
    )
    if regional_model in ("AIGFS", "AIGEFS Mittel"):
        from ai_models import show as show_regional_ai
        show_regional_ai(map_region, forced_model=regional_model)
    elif regional_model == "ICON-D2-RUC":
        from icon_d2_ruc import show as show_regional_ruc
        show_regional_ruc(map_region)
    elif regional_model in ("ECMWF IFS", "GEFS Mittel"):
        from global_models import show as show_regional_global
        show_regional_global(map_region, forced_model=regional_model)
    else:
        from icon_eu import show as show_icon_eu
        show_icon_eu(map_region, forced_model=regional_model)
    st.stop()
krumet_mode = map_region == 'KRUMET'
if krumet_mode:
    LAT_MIN, LAT_MAX = 50.0, 52.2
    LON_MIN, LON_MAX = 11.5, 15.4
    PLACES = {'Leipzig': (51.340, 12.375), 'Dresden': (51.051, 13.738), 'Chemnitz': (50.833, 12.925), 'Zwickau': (50.718, 12.493), 'Görlitz': (51.156, 14.989), 'Bautzen': (51.181, 14.429), 'Belgern': (51.483, 13.125), 'Mühlberg': (51.434, 13.222), 'Cottbus': (51.760, 14.335), 'Altenburg': (50.985, 12.434)}
    st.subheader('🔬 KRUMET v0.3 | Sachsen und Umgebung')
    st.caption('Gebiet: 50,0–52,2°N / 11,5–15,4°O. Experimentelle ICON-D2-Geländekorrektur; WorldCover für das größere Gebiet noch nicht verfügbar.')
else:
    st.caption('ICON-D2-Karten für Belgern, Mühlberg, Nünchritz, Glaubitz und Zeithain')
parameter_page = '🌡️ Temperatur' if krumet_mode else st.segmented_control('Parameter wählen', options=['🌡️ Temperatur','🌬️ Windböen','🌧️ Rate','💧 Summe','🌦️ Signifikantes Wetter','💦 Taupunkt','🌫️ Nebel'], default='🌡️ Temperatur', selection_mode='single')
is_gust = parameter_page == '🌬️ Windböen'
is_precip = parameter_page in ('🌧️ Rate','💧 Summe')
is_kind = parameter_page == '🌦️ Signifikantes Wetter'
is_dew = parameter_page == '💦 Taupunkt'
is_fog = parameter_page == '🌫️ Nebel'
st.subheader({'🌡️ Temperatur':'🌡️ 2-m-Temperatur (°C)','🌬️ Windböen':'🌬️ Windböen (km/h)','🌧️ Rate':'🌧️ Stündliche Niederschlagsintensität (mm/h)','💧 Summe':'💧 Niederschlagssumme (mm)','🌦️ Signifikantes Wetter':'🌦️ Signifikantes Wetter (DWD WW)','💦 Taupunkt':'💦 2-m-Taupunkt (°C)','🌫️ Nebel':'🌫️ Nebelpotenzial (experimentell)'}[parameter_page])
if not is_gust and not is_precip and not is_kind and not is_dew and not is_fog:
    st.info('Experimentelles Downscaling: Höhenkorrektur und optionale gespeicherte WorldCover-Landnutzung. Die Koeffizienten sind heuristisch und nicht validiert.')
if parameter_page == '🌡️ Temperatur':
  with st.expander('Landnutzung einmalig vorbereiten', expanded=False):
      st.write('Einmalige ESA-WorldCover-Abfrage; anschließend die erzeugte NPZ-Datei unter `data/landuse.npz` in GitHub hochladen. Bei regulären Prognosen wird kein Landnutzungsserver aufgerufen.')
      if st.button('ESA-WorldCover-Raster einmalig abrufen', disabled=krumet_mode):
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
  if LANDUSE_FILE.exists() and not krumet_mode:
      st.success('Gespeichertes Landnutzungsraster gefunden: data/landuse.npz')
  else:
      st.warning('Noch keine gespeicherte Landnutzungsdatei. Prognosen funktionieren vorerst nur mit Höhenkorrektur.')
  enable_landuse = st.checkbox('Gespeicherte Landnutzung verwenden (experimentell)', value=not krumet_mode, disabled=krumet_mode)
else:
    enable_landuse = False
    if is_gust:
        st.caption('DWD-ICON-D2-Windböen ohne experimentelle Temperaturkorrekturen.')
    elif is_dew:
        st.caption('ICON-D2-Taupunkt in 2 m Höhe, direktes Modellfeld ohne lokale Korrektur.')
    elif is_fog:
        st.caption('Experimenteller Nebel-/Hochnebelindikator aus Taupunktdifferenz, Wind und optional tiefer Bewölkung (CLCL). Niederschlag wird gesondert markiert; keine Sichtweitenprognose oder Nebelwahrscheinlichkeit.')
    elif is_kind:
        st.caption('Signifikantes Wetter aus dem originalen DWD-ICON-D2-WW-Feld. Nebel mit Reifbildung als Nebel zusammengefasst; keine Kategorie Gewitter in der Nähe. WW ist eine Diagnose der vergangenen Stunde.')
    else:
        st.caption('ICON-D2-Niederschlag als Wasseräquivalent. Rate = stündliche Akkumulationsdifferenz.')
# Display an indicative local valid time while choosing lead hours; the exact run is shown after loading.
preview_run = find_runs()[0]
st.caption(f'Orientierung: Modelllauf {preview_run:%d.%m.%Y %H:%M} UTC. Bei fehlenden Dateien wird automatisch ein älterer Lauf gewählt; die endgültigen Zeiten stehen an den Karten.')
start_hour, end_hour = st.select_slider('Prognosezeitraum auswählen (+0 bis +48 h)', options=list(range(49)), value=(0, 6), key='window_hour_'+('krumet' if krumet_mode else IMAGE_KEYS[parameter_page]), format_func=lambda h: f'+{h} h · {(preview_run+timedelta(hours=h)).astimezone(TZ):%d.%m. %H:%M %Z}')
st.caption(f'Gewählte Zeitspanne: {(preview_run+timedelta(hours=start_hour)).astimezone(TZ):%d.%m.%Y %H:%M %Z} bis {(preview_run+timedelta(hours=end_hour)).astimezone(TZ):%d.%m.%Y %H:%M %Z} (vorläufig). Es werden Karten nur für +{start_hour} bis +{end_hour} erstellt. Für Niederschlagsraten wird zusätzlich der jeweils vorherige Akkumulationsstand geladen.')
if st.button('DWD-Prognose manuell berechnen', type='primary'):
    with st.status('Suche verfügbaren DWD-Modelllauf …', expanded=True) as status:
        try:
            selected = None
            diagnostics = []
            for run in find_runs():
                try:
                    field_name = 'ww' if is_kind else ('tot_prec' if is_precip else ('td_2m' if is_dew else ('t_2m' if is_fog else ('vmax_10m' if is_gust else 't_2m'))))
                    files, examples, matching = list_run(run, field_name)
                    requested = list(range(start_hour, end_hour + 1))
                    if is_precip and parameter_page == '🌧️ Rate' and start_hour > 0:
                        requested = [start_hour-1] + requested
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
            krumet_met_files = {}
            if krumet_mode:
                for p in ('u_10m','v_10m','clct'):
                    try:
                        found,_,_ = list_run(run,p)
                        if all(h in found for h in hours):
                            krumet_met_files[p] = found
                    except requests.RequestException:
                        pass
                if len(krumet_met_files)<3:
                    st.warning('Wind-/Bewölkungsdaten fehlen teilweise: KRUMET verwendet dann nur die Höhenkorrektur.')
            if is_precip:
                hours = list(range(max(0,start_hour-1),end_hour+1)) if parameter_page == '🌧️ Rate' else list(range(start_hour,end_hour+1))
                if any(h not in files for h in hours):
                    raise RuntimeError('Benötigte Niederschlagsdateien fehlen.')
            if is_fog:
                fog_files = {}
                for p in ('t_2m','u_10m','v_10m'):
                    found,_,_ = list_run(run,p)
                    if any(h not in found for h in hours):
                        raise RuntimeError(f'Für Nebel fehlen ICON-D2-Dateien: {p}.')
                    fog_files[p] = found
                # Accumulated precipitation needs the preceding lead hour to form an hourly rate.
                fog_precip_files = {}
                try:
                    found,_,_ = list_run(run,'tot_prec')
                    needed = set(hours)
                    needed.update(h-1 for h in hours if h>0)
                    if all(h in found for h in needed):
                        fog_precip_files = found
                    else:
                        st.warning('Niederschlagsdateien unvollständig: Nebelindikator ohne Regenabgrenzung.')
                except requests.RequestException:
                    st.warning('Niederschlagsdaten nicht erreichbar: Nebelindikator ohne Regenabgrenzung.')
                fog_cloud_files = {}
                for p in ('clcl',):
                    try:
                        found,_,_ = list_run(run,p)
                        if all(h in found for h in hours):
                            fog_cloud_files[p] = found
                        else:
                            st.info('Tiefe Bewölkung ist nicht für alle Stunden verfügbar; Basisindikator bleibt aktiv.')
                    except requests.RequestException:
                        st.info('Tiefe Bewölkung aktuell nicht verfügbar; Basisindikator bleibt aktiv.')
                fog_precip_previous = None
                fog_precip_previous_hour = None
            st.write(f'Gefundener Modelllauf: {run:%Y-%m-%d %H:%M} UTC')
            images = []
            if krumet_mode:
                st.session_state['krumet_station_samples'] = []
            terrain = None
            try:
                if is_gust or is_precip or is_kind or is_dew or is_fog:
                    raise ValueError('Für diesen Parameter nicht benötigt')
                terrain = get_terrain((LAT_MIN,LAT_MAX,LON_MIN,LON_MAX) if krumet_mode else None)
                st.write(f'Geländedaten geladen: {terrain.min():.0f} bis {terrain.max():.0f} m')
            except Exception as terrain_error:
                if krumet_mode:
                    raise RuntimeError(f'KRUMET benötigt Geländedaten: {terrain_error}') from terrain_error
                if parameter_page == '🌡️ Temperatur':
                    st.warning(f'Geländedaten nicht verfügbar: {terrain_error}. Es werden nur ICON-D2-Karten erstellt.')
            landuse = None
            if enable_landuse and terrain is not None:
                try:
                    landuse, polygon_count = load_landuse_file()
                    st.write(f'WorldCover-Landnutzung geladen: {polygon_count} klassifizierte Rasterzellen; {(landuse > 0).mean():.0%} Rasterabdeckung')
                except Exception as land_error:
                    st.warning(f'Gespeicherte Landnutzung nicht verfügbar: {land_error}. Höhenkorrektur bleibt aktiv.')
            precip_previous = None
            for idx, hour in enumerate(hours):
                st.write(f'Lade Daten +{hour} h ({idx+1}/{len(hours)}) …')
                field_name = 'ww' if is_kind else ('tot_prec' if is_precip else ('td_2m' if is_dew or is_fog else ('vmax_10m' if is_gust else 't_2m')))
                messages = decode_grib(files[hour], field_name)
                # Select the exact hourly forecast time, never a neighboring quarter-hour.
                target = run + timedelta(hours=hour)
                msg = min(messages, key=lambda m: abs((m[3]-target).total_seconds()))
                if abs((msg[3]-target).total_seconds()) > 60:
                    raise RuntimeError(f'Kein exakter Prognosezeitpunkt für +{hour} h vorhanden.')
                X, Y, Z = interpolate(*msg[:3],grid_m=500 if is_fog else None,categorical=is_kind)
                if is_kind:
                    display=classify_ww(Z)
                    original_png=(render_precip(X,Y,display,run,msg[3],parameter_page),)*2
                elif is_precip:
                    # TOT_PREC is the accumulated water equivalent since model initialization.
                    # Each hourly difference is a one-hour mean rate in mm/h.
                    if not np.isfinite(Z).all():
                        raise RuntimeError('Ungültige Niederschlagswerte im Modellfeld.')
                    Z = np.maximum(Z,0)
                    if precip_previous is None:
                        hourly = np.zeros_like(Z)
                    else:
                        if np.nanmin(Z-precip_previous) < -.1:
                            raise RuntimeError('Niederschlagsakkumulation fällt unerwartet ab; Daten prüfen.')
                        hourly = np.maximum(Z-precip_previous,0)
                    precip_previous = Z.copy()
                    if hour < start_hour:
                        continue
                    display = hourly if parameter_page == '🌧️ Rate' else Z
                    original_png = tuple(render_precip(X,Y,display,run,msg[3],parameter_page,flag) for flag in (False,True))
                elif is_fog:
                    def load_fog_field(parameter):
                        records=decode_grib(fog_files[parameter][hour],parameter)
                        record=min(records,key=lambda m: abs((m[3]-target).total_seconds()))
                        if abs((record[3]-target).total_seconds())>60:
                            raise RuntimeError(f'Zeitstempel für {parameter} +{hour} h fehlt.')
                        return interpolate(*record[:3],grid_m=500)[2]
                    t2m=load_fog_field('t_2m')
                    u10=load_fog_field('u_10m')
                    v10=load_fog_field('v_10m')
                    wind=np.hypot(u10,v10)*3.6
                    spread=np.maximum(t2m-Z,0)
                    # Diagnostic indicator, not a probability or actual fog visibility.
                    score=np.clip((2.5-spread)/2.5,0,1)*np.clip((18-wind)/15,0,1)
                    categories=np.zeros(Z.shape,dtype=int)
                    categories[score>=.2]=1
                    categories[score>=.45]=2
                    categories[score>=.7]=3
                    # CLCL is low-level cloud fraction, not proof of fog at ground level.
                    if 'clcl' in fog_cloud_files:
                        try:
                            cloud_records=decode_grib(fog_cloud_files['clcl'][hour],'clcl')
                            cloud_record=min(cloud_records,key=lambda m:abs((m[3]-target).total_seconds()))
                            if abs((cloud_record[3]-target).total_seconds())<=60:
                                low_cloud=interpolate(*cloud_record[:3],grid_m=500)[2]
                                # Strong low-cloud cover with nearly saturated air: retain a fog/high-stratus signal.
                                categories[(low_cloud>=70)&(spread<=1.5)&(wind<=12)] = np.maximum(categories[(low_cloud>=70)&(spread<=1.5)&(wind<=12)],2)
                        except (requests.RequestException,ValueError,RuntimeError) as cloud_error:
                            st.warning(f'Tiefe Bewölkung +{hour} h nicht verarbeitet: {cloud_error}. Basisindikator bleibt aktiv.')
                    # Distinguish precipitation-associated saturation from fog-favourable conditions.
                    if fog_precip_files and hour>0:
                        def fog_accumulation(lead):
                            records=decode_grib(fog_precip_files[lead],'tot_prec')
                            expected=run+timedelta(hours=lead)
                            record=min(records,key=lambda m:abs((m[3]-expected).total_seconds()))
                            if abs((record[3]-expected).total_seconds())>60:
                                raise RuntimeError(f'Kein exakter Niederschlagszeitpunkt +{lead} h.')
                            return np.maximum(interpolate(*record[:3],grid_m=500)[2],0)
                        current_accum=fog_accumulation(hour)
                        if fog_precip_previous_hour==hour-1:
                            previous_accum=fog_precip_previous
                        else:
                            previous_accum=fog_accumulation(hour-1)
                        hourly_rain=np.maximum(current_accum-previous_accum,0)
                        categories[hourly_rain>=0.1]=4
                        fog_precip_previous=current_accum
                        fog_precip_previous_hour=hour
                    original_png=(render_fog(X,Y,categories,run,msg[3]),)*2
                elif is_dew:
                    original_png=tuple(render_png(X,Y,Z,run,msg[3],mode='ICON-D2 Taupunkt · keine lokale Korrektur',show_km_values=flag) for flag in (False,True))
                else:
                    original_png = (render_gust(X,Y,Z,run,msg[3]), render_gust(X,Y,Z,run,msg[3],show_km_values=True)) if is_gust else ((None,None) if krumet_mode else (render_png(X,Y,Z,run,msg[3]), render_png(X,Y,Z,run,msg[3],show_km_values=True)))
                local_png = None
                combined_png = None
                diff_png = None
                if terrain is not None:
                    local, correction = downscale_temperature(Z, terrain)
                    if krumet_mode and len(krumet_met_files)==3:
                        try:
                            met = {}
                            for p in ('u_10m','v_10m','clct'):
                                records = decode_grib(krumet_met_files[p][hour],p)
                                record = min(records,key=lambda m:abs((m[3]-target).total_seconds()))
                                if abs((record[3]-target).total_seconds())>60:
                                    raise ValueError(f'Zeitstempel {p} fehlt')
                                met[p] = interpolate(*record[:3])[2]
                            cold = krumet_cold_air_diagnostic(terrain,msg[3],met['u_10m'],met['v_10m'],met['clct'])
                            local += cold
                            correction += cold
                        except (requests.RequestException,ValueError,RuntimeError) as exc:
                            st.warning(f'KRUMET +{hour} h: Kaltluftdiagnostik ausgefallen ({exc}).')
                    local_png = (render_png(X,Y,local,run,msg[3],mode='KRUMET · Gelände + Kaltluft',correction=correction,model_name='KRUMET'),)*2 if krumet_mode else tuple(render_png(X,Y,local,run,msg[3],mode='Lokale Höhenkorrektur · experimentell',correction=correction,show_km_values=flag) for flag in (False,True))
                    if landuse is not None:
                        lu_corr = landuse_temperature_correction(landuse, msg[3])
                        combined = local + lu_corr
                        combined_png = tuple(render_png(X,Y,combined,run,msg[3],mode='KRUMET · Höhe + WorldCover' if krumet_mode else 'Höhe + WorldCover-Landnutzung · experimentell',correction=combined-Z,show_km_values=flag,model_name='KRUMET' if krumet_mode else 'ICON-D2') for flag in (False,True))
                        diff_png = render_difference(X, Y, combined-Z, run, msg[3],model_name='KRUMET' if krumet_mode else 'ICON-D2')
                if krumet_mode and terrain is not None:
                    try:
                        for sid, name, lat, lon, elevation in dwd_stations():
                            st.session_state['krumet_station_samples'].append({'Station':name, 'ID':sid, 'Zeit UTC':msg[3], 'ICON-D2':round(station_grid_value(X,Y,Z,lat,lon),2), 'KRUMET':round(station_grid_value(X,Y,local,lat,lon),2)})
                    except (requests.RequestException,ValueError) as station_error:
                        st.warning(f'Stationsmetadaten nicht verfügbar: {station_error}')
                images.append((hour, msg[3], original_png, local_png, combined_png, diff_png))
            st.session_state['krumet_images' if krumet_mode else IMAGE_KEYS[parameter_page]] = images
            status.update(label='Karten erstellt', state='complete')
        except Exception as exc:
            status.update(label='Abruf oder Verarbeitung fehlgeschlagen', state='error')
            st.error(f'{type(exc).__name__}: {exc}')

image_key = 'krumet_images' if krumet_mode else IMAGE_KEYS[parameter_page]
if image_key in st.session_state:
    images = st.session_state[image_key]
    labels = [f'+{item[0]:g} h · {item[1].astimezone(TZ):%d.%m.%Y %H:%M %Z}' for item in images]
    choice = st.select_slider('Vorhersagekarte', options=range(len(images)),
                              format_func=lambda i: labels[i], key='hour_'+image_key)
    st.caption(f'Ausgewählt: {labels[choice]}')
    h, valid, original_png, local_png, combined_png, diff_png = images[choice]
    variants = {parameter_page: original_png} if (is_precip or is_kind or is_dew or is_fog) else ({'ICON-D2 Windböen': original_png} if is_gust else {} if krumet_mode else {'ICON-D2 (interpoliert)': original_png})
    if local_png is not None and parameter_page == '🌡️ Temperatur':
        variants['KRUMET · Höhenkorrektur' if krumet_mode else 'Höhenkorrektur'] = local_png
    if combined_png is not None and parameter_page == '🌡️ Temperatur':
        variants['KRUMET · Höhe + Landnutzung' if krumet_mode else 'Höhe + Landnutzung'] = combined_png
        variants['Differenz zu ICON-D2'] = diff_png
    variant = st.radio('Kartentyp', list(variants), horizontal=True, key='variant_'+image_key)
    show_km_values = False if krumet_mode else st.checkbox('Werte alle ca. 1 km anzeigen', disabled=is_kind or is_fog, value=False, key='values_'+image_key, help='Interpolierte Modellwerte am jeweiligen Punkt, keine Messwerte.')
    selected_map = variants[variant]
    png = selected_map if variant == 'Differenz zu ICON-D2' else selected_map[int(show_km_values)]
    st.image(png, width='stretch')
    st.download_button('PNG herunterladen', data=png,
                       file_name=f'ELBTAL_{image_key}_{list(variants).index(variant)}_{valid:%Y%m%d_%H%M}_UTC.png', mime='image/png')


if krumet_mode and st.session_state.get('krumet_station_samples'):
    with st.expander('🔬 KRUMET vs. ICON-D2 vs. DWD-Messstationen', expanded=False):
        st.caption('Vergleich nur für bereits berechnete Prognosezeiten. Beobachtungen können verzögert vorliegen; ohne Messwert keine Bewertung. Stationsnahe Rasterwerte sind keine exakte Stationshöhenkorrektur.')
        import pandas as pd
        samples = pd.DataFrame(st.session_state['krumet_station_samples'])
        station_names = sorted(samples['Station'].unique())
        chosen = st.selectbox('DWD-Messstation', station_names, key='krumet_verify_station')
        subset = samples[samples['Station']==chosen].copy()
        sid = str(subset['ID'].iloc[0])
        if st.button('DWD-Beobachtungen laden und vergleichen', key='krumet_verify_load'):
            try:
                observations = dwd_hourly_temperature(sid)
                comparison = subset.merge(observations, left_on='Zeit UTC', right_on='time', how='left')
                comparison['ICON-D2-Fehler'] = comparison['ICON-D2'] - comparison['temperature']
                comparison['KRUMET-Fehler'] = comparison['KRUMET'] - comparison['temperature']
                display = comparison.rename(columns={'temperature':'DWD gemessen'})[['Zeit UTC','DWD gemessen','ICON-D2','KRUMET','ICON-D2-Fehler','KRUMET-Fehler']]
                st.dataframe(display, hide_index=True)
                valid_rows = comparison.dropna(subset=['temperature'])
                if len(valid_rows):
                    mae_icon = valid_rows['ICON-D2-Fehler'].abs().mean()
                    mae_krumet = valid_rows['KRUMET-Fehler'].abs().mean()
                    st.write(f'Verfügbare Vergleiche: {len(valid_rows)} · MAE ICON-D2: {mae_icon:.2f} °C · MAE KRUMET: {mae_krumet:.2f} °C')
                    st.caption('Kleine Stichprobe; noch kein Nachweis einer systematischen Verbesserung. Für Strahlungsnächte sind mehrere vollständige Nächte nötig.')
                else:
                    st.info('Für diese Prognosezeiten liegen noch keine DWD-Stundenmesswerte vor. Später erneut versuchen.')
            except (requests.RequestException,ValueError,KeyError,IndexError) as exc:
                st.warning(f'DWD-Stationsvergleich derzeit nicht verfügbar: {type(exc).__name__}: {exc}')


if krumet_mode:
    with st.expander('🌙 Historische Nächte | DWD-Messwerte', expanded=True):
        st.caption('Untersuche abgeschlossene Nächte (18–09 Uhr Ortszeit). Das sind echte Messwerte, noch kein historischer ICON-D2/KRUMET-Prognosevergleich.')
        try:
            import pandas as pd
            station_list = dwd_stations()
            if not station_list:
                st.warning('Keine DWD-Stationen im KRUMET-Gebiet gefunden. Bitte Stationsmetadaten prüfen.')
            else:
                station_options = {f'{name} ({sid})':sid for sid,name,lat,lon,elev in station_list}
                station_label = st.selectbox('Messstation für Nachtanalyse',list(station_options),key='krumet_night_station')
                yesterday = (datetime.now(TZ)-timedelta(days=1)).date()
                night_date = st.date_input('Nacht beginnt am',value=yesterday,max_value=yesterday,key='krumet_night_date')
                nights_count = st.slider('Anzahl Nächte für die Auswertung', min_value=1, max_value=14, value=5, key='krumet_nights_count')
                proxy_test = st.checkbox('🔬 KRUMET-v0.4-Testmodus (Karten bleiben unverändert)',value=True,key='krumet_v04_test')
                if st.button('Mehrere Nächte vergleichen',key='krumet_multi_night_go'):
                    try:
                        sid = station_options[station_label]
                        station = next(row for row in station_list if row[0]==sid)
                        lat,lon = station[2],station[3]
                        observations = dwd_hourly_temperature(sid).copy()
                        first_date = night_date-timedelta(days=nights_count-1)
                        archive = archived_icon_d2(lat,lon,first_date,night_date+timedelta(days=1))
                        joined = observations.merge(archive,on='time',how='inner')
                        joined['Ortszeit'] = joined['time'].dt.tz_convert('Europe/Berlin')
                        joined['Nachtbeginn'] = (joined['Ortszeit']-pd.Timedelta(hours=18)).dt.date
                        joined['Stunde'] = joined['Ortszeit'].dt.hour
                        joined = joined[(joined['Nachtbeginn']>=first_date)&(joined['Nachtbeginn']<=night_date)]
                        joined = joined[(joined['Stunde']>=18)|(joined['Stunde']<=9)].copy()
                        joined['Fehler'] = joined['temperature_2m']-joined['temperature']
                        joined = joined.dropna(subset=['Fehler'])
                        if joined.empty:
                            st.info('Für den Zeitraum sind noch keine gemeinsamen Mess- und Archivwerte vorhanden.')
                        else:
                            if proxy_test:
                                try:
                                    terrain_test = get_terrain((LAT_MIN,LAT_MAX,LON_MIN,LON_MAX))
                                    ny,nx = terrain_test.shape
                                    iy = int(np.clip(round((lat-LAT_MIN)/(LAT_MAX-LAT_MIN)*(ny-1)),0,ny-1))
                                    ix = int(np.clip(round((lon-LON_MIN)/(LON_MAX-LON_MIN)*(nx-1)),0,nx-1))
                                    _, terrain_delta = downscale_temperature(np.zeros_like(terrain_test),terrain_test)
                                    dy = (LAT_MAX-LAT_MIN)*111200/(ny-1)
                                    dx = (LON_MAX-LON_MIN)*111200*math.cos(math.radians((LAT_MIN+LAT_MAX)/2))/(nx-1)
                                    valley = float(np.clip((gaussian_filter(terrain_test,(10000/dy,10000/dx))[iy,ix]-terrain_test[iy,ix])/100,0,1))
                                    night_factor = np.maximum(0,np.cos((joined['Ortszeit'].dt.hour.to_numpy()-3)*np.pi/12))
                                    calm = np.clip((18-joined['wind_speed_10m'].to_numpy(dtype=float))/15,0,1)
                                    clear = np.clip((85-joined['cloud_cover'].to_numpy(dtype=float))/70,0,1)
                                    # Same bounded v0.3 physics proxy, not trained on observations.
                                    cooling = -1.5*night_factor*valley*calm*clear
                                    joined['KRUMET-Proxy'] = joined['temperature_2m']+terrain_delta[iy,ix]+cooling
                                    joined['KRUMET-Fehler'] = joined['KRUMET-Proxy']-joined['temperature']
                                    proxy_valid = joined.dropna(subset=['KRUMET-Fehler'])
                                    if not proxy_valid.empty:
                                        st.metric('KRUMET-v0.4-Test MAE',f"{proxy_valid['KRUMET-Fehler'].abs().mean():.2f} °C")
                                        st.metric('ICON-D2 MAE (gleiche Stunden)',f"{proxy_valid['Fehler'].abs().mean():.2f} °C")
                                        st.caption('Testmodus: unveränderte v0.3-Formeln als historischer Punktdaten-Proxy. Kein neu kalibriertes Modell, keine exakte GRIB-Rekonstruktion.')
                                        st.download_button('Stündlichen Testvergleich als CSV',data=proxy_valid[['Ortszeit','Nachtbeginn','temperature','temperature_2m','KRUMET-Proxy','Fehler','KRUMET-Fehler','wind_speed_10m','cloud_cover']].to_csv(index=False).encode('utf-8-sig'),file_name='KRUMET_v04_Testvergleich.csv',mime='text/csv',key='krumet_v04_csv')
                                except (requests.RequestException,ValueError,RuntimeError,KeyError) as proxy_error:
                                    st.warning(f'KRUMET-Testmodus nicht berechenbar: {type(proxy_error).__name__}: {proxy_error}')
                            summary = joined.groupby('Nachtbeginn').agg(
                                Stunden=('Fehler','size'),
                                Bias=('Fehler','mean'),
                                MAE=('Fehler',lambda x:x.abs().mean()),
                                Tmin_DWD=('temperature','min'),
                                Tmin_ICON=('temperature_2m','min'),
                                Wind=('wind_speed_10m','mean'),
                                Bewoelkung=('cloud_cover','mean')
                            ).reset_index()
                            summary['Tmin_Fehler'] = summary['Tmin_ICON']-summary['Tmin_DWD']
                            st.dataframe(summary.rename(columns={'Bias':'Bias (°C)','MAE':'MAE (°C)','Wind':'Wind (km/h)','Bewoelkung':'Bewölkung (%)','Tmin_Fehler':'Tmin-Fehler (°C)'}),hide_index=True)
                            complete = summary[summary['Stunden']>=12]
                            st.metric('Gesamt-MAE ICON-D2',f"{joined['Fehler'].abs().mean():.2f} °C")
                            st.metric('Mittlerer ICON-D2-Bias',f"{joined['Fehler'].mean():+.2f} °C")
                            if not complete.empty:
                                st.metric('Tmin-Bias vollständiger Nächte',f"{complete['Tmin_Fehler'].mean():+.2f} °C")
                            st.caption(f"{len(joined)} gemeinsame Stunden, {len(summary)} Nächte; davon {len(complete)} mit mindestens 12 Messstunden. Positive Fehler bedeuten zu mild. Tmin-Fehler nur für Nächte mit mindestens 12 Stunden zusammengefasst.")
                            st.download_button('Nachtvergleich als CSV',data=summary.to_csv(index=False).encode('utf-8-sig'),file_name='KRUMET_Nachtvergleich.csv',mime='text/csv',key='krumet_multi_csv')
                            st.info('ICON-D2: archivierte Punktprognosen mit festem 24-h-Vorlauf. KRUMET im Testmodus: nur Näherung mit denselben Formeln und Eingangsdaten; keine Anpassung der offiziellen Karten.')
                    except (requests.RequestException,ValueError,KeyError,IndexError,FileNotFoundError) as multi_error:
                        st.warning(f'Mehrnachtvergleich nicht verfügbar: {type(multi_error).__name__}: {multi_error}')
                if st.button('Abgeschlossene Nacht auswerten',key='krumet_night_go'):
                    measurements = dwd_hourly_temperature(station_options[station_label]).copy()
                    start_local = datetime.combine(night_date,datetime.min.time(),tzinfo=TZ).replace(hour=18)
                    end_local = start_local + timedelta(hours=15)
                    start_utc = pd.Timestamp(start_local.astimezone(timezone.utc))
                    end_utc = pd.Timestamp(end_local.astimezone(timezone.utc))
                    night = measurements[(measurements['time']>=start_utc)&(measurements['time']<=end_utc)].copy()
                    if night.empty:
                        st.info('Für diese Nacht liegen in der aktuellen DWD-Datei keine Stundenwerte vor. Bitte eine jüngere Nacht wählen.')
                    else:
                        night['Ortszeit'] = night['time'].dt.tz_convert('Europe/Berlin')
                        night = night.sort_values('time')
                        st.line_chart(night.set_index('Ortszeit')['temperature'],y_label='Temperatur (°C)')
                        minimum = night.loc[night['temperature'].idxmin()]
                        st.metric('Tiefste gemessene Stundentemperatur',f"{minimum['temperature']:.1f} °C")
                        st.caption(f"Minimum um {minimum['Ortszeit']:%d.%m.%Y %H:%M} Ortszeit · {len(night)} Stundenmessungen vorhanden")
                        st.markdown('**Archivierte ICON-D2-Prognose (24 Stunden Vorlauf)**')
                        try:
                            sid = station_options[station_label]
                            station = next(row for row in station_list if row[0]==sid)
                            lat, lon = station[2], station[3]
                            archived = archived_icon_d2(lat,lon,start_utc.date(),end_utc.date())
                            archived = archived[(archived['time']>=start_utc)&(archived['time']<=end_utc)]
                            comparison = night.merge(archived,on='time',how='left')
                            comparison['ICON-D2-Fehler'] = comparison['temperature_2m']-comparison['temperature']
                            valid = comparison.dropna(subset=['temperature','temperature_2m']).copy()
                            if valid.empty:
                                st.info('Für diese Nacht hat das ICON-D2-Archiv keine passenden 24-h-Prognosen geliefert.')
                            else:
                                # Historical KRUMET station-point approximation: same terrain and
                                # cold-pool equations, but Open-Meteo point forecasts instead of
                                # the original ICON-D2 GRIB fields; report separately.
                                try:
                                    terrain_hist = get_terrain((LAT_MIN,LAT_MAX,LON_MIN,LON_MAX))
                                    ny,nx = terrain_hist.shape
                                    iy = int(np.clip(round((lat-LAT_MIN)/(LAT_MAX-LAT_MIN)*(ny-1)),0,ny-1))
                                    ix = int(np.clip(round((lon-LON_MIN)/(LON_MAX-LON_MIN)*(nx-1)),0,nx-1))
                                    _, terrain_corr = downscale_temperature(np.zeros_like(terrain_hist),terrain_hist)
                                    local_times = valid['time'].dt.tz_convert('Europe/Berlin')
                                    night_factor = np.maximum(0,np.cos((local_times.dt.hour.to_numpy()-3)*np.pi/12))
                                    dy = (LAT_MAX-LAT_MIN)*111200/(ny-1)
                                    dx = (LON_MAX-LON_MIN)*111200*math.cos(math.radians((LAT_MIN+LAT_MAX)/2))/(nx-1)
                                    valley = np.clip((gaussian_filter(terrain_hist,(10000/dy,10000/dx))[iy,ix]-terrain_hist[iy,ix])/100,0,1)
                                    calm = np.clip((18-valid['wind_speed_10m'].to_numpy())/15,0,1)
                                    clear = np.clip((85-valid['cloud_cover'].to_numpy())/70,0,1)
                                    cold = -1.5*night_factor*valley*calm*clear
                                    valid['KRUMET-Proxy'] = valid['temperature_2m']+terrain_corr[iy,ix]+cold
                                    valid['KRUMET-Proxy-Fehler'] = valid['KRUMET-Proxy']-valid['temperature']
                                    st.metric('KRUMET-v0.3-Proxy MAE',f"{valid['KRUMET-Proxy-Fehler'].abs().mean():.2f} °C")
                                    st.caption('KRUMET-Proxy: dieselben Gelände-/Kaltluftformeln wie v0.3, aber historische Open-Meteo-Punktdaten statt originaler ICON-D2-GRIB-Felder. Daher kein exakter KRUMET-Reforecast.')
                                except (requests.RequestException,ValueError,RuntimeError) as proxy_error:
                                    st.warning(f'Historischer KRUMET-Proxy nicht berechenbar: {proxy_error}')
                                st.metric('ICON-D2 MAE (24 h Vorlauf)',f"{valid['ICON-D2-Fehler'].abs().mean():.2f} °C")
                                st.metric('ICON-D2 mittlere Abweichung',f"{valid['ICON-D2-Fehler'].mean():+.2f} °C",help='Positiv bedeutet: ICON-D2 zu mild.')
                                plot_columns = ['temperature','temperature_2m'] + (['KRUMET-Proxy'] if 'KRUMET-Proxy' in valid else [])
                                plot = valid.set_index('Ortszeit')[plot_columns].rename(columns={'temperature':'DWD gemessen','temperature_2m':'ICON-D2 archiviert'})
                                st.line_chart(plot,y_label='Temperatur (°C)')
                                st.dataframe(valid[['Ortszeit','temperature','temperature_2m','ICON-D2-Fehler','wind_speed_10m','cloud_cover']].rename(columns={'temperature':'DWD gemessen','temperature_2m':'ICON-D2 archiviert','wind_speed_10m':'Wind (km/h)','cloud_cover':'Bewölkung (%)'}),hide_index=True)
                                st.caption('Quelle: Open-Meteo Previous Runs, Modell ICON-D2, feste 24-h-Vorlaufzeit. Modellwerte können durch Open-Meteo-Interpolation und Höhenanpassung von den originalen DWD-GRIB-Rasterwerten abweichen.')
                                st.caption('Für eine exakte historische KRUMET-Rechnung benötigen wir weiterhin die damaligen vollständigen ICON-D2-GRIB-Felder.')
                        except (requests.RequestException,ValueError,KeyError,IndexError) as archive_error:
                            st.warning(f'Archivierte ICON-D2-Prognosen nicht abrufbar: {type(archive_error).__name__}: {archive_error}')
        except (requests.RequestException,ValueError,KeyError,FileNotFoundError) as exc:
            st.warning(f'Historische DWD-Messwerte derzeit nicht verfügbar: {type(exc).__name__}: {exc}')

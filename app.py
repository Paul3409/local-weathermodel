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


def interpolate(lats, lons, vals, grid_m=None):
    # Only a buffered area is needed; keep the download decoding manageable.
    keep = (lats >= LAT_MIN-.09) & (lats <= LAT_MAX+.09) & (lons >= LON_MIN-.09) & (lons <= LON_MAX+.09)
    if keep.sum() < 6:
        raise ValueError('Zu wenige ICON-D2-Rasterpunkte im Modellgebiet.')
    nlat = round((LAT_MAX-LAT_MIN)*111200/grid_m)+1 if grid_m else 245
    nlon = round((LON_MAX-LON_MIN)*111200*math.cos(math.radians((LAT_MIN+LAT_MAX)/2))/grid_m)+1 if grid_m else 235
    lat_axis = np.linspace(LAT_MIN, LAT_MAX, nlat)
    lon_axis = np.linspace(LON_MIN, LON_MAX, nlon)
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
    fig.suptitle('ELBTAL | '+('2-m-Taupunkt' if 'Taupunkt' in mode else '2-m-Temperatur'), fontsize=16, weight='bold', y=.98)
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
    out=io.BytesIO()
    fig.savefig(out,format='png',bbox_inches='tight',facecolor=fig.get_facecolor())
    plt.close(fig)
    return out.getvalue()



PREC_RATE = [(0,'#F2F5F7'),(.1,'#C8EAF7'),(.5,'#80C9EC'),(1,'#368DD1'),(2,'#2269A9'),(5,'#3B9A68'),(10,'#C4C948'),(15,'#F1D34E'),(20,'#F3A04A'),(30,'#E85A38'),(50,'#C72F43'),(75,'#83215F'),(100,'#C137A8')]
PREC_SUM = [(0,'#F2F5F7'),(.5,'#D1F2ED'),(1,'#91DCD8'),(2,'#51B9D4'),(5,'#3787C1'),(10,'#2465A0'),(15,'#3D986A'),(20,'#A8C34E'),(30,'#E6D14B'),(40,'#F1A24D'),(50,'#E46840'),(75,'#C43D53'),(100,'#793A94')]
PREC_KIND_COLORS = ['#F2F5F7','#278DCE','#F39A43','#F4A4CF','#D93455','#8C949F']


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
    if parameter_page == '❄️ Art':
        cmap=ListedColormap(PREC_KIND_COLORS)
        bounds=np.arange(-.5,6.5,1)
        filled=ax.contourf(X,Y,Z,levels=bounds,cmap=cmap,norm=BoundaryNorm(bounds,6))
        cb=fig.colorbar(filled,ax=ax,orientation='horizontal',fraction=.055,pad=.09)
        cb.set_ticks(range(6))
        cb.set_ticklabels(['Trocken','Regen','Schneeregen','Schnee','Gefr. Regen','Unbestimmt'])
        cb.ax.tick_params(labelsize=6)
    else:
        stops=PREC_RATE if parameter_page=='🌧️ Rate' else PREC_SUM
        vmax=100
        cmap=LinearSegmentedColormap.from_list('precip',[(v/vmax,col) for v,col in stops],N=2048)
        cmap.set_over(stops[-1][1])
        norm=Normalize(0,vmax)
        filled=ax.contourf(X,Y,Z,levels=np.linspace(0,vmax,501),cmap=cmap,norm=norm,extend='max')
        zmin,zmax=float(np.nanmin(Z)),float(np.nanmax(Z))
        boundaries=[v for v,_ in stops[1:] if zmin<v<zmax]
        if boundaries:
            ax.contour(X,Y,Z,levels=boundaries,colors='#293748',linewidths=.5,alpha=.7)
        if show_km_values:
            draw_km_temperatures(ax,X,Y,Z)
        cb=fig.colorbar(filled,ax=ax,orientation='horizontal',fraction=.055,pad=.09)
        cb.set_ticks([0,2,5,10,20,30,50,75,100])
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
    out=io.BytesIO()
    fig.savefig(out,format='png',bbox_inches='tight',facecolor=fig.get_facecolor())
    plt.close(fig)
    return out.getvalue()



IMAGE_KEYS = {'🌡️ Temperatur':'images_temp','🌬️ Windböen':'images_gust',
              '🌧️ Rate':'images_rate','💧 Summe':'images_sum','❄️ Art':'images_kind',
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
    out=io.BytesIO()
    fig.savefig(out,format='png',bbox_inches='tight',facecolor=fig.get_facecolor())
    plt.close(fig)
    return out.getvalue()


st.set_page_config(page_title='ELBTAL-T2M', page_icon='🌡️', layout='centered')
st.title('🌦️ ELBTAL · v0.9.0')
st.caption('ICON-D2-Karten für Belgern, Mühlberg, Nünchritz, Glaubitz und Zeithain')
parameter_page = st.segmented_control('Parameter wählen', options=['🌡️ Temperatur','🌬️ Windböen','🌧️ Rate','💧 Summe','❄️ Art','💦 Taupunkt','🌫️ Nebel'], default='🌡️ Temperatur', selection_mode='single')
is_gust = parameter_page == '🌬️ Windböen'
is_precip = parameter_page in ('🌧️ Rate','💧 Summe','❄️ Art')
is_kind = parameter_page == '❄️ Art'
is_dew = parameter_page == '💦 Taupunkt'
is_fog = parameter_page == '🌫️ Nebel'
st.subheader({'🌡️ Temperatur':'🌡️ 2-m-Temperatur (°C)','🌬️ Windböen':'🌬️ Windböen (km/h)','🌧️ Rate':'🌧️ Stündliche Niederschlagsintensität (mm/h)','💧 Summe':'💧 Niederschlagssumme (mm)','❄️ Art':'❄️ Niederschlagsart (Modellschätzung)','💦 Taupunkt':'💦 2-m-Taupunkt (°C)','🌫️ Nebel':'🌫️ Nebelpotenzial (experimentell)'}[parameter_page])
if not is_gust and not is_precip and not is_dew and not is_fog:
    st.info('Experimentelles Downscaling: Höhenkorrektur und optionale gespeicherte WorldCover-Landnutzung. Die Koeffizienten sind heuristisch und nicht validiert.')
if parameter_page == '🌡️ Temperatur':
  with st.expander('Landnutzung einmalig vorbereiten', expanded=False):
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
else:
    enable_landuse = False
    if is_gust:
        st.caption('DWD-ICON-D2-Windböen ohne experimentelle Temperaturkorrekturen.')
    elif is_dew:
        st.caption('ICON-D2-Taupunkt in 2 m Höhe, direktes Modellfeld ohne lokale Korrektur.')
    elif is_fog:
        st.caption('Experimenteller Nebel-/Hochnebelindikator aus Taupunktdifferenz, Wind und optional tiefer Bewölkung (CLCL). Niederschlag wird gesondert markiert; keine Sichtweitenprognose oder Nebelwahrscheinlichkeit.')
    elif is_kind:
        st.caption('Niederschlagsart aus stündlichen Regen-/Schneeanteilen; Graupel und unklare Fälle grau. Gefrierender Regen wird nicht diagnostiziert.')
    else:
        st.caption('ICON-D2-Niederschlag als Wasseräquivalent. Rate = stündliche Akkumulationsdifferenz.')
quarter_mode = (is_gust or parameter_page == '🌧️ Rate') and st.radio('Zeitschritte', ['1 Stunde','15 Minuten'],horizontal=True,key='steps_'+IMAGE_KEYS[parameter_page]) == '15 Minuten'
# Display an indicative local valid time while choosing lead hours; the exact run is shown after loading.
preview_run = find_runs()[0]
st.caption(f'Orientierung: Modelllauf {preview_run:%d.%m.%Y %H:%M} UTC. Bei fehlenden Dateien wird automatisch ein älterer Lauf gewählt; die endgültigen Zeiten stehen an den Karten.')
start_hour, end_hour = st.select_slider('Prognosezeitraum auswählen (+0 bis +24 h)', options=list(range(25)), value=(0, 6), format_func=lambda h: f'+{h} h · {(preview_run+timedelta(hours=h)).astimezone(TZ):%d.%m. %H:%M %Z}')
st.caption(f'Gewählte Zeitspanne: {(preview_run+timedelta(hours=start_hour)).astimezone(TZ):%d.%m.%Y %H:%M %Z} bis {(preview_run+timedelta(hours=end_hour)).astimezone(TZ):%d.%m.%Y %H:%M %Z} (vorläufig). Es werden Karten nur für +{start_hour} bis +{end_hour} erstellt. Für Niederschlagsraten wird zusätzlich der jeweils vorherige Akkumulationsstand geladen.')
if st.button('DWD-Prognose manuell berechnen', type='primary'):
    with st.status('Suche verfügbaren DWD-Modelllauf …', expanded=True) as status:
        try:
            selected = None
            diagnostics = []
            for run in find_runs():
                try:
                    files, examples, matching = list_run(run, 'tot_prec' if is_precip else ('td_2m' if is_dew else ('t_2m' if is_fog else ('vmax_10m' if is_gust else 't_2m'))))
                    requested = list(range(start_hour, end_hour + 1))
                    if is_precip and (is_kind or parameter_page == '🌧️ Rate') and start_hour > 0:
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
            if is_precip:
                hours = list(range(max(0,start_hour-1),end_hour+1)) if (is_kind or parameter_page == '🌧️ Rate') else list(range(start_hour,end_hour+1))
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
            terrain = None
            try:
                if is_gust or is_precip or is_dew or is_fog:
                    raise ValueError('Für diesen Parameter nicht benötigt')
                terrain = get_terrain()
                st.write(f'Geländedaten geladen: {terrain.min():.0f} bis {terrain.max():.0f} m')
            except Exception as terrain_error:
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
            phase_previous = {}
            phase_files = {}
            if is_kind:
                for component in ('rain_gsp','rain_con','snow_gsp','snow_con','grau_gsp'):
                    available_files,_,_ = list_run(run,component)
                    if any(h not in available_files for h in hours):
                        raise RuntimeError(f'Für Niederschlagsart fehlen Daten: {component} in ausgewählter Spanne.')
                    phase_files[component] = available_files
            if quarter_mode:
                # ICON-D2 quarter-hour forecasts are GRIB messages with exact validity times.
                # The containing file is selected by its lead hour; never invent sub-hourly values.
                quarter_images = []
                cached_messages = {}
                cached_grids = {}
                field = 'vmax_10m' if is_gust else 'tot_prec'
                def quarter_field(when):
                    lead = max(0,math.ceil((when-run).total_seconds()/3600-1e-9))
                    if lead not in files:
                        raise RuntimeError(f'DWD-Datei für +{lead} h fehlt.')
                    if lead not in cached_messages:
                        cached_messages.clear()
                        cached_grids.clear()
                        cached_messages[lead] = decode_grib(files[lead],field)
                    matches = [m for m in cached_messages[lead] if abs((m[3]-when).total_seconds())<=60]
                    if not matches:
                        raise RuntimeError(f'Keine echte 15-Minuten-Prognose für {when.astimezone(TZ):%d.%m.%Y %H:%M %Z} in DWD-Datei +{lead} h. Keine zeitliche Interpolation.')
                    if when not in cached_grids:
                        cached_grids[when] = interpolate(*matches[0][:3])
                    return cached_grids[when]
                minutes = range(start_hour*60,end_hour*60+1,15)
                if end_hour-start_hour > 3:
                    raise RuntimeError('15-Minuten-Modus aus Speicher- und Laufzeitgründen auf maximal 3 Stunden pro Berechnung begrenzt. Bitte Zeitraum verkleinern.')
                for mi in minutes:
                    when = run+timedelta(minutes=mi)
                    if not is_gust and mi==0:
                        continue
                    st.write(f'15-Minuten-Karte: {when.astimezone(TZ):%d.%m.%Y %H:%M %Z}')
                    X,Y,Z = quarter_field(when)
                    if is_gust:
                        png=(render_gust(X,Y,Z,run,when),render_gust(X,Y,Z,run,when,True))
                    else:
                        _,_,prior = quarter_field(when-timedelta(minutes=15))
                        if np.nanmin(Z-prior)<-.1:
                            raise RuntimeError('Akkumulation zwischen Viertelstunden fällt ab.')
                        rate=np.maximum(Z-prior,0)*4
                        png=(render_precip(X,Y,rate,run,when,'🌧️ Rate'),render_precip(X,Y,rate,run,when,'🌧️ Rate',True))
                    quarter_images.append((mi/60,when,png,None,None,None))
                if not quarter_images:
                    raise RuntimeError('Keine Viertelstundenkarten vorhanden.')
                st.session_state[IMAGE_KEYS[parameter_page]] = quarter_images
                status.update(label='15-Minuten-Karten erstellt',state='complete')
            if not quarter_mode:
                for idx, hour in enumerate(hours):
                    st.write(f'Lade Daten +{hour} h ({idx+1}/{len(hours)}) …')
                    messages = decode_grib(files[hour], 'tot_prec' if is_precip else ('td_2m' if is_dew or is_fog else ('vmax_10m' if is_gust else 't_2m')))
                    # Select the exact hourly forecast time, never a neighboring quarter-hour.
                    target = run + timedelta(hours=hour)
                    msg = min(messages, key=lambda m: abs((m[3]-target).total_seconds()))
                    if abs((msg[3]-target).total_seconds()) > 60:
                        raise RuntimeError(f'Kein exakter Prognosezeitpunkt für +{hour} h vorhanden.')
                    X, Y, Z = interpolate(*msg[:3],grid_m=500 if is_fog else None)
                    if is_precip:
                        if is_kind:
                            increments = {}
                            for component,component_files in phase_files.items():
                                component_messages = decode_grib(component_files[hour],component)
                                component_msg = min(component_messages,key=lambda m: abs((m[3]-target).total_seconds()))
                                if abs((component_msg[3]-target).total_seconds()) > 60:
                                    raise RuntimeError(f'Zeitstempel stimmt nicht: {component} +{hour} h.')
                                _,_,component_grid = interpolate(*component_msg[:3])
                                if component in phase_previous:
                                    increments[component] = np.maximum(component_grid-phase_previous[component],0)
                                else:
                                    increments[component] = np.zeros_like(component_grid)
                                phase_previous[component] = component_grid
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
                        if is_kind:
                            # Separate rain/snow components are needed; no temperature-only guessing.
                            display = classify_precipitation(hourly,
                                increments['rain_gsp']+increments['rain_con'],
                                increments['snow_gsp']+increments['snow_con'],
                                increments['grau_gsp'])
                            original_png = (render_precip(X,Y,display,run,msg[3],parameter_page),)*2
                        else:
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
                        original_png = (render_gust(X,Y,Z,run,msg[3]), render_gust(X,Y,Z,run,msg[3],show_km_values=True)) if is_gust else (render_png(X,Y,Z,run,msg[3]), render_png(X,Y,Z,run,msg[3],show_km_values=True))
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
            if not quarter_mode:
                st.session_state[IMAGE_KEYS[parameter_page]] = images
                status.update(label='Karten erstellt', state='complete')
        except Exception as exc:
            status.update(label='Abruf oder Verarbeitung fehlgeschlagen', state='error')
            st.error(f'{type(exc).__name__}: {exc}')

image_key = IMAGE_KEYS[parameter_page]
if image_key in st.session_state:
    images = st.session_state[image_key]
    labels = [f'+{item[0]:g} h · {item[1].astimezone(TZ):%d.%m.%Y %H:%M %Z}' for item in images]
    choice = st.select_slider('Vorhersagekarte', options=range(len(images)),
                              format_func=lambda i: labels[i], key='hour_'+image_key)
    st.caption(f'Ausgewählt: {labels[choice]}')
    h, valid, original_png, local_png, combined_png, diff_png = images[choice]
    variants = {parameter_page: original_png} if (is_precip or is_dew or is_fog) else ({'ICON-D2 Windböen': original_png} if is_gust else {'ICON-D2 (interpoliert)': original_png})
    if local_png is not None and parameter_page == '🌡️ Temperatur':
        variants['Höhenkorrektur'] = local_png
    if combined_png is not None and parameter_page == '🌡️ Temperatur':
        variants['Höhe + Landnutzung'] = combined_png
        variants['Differenz zu ICON-D2'] = diff_png
    variant = st.radio('Kartentyp', list(variants), horizontal=True, key='variant_'+image_key)
    show_km_values = st.checkbox('Werte alle ca. 1 km anzeigen', disabled=is_kind or is_fog, value=False, key='values_'+image_key, help='Interpolierte Modellwerte am jeweiligen Punkt, keine Messwerte.')
    selected_map = variants[variant]
    png = selected_map if variant == 'Differenz zu ICON-D2' else selected_map[int(show_km_values)]
    st.image(png, width='stretch')
    st.download_button('PNG herunterladen', data=png,
                       file_name=f'ELBTAL_{image_key}_{list(variants).index(variant)}_{valid:%Y%m%d_%H%M}_UTC.png', mime='image/png')

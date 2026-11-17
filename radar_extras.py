"""ELBTAL radar extensions: HYMEC precipitation type, RW totals and city labels."""
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from urllib.parse import urljoin
import io
import re

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
import requests
import streamlit as st
from pyproj import CRS, Transformer
from scipy.ndimage import map_coordinates

import radar
from map_design import frame_png

HYMEC_DIR = radar.BASE + "composite/hymecng/"
RW_DIR = radar.BASE + "radolan/rw/"
PATTERNS = {
    "HYMEC": re.compile(r"^composite_HymecNG_(\d{8}_\d{4})_000-hd5$", re.I),
    "RW": re.compile(r"^raa01-rw_10000-(\d{10})-dwd---bin\.hdf5$", re.I),
}
# German municipalities above ~500,000 inhabitants plus ELBTAL's Mühlberg.
CITIES = (
    ("Berlin", 13.405, 52.520), ("Hamburg", 9.993, 53.551),
    ("München", 11.582, 48.135), ("Köln", 6.961, 50.938),
    ("Frankfurt/M.", 8.682, 50.111), ("Stuttgart", 9.182, 48.776),
    ("Düsseldorf", 6.774, 51.228), ("Leipzig", 12.375, 51.340),
    ("Dortmund", 7.466, 51.514), ("Essen", 7.012, 51.456),
    ("Bremen", 8.802, 53.079), ("Dresden", 13.738, 51.050),
    ("Hannover", 9.732, 52.375), ("Nürnberg", 11.077, 49.454),
    ("Duisburg", 6.763, 51.434), ("Mühlberg/E.", 13.222, 51.434),
)
HYMEC_NAMES = (
    "Kein Niederschlag", "Nicht klassifiziert", "Sprühregen", "Regen",
    "Gefr. Sprühregen", "Gefrierender Regen", "Schneeregen", "Schnee",
    "Graupel", "Hagel", "Großer Hagel",
)
HYMEC_COLORS = (
    "#10273e", "#777777", "#86b59a", "#29cd3a", "#e7b44a",
    "#f44643", "#f8a04b", "#46d6e9", "#448bdf", "#c63ccf", "#9015a8",
)
SUM_EDGES = (0, .1, .5, 1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 60, 80, 100, 150)
SUM_COLORS = (
    "#29506d", "#247eab", "#2db0d4", "#57d9d8", "#3dc69c",
    "#50c64b", "#96d43d", "#e5e942", "#f4c63b", "#ee9c31",
    "#ef7332", "#eb4c35", "#d73255", "#b52985", "#8e28b0", "#65228a",
)

class _Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.names = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "a":
            self.names.extend(value.split("?")[0].split("/")[-1]
                              for key, value in attrs if key.lower() == "href" and value)

@st.cache_data(ttl=120, show_spinner=False)
def list_extra_frames(product):
    folder = HYMEC_DIR if product == "HYMEC" else RW_DIR
    response = requests.get(folder, headers=radar.HEADERS, timeout=30)
    response.raise_for_status()
    parser = _Links()
    parser.feed(response.text)
    results = []
    for name in parser.names:
        match = PATTERNS[product].fullmatch(name)
        if not match:
            continue
        fmt = "%Y%m%d_%H%M" if product == "HYMEC" else "%y%m%d%H%M"
        try:
            dt = datetime.strptime(match.group(1), fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        results.append((dt, urljoin(folder, name)))
    return sorted(set(results), reverse=True)[:36 if product == "HYMEC" else 320]

@st.cache_data(ttl=600, show_spinner=False, max_entries=100)
def read_extra_frame(url, product):
    response = requests.get(url, headers=radar.HEADERS, timeout=65)
    response.raise_for_status()
    with h5py.File(io.BytesIO(response.content), "r") as h5:
        where = h5["where"].attrs
        nx, ny = int(where["xsize"]), int(where["ysize"])
        projection = where["projdef"]
        if isinstance(projection, bytes):
            projection = projection.decode()
        crs = CRS.from_user_input(projection)
        convert = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        x0, y0 = convert.transform(float(where["LL_lon"]), float(where["LL_lat"]))
        x1, y1 = convert.transform(float(where["UR_lon"]), float(where["UR_lat"]))
        group = h5["dataset1"]["data1"]
        encoded = np.asarray(group["data"])
        if encoded.shape != (ny, nx):
            raise RuntimeError(f"Unerwartetes Radar-Raster: {encoded.shape}")
        attrs = group["what"].attrs
        quantity = attrs.get("quantity", b"")
        if isinstance(quantity, bytes):
            quantity = quantity.decode()
        expected = "CLASS" if product == "HYMEC" else "ACRR"
        if quantity != expected:
            raise RuntimeError(f"DWD-Feld {quantity!r}; erwartet {expected!r}.")
        nodata = float(attrs.get("nodata", -9999))
        undetect = float(attrs.get("undetect", -9998))
        missing = encoded == nodata
        values = encoded.astype(np.float32) * float(attrs["gain"]) + float(attrs["offset"])
        if product == "HYMEC":
            # ODIM undetect=254 is scanned absence of precipitation.
            values[encoded == undetect] = 0
            invalid = (values < 0) | (values > 10)
        else:
            # RW ACRR already holds a 60-minute precipitation total (mm).
            values[encoded == undetect] = 0
            invalid = values < 0
        values[missing | invalid] = np.nan
        return values, (crs.to_string(), x0, y0, x1, y1)

@st.cache_data(ttl=600, show_spinner=False, max_entries=10)
def accumulated_rw(hours, end_time):
    index = dict(list_extra_frames("RW"))
    # Consecutive non-overlapping 60-minute windows; RW also publishes
    # moving hourly sums every 10 minutes, which must NOT all be added.
    required = [end_time - timedelta(hours=back) for back in range(hours)]
    absent = [t for t in required if t not in index]
    if absent:
        raise RuntimeError(
            f"{len(absent)} der {hours} stündlichen RW-Felder fehlen. "
            "Keine unvollständige Niederschlagssumme dargestellt."
        )
    total = None
    valid = None
    geometry = None
    for stamp in required:
        field, this_geo = read_extra_frame(index[stamp], "RW")
        if total is None:
            total = np.zeros(field.shape, dtype=np.float32)
            valid = np.ones(field.shape, dtype=bool)
            geometry = this_geo
        if field.shape != total.shape or this_geo != geometry:
            raise RuntimeError("RW-Rastergeometrie der Stunden ist nicht identisch.")
        valid &= np.isfinite(field)
        total += np.nan_to_num(field, nan=0.0)
    total[~valid] = np.nan
    return total, geometry

def render_extra(data, geometry, product, stamp, cities=False, hours=None):
    X, Y, rows, cols = radar.sample_geometry(*geometry, data.shape)
    valid = (rows >= 0) & (rows < data.shape[0]) & (cols >= 0) & (cols < data.shape[1])
    grid = map_coordinates(np.nan_to_num(data, nan=-9999), [rows, cols],
                           order=0, mode="constant", cval=-9999)
    grid = np.where(valid & (grid >= 0), grid, np.nan)
    if product == "HYMEC":
        levels = np.arange(-.5, 11.5, 1)
        colors = HYMEC_COLORS
        ticks = np.arange(11)
        legend_title = "Niederschlagsart"
        title = "Live-Wetteranalyse"
    elif product == "RW":
        levels = SUM_EDGES
        colors = SUM_COLORS
        ticks = [0, 1, 3, 5, 10, 20, 40, 80, 150]
        legend_title = f"Niederschlagssumme {hours} h (mm)"
        title = f"Niederschlagssumme {hours} h"
    else:
        levels = radar.DBZ_BOUNDS if product == "WN" else radar.RATE_BOUNDS
        colors = radar.COLORS
        ticks = ([0, 10, 20, 30, 40, 50, 60, 70] if product == "WN"
                 else [0, .5, 1, 2, 5, 10, 20, 50, 100])
        legend_title = ("Radarreflektivität (dBZ)" if product == "WN"
                        else "Niederschlagsintensität (mm/h) · 5-Minuten-Mittel")
        title = "Radarreflektivität" if product == "WN" else "Radar-Regenrate"
    cmap = ListedColormap(colors)
    cmap.set_bad("#10273e")
    norm = BoundaryNorm(levels, cmap.N)
    fig, ax = plt.subplots(figsize=(15, 8.2), dpi=120)
    fig.patch.set_facecolor("#071e35")
    ax.set_facecolor("#10273e")
    image = ax.imshow(grid, origin="lower", extent=(5.4, 15.6, 46.9, 55.4),
                      cmap=cmap, norm=norm, interpolation="nearest", aspect="auto")
    radar.draw_boundaries(ax)
    if cities:
        for name, lon, lat in CITIES:
            ax.scatter([lon], [lat], s=9, marker="s", color="white", zorder=30)
            ax.annotate(name, (lon, lat), xytext=(3, 4), textcoords="offset points",
                        fontsize=8, color="white", zorder=31,
                        path_effects=[patheffects.withStroke(linewidth=2.5,
                                                              foreground="#061c35")])
    ax.set_xlim(5.4, 15.6)
    ax.set_ylim(46.9, 55.4)
    ax.tick_params(colors="#d6e5f2", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#78a4c2")
    ax.grid(color="white", alpha=.12, linewidth=.5)
    ax.set_xlabel("Längengrad (°O)", color="#d6e5f2")
    ax.set_ylabel("Breitengrad (°N)", color="#d6e5f2")
    fig.subplots_adjust(left=.055, right=.855, top=.97, bottom=.095)
    cax = fig.add_axes([.89, .15, .025, .76])
    bar = fig.colorbar(image, cax=cax, orientation="vertical",
                       spacing="uniform", ticks=ticks)
    bar.ax.tick_params(colors="#d6e5f2", labelsize=9)
    if product == "HYMEC":
        bar.set_ticklabels(HYMEC_NAMES)
        bar.ax.tick_params(labelsize=7)
    bar.set_label(legend_title, color="#d6e5f2")
    output = io.BytesIO()
    fig.savefig(output, format="png", facecolor=fig.get_facecolor(), dpi=120)
    plt.close(fig)
    return frame_png(output.getvalue(), title, "DWD RADOLAN", "Deutschland",
                     stamp, stamp, portrait_override=False)

def show():
    st.subheader("📡 ELBTAL | Regenradar Deutschland")
    mode = st.selectbox(
        "Radarparameter",
        ["Reflektivität (dBZ)", "Regenrate (mm/h)", "Live-Wetteranalyse", "Niederschlagssummen"],
        key="radar_mode_new",
    )
    product = {
        "Reflektivität (dBZ)": "WN", "Regenrate (mm/h)": "RY",
        "Live-Wetteranalyse": "HYMEC", "Niederschlagssummen": "RW",
    }[mode]
    hours = (st.selectbox("Zeitraum", [1, 3, 6, 12, 24],
                          format_func=lambda n: f"{n} Stunden", key="radar_rw_hours")
             if product == "RW" else None)
    cities = st.checkbox("Städtenamen einblenden (über 500.000 Einwohner + Mühlberg/E.)",
                         value=False, key="radar_city_overlay")
    st.caption("DWD-Radardaten · HYMEC: bodennahe Niederschlagsart; "
               "RW: stationskorrigierte stündliche Niederschlagssummen.")
    try:
        frames = (list_extra_frames(product) if product in ("HYMEC", "RW")
                  else radar.list_frames(product))
    except Exception as exc:
        st.error(f"Radarverzeichnis nicht erreichbar: {type(exc).__name__}: {exc}")
        return
    if not frames:
        st.warning("Zurzeit keine passenden Radardaten verfügbar.")
        return
    if product == "RW":
        stamp, url = frames[0]
        st.caption(f"Endzeitpunkt: {stamp.astimezone(radar.TZ):%d.%m.%Y %H:%M %Z} "
                   f"· Niederschlag der letzten {hours} Stunden")
    else:
        idx = st.select_slider(
            "Radarzeit (MEZ/MESZ)", options=list(range(len(frames))), value=0,
            format_func=lambda i: frames[i][0].astimezone(radar.TZ).strftime("%d.%m.%Y %H:%M %Z"),
        )
        stamp, url = frames[idx]
        st.caption(f"Messzeit: {stamp.astimezone(radar.TZ):%d.%m.%Y %H:%M %Z} "
                   f"· UTC {stamp:%H:%M}")
    cache_key = (product, url, hours, cities)
    if st.button("📡 Radaraufnahme laden", type="primary"):
        try:
            with st.spinner("Originale DWD-Radardaten laden …"):
                if product == "RW":
                    field, geo = accumulated_rw(hours, stamp)
                elif product == "HYMEC":
                    field, geo = read_extra_frame(url, product)
                else:
                    field, geo = radar.load_frame(url, product)
                png = render_extra(field, geo, product, stamp, cities=cities, hours=hours)
            st.session_state["radar_image_extra"] = (cache_key, png)
        except Exception as exc:
            st.error(f"Radaraufnahme nicht darstellbar: {type(exc).__name__}: {exc}")
    cached = st.session_state.get("radar_image_extra", (None, None))
    if cached[0] == cache_key:
        png = cached[1]
        st.image(png, width="stretch")
        st.download_button("PNG herunterladen", png,
                           file_name=f"ELBTAL_Radar_{product}_{stamp:%Y%m%d_%H%M}.png",
                           mime="image/png")
    if product == "RW":
        st.caption("RW: 60-Minuten-Mengen (mm) in genau einstündigem Abstand. "
                   "Unvollständige Reihen werden nicht als gültige Summe angezeigt.")
    elif product == "HYMEC":
        st.caption("HYMEC ist eine Radar- und Zusatzdatenanalyse der Niederschlagsart "
                   "in 2 m Höhe, keine direkte Einzelmessung der Niederschlagsart.")

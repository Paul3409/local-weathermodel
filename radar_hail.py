"""ELBTAL KONRAD3D hail-signal map, based on official DWD 5-minute XML."""
from datetime import datetime, timezone
from html.parser import HTMLParser
from urllib.parse import urljoin
import io
import re
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
import numpy as np
import requests
import streamlit as st

import radar
from map_design import frame_png

BASE = radar.BASE + "konrad3d/"
PATTERN = re.compile(r"^KONRAD3D_(\d{8}T\d{6})\.xml$", re.I)
# Five map classes: zero and two area subdivisions for each DWD hail flag.
# 5 km² is a display convention, not a validated probability threshold.
LABELS = ("Kein Hagelsignal", "Hagelsignal", "Hagelsignal (ausgedehnt)",
          "Großhagelsignal", "Großhagelsignal (ausgedehnt)")
COLORS = ("#10273e", "#f5dc52", "#f49a38", "#e94960", "#9026ad")


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.names = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "a":
            self.names.extend(v.split("?")[0].split("/")[-1]
                              for k, v in attrs if k.lower() == "href" and v)


@st.cache_data(ttl=120, show_spinner=False)
def list_frames():
    response = requests.get(BASE, headers=radar.HEADERS, timeout=30)
    response.raise_for_status()
    parser = Links()
    parser.feed(response.text)
    found = []
    for name in parser.names:
        match = PATTERN.fullmatch(name)
        if match:
            try:
                stamp = datetime.strptime(match.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            found.append((stamp, urljoin(BASE, name)))
    return sorted(set(found), reverse=True)[:36]


def child_text(element, path):
    child = element.find(path)
    return child.text.strip() if child is not None and child.text else None


def read_float(element, path, default=0.0):
    try:
        return float(child_text(element, path))
    except (ValueError, TypeError):
        return default


def risk_class(feature):
    """DWD flag 0/1/2, with area split only for visualization."""
    flag = int(read_float(feature, "./intensity/hail_flag", -1))
    if flag not in (0, 1, 2):
        return None
    if flag == 0:
        return 0
    area_path = ("./hymec/area_large_hail" if flag == 2
                 else "./hymec/area_hail_total")
    area = read_float(feature, area_path, 0)
    return (3 if flag == 2 else 1) + int(area >= 5.0)


@st.cache_data(ttl=300, show_spinner=False, max_entries=24)
def load_cells(url):
    response = requests.get(url, headers=radar.HEADERS, timeout=40)
    response.raise_for_status()
    # Reject malformed or unexpectedly large XML files.
    if len(response.content) > 12_000_000:
        raise RuntimeError("KONRAD3D-Datei ist unerwartet groß.")
    root = ET.fromstring(response.content)
    cells = []
    for feature in root.findall("./cells/feature"):
        grade = risk_class(feature)
        if grade is None:
            continue
        polygons = []
        for polygon in feature.findall("./geometry/polygons_projected/geodetic_coordinates/polygon"):
            lat = child_text(polygon, "latitudes")
            lon = child_text(polygon, "longitudes")
            if not lat or not lon:
                continue
            try:
                lats = [float(v) for v in lat.split()]
                lons = [float(v) for v in lon.split()]
            except ValueError:
                continue
            if len(lats) == len(lons) and len(lats) >= 3:
                polygons.append(list(zip(lons, lats)))
        lon = read_float(feature, "./geometry/centroid_3d/geodetic_coordinate/longitude", float("nan"))
        lat = read_float(feature, "./geometry/centroid_3d/geodetic_coordinate/latitude", float("nan"))
        cells.append((grade, polygons, lon, lat))
    return cells


def render(cells, stamp, cities=False):
    fig, ax = plt.subplots(figsize=(15, 8.2), dpi=120)
    fig.patch.set_facecolor("#071e35")
    ax.set_facecolor("#10273e")
    ax.set_xlim(5.4, 15.6)
    ax.set_ylim(46.9, 55.4)
    # Show colored footprint of each identified convective cell, not a
    # fictitious countrywide interpolated probability field.
    for grade, polygons, lon, lat in sorted(cells, key=lambda item: item[0]):
        if grade == 0:
            continue
        for vertices in polygons:
            ax.add_patch(Polygon(vertices, closed=True, facecolor=COLORS[grade],
                                 edgecolor="#f1f4fb", linewidth=.6, alpha=.85, zorder=8))
        if not polygons and np.isfinite(lon) and np.isfinite(lat):
            ax.scatter([lon], [lat], color=COLORS[grade], s=70,
                       edgecolors="white", linewidths=.6, zorder=9)
    radar.draw_boundaries(ax)
    if cities:
        from radar_extras import CITIES
        from matplotlib import patheffects
        for name, lon, lat in CITIES:
            ax.scatter([lon], [lat], s=9, marker="s", color="white", zorder=30)
            ax.annotate(name, (lon, lat), xytext=(3, 4), textcoords="offset points",
                        fontsize=8, color="white", zorder=31,
                        path_effects=[patheffects.withStroke(linewidth=2.5,
                                                              foreground="#061c35")])
    ax.tick_params(colors="#d6e5f2", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#78a4c2")
    ax.grid(color="white", alpha=.12, linewidth=.5)
    ax.set_xlabel("Längengrad (°O)", color="#d6e5f2")
    ax.set_ylabel("Breitengrad (°N)", color="#d6e5f2")
    fig.subplots_adjust(left=.055, right=.855, top=.97, bottom=.095)
    cax = fig.add_axes([.89, .15, .025, .76])
    palette = ListedColormap(COLORS)
    norm = BoundaryNorm(np.arange(-.5, 5.5), palette.N)
    from matplotlib.cm import ScalarMappable
    bar = fig.colorbar(ScalarMappable(norm=norm, cmap=palette), cax=cax,
                       orientation="vertical", ticks=range(5))
    bar.set_ticklabels(LABELS)
    bar.ax.tick_params(colors="#d6e5f2", labelsize=7)
    bar.set_label("Hagelsignal (DWD + ELBTAL-Flächenabstufung)", color="#d6e5f2")
    output = io.BytesIO()
    fig.savefig(output, format="png", facecolor=fig.get_facecolor(), dpi=120)
    plt.close(fig)
    return frame_png(output.getvalue(), "Hagelrisiko", "DWD KONRAD3D",
                     "Deutschland", stamp, stamp, portrait_override=False)

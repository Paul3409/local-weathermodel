"""ELBTAL radar-based hail potential indicator (NOT a measured hail probability).

The area shading uses DWD WN reflectivity only. Reflectivity is not
hail-specific; strong rain can also produce high values. KONRAD3D hail
flags are shown as separate, independently observed overlays.
"""
import io
from datetime import timedelta
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Polygon
from matplotlib import patheffects
import numpy as np
from scipy.ndimage import map_coordinates
import radar
from map_design import frame_png

# Lower classes distinguish no echo and weak echoes. These are not probabilities.
EDGES = [-.5, .5, 15, 25, 35, 45, 50, 55, 60, 70, 1000]
COLORS = ["#10273e", "#777777", "#a0a0a0", "#67aa69",
          "#138e39", "#ffd93b", "#bb9a2b", "#e36a20",
          "#e52d50", "#b635d0"]
LABELS = ["Kein Echo", "< 15 dBZ", "15–24 dBZ", "25–34 dBZ",
          "35–44 dBZ", "45–49 dBZ", "50–54 dBZ", "55–59 dBZ",
          "60–69 dBZ", "≥ 70 dBZ"]


def matching_cells(stamp):
    """Get KONRAD3D only if a scan close to the radar time is available."""
    import radar_hail
    frames = radar_hail.list_frames()
    if not frames:
        return []
    candidate = min(frames, key=lambda x: abs((x[0] - stamp).total_seconds()))
    if abs((candidate[0] - stamp).total_seconds()) > 5 * 60:
        return []
    return radar_hail.load_cells(candidate[1])


def render(values, geometry, stamp, cells=(), cities=False, region="Deutschland"):
    X, Y, row, col = radar.sample_geometry(*geometry, values.shape, region)
    valid = (row >= 0) & (row < values.shape[0]) & (col >= 0) & (col < values.shape[1])
    sampled = map_coordinates(np.nan_to_num(values, nan=-9999),
                              [row, col], order=0, mode="constant", cval=-9999)
    sampled = np.where(valid & (sampled >= 0), sampled, np.nan)
    # Distinguish coverage gaps from measured low/absent reflectivity.
    display = np.where(np.isfinite(sampled), sampled, np.nan)
    palette = ListedColormap(COLORS)
    palette.set_bad("#10273e")
    norm = BoundaryNorm(EDGES, palette.N)

    fig, ax = plt.subplots(figsize=(15, 8.2), dpi=120)
    fig.patch.set_facecolor("#071e35")
    ax.set_facecolor("#10273e")
    ax.imshow(display, origin="lower", extent=radar.RADAR_REGIONS[region],
              aspect="auto", cmap=palette, norm=norm, interpolation="nearest")

    # Outlines indicate actual KONRAD3D hail detection; no filled fake probabilities.
    for grade, polygons, lon, lat in cells:
        if grade <= 0:
            continue
        color = "#ff67f1" if grade >= 3 else "#ffffff"
        for vertices in polygons:
            ax.add_patch(Polygon(vertices, closed=True, facecolor="none",
                                 edgecolor=color, linewidth=2.2, zorder=18))
        if not polygons and np.isfinite(lon) and np.isfinite(lat):
            ax.scatter([lon], [lat], marker="o", facecolors="none",
                       edgecolors=color, s=90, linewidths=2, zorder=18)

    radar.draw_boundaries(ax, region)
    if cities:
        from radar_extras import CITIES
        for name, lon, lat in CITIES:
            w,e,s,n=radar.RADAR_REGIONS[region]
            if not (w<=lon<=e and s<=lat<=n): continue
            ax.scatter([lon], [lat], s=9, marker="s", color="white", zorder=30)
            ax.annotate(name, (lon, lat), xytext=(3, 4), textcoords="offset points",
                        fontsize=8, color="white", zorder=31,
                        path_effects=[patheffects.withStroke(linewidth=2.5,
                                                              foreground="#061c35")])
    ax.set_xlim(*radar.RADAR_REGIONS[region][:2])
    ax.set_ylim(*radar.RADAR_REGIONS[region][2:])
    ax.tick_params(colors="#d6e5f2", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#78a4c2")
    ax.grid(color="white", alpha=.12, linewidth=.5)
    ax.set_xlabel("Längengrad (°O)", color="#d6e5f2")
    ax.set_ylabel("Breitengrad (°N)", color="#d6e5f2")
    fig.subplots_adjust(left=.055, right=.855, top=.97, bottom=.095)
    cax = fig.add_axes([.89, .15, .025, .76])
    from matplotlib.cm import ScalarMappable
    cb = fig.colorbar(ScalarMappable(cmap=palette, norm=norm), cax=cax,
                      orientation="vertical", ticks=[0, 10, 20, 30, 40, 47.5, 52.5, 57.5, 65, 80])
    cb.set_ticklabels(LABELS)
    cb.ax.tick_params(colors="#d6e5f2", labelsize=8)
    cb.set_label("Radarindikator (dBZ) · keine Hagelwahrscheinlichkeit", color="#d6e5f2")
    output = io.BytesIO()
    fig.savefig(output, format="png", facecolor=fig.get_facecolor(), dpi=120)
    plt.close(fig)
    return frame_png(output.getvalue(), "Hagelrisiko HD", "DWD WN + KONRAD3D",
                     region, stamp, stamp, portrait_override=False)

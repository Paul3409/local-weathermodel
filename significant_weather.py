"""DWD ICON WW weather-interpretation categories; never interpolate code values linearly."""
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

# Order: cloud cover, fog, precipitation, thunderstorm; reference colors chosen by ELBTAL.
CATEGORIES = [
    ("Wolkenlos", "#F2F2F2"),
    ("Heiter", "#D0D0D0"),
    ("Wolkig", "#969696"),
    ("Bedeckt", "#5C5C5C"),
    ("Nebel", "#FFFF31"),
    ("Regen leicht", "#37FA35"),
    ("Regen mäßig", "#29C72A"),
    ("Regen stark", "#078C12"),
    ("Schneeregen leicht", "#F8BB81"),
    ("Schneeregen mäßig/stark", "#F49A39"),
    ("Schnee leicht", "#39E2ED"),
    ("Schnee mäßig", "#438CF6"),
    ("Schnee stark", "#2A56AD"),
    ("Gefr. Regen leicht", "#F24552"),
    ("Gefr. Regen mäßig/stark", "#CE070E"),
    ("Gewitter leicht/mäßig", "#EF52ED"),
    ("Gewitter stark", "#BD16BC"),
    ("Unbekannter WW-Code", "#85868D"),
]
# DWD WW: 0..3 cloud amount, 45/48 fog, 51..57 drizzle,
# 61..67 rain, 71..77 snow, 80..86 showers, 95/96/99 thunder.
# WW=48 (rime fog) is grouped into generic fog; no separate legend entry.
# DWD ICON normally does not emit 68/69 (sleet), but map them if present.
WW_TO_CATEGORY = {
    0:0,1:1,2:2,3:3,
    45:4,48:4,
    51:5,53:6,55:7,
    56:13,57:14,
    61:5,63:6,65:7,
    66:13,67:14,
    68:8,69:9,
    71:10,73:11,75:12,77:10,
    80:5,81:6,82:7,
    85:10,86:11,
    95:15,96:16,99:16,
}
def classify_ww(values):
    """Return categorical indices, preserving unknown/missing values explicitly."""
    result=np.full(np.shape(values),17,dtype=np.int16)
    rounded=np.rint(values)
    valid=np.isfinite(values)&(np.abs(values-rounded)<.01)
    for code,category in WW_TO_CATEGORY.items():
        result[valid&(rounded==code)]=category
    return result

def palette():
    cmap=ListedColormap([color for _,color in CATEGORIES])
    bounds=np.arange(-.5,len(CATEGORIES)+.5,1)
    return cmap,BoundaryNorm(bounds,cmap.N),bounds

def legend_handles():
    return [Patch(facecolor=color,edgecolor="#4B5563",linewidth=.35,label=name)
            for name,color in CATEGORIES[:-1]]

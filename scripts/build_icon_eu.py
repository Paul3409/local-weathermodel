"""Generate Central Europe ICON-EU temperature maps directly from DWD GRIB."""
import bz2, json, re, tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urljoin
import requests
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import BoundaryNorm, ListedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.spatial import cKDTree
from eccodes import codes_grib_new_from_file, codes_get_array, codes_get, codes_release

BASE = "https://opendata.dwd.de/weather/nwp/icon-eu/grib/"
OUT = Path("_site/icon-eu")
OUT.mkdir(parents=True, exist_ok=True)
hours = list(range(0,49,3))
now = datetime.now(timezone.utc)
cycle = now.replace(hour=now.hour//6*6, minute=0, second=0, microsecond=0)
for offset in range(0, 37, 6):
    run = cycle - timedelta(hours=offset)
    stamp = run.strftime("%Y%m%d%H")
    folder = urljoin(BASE, f"{run:%H}/t_2m/")
    response = requests.get(folder, timeout=40)
    if response.status_code == 404:
        continue
    response.raise_for_status()
    files = {}
    for name in re.findall(r'href="([^"]+)"', response.text):
        m = re.search(r"_"+stamp+r"_(\d{3})(?:_|\.)", name)
        if m and name.endswith(".grib2.bz2") and 'regular-lat-lon' in name:
            files[int(m[1])] = urljoin(folder, name)
    if all(h in files for h in hours):
        hours = [h for h in range(0,121,3) if h in files]
        break
else:
    raise RuntimeError("No complete ICON-EU cycle found")

PARAMS = {
    "t2": ("t_2m", "2-m-Temperatur", "°C", -273.15, 1, None),
    "gust": ("vmax_10m", "Windböen", "km/h", 0, 3.6, [0,20,40,60,80,100,120,150,200]),
    "precip": ("tot_prec", "Niederschlagssumme seit Modellstart", "mm", 0, 1, [0,.1,1,2,5,10,20,40,80,150]),
    "snow": ("h_snow", "Schneehöhe", "cm", 0, 100, [0,.1,1,2,5,10,20,40,80,150]),
}
products = []
grid_cache = {}
session = requests.Session()
def decode(param, url):
    response = session.get(url, timeout=90)
    response.raise_for_status()
    with tempfile.TemporaryFile() as f:
        f.write(bz2.decompress(response.content))
        f.seek(0)
        gid = codes_grib_new_from_file(f)
        if gid is None:
            raise ValueError("Empty GRIB")
        try:
            lat = np.asarray(codes_get_array(gid,"latitudes"))
            lon = np.asarray(codes_get_array(gid,"longitudes"))
            val = np.asarray(codes_get_array(gid,"values")) * PARAMS[param][4] + PARAMS[param][3]
            valid = datetime.strptime(str(codes_get(gid,"validityDate"))+f'{codes_get(gid,"validityTime"):04d}',"%Y%m%d%H%M").replace(tzinfo=timezone.utc)
        finally:
            codes_release(gid)
    return lat, lon, val, valid

def render(param, h, decoded):
    lat, lon, val, valid = decoded
    mask = (lat>=43.5)&(lat<=58.5)&(lon>=1.5)&(lon<=27.5)&np.isfinite(val)&(val>-100)&(val<10000)
    if mask.sum()<100:
        raise ValueError("Insufficient Central Europe grid points")
    x,y=np.meshgrid(np.linspace(2,27,500),np.linspace(44,58,350))
    key = (lat[mask].tobytes(), lon[mask].tobytes())
    if key not in grid_cache:
        if len(grid_cache) >= 4:
            grid_cache.clear()
        tree = cKDTree(np.column_stack((lon[mask],lat[mask])))
        dist, idx = tree.query(np.column_stack((x.ravel(),y.ravel())),k=4)
        weights = 1 / np.maximum(dist,1e-9)**2
        weights /= weights.sum(axis=1,keepdims=True)
        grid_cache[key] = (idx,weights)
    idx,weights = grid_cache[key]
    z = (val[mask][idx]*weights).sum(axis=1).reshape(x.shape)
    fig=plt.figure(figsize=(10,7),facecolor="#101a2b")
    ax=fig.add_axes([.035,.19,.93,.70],projection=ccrs.PlateCarree())
    cax=fig.add_axes([.12,.095,.76,.025])
    ax.set_extent([2,27,44,58],crs=ccrs.PlateCarree())
    levels=[-30,-20,-15,-10,-5,0,5,10,15,20,25,30,35,40,45]
    cmap=ListedColormap(["#602080","#843ab5","#3544b8","#3978db","#79c9f2","#d5f1fa","#b9e4a0","#74bc62","#d5df56","#f9df48","#f5a33b","#e85c31","#bd243a","#7e123f"])
    if PARAMS[param][5] is not None:
        levels=PARAMS[param][5]
        cmap=plt.get_cmap("YlGnBu" if param in ("precip","rate","snow") else "YlOrRd",len(levels)-1)
    norm=BoundaryNorm(levels,cmap.N)
    im=ax.pcolormesh(x,y,z,cmap=cmap,norm=norm,shading="auto",transform=ccrs.PlateCarree())
    ax.add_feature(cfeature.BORDERS.with_scale("50m"),linewidth=.8,edgecolor="#263345")
    ax.coastlines(resolution="50m",linewidth=.8,color="#263345")

    for name,a,b in [("Berlin",52.52,13.405),("Hamburg",53.55,9.99),("München",48.14,11.58),("Köln",50.94,6.96),("Frankfurt",50.11,8.68),("Leipzig",51.34,12.37),("Paris",48.86,2.35),("Wien",48.21,16.37),("Warschau",52.23,21.01),("Prag",50.08,14.44)]:
        ax.plot(b,a,"k.",ms=5)
        ax.annotate(name,(b,a),xytext=(4,5),textcoords="offset points",ha="left",fontsize=8,color="#142136",clip_on=True,path_effects=[pe.withStroke(linewidth=2.5,foreground="white")])
    fig.text(.035,.965,f"ICON-EU | Mitteleuropa | {PARAMS[param][1]}\n{valid:%d.%m.%Y %H:%M} UTC | +{h} h",color="white",fontsize=12,va="top")
    ax.tick_params(colors="white")
    cb=fig.colorbar(im,cax=cax,orientation="horizontal",label=PARAMS[param][2],extend="both")
    cb.ax.tick_params(colors="white")
    cb.ax.xaxis.label.set_color("white")
    cb.ax.tick_params(labelsize=8)
    fig.text(.5,.018,"DWD ICON-EU · Grenzen: Natural Earth · IDW-interpoliert",ha="center",color="white",fontsize=8)
    path=OUT/f"{param}_{h:03d}.png"
    fig.savefig(path,dpi=120,facecolor=fig.get_facecolor())
    plt.close(fig)
    products.append(dict(param=param,hours=h,valid=valid.isoformat(),path=f"icon-eu/{path.name}"))
    print(f"Generated {param} +{h:03d}h ({len(products)} maps)", flush=True)
PARAMS["rate"] = ("tot_prec", "Niederschlag der letzten 3 Stunden", "mm/3 h", 0, 1, [0,.1,.5,1,2,5,10,20,40,80])
for param in ("t2","gust","precip","snow"):
    available = files if param == "t2" else {}
    if param != "t2":
        folder = urljoin(BASE, f"{run:%H}/{PARAMS[param][0]}/")
        response = requests.get(folder, timeout=40)
        response.raise_for_status()
        for name in re.findall(r'href="([^"]+)"', response.text):
            m = re.search(r"_"+stamp+r"_(\d{3})(?:_|\.)", name)
            if m and name.endswith(".grib2.bz2"):
                available[int(m[1])] = urljoin(folder,name)
    previous = None
    previous_hour = None
    for h in hours:
        if h not in available:
            previous = None
            continue
        decoded = decode(param,available[h])
        render(param,h,decoded)
        if param == "precip" and previous is not None and h-previous_hour == 3:
            lat,lon,values,valid = decoded
            render("rate",h,(lat,lon,np.maximum(0,values-previous),valid))
        previous = decoded[2]
        previous_hour = h
    if not any(p["param"] == param for p in products):
        raise RuntimeError(f"No products for {param}")
(OUT/"index.json").write_text(json.dumps(dict(run=stamp,products=products)),encoding="utf-8")

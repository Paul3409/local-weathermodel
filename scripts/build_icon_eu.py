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
from matplotlib.colors import BoundaryNorm, ListedColormap, LinearSegmentedColormap
from matplotlib.patches import Circle, Polygon, FancyBboxPatch
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
    fig=plt.figure(figsize=(12,9),facecolor="#101d2d")
    ax=fig.add_axes([.03,.15,.94,.67],projection=ccrs.PlateCarree())
    cax=fig.add_axes([.09,.085,.82,.022])
    ax.set_extent([2,27,44,58],crs=ccrs.PlateCarree())
    palettes={
        "gust":["#f4f2dd","#d4df86","#f8c84c","#f28a32","#d63c45","#681b64"],
        "precip":["#f4f5f6","#b9e2ff","#388ed3","#193f90","#23834c","#f0d83b","#f58232","#c82945"],
        "rate":["#f5f5ef","#b6e2ff","#4c9fdd","#2459a8","#37a35a","#f0dc43","#ef8131","#c92643"],
        "snow":["#f3f5f6","#d5edff","#8ec8ef","#438bd0","#304da3","#633d9d","#b73e91"]}
    if param=="t2":
        ticks=[-30,-20,-15,-10,-5,0,5,10,15,20,25,30,35,40,45]
        levels=np.arange(-30,46,1)
        cold=LinearSegmentedColormap.from_list("cold",["#63208a","#2364c4","#a5e2ff"])(np.linspace(0,1,30))
        warm=LinearSegmentedColormap.from_list("warm",["#7bc66b","#ffe34b","#e54a27","#8b164d"])(np.linspace(0,1,45))
        levels=np.arange(-30,46)
        cmap=ListedColormap(np.vstack([cold,warm]))
    else:
        ticks=PARAMS[param][5]
        levels=np.concatenate([np.linspace(a,b,4,endpoint=False) for a,b in zip(ticks[:-1],ticks[1:])]+[[ticks[-1]]])
        cmap=LinearSegmentedColormap.from_list(param,palettes[param],N=len(levels)-1)
    norm=BoundaryNorm(levels,cmap.N,clip=True)
    im=ax.imshow(z,extent=[x.min(),x.max(),y.min(),y.max()],origin="lower",cmap=cmap,norm=norm,interpolation="nearest",transform=ccrs.PlateCarree())
    outline=[pe.Stroke(linewidth=1.05,foreground="white"),pe.Normal()]
    ax.add_feature(cfeature.BORDERS.with_scale("10m"),linewidth=.48,edgecolor="#10151c",path_effects=outline,zorder=4)
    ax.coastlines(resolution="10m",linewidth=.48,color="#10151c",path_effects=outline,zorder=4)
    ax.add_feature(cfeature.RIVERS.with_scale("50m"),edgecolor="#10151c",linewidth=.35,alpha=.35,zorder=3)
    ax.add_feature(cfeature.LAKES.with_scale("50m"),facecolor="#10151c",edgecolor="none",alpha=.17,zorder=3)
    
    cities=[("Berlin",52.52,13.405),("Hamburg",53.55,9.99),("München",48.14,11.58),("Köln",50.94,6.96),("Frankfurt",50.11,8.68),("Leipzig",51.34,12.37),("Paris",48.86,2.35),("Wien",48.21,16.37),("Warschau",52.23,21.01),("Prag",50.08,14.44),("Amsterdam",52.37,4.9),("Brüssel",50.85,4.35),("Kopenhagen",55.68,12.57),("Zürich",47.37,8.54),("Mailand",45.46,9.19),("Budapest",47.5,19.04)]
    for name,a,b in cities:
        ax.plot(b,a,"k.",ms=3,zorder=5)
        ax.annotate(name,(b,a),xytext=(3,4),textcoords="offset points",fontsize=7.5,color="#142136",clip_on=True,zorder=6,path_effects=[pe.withStroke(linewidth=1.8,foreground="white")])
    fig.add_artist(Circle((.05,.95),.018,transform=fig.transFigure,fill=False,edgecolor="#65d9f2",linewidth=1))
    fig.add_artist(Polygon([(.073,.95),(.038,.961),(.046,.95),(.038,.939)],transform=fig.transFigure,color="#65d9f2"))
    fig.text(.08,.95,"MeteoOST",fontsize=22,weight="bold",color="#edf5fc",va="center")
    fig.text(.97,.95,"ICON-EU  /  7 km · Mitteleuropa",ha="right",color="#a2b9ce",fontsize=12,va="center")
    fig.text(.03,.895,PARAMS[param][1],fontsize=20,weight="bold",color="white")
    fig.text(.03,.86,f"Modellstart {run:%d.%m.%Y %H:%M} UTC  |  Gültig {valid:%d.%m.%Y %H:%M} UTC",fontsize=10,color="#a2b9ce")
    fig.text(.97,.895,f"+{h} h",ha="right",fontsize=22,weight="bold",color="#65d9f2")
    cb=fig.colorbar(im,cax=cax,orientation="horizontal",ticks=ticks,extend="both")
    cb.ax.tick_params(colors="white",labelsize=8,length=2)
    cb.set_label(PARAMS[param][2],color="white",fontsize=10)
    fig.text(.03,.022,"DWD · Natural Earth · IDW-interpoliert",color="#a2b9ce",fontsize=8)
    fig.text(.97,.022,"METEOOST / SIGNATURE",ha="right",color="#a2b9ce",fontsize=8)
    path=OUT/f"{param}_{h:03d}.png"
    fig.savefig(path,dpi=200,facecolor=fig.get_facecolor(),pil_kwargs={"compress_level":3})
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

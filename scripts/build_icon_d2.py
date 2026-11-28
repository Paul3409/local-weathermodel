"""Generate local ICON-D2 temperature maps directly from DWD GRIB."""
import bz2, json, re, tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urljoin
import requests
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from eccodes import codes_grib_new_from_file, codes_get_array, codes_get, codes_release

BASE = "https://opendata.dwd.de/weather/nwp/icon-d2/grib/"
OUT = Path("_site/icon-d2")
OUT.mkdir(parents=True, exist_ok=True)
hours = [0, 3, 6, 9, 12, 15, 18, 21, 24]
now = datetime.now(timezone.utc)
cycle = now.replace(hour=now.hour//3*3, minute=0, second=0, microsecond=0)
for offset in range(0, 25, 3):
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
        if m and name.endswith(".grib2.bz2"):
            files[int(m[1])] = urljoin(folder, name)
    if all(h in files for h in hours):
        break
else:
    raise RuntimeError("No complete ICON-D2 cycle found")

products = []
for h in hours:
    response = requests.get(files[h], timeout=90)
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
            val = np.asarray(codes_get_array(gid,"values"))-273.15
            valid = datetime.strptime(str(codes_get(gid,"validityDate"))+f'{codes_get(gid,"validityTime"):04d}',"%Y%m%d%H%M").replace(tzinfo=timezone.utc)
        finally:
            codes_release(gid)
    mask = (lat>=51.19)&(lat<=51.59)&(lon>=13.01)&(lon<=13.49)&np.isfinite(val)&(val>-100)&(val<70)
    if mask.sum()<6:
        raise ValueError("Insufficient local grid points")
    x,y=np.meshgrid(np.linspace(13.10,13.40,235),np.linspace(51.28,51.50,245))
    z=griddata((lon[mask],lat[mask]),val[mask],(x,y),method="linear")
    if not np.isfinite(z).all():
        z=np.where(np.isfinite(z),z,griddata((lon[mask],lat[mask]),val[mask],(x,y),method="nearest"))
    fig,ax=plt.subplots(figsize=(8,7),layout="constrained",facecolor="#101a2b")
    im=ax.pcolormesh(x,y,z,cmap="turbo",vmin=-15,vmax=35,shading="auto")
    ax.set_aspect(1/np.cos(np.deg2rad(51.39)))
    for name,a,b in [("Belgern",51.483,13.125),("Mühlberg",51.434,13.222),("Zeithain",51.334,13.338),("Nünchritz",51.299,13.388),("Glaubitz",51.325,13.379)]:
        ax.plot(b,a,"k.",ms=5)
        ax.annotate(name,(b,a),xytext=(-3,5),textcoords="offset points",ha="right",fontsize=9,color="black",bbox=dict(fc="white",alpha=.8,ec="none",pad=1))
    ax.set_title(f"ICON-D2 | 2-m-Temperatur\n{valid:%d.%m.%Y %H:%M} UTC | +{h} h",color="white")
    ax.tick_params(colors="white")
    cb=fig.colorbar(im,ax=ax,label="°C",extend="both")
    cb.ax.tick_params(colors="white")
    cb.ax.yaxis.label.set_color("white")
    fig.text(.5,.005,"DWD ICON-D2 · interpoliert · keine Geländekorrektur",ha="center",color="white",fontsize=9)
    path=OUT/f"t2_{h:03d}.png"
    fig.savefig(path,dpi=120,facecolor=fig.get_facecolor())
    plt.close(fig)
    products.append(dict(hours=h,valid=valid.isoformat(),path=f"icon-d2/{path.name}"))
(OUT/"index.json").write_text(json.dumps(dict(run=stamp,products=products)),encoding="utf-8")

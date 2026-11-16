"""ICON-EU synoptic maps. Data and boundaries are downloaded only on manual request."""
import bz2
import io
import os
import re
import tempfile
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from urllib.parse import urljoin

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
import streamlit as st
from eccodes import codes_grib_new_from_file, codes_get, codes_get_array, codes_release

BASE = "https://opendata.dwd.de/weather/nwp/icon-eu/grib"
COUNTRIES = "https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson"
STATES = "https://raw.githubusercontent.com/isellsoap/deutschlandGeoJSON/main/2_bundeslaender/3_mittel.geo.json"
REGIONS = {"Deutschland": (5.5, 15.6, 47.0, 55.3), "Europa": (-12.0, 35.0, 34.0, 72.0)}
HEADERS = {"User-Agent": "ELBTAL weather visualization (educational)"}
MAP_STYLE_VERSION = "mslp-vivid-readable-ticks-v2"

class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.names = []
    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.names += [v for k,v in attrs if k=="href" and v]

@st.cache_data(ttl=600,show_spinner=False)
def discover(run, variable):
    folder=f"{BASE}/{run:%H}/{variable}/"
    response=requests.get(folder,headers=HEADERS,timeout=25)
    response.raise_for_status()
    parser=Links()
    parser.feed(response.text)
    result={}
    stamp=run.strftime("%Y%m%d%H")
    for name in parser.names:
        lower=name.lower()
        if not lower.endswith(".grib2.bz2") or stamp not in lower:
            continue
        if variable=="pmsl" and "single-level" not in lower:
            continue
        if variable=="t" and ("pressure-level" not in lower or not re.search(r"(?:^|_)850(?:_|\.)",lower)):
            continue
        match=re.search("_"+stamp+r"_(\d{3})(?:_|\.)",lower)
        if match:
            result[int(match.group(1))]=urljoin(folder,name)
    return result

@st.cache_data(ttl=86400,show_spinner=False)
def borders(url):
    response=requests.get(url,headers=HEADERS,timeout=30)
    response.raise_for_status()
    return response.json()

def plot_geometry(ax,geom,linewidth):
    if not geom:
        return
    typ=geom.get("type")
    coords=geom.get("coordinates",[])
    if typ=="Polygon":
        polygons=[coords]
    elif typ=="MultiPolygon":
        polygons=coords
    else:
        return
    for polygon in polygons:
        for ring in polygon:
            if len(ring)>1:
                arr=np.asarray(ring,dtype=float)
                ax.plot(arr[:,0],arr[:,1],color="#253449",linewidth=linewidth,alpha=.75,zorder=4)

def load_grid(url, valid, bounds, variable):
    response=requests.get(url,headers=HEADERS,timeout=90)
    response.raise_for_status()
    raw=bz2.decompress(response.content)
    path=None
    try:
        with tempfile.NamedTemporaryFile(suffix=".grib2",delete=False) as file:
            file.write(raw)
            path=file.name
        del raw
        with open(path,"rb") as stream:
            while True:
                gid=codes_grib_new_from_file(stream)
                if gid is None:
                    break
                try:
                    date=int(codes_get(gid,"validityDate"))
                    time=int(codes_get(gid,"validityTime"))
                    timestamp=datetime.strptime(f"{date:08d}{time:04d}","%Y%m%d%H%M").replace(tzinfo=timezone.utc)
                    if abs((timestamp-valid).total_seconds())>60:
                        continue
                    if variable=="t":
                        level=int(codes_get(gid,"level"))
                        if level!=850:
                            continue
                    lat=np.asarray(codes_get_array(gid,"latitudes"))
                    lon=np.asarray(codes_get_array(gid,"longitudes"))
                    vals=np.asarray(codes_get_array(gid,"values"))
                    lon=np.where(lon>180,lon-360,lon)
                    west,east,south,north=bounds
                    mask=(lon>=west-.5)&(lon<=east+.5)&(lat>=south-.5)&(lat<=north+.5)
                    if not mask.any():
                        raise RuntimeError("ICON-EU-Feld deckt die gewählte Region nicht ab.")
                    lat,lon,vals=lat[mask],lon[mask],vals[mask]
                    if variable=="pmsl":
                        vals=vals/100.0
                        if not 850<np.nanmedian(vals)<1100:
                            raise RuntimeError("MSLP-Einheiten unerwartet; Karte nicht erstellt.")
                    else:
                        vals=vals-273.15
                        if not -100<np.nanmedian(vals)<60:
                            raise RuntimeError("T850-Einheiten unerwartet; Karte nicht erstellt.")
                    from scipy.interpolate import griddata
                    nx,ny=(225,180) if bounds==REGIONS["Deutschland"] else (300,220)
                    x=np.linspace(west,east,nx)
                    y=np.linspace(south,north,ny)
                    X,Y=np.meshgrid(x,y)
                    Z=griddata(np.column_stack((lon,lat)),vals,(X,Y),method="linear")
                    if not np.isfinite(Z).all():
                        nearest=griddata(np.column_stack((lon,lat)),vals,(X,Y),method="nearest")
                        Z=np.where(np.isfinite(Z),Z,nearest)
                    return X,Y,Z
                finally:
                    codes_release(gid)
    finally:
        if path and os.path.exists(path):
            os.unlink(path)
    raise RuntimeError("Keine GRIB-Nachricht mit exakt passender Gültigkeitszeit und Druckfläche gefunden.")

def render(X,Y,Z,region,parameter,run,valid):
    bounds=REGIONS[region]
    fig,ax=plt.subplots(figsize=(9,8),dpi=120)
    fig.patch.set_facecolor("#f6f8fa")
    if parameter=="MSLP":
        from matplotlib.colors import LinearSegmentedColormap, Normalize
        # Fixed, intentionally nonlinear pressure color anchors: vivid reds from 1030 hPa.
        pressure_anchors=[
            (950,"#30105B"),
            (960,"#481A86"),
            (975,"#5141AF"),
            (985,"#2059B6"),
            (995,"#299BC6"),
            (1005,"#3CBF91"),
            (1015,"#A8D45B"),
            (1020,"#F3DE4C"),
            (1025,"#F59B37"),
            (1030,"#E43C38"),
            (1040,"#A51E3C"),
            (1050,"#681336"),
            (1060,"#410C30"),
        ]
        cmap=LinearSegmentedColormap.from_list(
            "elbtal_mslp_vivid",
            [((pressure-950)/110,color) for pressure,color in pressure_anchors],
            N=512,
        )
        norm=Normalize(vmin=950,vmax=1060,clip=False)
        levels=np.linspace(950,1060,221)
        im=ax.contourf(X,Y,Z,levels=levels,cmap=cmap,norm=norm,extend="both")
        contours=np.arange(940,1070,4)
        line=ax.contour(X,Y,Z,levels=contours,colors="#24364b",linewidths=.55)
        ax.clabel(line,inline=True,fontsize=6,fmt="%d")
        label="Luftdruck auf Meereshöhe (hPa)"
    else:
        from matplotlib.colors import LinearSegmentedColormap, Normalize
        # Fixed meteorological T850 palette, with a narrow transition at 0 °C.
        # Slightly negative values remain light blue; positive values start dark green.
        anchors=[
            (-30.0,"#F8F9FC"),  # nearly white
            (-25.0,"#F4A6D4"),  # pink
            (-20.0,"#A05BC6"),  # violet
            (-15.0,"#542178"),  # dark violet
            (-10.0,"#163A8C"),  # dark blue
            (-5.0,"#2678D3"),   # blue
            (-0.05,"#BDEBFA"),  # below zero: light blue
            (0.05,"#14572C"),   # above zero: dark green
            (5.0,"#39A447"),   # green
            (10.0,"#F4DE45"),  # yellow
            (15.0,"#F39839"),  # orange
            (20.0,"#D93032"),  # red
            (25.0,"#831727"),  # dark red
            (30.0,"#9937B8"),  # purple
        ]
        cmap=LinearSegmentedColormap.from_list(
            "elbtal_t850",
            [((value+30)/60,color) for value,color in anchors],
            N=512,
        )
        norm=Normalize(vmin=-30,vmax=30,clip=False)
        levels=np.linspace(-30,30,241)
        im=ax.contourf(X,Y,Z,levels=levels,cmap=cmap,norm=norm,extend="both")
        line=ax.contour(X,Y,Z,levels=np.arange(-30,31,5),colors="#24364b",linewidths=.45)
        ax.clabel(line,inline=True,fontsize=6,fmt="%d")
        # Highlight the freezing-level isotherm without changing the temperature field.
        ax.contour(X,Y,Z,levels=[0],colors="#111827",linewidths=1.15)
        label="Temperatur auf 850 hPa (°C)"
    try:
        data=borders(COUNTRIES)
        for feature in data.get("features",[]):
            plot_geometry(ax,feature.get("geometry"),.55)
        if region=="Deutschland":
            for feature in borders(STATES).get("features",[]):
                plot_geometry(ax,feature.get("geometry"),.35)
    except requests.RequestException as exc:
        raise RuntimeError(f"Grenzdaten nicht erreichbar: {exc}") from exc
    ax.set_xlim(bounds[0],bounds[1])
    ax.set_ylim(bounds[2],bounds[3])
    ax.set_xlabel("Längengrad (°O)")
    ax.set_ylabel("Breitengrad (°N)")
    ax.grid(alpha=.15)
    ax.set_title(f"ELBTAL | ICON-EU {parameter} | {region}\nLauf {run:%d.%m.%Y %H:%M} UTC · Gültig {valid:%d.%m.%Y %H:%M} UTC",fontsize=11)
    cb=fig.colorbar(im,ax=ax,orientation="horizontal",pad=.08,fraction=.05)
    if parameter=="T850":
        cb.set_ticks(np.arange(-30,31,5))
    else:
        # Avoid crowded labels around 1025/1030 while preserving the exact color mapping.
        cb.set_ticks([950,970,985,995,1005,1015,1030,1045,1060])
    cb.set_label(label)
    output=io.BytesIO()
    fig.savefig(output,format="png",bbox_inches="tight",facecolor=fig.get_facecolor())
    plt.close(fig)
    return output.getvalue()

def show(region):
    st.subheader(f"ICON-EU · {region}")
    st.caption("Stündliche Original-Modellzeitpunkte; Landesgrenzen und bei Deutschland Bundesländergrenzen. Keine künstliche zeitliche Verfeinerung.")
    parameter=st.radio("Parameter",["MSLP","T850"],horizontal=True,key="eu_parameter")
    lead=st.select_slider("Vorhersagestunde",options=list(range(25)),value=6,format_func=lambda h:f"+{h} h",key="eu_lead")
    if st.button("ICON-EU-Karte manuell berechnen",type="primary"):
        with st.status("ICON-EU-Datei und Grenzen laden …",expanded=True) as status:
            try:
                now=datetime.now(timezone.utc)
                runs=[]
                for offset in range(0,34,3):
                    run=(now-timedelta(hours=offset)).replace(minute=0,second=0,microsecond=0)
                    run=run.replace(hour=(run.hour//3)*3)
                    if run not in runs:
                        runs.append(run)
                variable="pmsl" if parameter=="MSLP" else "t"
                selected=None
                for run in runs:
                    try:
                        available=discover(run,variable)
                    except requests.RequestException:
                        continue
                    if lead in available:
                        selected=(run,available[lead])
                        break
                if selected is None:
                    raise RuntimeError(f"Keine passende ICON-EU-Datei für {parameter} +{lead} h gefunden. Bei T850 wird ausdrücklich eine 850-hPa-Druckflächen-Datei benötigt.")
                run,url=selected
                valid=run+timedelta(hours=lead)
                X,Y,Z=load_grid(url,valid,REGIONS[region],variable)
                png=render(X,Y,Z,region,parameter,run,valid)
                st.session_state["eu_map"]=(MAP_STYLE_VERSION,region,parameter,lead,valid,png)
                status.update(label="ICON-EU-Karte erstellt",state="complete")
            except Exception as exc:
                status.update(label="ICON-EU-Verarbeitung fehlgeschlagen",state="error")
                st.error(f"{type(exc).__name__}: {exc}")
    result=st.session_state.get("eu_map")
    if result and (len(result)!=6 or result[0]!=MAP_STYLE_VERSION):
        st.session_state.pop("eu_map",None)
        st.info("Die Karten-Farbskala wurde aktualisiert. Bitte die ICON-EU-Karte erneut berechnen.")
        result=None
    if result and len(result)==6 and result[:4]==(MAP_STYLE_VERSION,region,parameter,lead):
        _,_,_,_,valid,png=result
        st.image(png,width="stretch")
        st.download_button("PNG herunterladen",png,file_name=f"ELBTAL_ICON-EU_{region}_{parameter}_{valid:%Y%m%d_%H%M}.png",mime="image/png")

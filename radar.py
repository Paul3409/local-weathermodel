"""ELBTAL DWD radar: original RX reflectivity and RY 5-minute precipitation."""
import bz2
import io
import re
from datetime import datetime, timezone
from html.parser import HTMLParser
from urllib.parse import urljoin
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import requests
import streamlit as st
from scipy.ndimage import map_coordinates
from pyproj import Transformer
from map_design import frame_png

BASE="https://opendata.dwd.de/weather/radar/"
TZ=ZoneInfo("Europe/Berlin")
HEADERS={"User-Agent":"ELBTAL educational weather radar visualization"}
# RADOLAN 900x900 legacy national composite: 1 km polar stereographic grid.
PROJ="+proj=stere +lat_0=90 +lat_ts=60 +lon_0=10 +a=6370040 +b=6370040 +units=m +no_defs"
TRANSFORMER=Transformer.from_crs("EPSG:4326",PROJ,always_xy=True)
X0,Y0=-523462.2,-4658644.7
COLORS=["#666666","#858585","#4c999f","#08c4d9","#00a9f4","#0072ed",
        "#0035c8","#00f000","#00c900","#00a300","#ffff00","#ffc000",
        "#ff8000","#ff3c00","#ff0000","#c90000","#990000",
        "#f5b5ee","#cf70d4","#9d009e"]
# Same palette for both modes, but separate physically meaningful units.
DBZ_BOUNDS=[0,4,7,10,13,16,19,21,25,29,33,35,39,43,47,51,55,59,63,67,71]
RATE_BOUNDS=[0,.05,.1,.2,.4,.7,1,1.5,2,3,4,6,9,13,18,25,35,50,70,100,150]

class Links(HTMLParser):
    def __init__(self):
        super().__init__();self.names=[]
    def handle_starttag(self,tag,attrs):
        if tag=="a":
            self.names += [v.split("?")[0].split("/")[-1] for k,v in attrs if k=="href" and v]

@st.cache_data(ttl=120,show_spinner=False)
def list_frames(product):
    # RX: measured reflectivity, RY: quality-corrected 5-min precipitation.
    folder=BASE+("composite/rx/" if product=="RX" else "radolan/ry/")
    response=requests.get(folder,headers=HEADERS,timeout=30)
    response.raise_for_status()
    parser=Links();parser.feed(response.text)
    found=[]
    pattern=re.compile(r"raa01-"+product.lower()+r"_10000-(\d{10})-dwd---bin(?:\.bz2)?$",re.I)
    for name in parser.names:
        match=pattern.search(name)
        if match:
            try:
                stamp=datetime.strptime(match.group(1),"%y%m%d%H%M").replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            found.append((stamp,urljoin(folder,name)))
    return sorted(set(found),reverse=True)[:24]

@st.cache_data(ttl=900,show_spinner=False,max_entries=30)
def load_frame(url,product):
    response=requests.get(url,headers=HEADERS,timeout=55)
    response.raise_for_status()
    raw=bz2.decompress(response.content) if url.endswith(".bz2") else response.content
    pos=raw.find(b"\x03")
    if pos<0:
        raise RuntimeError("RADOLAN-Datei ohne Header-Endmarke (ETX).")
    header=raw[:pos].decode("latin1","replace")
    if not header.startswith(product):
        raise RuntimeError("Radarprodukt im Header stimmt nicht mit der Auswahl überein.")
    payload=raw[pos+1:]
    # RX is 8-bit rvp6 reflectivity; RY is unsigned 16-bit rainfall with flag bits.
    if product=="RX":
        if len(payload)!=900*900:
            raise RuntimeError(f"RX-Rastergröße unerwartet: {len(payload)} Bytes.")
        arr=np.frombuffer(payload,dtype=np.uint8).reshape((900,900))
        values=arr.astype(np.float32)*.5-32.5
        values[arr>=250]=np.nan
        values[values<0]=np.nan
    else:
        if len(payload)!=900*900*2:
            raise RuntimeError(f"RY-Rastergröße unerwartet: {len(payload)} Bytes.")
        arr=np.frombuffer(payload,dtype="<u2").reshape((900,900))
        # RADOLAN 16-bit flag masks: 0x1000 secondary, 0x2000 nodata,
        # 0x4000 negative, 0x8000 clutter.
        bad=((arr&0x2000)!=0)|((arr&0x8000)!=0)
        precision_match=re.search(r"PR\s*E([+-]?\d+)",header)
        if not precision_match:
            raise RuntimeError("RY-Präzision (PR) fehlt im RADOLAN-Header.")
        precision=10.0**int(precision_match.group(1))
        values=(arr&0x0fff).astype(np.float32)*precision*12.0
        values[bad]=np.nan
    # RADOLAN file rows are ordered north to south.
    return np.flipud(values.copy())

@st.cache_data(ttl=86400,show_spinner=False)
def sample_geometry():
    # Build lat/lon grid and project to the RADOLAN native 1 km raster.
    lon=np.linspace(5.4,15.6,800)
    lat=np.linspace(46.9,55.4,720)
    X,Y=np.meshgrid(lon,lat)
    px,py=TRANSFORMER.transform(X,Y)
    col=(px-X0)/1000
    row=(py-Y0)/1000
    return X,Y,row,col

def draw_boundaries(ax):
    from icon_eu import borders,plot_geometry,COUNTRIES,STATES
    for url,lw in ((COUNTRIES,.6),(STATES,.38)):
        try:
            features=borders(url).get("features",[])
            for feature in features:
                plot_geometry(ax,feature.get("geometry"),lw)
        except requests.RequestException:
            # Missing boundary service must not change actual radar values.
            pass

def render(values,product,stamp):
    X,Y,row,col=sample_geometry()
    valid=(row>=0)&(row<=899)&(col>=0)&(col<=899)
    sampled=map_coordinates(np.nan_to_num(values,nan=-9999),[row,col],
                            order=0,mode="constant",cval=-9999)
    sampled=np.where(valid & (sampled>=0),sampled,np.nan)
    bounds=DBZ_BOUNDS if product=="RX" else RATE_BOUNDS
    cmap=ListedColormap(COLORS)
    cmap.set_bad("#10273e")
    norm=BoundaryNorm(bounds,cmap.N)
    fig,ax=plt.subplots(figsize=(9,8),dpi=120)
    fig.patch.set_facecolor("#071e35")
    ax.set_facecolor("#10273e")
    image=ax.imshow(sampled,origin="lower",extent=(5.4,15.6,46.9,55.4),
                    cmap=cmap,norm=norm,interpolation="nearest",aspect="auto")
    draw_boundaries(ax)
    ax.set_xlim(5.4,15.6);ax.set_ylim(46.9,55.4)
    ax.tick_params(colors="#d6e5f2",labelsize=9)
    for spine in ax.spines.values():spine.set_color("#78a4c2")
    ax.grid(color="white",alpha=.12,linewidth=.5)
    ax.set_xlabel("Längengrad (°O)",color="#d6e5f2")
    ax.set_ylabel("Breitengrad (°N)",color="#d6e5f2")
    fig.subplots_adjust(left=.09,right=.98,top=.96,bottom=.19)
    cb=fig.colorbar(image,ax=ax,orientation="horizontal",pad=.09,
                    fraction=.045,aspect=35,spacing="uniform",
                    ticks=[0,10,20,30,40,50,60,70] if product=="RX"
                    else [0,.5,1,2,5,10,20,50,100])
    cb.ax.tick_params(colors="#d6e5f2",labelsize=8)
    cb.set_label("Radarreflektivität (dBZ)" if product=="RX" else "Niederschlagsintensität (mm/h) · 5-Minuten-Mittel",color="#d6e5f2")
    buf=io.BytesIO()
    fig.savefig(buf,format="png",facecolor=fig.get_facecolor(),dpi=120)
    plt.close(fig)
    label="Radarreflektivität" if product=="RX" else "Radar-Regenrate"
    return frame_png(buf.getvalue(),label,"DWD RADOLAN","Deutschland",stamp,stamp)

def show():
    st.subheader("📡 ELBTAL | Regenradar Deutschland")
    mode=st.segmented_control("Radarprodukt",["Reflektivität (dBZ)","Regenrate (mm/h)"],
                              default="Reflektivität (dBZ)",key="radar_mode")
    product="RX" if mode=="Reflektivität (dBZ)" else "RY"
    st.caption("DWD-Radarverbund · gemessene Radarreflektivität (RX) oder qualitätskorrigierte Niederschlagsmenge (RY), auf mm/h umgerechnet. 1-km-Raster, neue Aufnahme ungefähr alle 5 Minuten. Kein Wettermodell.")
    try:
        frames=list_frames(product)
    except Exception as exc:
        st.error(f"Radarverzeichnis nicht erreichbar: {exc}")
        return
    if not frames:
        st.warning("Der DWD bietet derzeit keine passenden Radardateien in diesem Verzeichnis an.")
        return
    options=list(range(len(frames)))
    idx=st.select_slider("Radarzeit (MEZ/MESZ)",options=options,value=0,
                         format_func=lambda i:frames[i][0].astimezone(TZ).strftime("%d.%m.%Y %H:%M %Z"))
    stamp,url=frames[idx]
    st.caption(f"Messzeit: {stamp.astimezone(TZ):%d.%m.%Y %H:%M %Z} · UTC {stamp:%H:%M} · kein Vorhersagezeitpunkt")
    if st.button("📡 Radaraufnahme laden",type="primary"):
        try:
            with st.spinner("Originale DWD-Radardaten werden geladen …"):
                data=load_frame(url,product)
                png=render(data,product,stamp)
            st.session_state["radar_image"]=(url,png)
        except Exception as exc:
            st.error(f"Radaraufnahme nicht darstellbar: {type(exc).__name__}: {exc}")
    if st.session_state.get("radar_image",(None,None))[0]==url:
        png=st.session_state["radar_image"][1]
        st.image(png,width="stretch")
        st.download_button("PNG herunterladen",png,file_name=f"ELBTAL_Radar_{product}_{stamp:%Y%m%d_%H%M}.png",mime="image/png")
    st.caption("Farbskala nach deiner Vorlage. dBZ und mm/h verwenden unterschiedliche physikalische Schwellen. Fehlende oder ungültige Radardaten werden nicht als Regen dargestellt.")

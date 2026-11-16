"""ELBTAL DWD radar: original RX reflectivity and RY 5-minute precipitation."""
import bz2
import tarfile
import h5py
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
from pyproj import Transformer, CRS
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
    folder=BASE+("composite/wn/" if product=="WN" else "radolan/ry/")
    response=requests.get(folder,headers=HEADERS,timeout=30)
    response.raise_for_status()
    parser=Links();parser.feed(response.text)
    found=[]
    pattern=(re.compile(r"composite_wn_(\d{8}_\d{4})\.tar$",re.I) if product=="WN" else re.compile(r"raa01-ry_10000-(\d{10})-dwd---bin\.hdf5$",re.I))
    for name in parser.names:
        match=pattern.search(name)
        if match:
            try:
                stamp=datetime.strptime(match.group(1),"%Y%m%d_%H%M" if product=="WN" else "%y%m%d%H%M").replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            found.append((stamp,urljoin(folder,name)))
    return sorted(set(found),reverse=True)[:24]

@st.cache_data(ttl=900,show_spinner=False,max_entries=30)
def load_frame(url,product):
    response=requests.get(url,headers=HEADERS,timeout=65)
    response.raise_for_status()
    raw=response.content
    if product=="WN":
        with tarfile.open(fileobj=io.BytesIO(raw),mode="r:*") as archive:
            members=[m for m in archive.getmembers() if m.isfile() and (m.name.endswith((".h5",".hdf5","-hd5")))]
            if not members:
                raise RuntimeError("WN-TAR enthält keine ODIM-HDF5-Radardatei.")
            raw=archive.extractfile(members[0]).read()
    with h5py.File(io.BytesIO(raw),"r") as h5:
        where=h5["where"].attrs
        xsize=int(where["xsize"]);ysize=int(where["ysize"])
        proj=where["projdef"]
        if isinstance(proj,bytes):proj=proj.decode()
        crs=CRS.from_user_input(proj)
        # ODIM defines the lower-left and upper-right pixel corner coordinates.
        transformer=Transformer.from_crs("EPSG:4326",crs,always_xy=True)
        x0,y0=transformer.transform(float(where["LL_lon"]),float(where["LL_lat"]))
        x1,y1=transformer.transform(float(where["UR_lon"]),float(where["UR_lat"]))
        group=h5["dataset1"]["data1"]
        data=np.asarray(group["data"],dtype=np.float32)
        if data.shape!=(ysize,xsize):
            raise RuntimeError(f"ODIM-Raster hat {data.shape}, erwartet {(ysize,xsize)}.")
        attrs=group["what"].attrs
        quantity=attrs.get("quantity",b"")
        if isinstance(quantity,bytes):quantity=quantity.decode()
        if product=="WN" and quantity not in ("DBZH","TH","DBZ","DBZV"):
            raise RuntimeError(f"WN-Feld {quantity!r} ist keine Reflektivität.")
        if product=="RY" and quantity not in ("RATE","ACRR","DBZH",""):
            raise RuntimeError(f"Unbekannte RY-Einheit {quantity!r}; keine sichere Umrechnung.")
        nodata=float(attrs.get("nodata",np.nan))
        undetect=float(attrs.get("undetect",np.nan))
        missing=(data==nodata)|(data==undetect)
        values=data*float(attrs["gain"])+float(attrs["offset"])
        if product=="RY":
            # RATE is already in mm/h; ACRR denotes a 5-minute accumulation.
            if quantity=="ACRR":values*=12
            elif quantity!="RATE":
                raise RuntimeError(f"RY-Feld {quantity!r} hat keine gesicherte mm/h-Einheit.")
        values[missing]=np.nan
        values[values<0]=np.nan
        return values,(crs.to_string(),x0,y0,x1,y1)

@st.cache_data(ttl=86400,show_spinner=False)
def sample_geometry(proj,x0,y0,x1,y1,shape):
    lon=np.linspace(5.4,15.6,800)
    lat=np.linspace(46.9,55.4,720)
    X,Y=np.meshgrid(lon,lat)
    transformer=Transformer.from_crs("EPSG:4326",CRS.from_user_input(proj),always_xy=True)
    px,py=transformer.transform(X,Y)
    ny,nx=shape
    col=(px-x0)/(x1-x0)*nx-.5
    row=(y1-py)/(y1-y0)*ny-.5
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

def render(values,geo,product,stamp):
    X,Y,row,col=sample_geometry(*geo,values.shape)
    valid=(row>=0)&(row<values.shape[0])&(col>=0)&(col<values.shape[1])
    sampled=map_coordinates(np.nan_to_num(values,nan=-9999),[row,col],
                            order=0,mode="constant",cval=-9999)
    sampled=np.where(valid & (sampled>=0),sampled,np.nan)
    bounds=DBZ_BOUNDS if product=="WN" else RATE_BOUNDS
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
                    ticks=[0,10,20,30,40,50,60,70] if product=="WN"
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
    product="WN" if mode=="Reflektivität (dBZ)" else "RY"
    st.caption("DWD-Radarverbund · gemessene Radarreflektivität (WN) oder qualitätskorrigierte Niederschlagsmenge (RY), auf mm/h umgerechnet. 1-km-Raster, neue Aufnahme ungefähr alle 5 Minuten. Kein Wettermodell.")
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
                data,geo=load_frame(url,product)
                png=render(data,geo,product,stamp)
            st.session_state["radar_image"]=(url,png)
        except Exception as exc:
            st.error(f"Radaraufnahme nicht darstellbar: {type(exc).__name__}: {exc}")
    if st.session_state.get("radar_image",(None,None))[0]==url:
        png=st.session_state["radar_image"][1]
        st.image(png,width="stretch")
        st.download_button("PNG herunterladen",png,file_name=f"ELBTAL_Radar_{product}_{stamp:%Y%m%d_%H%M}.png",mime="image/png")
    st.caption("Farbskala nach deiner Vorlage. dBZ und mm/h verwenden unterschiedliche physikalische Schwellen. Fehlende oder ungültige Radardaten werden nicht als Regen dargestellt.")

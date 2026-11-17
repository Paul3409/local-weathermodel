"""MeteoOst DWD radar: original RX reflectivity and RY 5-minute precipitation."""
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
HEADERS={"User-Agent":"MeteoOst educational weather radar visualization"}
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
    """Fast high-contrast outlines, batched instead of thousands of plot artists."""
    from icon_eu import borders, COUNTRIES, STATES
    from matplotlib.collections import LineCollection
    for url,is_country in ((COUNTRIES,True),(STATES,False)):
        try:
            segments=[]
            for feature in borders(url).get("features",[]):
                geometry=feature.get("geometry") or {}
                kind=geometry.get("type")
                coords=geometry.get("coordinates",[])
                polygons=[coords] if kind=="Polygon" else coords if kind=="MultiPolygon" else []
                for polygon in polygons:
                    for ring in polygon:
                        if len(ring)<2:
                            continue
                        points=np.asarray(ring,dtype=float)
                        if points.ndim!=2 or points.shape[1]<2:
                            continue
                        # Skip features entirely outside the German radar viewport.
                        if (np.nanmax(points[:,0])<5.4 or np.nanmin(points[:,0])>15.6
                            or np.nanmax(points[:,1])<46.9 or np.nanmin(points[:,1])>55.4):
                            continue
                        # Screen resolution cannot resolve subpixel border vertices.
                        stride=max(1,len(points)//1400)
                        reduced=points[::stride]
                        if stride>1:
                            reduced=np.vstack((reduced,points[-1]))
                        segments.append(reduced)
            if segments:
                # Two batched collections: dark outline + visible light line.
                ax.add_collection(LineCollection(segments,colors="#061b2b",
                                  linewidths=1.8 if is_country else 1.05,
                                  alpha=.9,zorder=19))
                ax.add_collection(LineCollection(segments,
                                  colors="#e8f4fa" if is_country else "#a8c8da",
                                  linewidths=.85 if is_country else .45,
                                  alpha=.9,zorder=20))
        except (requests.RequestException,ValueError,TypeError):
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
    fig,ax=plt.subplots(figsize=(15,8.2),dpi=120)
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
    # Radar-only wide export: large map and a right-hand vertical colorbar.
    fig.subplots_adjust(left=.055,right=.855,top=.97,bottom=.095)
    cax=fig.add_axes([.89,.15,.025,.76])
    cb=fig.colorbar(image,cax=cax,orientation="vertical",spacing="uniform",
                    ticks=[0,10,20,30,40,50,60,70] if product=="WN"
                    else [0,.5,1,2,5,10,20,50,100])
    cb.ax.tick_params(colors="#d6e5f2",labelsize=10)
    cb.set_label("Radarreflektivität (dBZ)" if product=="WN" else "Niederschlagsintensität (mm/h) · 5-Minuten-Mittel",color="#d6e5f2")
    buf=io.BytesIO()
    fig.savefig(buf,format="png",facecolor=fig.get_facecolor(),dpi=120)
    plt.close(fig)
    label="Radarreflektivität" if product=="WN" else "Radar-Regenrate"
    return frame_png(buf.getvalue(),label,"DWD RADOLAN","Deutschland",stamp,stamp,portrait_override=False)

def show():
    """Radar page with all original and extended measurement products."""
    from radar_extras import show as show_extended_radar
    return show_extended_radar()

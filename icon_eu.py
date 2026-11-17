from map_design import style_figure, frame_png
"""ICON-EU synoptic maps. Data and boundaries are downloaded only on manual request."""
import bz2
import io
import os
import re
import tempfile
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from significant_weather import classify_ww, palette, legend_handles
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
REGIONS = {"Deutschland": (5.5, 15.6, 47.0, 55.3), "Mitteleuropa": (2.0, 27.0, 44.0, 58.0), "Europa & Nordatlantik": (-65.0, 65.0, 32.0, 72.0)}
HEADERS = {"User-Agent": "ELBTAL weather visualization (educational)"}
MAP_STYLE_VERSION = "windchill-v1"
LOCAL_TZ=ZoneInfo("Europe/Berlin")
EU_LEADS=list(range(79))+list(range(81,121,3))
def previous_eu_lead(lead):
    if lead==0:
        return None
    if lead<=78:
        return lead-1
    return lead-3
SURFACE_FIELDS = {"2-m-Temperatur":"t_2m","Windböen":"vmax_10m","Niederschlagsrate":"tot_prec","Niederschlagssumme":"tot_prec","Taupunkt":"td_2m","Signifikantes Wetter":"ww","Schneehöhe":"h_snow","Windchill":"t_2m"}
PHASE_COMPONENTS=("rain_gsp","rain_con","snow_gsp","snow_con")
PHASE_COLORS=["#F2F5F7","#278DCE","#F39A43","#F4A4CF","#8C949F"]
PHASE_LABELS=["Trocken","Regen","Schneeregen","Schnee","Unklar"]
TEMP_STOPS = [(-20,"#9851BD"),(-15,"#4C207F"),(-10,"#16346E"),(-5,"#3377C6"),(0,"#B8E5F4"),(.4999,"#B8E5F4"),(.5,"#174C35"),(5,"#438A49"),(10,"#A9C64D"),(15,"#F0D64B"),(20,"#F5B16C")]
GUST_STOPS = [(0,"#499FE1"),(10,"#204D99"),(20,"#347B4A"),(30,"#A3BE4B"),(40,"#E8D24A"),(50,"#F3A143"),(60,"#E96A38"),(70,"#DB3834"),(80,"#B9252E"),(90,"#811A2C"),(100,"#48205F"),(110,"#853AA8"),(120,"#C52D9A")]
PREC_RATE = [(0,"#F2F5F7"),(.1,"#C8EAF7"),(.5,"#80C9EC"),(1,"#368DD1"),(2,"#2269A9"),(5,"#3B9A68"),(10,"#C4C948"),(15,"#F1D34E"),(20,"#F3A04A"),(30,"#E85A38"),(50,"#C72F43"),(75,"#83215F"),(100,"#C137A8")]
PREC_SUM = [(0,"#F2F5F7"),(.5,"#D1F2ED"),(1,"#91DCD8"),(2,"#51B9D4"),(5,"#3787C1"),(10,"#2465A0"),(15,"#3D986A"),(20,"#A8C34E"),(30,"#E6D14B"),(40,"#F1A24D"),(50,"#E46840"),(75,"#C43D53"),(100,"#793A94")]

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
        if variable in SURFACE_FIELDS.values() or variable in PHASE_COMPONENTS or variable in ("snow_gsp","snow_con"):
            if variable not in lower:
                continue
        if variable=="pmsl" and "single-level" not in lower:
            continue
        if variable in ("t","fi") and ("pressure-level" not in lower or not re.search(r"(?:^|_)"+("850" if variable=="t" else "500")+r"(?:_|\.)",lower)):
            continue
        match=re.search("_"+stamp+r"_(\d{3})(?:_|\.)",lower)
        if match:
            result[int(match.group(1))]=urljoin(folder,name)
    return result


D2_BASE="https://opendata.dwd.de/weather/nwp/icon-d2/grib"

@st.cache_data(ttl=600,show_spinner=False)
def discover_d2(run,variable):
    folder=f"{D2_BASE}/{run:%H}/{variable}/"
    response=requests.get(folder,headers=HEADERS,timeout=25)
    response.raise_for_status()
    parser=Links()
    parser.feed(response.text)
    result={}
    stamp=run.strftime("%Y%m%d%H")
    for name in parser.names:
        lower=name.lower()
        if not lower.endswith(".grib2.bz2") or stamp not in lower or variable not in lower:
            continue
        if variable in ("t","fi"):
            if "pressure-level" not in lower or not re.search(r"(?:^|_)"+("850" if variable=="t" else "500")+r"(?:_|\.)",lower):
                continue
        match=re.search("_"+stamp+r"_(\d{3})(?:_|\.)",lower)
        if match:
            result[int(match.group(1))]=urljoin(folder,name)
    return result

@st.cache_data(ttl=3600,show_spinner=False)
def d2_native_coordinates(run):
    """Fetch time-invariant ICON-D2 cell-center coordinates in GRIB value order."""
    coords=[]
    for field in ("clat","clon"):
        folder=f"{D2_BASE}/{run:%H}/{field}/"
        response=requests.get(folder,headers=HEADERS,timeout=30)
        response.raise_for_status()
        parser=Links()
        parser.feed(response.text)
        stamp=run.strftime("%Y%m%d%H")
        names=[name for name in parser.names
               if name.lower().endswith(".grib2.bz2")
               and stamp in name and "icosahedral" in name.lower()]
        if not names:
            raise RuntimeError(f"ICON-D2-Gitterkoordinate {field.upper()} für Lauf {stamp} fehlt.")
        name=next((name for name in names if "time-invariant" in name.lower()),names[0])
        raw=requests.get(urljoin(folder,name),headers=HEADERS,timeout=90)
        raw.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".grib2",delete=False) as tmp:
            tmp.write(bz2.decompress(raw.content))
            filename=tmp.name
        try:
            with open(filename,"rb") as stream:
                gid=codes_grib_new_from_file(stream)
                if gid is None:
                    raise RuntimeError(f"Leere ICON-D2-{field.upper()}-Datei.")
                try:
                    values=np.asarray(codes_get_array(gid,"values"),dtype=np.float64)
                    try:
                        unit=str(codes_get(gid,"units")).lower()
                    except Exception:
                        unit=""
                    if "rad" in unit:
                        values=np.rad2deg(values)
                    elif not ("deg" in unit or "degree" in unit):
                        if field=="clat" and np.nanmax(np.abs(values))<=1.7:
                            values=np.rad2deg(values)
                        elif field=="clon" and np.nanmax(np.abs(values))<=6.4:
                            values=np.rad2deg(values)
                    coords.append(values)
                finally:
                    codes_release(gid)
        finally:
            os.unlink(filename)
    lat,lon=coords
    if len(lat)!=len(lon) or not np.isfinite(lat).all() or not np.isfinite(lon).all():
        raise RuntimeError("ICON-D2 CLAT/CLON haben ungleiche Länge oder ungültige Werte.")
    if not (np.nanmin(lat)>=-90 and np.nanmax(lat)<=90):
        raise RuntimeError("ICON-D2 CLAT liegt außerhalb gültiger Breitengrade.")
    return lat,lon


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
                    # Some ICON-D2 GRIB templates omit computed validity keys.
                    # Fall back to reference time plus forecast step.
                    try:
                        date=int(codes_get(gid,"validityDate"))
                        time=int(codes_get(gid,"validityTime"))
                        timestamp=datetime.strptime(f"{date:08d}{time:04d}","%Y%m%d%H%M").replace(tzinfo=timezone.utc)
                    except Exception as validity_error:
                        try:
                            date=int(codes_get(gid,"dataDate"))
                            time=int(codes_get(gid,"dataTime"))
                            step=int(codes_get(gid,"endStep"))
                            reference=datetime.strptime(f"{date:08d}{time:04d}","%Y%m%d%H%M").replace(tzinfo=timezone.utc)
                            timestamp=reference+timedelta(hours=step)
                        except Exception as fallback_error:
                            raise RuntimeError(
                                f"GRIB-Zeitmetadaten fehlen für {variable}: "
                                f"validityDate/validityTime ({validity_error}); "
                                f"dataDate/dataTime/endStep ({fallback_error})"
                            ) from fallback_error
                    if abs((timestamp-valid).total_seconds())>60:
                        continue
                    if variable in ("t","fi"):
                        level=int(codes_get(gid,"level"))
                        if level!=(850 if variable=="t" else 500):
                            continue
                    vals=np.asarray(codes_get_array(gid,"values"))
                    try:
                        lat=np.asarray(codes_get_array(gid,"latitudes"))
                        lon=np.asarray(codes_get_array(gid,"longitudes"))
                    except Exception as grid_error:
                        if "/icon-d2/" not in url:
                            raise RuntimeError(
                                f"GRIB-Koordinaten für {variable} nicht lesbar: {grid_error}"
                            ) from grid_error
                        # Native ICON-D2 GRIB (unstructured grid) has no embedded lat/lon.
                        # The DWD CLAT/CLON time-invariant fields share the value order.
                        stamp_match=re.search(r"_(\d{10})_\d{3}(?:_|\.)",url)
                        if not stamp_match:
                            raise RuntimeError("ICON-D2-Modelllauf im Dateinamen nicht erkennbar.")
                        model_run=datetime.strptime(stamp_match.group(1),"%Y%m%d%H").replace(tzinfo=timezone.utc)
                        lat,lon=d2_native_coordinates(model_run)
                    if len(vals)!=len(lat) or len(vals)!=len(lon):
                        raise RuntimeError(
                            f"ICON-D2-Gitter nicht kompatibel: {len(vals)} Werte, "
                            f"{len(lat)} CLAT und {len(lon)} CLON. "
                            "Kein räumlich falsches Kartenbild erzeugt."
                        )
                    lon=np.where(lon>180,lon-360,lon)
                    west,east,south,north=bounds
                    mask=(lon>=west-.5)&(lon<=east+.5)&(lat>=south-.5)&(lat<=north+.5)
                    if not mask.any():
                        raise RuntimeError("ICON-EU-Feld deckt die gewählte Region nicht ab.")
                    lat,lon,vals=lat[mask],lon[mask],vals[mask]
                    if variable=="fi":
                        units=str(codes_get(gid,"units")).lower().replace(" ","")
                        if "m2" in units or "m**2" in units or "m^2" in units: vals=vals/9.80665/10
                        elif units in ("gpm","m"): vals=vals/10
                        else: raise RuntimeError(f"Unbekannte FI-Einheit: {units}")
                        if not 400<np.nanmedian(vals)<650: raise RuntimeError("Z500-Werte unplausibel.")
                    elif variable=="pmsl":
                        vals=vals/100.0
                        if not 850<np.nanmedian(vals)<1100:
                            raise RuntimeError("MSLP-Einheiten unerwartet; Karte nicht erstellt.")
                    elif variable in ("t","t_2m","td_2m"):
                        vals=vals-273.15
                        if not -100<np.nanmedian(vals)<60:
                            raise RuntimeError("Temperatur-Einheiten unerwartet; Karte nicht erstellt.")
                    elif variable=="vmax_10m":
                        vals=vals*3.6
                        if not 0<=np.nanmedian(vals)<250:
                            raise RuntimeError("Windböen-Einheiten unerwartet; Karte nicht erstellt.")
                    elif variable=="h_snow":
                        vals=vals*100.0
                        if not 0<=np.nanmedian(vals)<2000: raise RuntimeError("Schneehöhe außerhalb plausibler Werte.")
                    elif variable=="tot_prec" or variable in PHASE_COMPONENTS:
                        if not 0<=np.nanmedian(vals)<2000:
                            raise RuntimeError("Niederschlagswerte unplausibel; Karte nicht erstellt.")
                    from scipy.interpolate import griddata
                    nx,ny=(225,180) if bounds in (REGIONS["Deutschland"],REGIONS["Mitteleuropa"]) else (300,220)
                    x=np.linspace(west,east,nx)
                    y=np.linspace(south,north,ny)
                    X,Y=np.meshgrid(x,y)
                    Z=griddata(np.column_stack((lon,lat)),vals,(X,Y),method="nearest" if variable=="ww" else "linear")
                    if variable=="ww" and not np.isfinite(Z).all():
                        raise RuntimeError("Ungültige WW-Werte im Modellfeld.")
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


# NOAA NOMADS GFS 0.25-degree subset; request only the necessary field and area.
GFS_FILTER="https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
GFS_FIELDS={
    "MSLP":("PRMSL","mean_sea_level"),
    "Z500 + MSLP":("HGT","500_mb"),
    "T850":("TMP","850_mb"),
    "T850 + MSLP":("TMP","850_mb"),
    "2-m-Temperatur":("TMP","2_m_above_ground"),
    "Taupunkt":("DPT","2_m_above_ground"),
    "Windböen":("GUST","surface"),
    "Schneehöhe":("SNOD","surface"),
    "Neuschneemenge":("ASNOW","surface"),
    "Windchill":("TMP","2_m_above_ground"),
    "Windchill-U":("UGRD","10_m_above_ground"),
    "Windchill-V":("VGRD","10_m_above_ground"),
}
@st.cache_data(ttl=600,show_spinner=False)
def discover_gfs(run,parameter,bounds):
    if parameter not in GFS_FIELDS:
        return {}
    # NOMADS exposes 3-hourly files for this interface; 0.25 degree is native.
    from urllib.parse import urlencode
    var,level=GFS_FIELDS[parameter]
    west,east,south,north=bounds
    result={}
    for lead in range(0,385,3):
        args={
            "file":f"gfs.t{run:%H}z.pgrb2.0p25.f{lead:03d}",
            f"lev_{level}":"on",
            f"var_{var}":"on",
            "subregion":"",
            "leftlon":str(west-.5),"rightlon":str(east+.5),
            "toplat":str(north+.5),"bottomlat":str(south-.5),
            "dir":f"/gfs.{run:%Y%m%d}/{run:%H}/atmos",
        }
        result[lead]=GFS_FILTER+"?"+urlencode(args)
    return result

@st.cache_data(ttl=900,show_spinner=False)
def load_gfs(url,valid,bounds,parameter):
    from scipy.interpolate import griddata
    response=requests.get(url,headers=HEADERS,timeout=100)
    response.raise_for_status()
    data=response.content
    if not data.startswith(b"GRIB"):
        raise RuntimeError("NOAA-NOMADS liefert aktuell keine GFS-GRIB-Daten (Modelllauf möglicherweise noch nicht verfügbar).")
    path=None
    try:
        with tempfile.NamedTemporaryFile(suffix=".grib2",delete=False) as file:
            file.write(data)
            path=file.name
        with open(path,"rb") as stream:
            while True:
                gid=codes_grib_new_from_file(stream)
                if gid is None:break
                try:
                    date=int(codes_get(gid,"validityDate"))
                    time=int(codes_get(gid,"validityTime"))
                    stamp=datetime.strptime(f"{date:08d}{time:04d}","%Y%m%d%H%M").replace(tzinfo=timezone.utc)
                    if abs((stamp-valid).total_seconds())>60:continue
                    lat=np.asarray(codes_get_array(gid,"latitudes"))
                    lon=np.asarray(codes_get_array(gid,"longitudes"))
                    vals=np.asarray(codes_get_array(gid,"values"))
                    lon=np.where(lon>180,lon-360,lon)
                    west,east,south,north=bounds
                    mask=(lon>=west-.5)&(lon<=east+.5)&(lat>=south-.5)&(lat<=north+.5)
                    if not mask.any():continue
                    lat,lon,vals=lat[mask],lon[mask],vals[mask]
                    if parameter=="MSLP":vals=vals/100.
                    elif parameter in ("T850","T850 + MSLP","2-m-Temperatur","Taupunkt","Windchill"):vals=vals-273.15
                    elif parameter=="Z500 + MSLP":vals=vals/10
                    elif parameter=="Windböen":vals=vals*3.6
                    elif parameter in ("Schneehöhe","Neuschneemenge"):vals=vals*100.0
                    if not np.isfinite(vals).all():raise RuntimeError("GFS-Feld enthält fehlende Werte.")
                    if parameter=="Z500 + MSLP" and not 400<np.median(vals)<650:raise RuntimeError("GFS-Z500 unplausibel.")
                    if parameter=="MSLP" and not 850<np.median(vals)<1100:raise RuntimeError("GFS-MSLP unplausibel.")
                    if parameter in ("T850","T850 + MSLP","2-m-Temperatur","Taupunkt") and not -100<np.median(vals)<60:raise RuntimeError("GFS-Temperatur unplausibel.")
                    if parameter=="Windböen" and not 0<=np.median(vals)<250:raise RuntimeError("GFS-Windböen unplausibel.")
                    nx,ny=(225,180) if bounds==REGIONS["Deutschland"] else (300,220)
                    X,Y=np.meshgrid(np.linspace(west,east,nx),np.linspace(south,north,ny))
                    Z=griddata(np.column_stack((lon,lat)),vals,(X,Y),method="linear")
                    if not np.isfinite(Z).all():
                        nearest=griddata(np.column_stack((lon,lat)),vals,(X,Y),method="nearest")
                        Z=np.where(np.isfinite(Z),Z,nearest)
                    return X,Y,Z
                finally:codes_release(gid)
    finally:
        if path and os.path.exists(path):os.unlink(path)
    raise RuntimeError("GFS-GRIB-Datei enthält kein Feld mit der angeforderten Gültigkeitszeit.")

def windchill_celsius(temp_c, u_ms, v_ms):
    """North-American/Canadian wind chill, using 10-m model wind as a proxy.

    Only defined for T <= 10 C and wind > 4.8 km/h. Outside that domain,
    show air temperature rather than extrapolating the wind-chill formula.
    """
    speed_kmh=np.hypot(u_ms,v_ms)*3.6
    if not (np.isfinite(temp_c).all() and np.isfinite(speed_kmh).all()):
        raise RuntimeError("Windchill-Eingangsdaten enthalten ungültige Werte.")
    calculated=13.12+0.6215*temp_c-11.37*np.power(np.maximum(speed_kmh,0.01),0.16)+0.3965*temp_c*np.power(np.maximum(speed_kmh,0.01),0.16)
    return np.where((temp_c<=10)&(speed_kmh>4.8),calculated,temp_c)

SNOW_STOPS=[(0.1,"#1BC33A"),(0.5,"#D8F6FC"),(1,"#A5E8F8"),(2,"#6DCEEF"),(5,"#339EDB"),(10,"#1765B9"),(20,"#5949BA"),(30,"#9853C8"),(50,"#D95AB6"),(75,"#EF779A"),(100,"#F7C7DF"),(150,"#FFFFFF")]
def snow_water_mask(X,Y):
    """Rasterize actual land polygons; ocean and lakes outside polygons remain blue."""
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds
    # Geo-countries covers land only; this is a land/sea mask, not an inland-lake inventory.
    geo=borders(COUNTRIES)
    geoms=[(f["geometry"],1) for f in geo.get("features",[]) if f.get("geometry")]
    ny,nx=X.shape
    transform=from_bounds(float(X.min()),float(Y.min()),float(X.max()),float(Y.max()),nx,ny)
    return np.flipud(rasterize(geoms,out_shape=(ny,nx),transform=transform,fill=0,dtype="uint8"))==0

def render(X,Y,Z,region,parameter,run,valid,model="ICON-EU",mslp=None):
    bounds=REGIONS[region]
    # A slightly shorter canvas for WW maps; preserve the full geographic extent and native data grid.
    fig,ax=plt.subplots(figsize=(10,10) if region=="Deutschland" else (12,8.5),dpi=120)
    fig.patch.set_facecolor("#f6f8fa")
    if parameter in ("Neuschneemenge","Schneehöhe"):
        from matplotlib.colors import ListedColormap, BoundaryNorm
        edges=[0,.1,.5,1,2,5,10,20,30,50,75,100,150,10000]
        colors=[c for _,c in SNOW_STOPS]
        colors.append("#FFFFFF")
        cmap=ListedColormap(colors)
        norm=BoundaryNorm(edges,cmap.N)
        im=ax.pcolormesh(X,Y,np.maximum(Z,0),cmap=cmap,norm=norm,shading="nearest",rasterized=True)
        # Water blue is a geographic background category, never a snow measurement.
        from matplotlib.colors import ListedColormap as _Listed
        water=snow_water_mask(X,Y)
        ax.pcolormesh(X,Y,np.ma.masked_where(~water,np.ones(Z.shape)),cmap=_Listed(["#2368AD"]),vmin=0,vmax=1,shading="nearest",zorder=3,rasterized=True)
        label="Schneehöhe (cm)" if parameter=="Schneehöhe" else "Berechnete Neuschneehöhe seit Modellstart (cm; 10:1-Schätzung bei ICON)"
    elif parameter=="Z500 + MSLP":
        from matplotlib.colors import LinearSegmentedColormap, Normalize
        if mslp is None or mslp.shape!=Z.shape: raise RuntimeError("MSLP-Raster fehlt.")
        anchors=[(480,"#65058C"),(490,"#8500F4"),(500,"#0033E9"),(510,"#008FF7"),
                 (520,"#00D6EC"),(530,"#00DF81"),(540,"#44E000"),(550,"#E7F500"),
                 (560,"#FFE600"),(570,"#FFAD00"),(580,"#FF5900"),(590,"#F10B12"),(600,"#DB008D")]
        cmap=LinearSegmentedColormap.from_list("elbtal_z500",[((v-480)/120,c) for v,c in anchors],N=512)
        im=ax.contourf(X,Y,Z,levels=np.linspace(480,600,241),cmap=cmap,norm=Normalize(480,600),extend="both")
        if not np.isfinite(mslp).all() or not 850<np.nanmedian(mslp)<1100:
            raise RuntimeError("MSLP-Werte für Isobaren ungültig.")
        lines=ax.contour(X,Y,mslp,levels=np.arange(940,1081,4),colors="#19243A",linewidths=.85,alpha=.9)
        ax.clabel(lines,inline=True,fontsize=6,fmt="%d")
        # Standard 552-gpdam contour (500-hPa geopotential), not a sea-level isobar.
        if float(np.nanmin(Z)) <= 552 <= float(np.nanmax(Z)):
            ridge=ax.contour(X,Y,Z,levels=[552],colors="#05070B",linewidths=1.8,zorder=6)
            ax.clabel(ridge,inline=True,fontsize=8,fmt={552:"552"},colors="#05070B")
        label="Geopotenzielle Höhe 500 hPa (gpdam) · Isobaren MSLP (hPa)"
    elif parameter=="MSLP":
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
    elif parameter in ("T850","T850 + MSLP"):
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
        if parameter=="T850 + MSLP":
            if mslp is None or mslp.shape!=Z.shape or not 850<float(np.nanmedian(mslp))<1100:
                raise RuntimeError("MSLP-Raster für weiße Isobaren fehlt oder ist ungültig.")
            lines=ax.contour(X,Y,mslp,levels=np.arange(940,1081,4),
                             colors="white",linewidths=1.05,alpha=.98)
            ax.clabel(lines,inline=True,fontsize=7,fmt="%d",colors="white")
        label="Temperatur auf 850 hPa (°C)" + (" · weiße MSLP-Isobaren (hPa)" if parameter=="T850 + MSLP" else "")
    elif parameter=="Signifikantes Wetter":
        cmap,norm,bounds_ww=palette()
        im=ax.pcolormesh(X,Y,Z,cmap=cmap,norm=norm,shading="nearest",rasterized=True)
        label="DWD ICON-EU WW (Wetterinterpretation)"
    else:
        from matplotlib.colors import LinearSegmentedColormap, Normalize
        if parameter=="2-m-Temperatur":
            stops,vmin,vmax=TEMP_STOPS,-20,20
            label="Lufttemperatur in 2 m (°C)"
        elif parameter=="Windchill":
            stops,vmin,vmax=TEMP_STOPS,-20,20
            label="Windchill (°C; T2m + Wind 10 m)"
        elif parameter=="Taupunkt":
            stops,vmin,vmax=TEMP_STOPS,-20,20
            label="Taupunkttemperatur in 2 m (°C)"
        elif parameter=="Windböen":
            stops,vmin,vmax=GUST_STOPS,0,120
            label="Windböen (km/h)"
        elif parameter=="Niederschlagsrate":
            stops,vmin,vmax=PREC_RATE,0,100
            label="Niederschlagsrate (mm/h; Mittel des Akkumulationsintervalls)"
        elif parameter=="Niederschlagssumme":
            stops,vmin,vmax=PREC_SUM,0,100
            label="Niederschlagssumme seit Modellstart (mm = l/m²)"
        else:
            raise RuntimeError(f"Unbekannter Parameter: {parameter}")
        cmap=LinearSegmentedColormap.from_list("elbtal_germany",[((v-vmin)/(vmax-vmin),color) for v,color in stops],N=2048)
        norm=Normalize(vmin=vmin,vmax=vmax)
        im=ax.contourf(X,Y,Z,levels=np.linspace(vmin,vmax,161),cmap=cmap,norm=norm,extend="both")
        if parameter in ("2-m-Temperatur","Taupunkt","Windchill"):
            contours=np.arange(-20,21,2)
        elif parameter=="Windböen":
            contours=np.arange(0,121,10)
        else:
            contours=[.5,1,2,5,10,20,30,50,75]
        lo,hi=float(np.nanmin(Z)),float(np.nanmax(Z))
        contours=[v for v in contours if lo<v<hi]
        if contours:
            lines=ax.contour(X,Y,Z,levels=contours,colors="#24364b",linewidths=.45,alpha=.7)
            ax.clabel(lines,inline=True,fontsize=6,fmt="%g")
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
    ax.set_title("")
    if parameter=="Signifikantes Wetter":
        ax.set_title("")
        # Grouped WW legend is drawn in the navy outer frame.
        fig.subplots_adjust(left=.095,right=.985,top=.975,bottom=.095)
    else:
        cb=fig.colorbar(im,ax=ax,orientation="vertical" if region!="Deutschland" else "horizontal",pad=.035 if region!="Deutschland" else .08,fraction=.045 if region!="Deutschland" else .05)
    if parameter=="Signifikantes Wetter":
        pass
    elif parameter in ("Neuschneemenge","Schneehöhe"):
        cb.set_ticks([.1,1,5,10,20,30,50,75,100,150])
    elif parameter=="Z500 + MSLP":
        cb.set_ticks(np.arange(480,601,10))
    elif parameter in ("T850","T850 + MSLP"):
        cb.set_ticks(np.arange(-30,31,5))
    elif parameter=="MSLP":
        cb.set_ticks([950,970,985,995,1005,1015,1030,1045,1060])
    elif parameter in ("2-m-Temperatur","Taupunkt","Windchill"):
        cb.set_ticks(np.arange(-20,21,5))
    elif parameter=="Windböen":
        cb.set_ticks(np.arange(0,121,10))
    else:
        cb.set_ticks([0,10,20,30,50,75,100])
    if parameter!="Signifikantes Wetter":
        cb.set_label(label)
    style_figure(fig, ax)
    if region=="Deutschland":
        # Dedicated 3:4 detail card: make the actual map occupy nearly the
        # entire plot width; reserve a compact strip for the color scale.
        # This fixes the excessive empty space that cropping the PNG cannot.
        if parameter=="Signifikantes Wetter":
            ax.set_position([.095,.085,.875,.89])
        else:
            ax.set_position([.095,.18,.875,.795])
            cb.ax.set_position([.14,.085,.78,.028])
    if region!="Deutschland":
        # Fill the card vertically, leaving a narrow, dedicated strip for
        # a right-hand temperature/parameter scale. Never stretch geography.
        fig.set_size_inches(15.2,7.8) if region=="Europa & Nordatlantik" else fig.set_size_inches(13.2,8.8)
        ax.set_aspect("auto")
        if parameter=="Signifikantes Wetter":
            ax.set_position([.07,.08,.89,.88])
        else:
            ax.set_position([.07,.09,.79,.87])
            cb.ax.set_position([.885,.13,.028,.79])
            cb.ax.yaxis.set_ticks_position("right")
            cb.ax.yaxis.set_label_position("right")
            cb.ax.tick_params(axis="y",labelsize=10,labelrotation=0,pad=5)
            # Keep tick numbers horizontal for easy reading, from top to bottom.
            cb.set_label(label,rotation=270,labelpad=22)
    output=io.BytesIO()
    fig.savefig(output,format="png",bbox_inches="tight",facecolor=fig.get_facecolor())
    plt.close(fig)
    return frame_png(output.getvalue(), parameter, model, region, run, valid)

def show(region, forced_model=None):
    st.subheader(f"Wettermodelle · {region}")
    model=forced_model or st.segmented_control("Wettermodell",["ICON-EU","ICON-D2","GFS"] if region=="Deutschland" else ["ICON-EU","GFS"],default="ICON-EU",key="synoptic_model")
    st.caption("ICON-D2: Deutschland bis +48 h stündlich (sofern die DWD-Dateien verfügbar sind). ICON-EU: bis +78 h stündlich, danach bis +120 h alle 3 Stunden. GFS: 0,25°-Gitter, +0 bis +384 h in 3-Stunden-Schritten; nur die hier unterstützten Felder. Grenzen bleiben erhalten.")
    st.caption("Signifikantes Wetter: originale DWD-WW-Codes, räumlich nur nächster Modellgitterpunkt. Nebel mit Reifbildung wird als Nebel gezeigt; Gewitter in der Nähe nicht als eigene Klasse.")
    choices=(["MSLP","T850","T850 + MSLP","Z500 + MSLP"] + (list(SURFACE_FIELDS) if region=="Deutschland" else [])) if model in ("ICON-EU","ICON-D2") else (["MSLP","T850","T850 + MSLP","Z500 + MSLP","2-m-Temperatur","Taupunkt","Windböen"] if region=="Deutschland" else ["MSLP","T850","T850 + MSLP","Z500 + MSLP"])
    choices += ["Neuschneemenge","Schneehöhe"]
    if "Windchill" not in choices: choices.append("Windchill")
    parameter=st.radio("Parameter",choices,horizontal=True,key="eu_parameter_"+model)
    if parameter=="Neuschneemenge":
        st.caption("ICON: aus akkumuliertem Schneewasseräquivalent (SNOW_GSP + SNOW_CON), 10:1-Umrechnung; Schätzung, keine prognostizierte Schneedeckenänderung. GFS: akkumulierte Schneefallhöhe (ASNOW), falls verfügbar.")
    elif parameter=="Schneehöhe":
        st.caption("Schneedecke zum Gültigkeitszeitpunkt: ICON H_SNOW / GFS SNOD; nicht der seit Modellstart gefallene Schnee.")
    elif parameter=="Windchill":
        st.caption("Berechnete Windchill-Temperatur aus 2-m-Lufttemperatur und 10-m-Wind (U/V), nicht aus Böen. Formel nur bei ≤ 10 °C und Wind > 4,8 km/h; sonst Lufttemperatur. Keine vollständige gefühlte Temperatur (Sonne/Feuchte fehlen).")
    from model_cycles import recent_cycles
    run_options=recent_cycles(model)
    run_choice=st.selectbox(
        "Modelllauf auswählen · vier letzte berechnete Zyklen (UTC)",
        options=[None]+run_options,
        format_func=lambda r: "Automatisch · jüngster verfügbarer Lauf" if r is None else f"{r:%d.%m.%Y %H} UTC",
        key="eu_run_"+model,
    )
    preview_run=run_choice or run_options[0]
    lead=st.select_slider(
        "Vorhersagestunde · Datum und Uhrzeit (MEZ/MESZ)",
        options=(list(range(49)) if model=="ICON-D2" else EU_LEADS) if model!="GFS" else list(range(0,385,3)),value=6,
        format_func=lambda h:f"+{h} h · {(preview_run+timedelta(hours=h)).astimezone(LOCAL_TZ):%d.%m.%Y %H:%M %Z}",
        key="eu_lead_"+model,
    )
    st.caption("Vier UTC-Zyklen nach modellabhängiger Bereitstellungsfrist; tatsächliche Verfügbarkeit wird beim Abruf geprüft.")
    if st.button(f"{model}-Karte manuell berechnen",type="primary"):
        with st.status(f"{model}-Datei und Grenzen laden …",expanded=True) as status:
            try:
                runs=[run_choice] if run_choice is not None else recent_cycles(model, count=6)
                variable="snow_gsp" if parameter=="Neuschneemenge" else ("pmsl" if parameter=="MSLP" else ("t" if parameter in ("T850","T850 + MSLP") else ("fi" if parameter=="Z500 + MSLP" else SURFACE_FIELDS.get(parameter,"t_2m"))))
                selected=None
                for run in runs:
                    try:
                        available=(discover_d2(run,variable) if model=="ICON-D2" else discover(run,variable)) if model!="GFS" else discover_gfs(run,parameter,REGIONS[region])
                    except requests.RequestException:
                        continue
                    previous_lead=previous_eu_lead(lead)
                    required=[previous_lead,lead] if parameter=="Niederschlagsrate" and previous_lead is not None else [lead]
                    if all(hour in available for hour in required):
                        if parameter=="Windchill" and model!="GFS":
                            try:
                                getfiles=discover_d2 if model=="ICON-D2" else discover
                                u_files=getfiles(run,"u_10m")
                                v_files=getfiles(run,"v_10m")
                            except requests.RequestException: continue
                            if lead not in u_files or lead not in v_files: continue
                        if parameter=="Neuschneemenge" and model!="GFS":
                            try: extra=discover_d2(run,"snow_con") if model=="ICON-D2" else discover(run,"snow_con")
                            except requests.RequestException: continue
                            if lead not in extra: continue
                        if parameter in ("Z500 + MSLP","T850 + MSLP") and model!="GFS":
                            try: pressure_files=discover_d2(run,"pmsl") if model=="ICON-D2" else discover(run,"pmsl")
                            except requests.RequestException: continue
                            if lead not in pressure_files: continue
                        if model=="GFS":
                            try:
                                if parameter=="Windchill":
                                    for component in ("Windchill-U","Windchill-V"):
                                        wind_urls=discover_gfs(run,component,REGIONS[region])
                                        load_gfs(wind_urls[lead],run+timedelta(hours=lead),REGIONS[region],component)
                                # Check the actual GRIB; a URL may precede publication.
                                load_gfs(available[lead],run+timedelta(hours=lead),REGIONS[region],parameter)
                            except (requests.RequestException,RuntimeError,ValueError,OSError) as exc:
                                st.write(f"GFS-Lauf {run:%d.%m. %H} UTC nicht verfügbar: {type(exc).__name__}")
                                continue
                        selected=(run,available[lead],available.get(previous_lead))
                        break
                if selected is None:
                    raise RuntimeError(f"Keine passende {model}-Datei für {parameter} +{lead} h im "+("gewählten Lauf" if run_choice is not None else "verfügbaren Lauf")+" gefunden. Bitte einen anderen Lauf oder Prognoseschritt wählen.")
                run,url,previous_url=selected
                valid=run+timedelta(hours=lead)
                X,Y,Z=load_grid(url,valid,REGIONS[region],variable) if model!="GFS" else load_gfs(url,valid,REGIONS[region],parameter)
                if parameter=="Windchill":
                    if model=="GFS":
                        u_url=discover_gfs(run,"Windchill-U",REGIONS[region])[lead]
                        v_url=discover_gfs(run,"Windchill-V",REGIONS[region])[lead]
                        _,_,u=load_gfs(u_url,valid,REGIONS[region],"Windchill-U")
                        _,_,v=load_gfs(v_url,valid,REGIONS[region],"Windchill-V")
                    else:
                        getfiles=discover_d2 if model=="ICON-D2" else discover
                        _,_,u=load_grid(getfiles(run,"u_10m")[lead],valid,REGIONS[region],"u_10m")
                        _,_,v=load_grid(getfiles(run,"v_10m")[lead],valid,REGIONS[region],"v_10m")
                    Z=windchill_celsius(Z,u,v)
                if parameter=="Neuschneemenge" and model!="GFS":
                    extra=discover_d2(run,"snow_con") if model=="ICON-D2" else discover(run,"snow_con")
                    _,_,convective=load_grid(extra[lead],valid,REGIONS[region],"snow_con")
                    # Both components are accumulated mm water equivalent. 10:1 -> cm snow.
                    Z=np.maximum(Z+convective,0)
                if parameter=="Niederschlagsrate":
                    if lead==0:
                        Z=np.zeros_like(Z)
                    else:
                        interval=lead-previous_lead
                        _,_,prior=load_grid(previous_url,run+timedelta(hours=previous_lead),REGIONS[region],variable)
                        diff=(Z-prior)/interval
                        if np.nanmin(diff)<-.1:
                            raise RuntimeError("Niederschlagsakkumulation fällt ab. Keine Rate berechnet.")
                        Z=np.maximum(diff,0)
                elif parameter=="Niederschlagssumme":
                    Z=np.maximum(Z,0)
                elif parameter=="Signifikantes Wetter":
                    Z=classify_ww(Z)
                if parameter=="Niederschlagsrate" and lead>78:
                    st.info(f"Niederschlagsrate: Mittelwert über die letzten {lead-previous_lead} Stunden (mm/h), nicht einstündiger Niederschlag.")
                mslp=None
                if parameter in ("Z500 + MSLP","T850 + MSLP"):
                    if model!="GFS":
                        pressure_files=discover_d2(run,"pmsl") if model=="ICON-D2" else discover(run,"pmsl")
                        _,_,mslp=load_grid(pressure_files[lead],valid,REGIONS[region],"pmsl")
                    else:
                        pressure_urls=discover_gfs(run,"MSLP",REGIONS[region])
                        _,_,mslp=load_gfs(pressure_urls[lead],valid,REGIONS[region],"MSLP")
                png=render(X,Y,Z,region,parameter,run,valid,model=model,mslp=mslp)
                st.session_state["eu_map"]=(MAP_STYLE_VERSION,model,region,parameter,lead,valid,png)
                st.caption(f"Verwendeter Modelllauf: {run:%d.%m.%Y %H:%M} UTC · Gültig: {valid.astimezone(LOCAL_TZ):%d.%m.%Y %H:%M %Z}")
                status.update(label=f"{model}-Karte erstellt",state="complete")
            except Exception as exc:
                status.update(label=f"{model}-Verarbeitung fehlgeschlagen",state="error")
                st.error(f"{type(exc).__name__}: {exc}")
    result=st.session_state.get("eu_map")
    if result and (len(result)!=7 or result[0]!=MAP_STYLE_VERSION):
        st.session_state.pop("eu_map",None)
        st.info("Die Karten-Farbskala wurde aktualisiert. Bitte die ICON-EU-Karte erneut berechnen.")
        result=None
    if result and len(result)==7 and result[:5]==(MAP_STYLE_VERSION,model,region,parameter,lead):
        _,_,_,_,_,valid,png=result
        st.caption(f"Gültig: {valid.astimezone(LOCAL_TZ):%d.%m.%Y %H:%M %Z} · Modellzeitpunkt UTC: {valid:%d.%m.%Y %H:%M}")
        st.image(png,width="stretch")
        st.download_button("PNG herunterladen",png,file_name=f"ELBTAL_{model}_{region}_{parameter}_{valid:%Y%m%d_%H%M}.png",mime="image/png")

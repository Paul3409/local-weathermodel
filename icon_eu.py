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
REGIONS = {"Deutschland": (5.5, 15.6, 47.0, 55.3), "Europa": (-12.0, 35.0, 34.0, 72.0)}
HEADERS = {"User-Agent": "ELBTAL weather visualization (educational)"}
MAP_STYLE_VERSION = "time-label-gfs-v1"
LOCAL_TZ=ZoneInfo("Europe/Berlin")
EU_LEADS=list(range(79))+list(range(81,121,3))
def previous_eu_lead(lead):
    if lead==0:
        return None
    if lead<=78:
        return lead-1
    return lead-3
SURFACE_FIELDS = {"2-m-Temperatur":"t_2m","Windböen":"vmax_10m","Niederschlagsrate":"tot_prec","Niederschlagssumme":"tot_prec","Taupunkt":"td_2m","Signifikantes Wetter":"ww"}
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
        if variable in SURFACE_FIELDS.values() or variable in PHASE_COMPONENTS:
            if variable not in lower:
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
                    elif variable in ("t","t_2m","td_2m"):
                        vals=vals-273.15
                        if not -100<np.nanmedian(vals)<60:
                            raise RuntimeError("Temperatur-Einheiten unerwartet; Karte nicht erstellt.")
                    elif variable=="vmax_10m":
                        vals=vals*3.6
                        if not 0<=np.nanmedian(vals)<250:
                            raise RuntimeError("Windböen-Einheiten unerwartet; Karte nicht erstellt.")
                    elif variable=="tot_prec" or variable in PHASE_COMPONENTS:
                        if not 0<=np.nanmedian(vals)<2000:
                            raise RuntimeError("Niederschlagswerte unplausibel; Karte nicht erstellt.")
                    from scipy.interpolate import griddata
                    nx,ny=(225,180) if bounds==REGIONS["Deutschland"] else (300,220)
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
    "T850":("TMP","850_mb"),
    "2-m-Temperatur":("TMP","2_m_above_ground"),
    "Taupunkt":("DPT","2_m_above_ground"),
    "Windböen":("GUST","surface"),
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
    for lead in range(0,121,3):
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
                    elif parameter in ("T850","2-m-Temperatur","Taupunkt"):vals=vals-273.15
                    elif parameter=="Windböen":vals=vals*3.6
                    if not np.isfinite(vals).all():raise RuntimeError("GFS-Feld enthält fehlende Werte.")
                    if parameter=="MSLP" and not 850<np.median(vals)<1100:raise RuntimeError("GFS-MSLP unplausibel.")
                    if parameter in ("T850","2-m-Temperatur","Taupunkt") and not -100<np.median(vals)<60:raise RuntimeError("GFS-Temperatur unplausibel.")
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

def render(X,Y,Z,region,parameter,run,valid,model="ICON-EU"):
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
    elif parameter=="T850":
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
    elif parameter=="Signifikantes Wetter":
        cmap,norm,bounds_ww=palette()
        im=ax.pcolormesh(X,Y,Z,cmap=cmap,norm=norm,shading="nearest",rasterized=True)
        label="DWD ICON-EU WW (Wetterinterpretation)"
    else:
        from matplotlib.colors import LinearSegmentedColormap, Normalize
        if parameter=="2-m-Temperatur":
            stops,vmin,vmax=TEMP_STOPS,-20,20
            label="2-m-Temperatur (°C)"
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
        if parameter in ("2-m-Temperatur","Taupunkt"):
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
    ax.set_title(f"ELBTAL | {model} {parameter} | {region}\nLauf {run:%d.%m.%Y %H:%M} UTC · Gültig {valid.astimezone(LOCAL_TZ):%d.%m.%Y %H:%M %Z}",fontsize=11)
    if parameter=="Signifikantes Wetter":
        fig.subplots_adjust(bottom=.27)
        fig.legend(handles=legend_handles(),loc="lower center",ncol=4,fontsize=7,
                   frameon=False,bbox_to_anchor=(.5,.015),columnspacing=.9,handlelength=1.6)
    else:
        cb=fig.colorbar(im,ax=ax,orientation="horizontal",pad=.08,fraction=.05)
    if parameter=="Signifikantes Wetter":
        pass
    elif parameter=="T850":
        cb.set_ticks(np.arange(-30,31,5))
    elif parameter=="MSLP":
        cb.set_ticks([950,970,985,995,1005,1015,1030,1045,1060])
    elif parameter in ("2-m-Temperatur","Taupunkt"):
        cb.set_ticks(np.arange(-20,21,5))
    elif parameter=="Windböen":
        cb.set_ticks(np.arange(0,121,10))
    else:
        cb.set_ticks([0,10,20,30,50,75,100])
    if parameter!="Signifikantes Wetter":
        cb.set_label(label)
    output=io.BytesIO()
    fig.savefig(output,format="png",bbox_inches="tight",facecolor=fig.get_facecolor())
    plt.close(fig)
    return output.getvalue()

def show(region):
    st.subheader(f"Wettermodelle · {region}")
    model=st.segmented_control("Wettermodell",["ICON-EU","GFS"],default="ICON-EU",key="synoptic_model")
    st.caption("ICON-EU: bis +78 h stündlich, danach bis +120 h alle 3 Stunden. GFS: 0,25°-Gitter, +0 bis +120 h; nur die hier unterstützten Felder. Grenzen bleiben erhalten.")
    st.caption("Signifikantes Wetter: originale DWD-WW-Codes, räumlich nur nächster Modellgitterpunkt. Nebel mit Reifbildung wird als Nebel gezeigt; Gewitter in der Nähe nicht als eigene Klasse.")
    choices=(["MSLP","T850"] + (list(SURFACE_FIELDS) if region=="Deutschland" else [])) if model=="ICON-EU" else (["MSLP","T850","2-m-Temperatur","Taupunkt","Windböen"] if region=="Deutschland" else ["MSLP","T850"])
    parameter=st.radio("Parameter",choices,horizontal=True,key="eu_parameter_"+model)
    now_preview=datetime.now(timezone.utc)
    preview_run=now_preview.replace(minute=0,second=0,microsecond=0)
    preview_run=preview_run.replace(hour=(preview_run.hour//6)*6)
    lead=st.select_slider(
        "Vorhersagestunde · Datum und Uhrzeit (MEZ/MESZ)",
        options=EU_LEADS if model=="ICON-EU" else list(range(0,121,3)),value=6,
        format_func=lambda h:f"+{h} h · {(preview_run+timedelta(hours=h)).astimezone(LOCAL_TZ):%d.%m.%Y %H:%M %Z}",
        key="eu_lead_"+model,
    )
    st.caption("Die Uhrzeit an der Auswahl ist eine Vorschau. Maßgeblich ist der beim Abruf tatsächlich gefundene Modelllauf; dessen Gültigkeitszeit steht auf der Karte.")
    if st.button(f"{model}-Karte manuell berechnen",type="primary"):
        with st.status(f"{model}-Datei und Grenzen laden …",expanded=True) as status:
            try:
                now=datetime.now(timezone.utc)
                runs=[]
                for offset in range(0,34,3):
                    run=(now-timedelta(hours=offset)).replace(minute=0,second=0,microsecond=0)
                    run=run.replace(hour=(run.hour//3)*3)
                    if run not in runs and (model=="GFS" or lead<=78 or run.hour in (0,6,12,18)) and (model!="GFS" or run.hour in (0,6,12,18)):
                        runs.append(run)
                variable="pmsl" if parameter=="MSLP" else ("t" if parameter=="T850" else SURFACE_FIELDS[parameter])
                selected=None
                for run in runs:
                    try:
                        available=discover(run,variable) if model=="ICON-EU" else discover_gfs(run,parameter,REGIONS[region])
                    except requests.RequestException:
                        continue
                    previous_lead=previous_eu_lead(lead)
                    required=[previous_lead,lead] if parameter=="Niederschlagsrate" and previous_lead is not None else [lead]
                    if all(hour in available for hour in required):
                        selected=(run,available[lead],available.get(previous_lead))
                        break
                if selected is None:
                    raise RuntimeError(f"Keine passende ICON-EU-Datei für {parameter} +{lead} h gefunden. Bei T850 wird ausdrücklich eine 850-hPa-Druckflächen-Datei benötigt.")
                run,url,previous_url=selected
                valid=run+timedelta(hours=lead)
                X,Y,Z=load_grid(url,valid,REGIONS[region],variable) if model=="ICON-EU" else load_gfs(url,valid,REGIONS[region],parameter)
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
                png=render(X,Y,Z,region,parameter,run,valid,model=model)
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

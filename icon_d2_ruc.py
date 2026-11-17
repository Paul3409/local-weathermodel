"""DWD ICON-D2-RUC direct native-grid maps (official Open Data v1).
Hourly cycles, forecast hours 0..27. No Open-Meteo.
"""
import io
import os
import tempfile
from datetime import datetime,timedelta,timezone
from urllib.parse import quote
import numpy as np
import requests
import streamlit as st
from eccodes import codes_grib_new_from_file,codes_get,codes_get_array,codes_release
from scipy.interpolate import griddata
from icon_eu import REGIONS,LOCAL_TZ,render,plot_geometry,borders,COUNTRIES
from map_design import style_figure,frame_png
from global_models import climatology_on_grid,anomaly_map

BASE="https://opendata.dwd.de/weather/nwp/v1/m/icon-d2-ruc/p"
HEADERS={"User-Agent":"MeteoOst-Wetterkarten/1.20 (educational)"}
PARAMS={
    "MSLP":("PMSL",None,None),
    "T850":("T",100,85000),
    "T850 + MSLP":("T",100,85000),
    "T2":("T_2M",None,None),
    "Z500 + MSLP":("FI",100,50000),
    "Schneehöhe":("H_SNOW",None,None),
    "Windböen":("VMAX_10M",None,None),
    "Niederschlagssumme":("TOT_PREC",None,None),
    "Niederschlagsrate":("TOT_PREC",None,None),
    "CAPE":("CAPE_ML",None,None),
    "Simulierte Radarreflektivität":("DBZ_CMAX",None,None),
}
def url(run,lead,param,level_type=None,level=None):
    root=f"{BASE}/{quote(param,safe='')}"
    if level_type is not None:
        root+=f"/lvt1/{level_type}/lv1/{level}"
    stamp=quote(run.strftime("%Y-%m-%dT%H:00"),safe="-T")
    return f"{root}/r/{stamp}/s/PT{lead:03d}H00M.grib2"

@st.cache_data(ttl=900,show_spinner=False)
def download(url):
    response=requests.get(url,headers=HEADERS,timeout=100)
    response.raise_for_status()
    if not response.content.startswith(b"GRIB"):
        raise RuntimeError("DWD hat keine GRIB2-Datei geliefert.")
    return response.content

@st.cache_data(ttl=3600,show_spinner=False)
def coordinates(run):
    coords=[]
    for param in ("CLAT","CLON"):
        raw=download(url(run,0,param))
        vals,_,_=grib_values(raw)
        coords.append(vals)
    lat,lon=coords
    # ICON native CLAT/CLON are often in radians.
    if np.nanmax(np.abs(lat))<2:lat=np.rad2deg(lat)
    if np.nanmax(np.abs(lon))<7:lon=np.rad2deg(lon)
    if len(lat)!=len(lon) or np.nanmax(np.abs(lat))>90:
        raise RuntimeError("DWD-Gitterkoordinaten sind ungültig.")
    return lat,lon

def grib_values(raw):
    path=None
    try:
        with tempfile.NamedTemporaryFile(suffix=".grib2",delete=False) as f:
            f.write(raw);path=f.name
        with open(path,"rb") as stream:
            gid=codes_grib_new_from_file(stream)
            if gid is None:raise RuntimeError("Leere GRIB-Datei.")
            try:
                vals=np.asarray(codes_get_array(gid,"values"),dtype=float)
                units=str(codes_get(gid,"units"))
                valid=None
                try:
                    valid=(int(codes_get(gid,"validityDate")),int(codes_get(gid,"validityTime")))
                except Exception:pass
                return vals,units,valid
            finally:codes_release(gid)
    finally:
        if path and os.path.exists(path):os.unlink(path)

@st.cache_data(ttl=900,show_spinner=False)
def field(run,lead,parameter,region):
    param,level_type,level=PARAMS[parameter]
    raw=download(url(run,lead,param,level_type,level))
    vals,units,valid=grib_values(raw)
    lat,lon=coordinates(run)
    if len(vals)!=len(lat):
        raise RuntimeError("ICON-D2-RUC-Werte und Gitter passen nicht zusammen.")
    west,east,south,north=REGIONS[region]
    mask=(lat>=south-.7)&(lat<=north+.7)&(lon>=west-.7)&(lon<=east+.7)&np.isfinite(vals)&(np.abs(vals)<9000)
    if mask.sum()<20:raise RuntimeError("Keine gültigen RUC-Daten für das Kartengebiet.")
    if param in ("T","T_2M"):vals=vals-273.15
    elif param=="PMSL":vals=vals/100
    elif param=="H_SNOW":vals=vals*100
    elif param=="VMAX_10M":vals=vals*3.6
    elif param=="FI":
        vals=vals/(9.80665*10) if "m2" in units.replace(" ","").lower() else vals/10
    nx,ny=(225,180) if region=="Deutschland" else (300,220)
    X,Y=np.meshgrid(np.linspace(west,east,nx),np.linspace(south,north,ny))
    points=np.column_stack((lon[mask],lat[mask]))
    Z=griddata(points,vals[mask],(X,Y),method="linear")
    if not np.isfinite(Z).all():
        near=griddata(points,vals[mask],(X,Y),method="nearest")
        Z=np.where(np.isfinite(Z),Z,near)
    return X,Y,Z

def render_diagnostic(X,Y,Z,region,parameter,run,valid):
    """Dedicated CAPE and reflectivity palette without modifying existing map renderer."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm,ListedColormap
    if parameter=="CAPE":
        levels=[0,50,100,250,500,750,1000,1500,2000,3000,4000,6000]
        colors=["#f4f6fa","#c9e7f7","#83c3eb","#458cc9","#3462aa","#46a36b","#b9c750","#f3cf49","#ef9139","#df4c3b","#a32669"]
        label="Mixed-Layer CAPE (J/kg)"
    else:
        levels=[-10,0,5,10,15,20,25,30,35,40,45,50,55,60,65,75,90]
        colors=["#f5f7fa","#d8edf8","#abd9f1","#74b5e5","#387fcb","#2366a4","#3a9c74","#77bc4e","#c4d34a","#f2cf42","#f6a23b","#e96c36","#db3e35","#b92655","#90266e","#652e8e"]
        label="Simulierte maximale Radarreflektivität (dBZ)"
    fig,ax=plt.subplots(figsize=(10,10) if region=="Deutschland" else (12,8.5),dpi=120)
    cmap=ListedColormap(colors)
    im=ax.contourf(X,Y,Z,levels=levels,cmap=cmap,norm=BoundaryNorm(levels,cmap.N),extend="max")
    try:
        for feature in borders(COUNTRIES).get("features",[]):
            plot_geometry(ax,feature.get("geometry"),.45)
    except Exception:pass
    west,east,south,north=REGIONS[region]
    ax.set_xlim(west,east);ax.set_ylim(south,north)
    cb=fig.colorbar(im,ax=ax,shrink=.75);cb.set_label(label)
    style_figure(fig,ax)
    out=io.BytesIO();fig.savefig(out,format="png",bbox_inches="tight");plt.close(fig)
    return frame_png(out.getvalue(),parameter,"ICON-D2-RUC",region,run,valid)

def recent_runs():
    cutoff=datetime.now(timezone.utc)-timedelta(minutes=55)
    latest=cutoff.replace(minute=0,second=0,microsecond=0)
    return [latest-timedelta(hours=i) for i in range(49)]

def show(region,forced_parameter=None):
    if forced_parameter is None:
        st.subheader(f"ICON-D2-RUC · {region}")
    st.caption("DWD Open Data v1 · stündliche Modellläufe · bis +27 h · native ICON-Gitterdaten. Kein Open-Meteo.")
    from v2_model_runs import run_picker
    run=run_picker("ICON-D2-RUC","ruc_run")
    lead=st.select_slider("Vorhersagestunde",options=list(range(28)),value=6,
        format_func=lambda h:f"+{h} h · {(run+timedelta(hours=h)).astimezone(LOCAL_TZ):%d.%m.%Y %H:%M %Z}",key="ruc_lead")
    parameter=forced_parameter if forced_parameter in (list(PARAMS)+["MSLP-Anomalie"]) else st.radio("Parameter",list(PARAMS)+["MSLP-Anomalie"],horizontal=True,key="ruc_param")
    if parameter=="Niederschlagsrate":
        st.caption("Mittlere Niederschlagsrate aus zwei aufeinanderfolgenden Stundenakkumulationen (mm/h).")
    def _calculate_one(lead):
        try:
            with st.spinner("DWD-RUC-Originaldaten laden …"):
                target="MSLP" if parameter=="MSLP-Anomalie" else parameter
                X,Y,Z=field(run,lead,target,region)
                valid=run+timedelta(hours=lead)
                if parameter=="Niederschlagsrate":
                    if lead==0:Z=np.zeros_like(Z)
                    else:
                        previous=field(run,lead-1,"Niederschlagssumme",region)[2]
                        if np.nanmin(Z-previous)<-.1:raise RuntimeError("Niederschlagssumme ist nicht monoton.")
                        Z=np.maximum(0,Z-previous)
                if parameter=="MSLP-Anomalie":
                    png=anomaly_map(X,Y,Z,climatology_on_grid(X,Y,valid.month),region,"ICON-D2-RUC",run,valid)
                else:
                    pressure=field(run,lead,"MSLP",region)[2] if parameter in ("T850 + MSLP","Z500 + MSLP") else None
                    name={"T2":"2-m-Temperatur","CAPE":"CAPE","Simulierte Radarreflektivität":"Simulierte Radarreflektivität"}.get(parameter,parameter)
                    png=(render_diagnostic(X,Y,Z,region,parameter,run,valid) if parameter in ("CAPE","Simulierte Radarreflektivität") else render(X,Y,Z,region,name,run,valid,model="ICON-D2-RUC",mslp=pressure))
                st.session_state["ruc_map"]=(region,run,lead,parameter,valid,png)
                st.session_state.setdefault("ruc_map_batch",{})[(region,run,lead,parameter)]=png
        except Exception as exc:
            st.error(f"DWD-RUC-Abruf fehlgeschlagen: {type(exc).__name__}: {exc}")
    st.markdown("**⚡ Prognosezeitraum vorladen**")
    _opts=list(range(28))
    _first,_last=st.select_slider("Zeitraum auswählen",options=_opts,value=(_opts[0],_opts[min(3,len(_opts)-1)]),key="ruc_map_range",format_func=lambda h:f"+{h} h · {(run+timedelta(hours=h)).astimezone(LOCAL_TZ):%d.%m. %H:%M}")
    _interval=st.selectbox("Kartenabstand (Stunden)",[1,2,3,6,12],index=2,key="ruc_map_interval")
    _hours=[h for h in _opts if _first<=h<=_last and (h-_first)%_interval==0]
    st.caption(f"{len(_hours)} Karten · fertige Karten erscheinen sofort.")
    # A fragment runs independently of the forecast-hour slider's full-app reruns.
    _batch_identity=(region,run,parameter)
    if st.button(f"⚡ {len(_hours)} Karten vorladen",key="ruc_map_batch_queue_start"):
        st.session_state["ruc_map_batch_queue"]={"identity":_batch_identity,"hours":list(_hours)}

    @st.fragment(run_every="2s")
    def _batch_worker():
        job=st.session_state.get("ruc_map_batch_queue")
        if not job or job["identity"]!=_batch_identity:
            return
        _all=job["hours"]
        _cache=st.session_state.get("ruc_map_batch",{})
        def _key(hour):
            return (region,run,hour,parameter)
        _done=sum(_key(h) in _cache for h in _all)
        if _done<len(_all):
            hour=next(h for h in _all if _key(h) not in _cache)
            _calculate_one(hour)
            _cache=st.session_state.get("ruc_map_batch",{})
            _done=sum(_key(h) in _cache for h in _all)
            # A failed forecast hour must not block the remaining hours.
            if _key(hour) not in _cache:
                job.setdefault("failed",[]).append(hour)
                job["hours"]=[h for h in _all if h!=hour]
        _total=len(_all)+len(job.get("failed",[]))
        _finished=_done+len(job.get("failed",[]))
        st.progress(min(1.0,_finished/max(1,_total)),text=f"{_finished}/{_total} verarbeitet · {_done} Karten fertig")
        if _done>=len(job["hours"]):
            st.success(f"{_done} Karten vorgeladen. Du kannst sie oben einzeln auswählen.")

    _batch_worker()
    if st.button("RUC-Karte berechnen",type="primary",key="ruc_generate"):
        _calculate_one(lead)
    _cached_png=st.session_state.get("ruc_map_batch",{}).get((region,run,lead,parameter))
    if _cached_png is not None:
        _current=st.session_state.get("ruc_map")
        if not _current or _current[:4]!=(region,run,lead,parameter):
            st.session_state["ruc_map"]=(region,run,lead,parameter,run+timedelta(hours=lead),_cached_png)
    result=st.session_state.get("ruc_map")
    if result and result[:4]==(region,run,lead,parameter):
        st.image(result[5],use_container_width=True)
        st.download_button("PNG herunterladen",result[5],file_name=f"MeteoOst_ICON-D2-RUC_{parameter}_{result[4]:%Y%m%d%H}.png",mime="image/png")

"""Direct ECMWF IFS and NOAA GEFS ensemble-mean maps; no Open-Meteo."""
import io
import os
import tempfile
from datetime import datetime, timedelta, timezone

import numpy as np
import requests
import streamlit as st
from eccodes import codes_grib_new_from_file, codes_get, codes_get_array, codes_release
from scipy.interpolate import griddata
from icon_eu import REGIONS, LOCAL_TZ, render

PARAMS={"MSLP":"msl","T850":"t","T850 + MSLP":"t","Z500 + MSLP":"z","T2":"2t","Schneehöhe":"sd"}
HEADERS={"User-Agent":"ELBTAL-Wetterkarten/1.0 (educational)"}

from model_cycles import recent_cycles

def cycles(model):
    return recent_cycles(model)

def steps(model,run):
    if model=="ECMWF IFS":
        return list(range(0,145,3))+ (list(range(150,361,6)) if run.hour in (0,12) else [])
    return list(range(0,241,3))+list(range(246,385,6))

@st.cache_data(ttl=1800,show_spinner=False)
def ecmwf_bytes(run,step,parameter):
    from ecmwf.opendata import Client
    client=Client(source="ecmwf",model="ifs",resol="0p25")
    path=None
    try:
        with tempfile.NamedTemporaryFile(suffix=".grib2",delete=False) as f:path=f.name
        req={"date":run.strftime("%Y%m%d"),"time":run.hour,"step":step,
             "stream":"oper","type":"fc","param":parameter}
        if parameter in ("t","z"):req.update(levelist=850 if parameter=="t" else 500)
        client.retrieve(**req,target=path)
        with open(path,"rb") as f:return f.read()
    finally:
        if path and os.path.exists(path):os.unlink(path)

@st.cache_data(ttl=1800,show_spinner=False)
def gefs_bytes(run,step,parameter,bounds):
    # NOAA NOMADS filter for precomputed GEFS ensemble mean (geavg).
    west,east,south,north=bounds
    var={"msl":"PRMSL","t":"TMP","z":"HGT"}[parameter]
    level={"msl":"mean sea level","t":"850 mb","z":"500 mb"}[parameter]
    filename=f"geavg.t{run:%H}z.pgrb2a.0p50.f{step:03d}"
    params={"file":filename,"dir":f"/gefs.{run:%Y%m%d}/{run:%H}/atmos/pgrb2ap5",
            f"var_{var}":"on",f"lev_{level.replace(' ','_')}":"on",
            "subregion":"","leftlon":west-1,"rightlon":east+1,
            "toplat":north+1,"bottomlat":south-1}
    url="https://nomads.ncep.noaa.gov/cgi-bin/filter_gefs_atmos_0p50a.pl"
    response=requests.get(url,params=params,headers=HEADERS,timeout=110)
    response.raise_for_status()
    if not response.content.startswith(b"GRIB"):
        raise RuntimeError("NOAA hat für diesen Lauf/Parameter keine GEFS-Mittelwertdatei geliefert.")
    return response.content

def decode(raw,valid,bounds,parameter):
    path=None
    try:
        with tempfile.NamedTemporaryFile(suffix=".grib2",delete=False) as f:
            f.write(raw);path=f.name
        with open(path,"rb") as stream:
            while True:
                gid=codes_grib_new_from_file(stream)
                if gid is None:break
                try:
                    date=int(codes_get(gid,"validityDate"))
                    tm=int(codes_get(gid,"validityTime"))
                    stamp=datetime.strptime(f"{date:08d}{tm:04d}","%Y%m%d%H%M").replace(tzinfo=timezone.utc)
                    if abs((stamp-valid).total_seconds())>60:continue
                    lat=np.asarray(codes_get_array(gid,"latitudes"),dtype=float)
                    lon=np.asarray(codes_get_array(gid,"longitudes"),dtype=float)
                    vals=np.asarray(codes_get_array(gid,"values"),dtype=float)
                    lon=np.where(lon>180,lon-360,lon)
                    west,east,south,north=bounds
                    mask=(lon>=west-1)&(lon<=east+1)&(lat>=south-1)&(lat<=north+1)&np.isfinite(vals)
                    if mask.sum()<10:continue
                    vals=vals[mask]
                    if parameter=="msl":vals=vals/100
                    elif parameter in ("t","2t"):vals=vals-273.15
                    elif parameter=="z":
                        # ECMWF z is geopotential (m2/s2), NOAA HGT is height (m).
                        short=str(codes_get(gid,"shortName"))
                        vals=vals/(9.80665*10) if short=="z" else vals/10
                    nx,ny=(225,180) if bounds==REGIONS["Deutschland"] else (300,220)
                    X,Y=np.meshgrid(np.linspace(west,east,nx),np.linspace(south,north,ny))
                    Z=griddata(np.column_stack((lon[mask],lat[mask])),vals,(X,Y),method="linear")
                    if not np.isfinite(Z).all():
                        near=griddata(np.column_stack((lon[mask],lat[mask])),vals,(X,Y),method="nearest")
                        Z=np.where(np.isfinite(Z),Z,near)
                    if parameter=="msl" and not 850<float(np.nanmedian(Z))<1100:
                        raise RuntimeError("MSLP-Daten unplausibel.")
                    return X,Y,Z
                finally:codes_release(gid)
    finally:
        if path and os.path.exists(path):os.unlink(path)
    raise RuntimeError("GRIB enthält kein passendes Feld für den gewählten Gültigkeitszeitpunkt.")

def get_field(model,run,lead,parameter,bounds):
    if model=="ECMWF IFS" and parameter=="sd":
        # ECMWF sd is metres of water equivalent, not snow depth in metres.
        # Combine with forecast snow density (kg/m³) for physical depth in cm.
        X,Y,swe=decode(ecmwf_bytes(run,lead,"sd"),run+timedelta(hours=lead),bounds,"sd")
        _,_,density=decode(ecmwf_bytes(run,lead,"rsn"),run+timedelta(hours=lead),bounds,"rsn")
        snow_cm=np.where((swe>0)&(density>0),100000*np.maximum(swe,0)/np.maximum(density,1),0)
        return X,Y,snow_cm
    raw=ecmwf_bytes(run,lead,parameter) if model=="ECMWF IFS" else gefs_bytes(run,lead,parameter,bounds)
    return decode(raw,run+timedelta(hours=lead),bounds,parameter)

@st.cache_data(ttl=7*86400,show_spinner=False)
def monthly_pressure_climatology(month):
    """NOAA PSL NCEP/NCAR Reanalysis 1, monthly MSLP normals 1991-2020 (hPa)."""
    import h5py
    urls=[
        "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis.derived/surface/slp.mon.ltm.1991-2020.nc",
        "https://psl.noaa.gov/thredds/fileServer/Datasets/ncep.reanalysis/Monthlies/surface/slp.mon.ltm.1991-2020.nc",
    ]
    failures=[]
    for url in urls:
        try:
            response=requests.get(url,headers=HEADERS,timeout=45)
            response.raise_for_status()
            with h5py.File(io.BytesIO(response.content),"r") as nc:
                lat=np.asarray(nc["lat"][:],dtype=float)
                lon=np.asarray(nc["lon"][:],dtype=float)
                values=np.asarray(nc["slp"][month-1],dtype=float)
                scale=np.asarray(nc["slp"].attrs.get("scale_factor",1),dtype=float).reshape(-1)[0]
                offset=np.asarray(nc["slp"].attrs.get("add_offset",0),dtype=float).reshape(-1)[0]
                values=values*scale+offset
            if values.shape!=(len(lat),len(lon)) or not 950<float(np.nanmedian(values))<1060:
                raise ValueError("Ungültige MSLP-Klimatologie.")
            return lat,lon,values
        except (requests.RequestException,OSError,KeyError,ValueError) as exc:
            failures.append(f"{type(exc).__name__}: {exc}")
    raise RuntimeError("NOAA-Klimatologie nicht erreichbar: "+"; ".join(failures))

def climatology_on_grid(X,Y,month):
    from scipy.interpolate import RegularGridInterpolator
    lat,lon,values=monthly_pressure_climatology(month)
    # NOAA longitudes run from 0 to <360; extend periodically across Greenwich.
    # Europe contains negative longitudes, so modulo 360 alone puts western
    # points beyond the last grid longitude and creates artificial NaNs.
    il=np.argsort(lat);jl=np.argsort(np.mod(lon,360))
    lon_sorted=np.mod(lon[jl],360)
    field=values[np.ix_(il,jl)]
    periodic_lon=np.concatenate((lon_sorted[-1:]-360,lon_sorted,lon_sorted[:1]+360))
    periodic_field=np.concatenate((field[:,-1:],field,field[:,:1]),axis=1)
    interpolator=RegularGridInterpolator((lat[il],periodic_lon),periodic_field,bounds_error=False,fill_value=np.nan)
    query_lon=np.mod(X.ravel(),360)
    query_lon=np.where(query_lon>periodic_lon[-2],query_lon-360,query_lon)
    result=interpolator(np.column_stack((Y.ravel(),query_lon))).reshape(X.shape)
    if not np.isfinite(result).all():
        raise RuntimeError("Klimatologie deckt das Kartengebiet nicht vollständig ab.")
    return result

def anomaly_map(X,Y,pressure,reference,region,model,run,valid):
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from map_design import frame_png, style_figure
    delta=pressure-reference
    fig,ax=plt.subplots(figsize=(9,11) if region=="Deutschland" else (12,8.5),dpi=120)
    fig.patch.set_facecolor("#061C35")
    ax.set_facecolor("#061C35")
    bounds=REGIONS[region]
    levels=np.arange(-32,34,2)
    im=ax.contourf(X,Y,delta,levels=levels,cmap="RdBu_r",norm=TwoSlopeNorm(vcenter=0,vmin=-32,vmax=32),extend="both")
    lines=ax.contour(X,Y,pressure,levels=np.arange(940,1081,4),colors="#1b2735",linewidths=.6)
    ax.clabel(lines,inline=True,fontsize=6,fmt="%d")
    from icon_eu import borders, COUNTRIES
    try:
        import matplotlib.path as mpath
        from matplotlib.patches import PathPatch
        for feature in borders(COUNTRIES).get("features",[]):
            geom=feature.get("geometry") or {}
            polys=geom.get("coordinates",[]) if geom.get("type")=="MultiPolygon" else [geom.get("coordinates",[])]
            for poly in polys:
                for ring in poly:
                    if len(ring)>2:
                        arr=np.asarray(ring)
                        ax.plot(arr[:,0],arr[:,1],color="#343e4d",linewidth=.45,alpha=.8)
    except (requests.RequestException,ValueError,KeyError):
        st.caption("Grenzdaten momentan nicht verfügbar.")
    ax.set_xlim(bounds[0],bounds[1]);ax.set_ylim(bounds[2],bounds[3])
    ax.set_xlabel("Längengrad");ax.set_ylabel("Breitengrad")
    colorbar=fig.colorbar(im,ax=ax,shrink=.75,label="MSLP-Anomalie (hPa)")
    ax.set_title("")
    style_figure(fig,ax)
    colorbar.outline.set_edgecolor("#B5C8DC")
    if region!="Deutschland":
        # Wide Atlantic-European geographic field: allocate most of the
        # horizontal canvas to the map instead of a near-square axes box.
        fig.set_size_inches(15,7.4)
        ax.set_aspect("auto")
        ax.set_position([.055,.11,.82,.84])
        colorbar.ax.set_position([.895,.18,.023,.66])
    out=io.BytesIO();fig.savefig(out,format="png",bbox_inches="tight");plt.close(fig)
    return frame_png(out.getvalue(),"MSLP-Anomalie",model,region,run,valid)

def show(region, forced_model=None, forced_parameter=None):
    if forced_model is None:
        st.subheader(f"Globale Modelle · {region}")
    model=forced_model or st.segmented_control("Modell",["ECMWF IFS","GEFS Mittel"],default="ECMWF IFS",key="global_model")
    st.caption("Direktdaten: ECMWF Open Data (IFS oper) / NOAA NOMADS (GEFS geavg). Kein Open-Meteo.")
    from v2_model_runs import run_picker
    run=run_picker(model,"global_run_"+model)
    st.caption("UTC-Initialisierung, nicht Fertigstellungszeit. Modellabhängige Verzögerung berücksichtigt; Dateiverfügbarkeit beim Abruf geprüft.")
    options=steps(model,run)
    lead=st.select_slider("Prognosestunde",options=options,value=options[min(2,len(options)-1)],
        format_func=lambda h:f"+{h} h · {(run+timedelta(hours=h)).astimezone(LOCAL_TZ):%d.%m.%Y %H:%M %Z}",
        key="global_lead_"+model)
    parameters=["MSLP","T850","T850 + MSLP","Z500 + MSLP","MSLP-Anomalie"]
    if model=="ECMWF IFS":parameters+=["T2","Schneehöhe"]
    parameter=forced_parameter if forced_parameter in parameters else st.radio("Parameter",parameters,horizontal=True,key="global_parameter_"+model)
    if parameter=="Schneehöhe":
        st.caption("ECMWF-Schneehöhe in cm, berechnet aus Schneewasseräquivalent (sd) und prognostizierter Schneedichte (rsn).")
    if parameter=="MSLP-Anomalie":
        st.caption("Referenz: NOAA PSL NCEP/NCAR Reanalysis 1, Monatsmittel 1991–2020 (2,5°). Automatischer Download; Monatsklimatologie, kein tagesgenauer ERA5-Vergleich.")
    def _calculate_one(lead):
        try:
            with st.spinner(f"{model} direkt herunterladen und berechnen …"):
                valid=run+timedelta(hours=lead)
                variable=PARAMS["MSLP" if parameter=="MSLP-Anomalie" else parameter]
                X,Y,Z=get_field(model,run,lead,variable,REGIONS[region])
                if parameter in ("Z500 + MSLP","T850 + MSLP"):
                    _,_,mslp=get_field(model,run,lead,"msl",REGIONS[region])
                else:mslp=None
                if parameter=="MSLP-Anomalie":
                    baseline=climatology_on_grid(X,Y,valid.month)
                    png=anomaly_map(X,Y,Z,baseline,region,model,run,valid)
                else:png=render(X,Y,Z,region,"2-m-Temperatur" if parameter=="T2" else parameter,run,valid,model=model,mslp=mslp)
                st.session_state["global_map"]=(model,region,run,lead,parameter,valid,png)

                st.session_state.setdefault("global_map_batch",{})[(model,region,run,lead,parameter)]=png
        except Exception as exc:
            st.error(f"Modellabruf fehlgeschlagen: {type(exc).__name__}: {exc}")
    st.markdown("**⚡ Prognosezeitraum vorladen**")
    _batch_options=options
    _first,_last=st.select_slider("Zeitraum auswählen",options=_batch_options,
        value=(_batch_options[0],_batch_options[min(3,len(_batch_options)-1)]),
        key="global_map_batch_range",
        format_func=lambda h:f"+{h} h · {(run+timedelta(hours=h)).astimezone(LOCAL_TZ):%d.%m. %H:%M}")
    _interval=st.selectbox("Kartenabstand (Stunden)",[1,2,3,6,12],index=2,
        key="global_map_batch_interval")
    _hours=[h for h in _batch_options if _first<=h<=_last and (h-_first)%_interval==0]
    st.caption(f"{len(_hours)} Karten · fertig berechnete Karten erscheinen sofort.")
    if st.button(f"⚡ {len(_hours)} Karten vorladen",key="global_map_batch_start"):
        _progress=st.progress(0)
        _ok=0
        for _i,_hour in enumerate(_hours):
            _batch_key=(model,region,run,_hour,parameter)
            _cached=st.session_state.get("global_map_batch",{})
            if _batch_key not in _cached:
                _calculate_one(_hour)
                _cached=st.session_state.get("global_map_batch",{})
            if _batch_key in _cached:
                _ok+=1
            _progress.progress((_i+1)/len(_hours),text=f"{_i+1}/{len(_hours)} verarbeitet · {_ok} fertig")
        st.success(f"{_ok}/{len(_hours)} Karten verfügbar.")
    if st.button("Globale Karte berechnen",type="primary",key="global_generate"):
        _calculate_one(lead)
    _cached_png=st.session_state.get("global_map_batch",{}).get((model,region,run,lead,parameter))
    if _cached_png is not None:
        _current=st.session_state.get("global_map")
        if not _current or _current[:5]!=(model,region,run,lead,parameter):
            st.session_state["global_map"]=(model,region,run,lead,parameter,run+timedelta(hours=lead),_cached_png)
    result=st.session_state.get("global_map")
    if result and result[:5]==(model,region,run,lead,parameter):
        st.image(result[6],use_container_width=True)
        st.download_button("PNG herunterladen",result[6],file_name=f"ELBTAL_{model.replace(' ','_')}_{result[5]:%Y%m%d%H}_{lead:03d}.png",mime="image/png")

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

PARAMS={"MSLP":"msl","T850":"t","Z500 + MSLP":"z"}
HEADERS={"User-Agent":"ELBTAL-Wetterkarten/1.0 (educational)"}

def cycles():
    now=datetime.now(timezone.utc)
    result=[]
    for back in range(0,30,3):
        d=(now-timedelta(hours=back)).replace(minute=0,second=0,microsecond=0)
        d=d.replace(hour=d.hour//6*6)
        if d not in result and (now-d).total_seconds()<=86400:result.append(d)
    return result

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
                    elif parameter=="t":vals=vals-273.15
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
    raw=ecmwf_bytes(run,lead,parameter) if model=="ECMWF IFS" else gefs_bytes(run,lead,parameter,bounds)
    return decode(raw,run+timedelta(hours=lead),bounds,parameter)

def anomaly_map(X,Y,pressure,reference,region,model,run,valid):
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from map_design import frame_png
    delta=pressure-reference
    fig,ax=plt.subplots(figsize=(9,8),dpi=120)
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
    fig.colorbar(im,ax=ax,shrink=.75,label="MSLP-Anomalie (hPa)")
    ax.set_title(f"ELBTAL | {model} MSLP-Anomalie | {region}\\nLauf {run:%d.%m.%Y %H} UTC · Gültig {valid.astimezone(LOCAL_TZ):%d.%m.%Y %H:%M %Z}")
    out=io.BytesIO();fig.savefig(out,format="png",bbox_inches="tight");plt.close(fig)
    return frame_png(out.getvalue(),"MSLP-Anomalie",model,region,run,valid)

def show(region):
    st.subheader(f"Globale Modelle · {region}")
    model=st.segmented_control("Modell",["ECMWF IFS","GEFS Mittel"],default="ECMWF IFS",key="global_model")
    st.caption("Direktdaten: ECMWF Open Data (IFS oper) / NOAA NOMADS (GEFS geavg). Kein Open-Meteo.")
    available=cycles()
    run=st.selectbox("Modelllauf der letzten 24 Stunden (UTC)",available,
        format_func=lambda d:f"{d:%d.%m.%Y %H} UTC",key="global_run_"+model)
    options=steps(model,run)
    lead=st.select_slider("Prognosestunde",options=options,value=options[min(2,len(options)-1)],
        format_func=lambda h:f"+{h} h · {(run+timedelta(hours=h)).astimezone(LOCAL_TZ):%d.%m.%Y %H:%M %Z}",
        key="global_lead_"+model)
    parameter=st.radio("Parameter",["MSLP","T850","Z500 + MSLP","MSLP-Anomalie"],horizontal=True,key="global_parameter_"+model)
    ref=None
    if parameter=="MSLP-Anomalie":
        st.info("Für eine echte Anomalie ist eine externe Druck-Klimatologie erforderlich. Keine Abweichung von einem erfundenen Standardwert.")
        upload=st.file_uploader("Referenz-Klimatologie (NPZ mit lat, lon und msl_hpa; passender Kalendermonat)",type=["npz"],key="global_climate")
        if upload is not None:
            with np.load(io.BytesIO(upload.getvalue())) as data:
                ref={k:data[k].copy() for k in ("lat","lon","msl_hpa")}
    if st.button("Globale Karte berechnen",type="primary",key="global_generate"):
        try:
            with st.spinner(f"{model} direkt herunterladen und berechnen …"):
                valid=run+timedelta(hours=lead)
                variable=PARAMS["MSLP" if parameter=="MSLP-Anomalie" else parameter]
                X,Y,Z=get_field(model,run,lead,variable,REGIONS[region])
                if parameter=="Z500 + MSLP":
                    _,_,mslp=get_field(model,run,lead,"msl",REGIONS[region])
                else:mslp=None
                if parameter=="MSLP-Anomalie":
                    if ref is None:raise ValueError("Bitte zuerst eine passende MSLP-Klimatologie hochladen.")
                    from scipy.interpolate import RegularGridInterpolator
                    la=np.asarray(ref["lat"]);lo=np.asarray(ref["lon"]);cl=np.asarray(ref["msl_hpa"])
                    if la.ndim!=1 or lo.ndim!=1 or cl.shape!=(len(la),len(lo)):
                        raise ValueError("Klimatologie benötigt 1D lat/lon und msl_hpa mit Form (lat,lon).")
                    il=np.argsort(la);jl=np.argsort(lo)
                    interp=RegularGridInterpolator((la[il],lo[jl]),cl[np.ix_(il,jl)],bounds_error=False,fill_value=np.nan)
                    baseline=interp(np.column_stack((Y.ravel(),X.ravel()))).reshape(X.shape)
                    if not np.isfinite(baseline).all() or not 850<np.median(baseline)<1100:
                        raise ValueError("Klimatologie deckt Kartengebiet nicht ab oder hat falsche Einheit.")
                    png=anomaly_map(X,Y,Z,baseline,region,model,run,valid)
                else:png=render(X,Y,Z,region,parameter,run,valid,model=model,mslp=mslp)
                st.session_state["global_map"]=(model,region,run,lead,parameter,valid,png)
        except Exception as exc:
            st.error(f"Modellabruf fehlgeschlagen: {type(exc).__name__}: {exc}")
    result=st.session_state.get("global_map")
    if result and result[:5]==(model,region,run,lead,parameter):
        st.image(result[6],use_container_width=True)
        st.download_button("PNG herunterladen",result[6],file_name=f"ELBTAL_{model.replace(' ','_')}_{result[5]:%Y%m%d%H}_{lead:03d}.png",mime="image/png")

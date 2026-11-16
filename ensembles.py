"""ELBTAL point ensemble meteograms from official GEFS / ECMWF open GRIB data."""
import io
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
import numpy as np
import requests
import streamlit as st
from eccodes import codes_grib_new_from_file, codes_get, codes_get_array, codes_release
from matplotlib import pyplot as plt
from matplotlib.dates import DateFormatter, DayLocator
from icon_eu import LOCAL_TZ
from map_design import NAVY
from app_version import VERSION

PLACES={
    "Mühlberg (Elbe)":(51.434,13.222),"Berlin":(52.52,13.405),
    "Leipzig":(51.34,12.375),"Dresden":(51.05,13.74),"Hamburg":(53.55,10.0),
    "München":(48.14,11.58),"Köln":(50.94,6.96),"Frankfurt am Main":(50.11,8.68),
    "Stuttgart":(48.78,9.18),"Hannover":(52.38,9.73),"Bremen":(53.08,8.8),
    "Wien":(48.21,16.37),"Prag":(50.08,14.44),"Warschau":(52.23,21.01),
    "Krakau":(50.06,19.94),"Amsterdam":(52.37,4.90),"Brüssel":(50.85,4.35),
    "Paris":(48.86,2.35),"London":(51.51,-0.13),"Zürich":(47.38,8.54),
    "Rom":(41.90,12.50),"Mailand":(45.46,9.19),"Madrid":(40.42,-3.70),
    "Barcelona":(41.39,2.17),"Lissabon":(38.72,-9.14),"Kopenhagen":(55.68,12.57),
    "Stockholm":(59.33,18.07),"Oslo":(59.91,10.75),"Helsinki":(60.17,24.94),
    "Budapest":(47.50,19.04),"Bratislava":(48.15,17.11),"Ljubljana":(46.06,14.51),
    "Zagreb":(45.81,15.98),"Belgrad":(44.82,20.46),"Bukarest":(44.43,26.10),
    "Sofia":(42.70,23.32),"Athen":(37.98,23.73),"Dublin":(53.35,-6.26),
    "Edinburgh":(55.95,-3.19),"Reykjavík":(64.15,-21.94),
}
FIELDS={"MSLP":"msl","T850":"t850","T2":"t2","T2 + Niederschlag":"t2","Neuschnee":"snow"}
COLORS=["#1DC6EE","#30B79C","#81C56D","#E2BF61","#E77D5F","#B07DD3"]
HEADERS={"User-Agent":"ELBTAL-Wetterkarten/1.0"}

def runs(model):
    now=datetime.now(timezone.utc)
    hours=(0,6,12,18) if model=="GEFS" else (0,6,12,18)
    result=[]
    for back in range(0,30,3):
        dt=(now-timedelta(hours=back)).replace(minute=0,second=0,microsecond=0)
        dt=dt.replace(hour=max(h for h in hours if h<=dt.hour))
        if dt not in result and now-dt<=timedelta(hours=24):result.append(dt)
    return result

def read_point(data,lat,lon,expected):
    if not data.startswith(b"GRIB"):raise RuntimeError("Keine GRIB-Daten erhalten")
    path=None
    try:
        with tempfile.NamedTemporaryFile(suffix=".grib2",delete=False) as tmp:
            tmp.write(data);path=tmp.name
        with open(path,"rb") as f:
            while True:
                gid=codes_grib_new_from_file(f)
                if gid is None:break
                try:
                    short=str(codes_get(gid,"shortName"))
                    if expected=="msl" and short not in ("prmsl","msl"):continue
                    if expected=="t850" and short not in ("t","t850"):continue
                    if expected=="t2" and short not in ("2t","t2m"):continue
                    if expected=="precip" and short not in ("tp","apcp"):continue
                    if expected=="snow" and short not in ("sd","sde","snod","sf","weasd"):continue
                    if expected=="t850" and int(codes_get(gid,"level"))!=850:continue
                    lats=np.asarray(codes_get_array(gid,"latitudes"))
                    lons=np.asarray(codes_get_array(gid,"longitudes"))
                    distance=(lats-lat)**2+(np.mod(lons-lon+180,360)-180)**2*np.cos(np.deg2rad(lat))**2
                    value=float(codes_get_array(gid,"values")[int(np.argmin(distance))])
                    if expected=="msl":value/=100
                    elif expected in ("t850","t2"):value-=273.15
                    elif expected=="snow" and short in ("sd","sde","snod"):value*=100
                    elif expected in ("precip","snow") and short in ("tp","sf"):value*=1000
                    return value
                finally:codes_release(gid)
    finally:
        if path and os.path.exists(path):os.unlink(path)
    raise RuntimeError(f"Feld {expected} in GRIB nicht gefunden")

@st.cache_data(ttl=3600,show_spinner=False)
def gefs_point(run,step,member,field,lat,lon):
    # Use NOAA's point-sized GRIB subregion for individual members.
    specs={
        "msl":("PRMSL","mean_sea_level"),
        "t850":("TMP","850_mb"),
        "t2":("TMP","2_m_above_ground"),
        "precip":("APCP","surface"),
        "snow":("SNOD","surface"),
    }
    variable,level=specs[field]
    name=("gec00" if member==0 else f"gep{member:02d}")+f".t{run:%H}z.pgrb2a.0p50.f{step:03d}"
    params={"file":name,"dir":f"/gefs.{run:%Y%m%d}/{run:%H}/atmos/pgrb2ap5",
            f"var_{variable}":"on",f"lev_{level}":"on",
            "subregion":"","leftlon":round(lon-1.1,3),"rightlon":round(lon+1.1,3),
            "toplat":round(lat+1.1,3),"bottomlat":round(lat-1.1,3)}
    response=requests.get("https://nomads.ncep.noaa.gov/cgi-bin/filter_gefs_atmos_0p50a.pl",
                          params=params,headers=HEADERS,timeout=45)
    response.raise_for_status()
    return read_point(response.content,lat,lon,field)

@st.cache_data(ttl=3600,show_spinner=False)
def ecmwf_step(run,step,field,lat,lon):
    from ecmwf.opendata import Client
    client=Client(source="ecmwf",model="ifs",resol="0p25")
    param={"msl":"msl","t850":"t","t2":"2t","precip":"tp","snow":"sf"}[field]
    path=None
    try:
        with tempfile.NamedTemporaryFile(suffix=".grib2",delete=False) as f:path=f.name
        kwargs={"date":run.strftime("%Y%m%d"),"time":run.hour,"step":step,
                "stream":"enfo","type":"pf","param":param,"number":list(range(1,51))}
        if field=="t850":kwargs["levelist"]=850
        client.retrieve(**kwargs,target=path)
        with open(path,"rb") as f:raw=f.read()
        # Each GRIB message represents a different ensemble member.
        results={}
        with open(path,"rb") as f:
            while True:
                gid=codes_grib_new_from_file(f)
                if gid is None:break
                try:
                    member=int(codes_get(gid,"number"))
                    lats=np.asarray(codes_get_array(gid,"latitudes"))
                    lons=np.asarray(codes_get_array(gid,"longitudes"))
                    distance=(lats-lat)**2+(np.mod(lons-lon+180,360)-180)**2*np.cos(np.deg2rad(lat))**2
                    value=float(codes_get_array(gid,"values")[int(np.argmin(distance))])
                    if field=="msl":value/=100
                    elif field in ("t850","t2"):value-=273.15
                    elif field in ("precip","snow"):value*=1000
                    results[member]=value
                finally:codes_release(gid)
        if not results:raise RuntimeError("Keine ECMWF-Ensemblemitglieder in der Datei")
        return results
    finally:
        if path and os.path.exists(path):os.unlink(path)

def fetch(model,run,field,lat,lon,steps):
    members=list(range(31)) if model=="GEFS" else list(range(1,51))
    data=np.full((len(members),len(steps)),np.nan)
    failures=[]
    if model=="GEFS":
        tasks=[(i,j) for i in range(len(members)) for j in range(len(steps))]
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures={pool.submit(gefs_point,run,steps[j],members[i],field,lat,lon):(i,j) for i,j in tasks}
            for future in as_completed(futures):
                i,j=futures[future]
                try:data[i,j]=future.result()
                except Exception as exc:failures.append(str(exc))
    else:
        for j,step in enumerate(steps):
            try:
                result=ecmwf_step(run,step,field,lat,lon)
                for i,member in enumerate(members):data[i,j]=result.get(member,np.nan)
            except Exception as exc:failures.append(str(exc))
    return data,failures

def draw(model,place,run,field,steps,data,precip=None):
    from matplotlib.ticker import MaxNLocator
    times=[(run+timedelta(hours=h)).astimezone(LOCAL_TZ) for h in steps]
    plt.rcParams.update({"font.family":"DejaVu Sans"})
    fig,ax=plt.subplots(figsize=(12,8),dpi=135)
    fig.patch.set_facecolor(NAVY);ax.set_facecolor("#0C2946")
    fig.subplots_adjust(left=.085,right=.96,top=.79,bottom=.20)
    for i,row in enumerate(data):
        ax.plot(times,row,color=COLORS[i%len(COLORS)],lw=.85,alpha=.52)
    mean=np.nanmean(data,axis=0)
    ax.plot(times,mean,color="white",lw=3,label="Ensemble-Mittel",zorder=6)
    if model=="GEFS" and np.isfinite(data[0]).any():
        ax.plot(times,data[0],color="#41F277",lw=2,label="Kontrolllauf",zorder=7)
    ax.grid(color="#6C8BA5",alpha=.3,linestyle="--")
    ax.tick_params(colors="#D9E8F6")
    for spine in ax.spines.values():spine.set_color("#5CA8D4")
    ylabel={"msl":"Luftdruck (hPa)","t850":"850-hPa-Temperatur (°C)",
            "t2":"2-m-Temperatur (°C)","snow":"Schneehöhe (cm)"}[field]
    ax.set_ylabel(ylabel,color="white")
    if field in ("t2","t850"):ax.axhline(0,color="#D9E8F6",ls="--",lw=1)
    ax.xaxis.set_major_locator(DayLocator(interval=2))
    ax.xaxis.set_major_formatter(DateFormatter("%d.%m",tz=LOCAL_TZ))
    ax.yaxis.set_major_locator(MaxNLocator(8))
    if precip is not None:
        rain=np.nanmean(precip,axis=0)
        ax2=ax.twinx()
        ax2.bar(times,rain,width=.55,color="#44B9F0",alpha=.23)
        ax2.set_ylabel("Niederschlag (mm / Intervall)",color="#83D8FF")
        ax2.tick_params(axis="y",colors="#83D8FF")
    fig.text(.045,.952,"ELBTAL",color="white",fontsize=25,fontweight="bold")
    fig.text(.045,.916,"W E T T E R K A R T E N",color="#45C8F5",fontsize=9)
    title={"msl":"MSLP","t850":"T850","t2":"T2","snow":"Schneehöhe"}[field]
    if precip is not None:title="T2 + Niederschlag"
    fig.text(.42,.95,f"Ensemble · {title}",color="white",fontsize=20,fontweight="bold")
    fig.text(.42,.91,f"{model} · {place}",color="#A9C3D9",fontsize=12)
    fig.text(.045,.85,f"Modelllauf: {run:%d.%m.%Y %H} UTC  ·  Zeitraum: +{steps[0]} bis +{steps[-1]} h",color="#C5D8E9",fontsize=11)
    fig.text(.045,.055,f"Daten: {'NOAA' if model=='GEFS' else 'ECMWF'} · {place}",color="#A9C3D9",fontsize=10)
    fig.text(.79,.055,f"ELBTAL V{VERSION}",color="white",fontsize=11,fontweight="bold")
    ax.legend(loc="upper right",facecolor="#0C2946",labelcolor="white",edgecolor="#406080",fontsize=9)
    out=io.BytesIO();fig.savefig(out,format="png",facecolor=fig.get_facecolor());plt.close(fig)
    return out.getvalue()

def show():
    st.subheader("📊 ELBTAL · Ensemble-Diagramme")
    model=st.segmented_control("Ensemble",["GEFS","ECMWF ENS"],default="GEFS",key="ens_model")
    place=st.selectbox("Ort",list(PLACES),key="ens_place")
    parameter=st.selectbox("Parameter",list(FIELDS),key="ens_param")
    run=st.selectbox("Modelllauf (UTC)",runs(model),format_func=lambda x:x.strftime("%d.%m.%Y %H UTC"),key="ens_run")
    horizon=st.select_slider("Prognosezeitraum",options=[48,96,144,192,240,288,336,384] if model=="GEFS" else [48,96,144,192,240,288,336,360] if run.hour in (0,12) else [48,96,144],value=96,key="ens_horizon")
    interval=12 if horizon<=144 else 24
    steps=list(range(0,horizon+1,interval))
    lat,lon=PLACES[place]
    st.caption("Offizielle Ensemble-Einzelmitglieder · GEFS: 30 + Kontrolllauf; ECMWF: 50. Die Berechnung kann wegen der Datenmenge länger dauern.")
    if parameter=="Neuschnee":
        st.info("GEFS SNOD ist Schneehöhe am Boden, nicht Neuschneezuwachs. Neuschnee wird erst nach verifizierter Differenzbildung angeboten.")
    if st.button("Ensemble berechnen",type="primary",key="ens_go"):
        with st.spinner("Ensemble-Einzelmitglieder werden direkt von NOAA / ECMWF geladen …"):
            try:
                field=FIELDS[parameter]
                if parameter=="Neuschnee":
                    raise ValueError("Neuschnee aus SNOD ist noch nicht fachlich korrekt implementiert.")
                data,errors=fetch(model,run,field,lat,lon,steps)
                if np.isfinite(data).sum()==0:raise RuntimeError("Keine gültigen Ensemble-Daten: "+("; ".join(errors[:2]) if errors else "Unbekannter Abruffehler"))
                precipitation=None
                if parameter=="T2 + Niederschlag":
                    precipitation,precip_errors=fetch(model,run,"precip",lat,lon,steps)
                    if np.isfinite(precipitation).sum()==0:
                        raise RuntimeError("Niederschlagsdaten fehlen: "+"; ".join(precip_errors[:2]))
                png=draw(model,place,run,field,steps,data,precipitation)
                st.session_state["ens_chart"]=(model,place,parameter,run,horizon,png)
                if errors:st.warning(f"{len(errors)} Teilabrufe fehlgeschlagen. Nur verfügbare Ensemblewerte dargestellt.")
            except Exception as exc:st.error(f"Ensemble-Abruf fehlgeschlagen: {type(exc).__name__}: {exc}")
    chart=st.session_state.get("ens_chart")
    if chart and chart[:5]==(model,place,parameter,run,horizon):
        st.image(chart[5],use_container_width=True)
        st.download_button("PNG herunterladen",chart[5],file_name=f"ELBTAL_Ensemble_{model.replace(' ','_')}_{run:%Y%m%d%H}.png",mime="image/png")

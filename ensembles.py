"""MeteoOst point ensemble meteograms from official GEFS / ECMWF open GRIB data."""
import io
import os
import tempfile
import time
from collections import Counter
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
HEADERS={"User-Agent":"MeteoOst-Wetterkarten/1.0"}

from model_cycles import recent_cycles

def runs(model):
    return recent_cycles(model)

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
                    elif expected=="precip" and short=="tp":value*=1000
                    elif expected=="snow" and short=="sf":value*=1000
                    return value
                finally:codes_release(gid)
    finally:
        if path and os.path.exists(path):os.unlink(path)
    raise RuntimeError(f"Feld {expected} in GRIB nicht gefunden")

def _noaa_get(url,params):
    """Retry transient NOAA overload; do not cache failed responses."""
    last=None
    for attempt in range(3):
        try:
            response=requests.get(url,params=params,headers=HEADERS,timeout=(8,35))
            if response.status_code in (429,500,502,503,504):
                last=RuntimeError(f"NOAA HTTP {response.status_code}")
                if attempt<2:
                    time.sleep(1.5*(attempt+1))
                    continue
            response.raise_for_status()
            if not response.content.startswith(b"GRIB"):
                preview=response.content[:180].decode("utf-8",errors="replace").replace("\\n"," ")
                last=RuntimeError(f"NOAA lieferte statt GRIB: HTTP {response.status_code}, "
                                  f"{len(response.content)} Bytes, {preview!r}")
                if attempt<2:
                    time.sleep(1.5*(attempt+1))
                    continue
                raise last
            return response
        except requests.RequestException as exc:
            last=exc
            if attempt<2:time.sleep(1.5*(attempt+1))
    raise RuntimeError(f"NOAA-Abruf fehlgeschlagen: {last}")


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
    response=_noaa_get("https://nomads.ncep.noaa.gov/cgi-bin/filter_gefs_atmos_0p50a.pl",params)
    return read_point(response.content,lat,lon,field)

@st.cache_data(ttl=3600,show_spinner=False)
def gfs_main_point(run,step,field,lat,lon):
    """Independent deterministic GFS run, not the GEFS control member."""
    specs={
        "msl":("PRMSL","mean_sea_level"),
        "t850":("TMP","850_mb"),
        "t2":("TMP","2_m_above_ground"),
        "precip":("APCP","surface"),
        "snow":("SNOD","surface"),
    }
    variable,level=specs[field]
    params={"file":f"gfs.t{run:%H}z.pgrb2.0p25.f{step:03d}",
            "dir":f"/gfs.{run:%Y%m%d}/{run:%H}/atmos",
            f"var_{variable}":"on",f"lev_{level}":"on",
            "subregion":"","leftlon":round(lon-1.1,3),"rightlon":round(lon+1.1,3),
            "toplat":round(lat+1.1,3),"bottomlat":round(lat-1.1,3)}
    response=_noaa_get("https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl",params)
    return read_point(response.content,lat,lon,field)


def fetch_gfs_main(run,field,lat,lon,steps):
    """Fetch deterministic GFS in parallel; gaps remain NaN."""
    result=np.full(len(steps),np.nan)
    failures=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(gfs_main_point,run,step,field,lat,lon):i
                 for i,step in enumerate(steps)}
        for future in as_completed(futures):
            try:result[futures[future]]=future.result()
            except Exception as exc:failures.append(str(exc))
    return result,failures


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
                except Exception as exc:
                    failures.append(f"+{steps[j]}h / Mitglied {members[i]:02d}: {exc}")
    else:
        for j,step in enumerate(steps):
            try:
                result=ecmwf_step(run,step,field,lat,lon)
                for i,member in enumerate(members):data[i,j]=result.get(member,np.nan)
            except Exception as exc:failures.append(str(exc))
    return data,failures

def draw(model,place,run,field,steps,data,precip=None,main=None):
    from matplotlib.ticker import MaxNLocator
    from matplotlib.patches import FancyBboxPatch, Arc
    from matplotlib.lines import Line2D
    times=[(run+timedelta(hours=h)).astimezone(LOCAL_TZ) for h in steps]
    plt.rcParams.update({"font.family":"DejaVu Sans"})
    fig=plt.figure(figsize=(12,9),dpi=135,facecolor=NAVY)
    ax=fig.add_axes([.075,.25,.88,.45],facecolor="#0C2946")
    for i,row in enumerate(data):
        # Control is highlighted separately, not duplicated as an ensemble perturbation.
        if model=="GEFS" and i==0:continue
        ax.plot(times,row,color=COLORS[i%len(COLORS)],lw=.9,alpha=.62)
    mean=np.nanmean(data,axis=0)
    ax.plot(times,mean,color="white",lw=3,label="Mittelwert",zorder=6)
    if model=="GEFS" and np.isfinite(data[0]).any():
        ax.plot(times,data[0],color="#2196FF",lw=2.8,label="Kontrolllauf",zorder=8)
    if main is not None and np.isfinite(main).any():
        ax.plot(times,main,color="#41F277",lw=2.8,label="GFS-Hauptlauf",zorder=9)
    ax.grid(color="#6C8BA5",alpha=.3,linestyle="--")
    ax.tick_params(colors="#D9E8F6",labelsize=10)
    for spine in ax.spines.values():spine.set_color("#5CA8D4")
    ylabel={"msl":"Luftdruck (hPa)","t850":"850-hPa-Temperatur (°C)",
            "t2":"2-m-Temperatur (°C)","snow":"Schneehöhenzuwachs (cm)" if model=="GEFS" else "Schneefall-Wasseräquivalent (mm)"}[field]
    ax.set_ylabel(ylabel,color="white",fontsize=11)
    if field in ("t2","t850"):ax.axhline(0,color="#D9E8F6",ls="--",lw=1)
    ax.xaxis.set_major_locator(DayLocator(interval=2))
    ax.xaxis.set_major_formatter(DateFormatter("%d.%m",tz=LOCAL_TZ))
    ax.yaxis.set_major_locator(MaxNLocator(8))
    if precip is not None:
        # NOAA APCP is already an interval total in kg/m² = mm; ECMWF tp is cumulative.
        rain=precip if model=="GEFS" else np.maximum(np.diff(precip,axis=1,prepend=np.nan),0)
        # Filter missing-value sentinels / implausible totals instead of
        # letting invalid precipitation values dominate the axis.
        rain=np.where((rain>=0)&(rain<=300),rain,np.nan)
        rain=np.nanmean(rain,axis=0)
        ax2=fig.add_axes([.075,.245,.88,.105],sharex=ax,facecolor="#0C2946")
        ax2.plot(times,rain,color="#50C8F0",lw=1.9,marker=".",markersize=3)
        ax2.fill_between(times,0,rain,color="#50C8F0",alpha=.13)
        ax2.set_ylabel("NS (mm/6h)",color="#83D8FF",fontsize=9)
        ax2.tick_params(axis="y",colors="#83D8FF",labelsize=8)
        ax2.tick_params(axis="x",labelbottom=True,bottom=True,colors="#D9E8F6",labelsize=9,pad=5)
        ax2.xaxis.set_major_locator(DayLocator(interval=2))
        ax2.xaxis.set_major_formatter(DateFormatter("%d.%m",tz=LOCAL_TZ))
        ax.tick_params(axis="x",labelbottom=False,bottom=False)
        ax2.grid(color="#6C8BA5",alpha=.2,linestyle="--")
        for spine in ax2.spines.values():spine.set_color("#5CA8D4")
        ax2.set_ylim(bottom=0)
        ax.set_position([.075,.365,.88,.335])
    # MeteoOst cloud mark in the same cyan line style as the weather map frames.
    for x,y,w,h,t1,t2 in [(.023,.919,.036,.037,0,180),(.039,.931,.051,.055,0,180),(.075,.918,.037,.039,0,180)]:
        fig.add_artist(Arc((x+w/2,y+h/2),w,h,theta1=t1,theta2=t2,
                           color="#45C8F5",lw=2.4,transform=fig.transFigure))
    fig.add_artist(Line2D([.032,.098],[.92,.92],color="#45C8F5",lw=2.4,transform=fig.transFigure))
    fig.text(.125,.94,"MeteoOST V2",color="white",fontsize=24,fontweight="bold")
    fig.text(.125,.91,"W E T T E R K A R T E N",color="#45C8F5",fontsize=9)
    fig.add_artist(Line2D([.37,.37],[.906,.977],color="#45C8F5",lw=1.5,transform=fig.transFigure))
    title={"msl":"MSLP","t850":"T850","t2":"T2","snow":"Schneehöhenzuwachs" if model=="GEFS" else "Schneefall-Wasseräquivalent"}[field]
    if precip is not None:title="T2 + Niederschlag"
    fig.text(.40,.948,f"Ensemble · {title}",color="white",fontsize=19,fontweight="bold")
    fig.text(.40,.913,f"{model}  ·  {place}",color="#A9C3D9",fontsize=11)
    valid=(run+timedelta(hours=steps[-1])).astimezone(LOCAL_TZ)
    cells=[
        ("MODELLLAUF",run.strftime("%d.%m.%Y · %H:%M UTC")),
        ("GÜLTIG BIS",valid.strftime("%d.%m.%Y · %H:%M %Z")),
        ("PROGNOSE",f"+{steps[-1]} h"),
    ]
    for i,(label,value) in enumerate(cells):
        x=.026+i*.324
        fig.add_artist(FancyBboxPatch((x,.79),.303,.093,
             boxstyle="round,pad=0.008,rounding_size=0.01",
             linewidth=1.3,edgecolor="#287CA6",facecolor="#0C2946",
             transform=fig.transFigure))
        fig.text(x+.015,.855,label,color="#A9C3D9",fontsize=9,fontweight="bold")
        fig.text(x+.015,.815,value,color="white",fontsize=11,fontweight="bold")
    # Full P01-P30 legend, with a separate control and ensemble mean.
    handles=[Line2D([0],[0],color=COLORS[i%len(COLORS)],lw=1.8,
                     label=f"P{i:02d}") for i in range(1,(31 if model=="GEFS" else 51))]
    handles.extend([Line2D([0],[0],color="#2196FF",lw=2.8,label="CONTROL")] if model=="GEFS" else [])
    if main is not None and np.isfinite(main).any():
        handles.append(Line2D([0],[0],color="#41F277",lw=2.8,label="GFS HAUPTLAUF"))
    handles.append(Line2D([0],[0],color="white",lw=3,label="AVG"))
    legend=fig.legend(handles=handles,loc="lower center",
             bbox_to_anchor=(.5,.125),ncol=11 if model=="GEFS" else 13,
             fontsize=7.5,frameon=False,columnspacing=.9,handlelength=1.7)
    for t in legend.get_texts():t.set_color("#D9E8F6")
    fig.add_artist(FancyBboxPatch((.025,.027),.95,.055,
        boxstyle="round,pad=0.007,rounding_size=0.009",
        linewidth=1.2,edgecolor="#287CA6",facecolor="#0C2946",
        transform=fig.transFigure))
    coverage=np.isfinite(data).mean()*100
    fig.text(.045,.046,f"Daten: {'NOAA' if model=='GEFS' else 'ECMWF'}  |  Ort: {place}  |  Abdeckung: {coverage:.0f} %",
             color="#A9C3D9",fontsize=10)
    fig.text(.82,.046,f"MeteoOST V2 | v{VERSION}",color="white",fontsize=11,fontweight="bold")
    out=io.BytesIO();fig.savefig(out,format="png",facecolor=fig.get_facecolor())
    plt.close(fig)
    return out.getvalue()

def _rebrand_archived_png(raw):
    """Update the two legacy branding labels on already published ensemble PNGs.

    The meteorological plot, axes, legend, and data remain untouched.
    """
    from PIL import Image, ImageDraw, ImageFont
    from matplotlib import font_manager
    picture=Image.open(io.BytesIO(raw)).convert("RGB")
    width,height=picture.size
    draw=ImageDraw.Draw(picture)
    font_path=font_manager.findfont("DejaVu Sans")
    title_font=ImageFont.truetype(font_path,max(12,round(width*24/1620)))
    footer_font=ImageFont.truetype(font_path,max(10,round(width*11/1620)))
    # Matplotlib figure positions are bottom-origin; Pillow is top-origin.
    draw.rectangle((int(width*.122),int(height*.025),int(width*.365),int(height*.081)),fill="#071B30")
    draw.text((int(width*.125),int(height*.033)),"MeteoOST V2",font=title_font,fill="white",stroke_width=0)
    draw.rectangle((int(width*.815),int(height*.930),int(width*.965),int(height*.969)),fill="#0C2946")
    draw.text((int(width*.82),int(height*.944)),f"MeteoOST V2 | v{VERSION}",font=footer_font,fill="white")
    out=io.BytesIO()
    picture.save(out,format="PNG")
    return out.getvalue()


def show():
    """Instant display of already published diagrams, without model downloads."""
    from ensemble_archive import available_images
    st.subheader("📊 MeteoOST V2 | v2.0.0 · Ensemble-Diagramme")
    model=st.segmented_control("Ensemble",["GEFS","ECMWF ENS"],default="GEFS",key="ens_model")
    place=st.selectbox("Ort",list(PLACES),key="ens_place")
    parameter=st.selectbox("Parameter",list(FIELDS),key="ens_param")
    available=available_images(model,place,parameter)
    if not available:
        st.info("Für diese Auswahl ist noch kein vorab berechnetes Ensemble veröffentlicht. "
                "Die automatische GitHub-Aktion bereitet neue Diagramme unabhängig von deinem Besuch vor.")
        st.caption("Es wird bewusst kein unvollständiger oder erfundener Modelllauf angezeigt. "
                   "Als erste automatische Auswahl wird GEFS für Mühlberg (Elbe) vorbereitet.")
        return

    archived_runs=list(dict.fromkeys(item[0] for item in available))
    # Switch to a newly published run automatically, even if Streamlit
    # preserved the previously selected (now older) run in session state.
    newest=archived_runs[0]
    if (st.session_state.get("ens_archive_latest_seen") != newest or
            st.session_state.get("ens_archive_run") not in archived_runs):
        st.session_state["ens_archive_run"]=newest
        st.session_state.pop("ens_archive_horizon",None)
    st.session_state["ens_archive_latest_seen"]=newest
    run=st.selectbox("Verfügbare Modellläufe (UTC)",archived_runs,
                     format_func=lambda value:value.strftime("%d.%m.%Y · %H UTC"),
                     key="ens_archive_run")
    horizons=sorted({item[1] for item in available if item[0]==run},reverse=True)
    horizon=st.selectbox("Verfügbare Prognosezeiträume",horizons,
                         format_func=lambda hours:f"+{hours} Stunden",
                         key="ens_archive_horizon")
    result=next((path for cycle,hours,path in available if cycle==run and hours==horizon),None)
    if result is None:
        st.warning("Die gewünschte Archivdatei ist nicht vorhanden.")
        return
    latest=archived_runs[0]
    if run==latest:
        st.caption(f"Neuester veröffentlichter und tatsächlich vorhandener Modelllauf: "
                   f"{run:%d.%m.%Y · %H} UTC · +{horizon} h")
    else:
        st.caption(f"Archivlauf: {run:%d.%m.%Y · %H} UTC · +{horizon} h. "
                   f"Neuester verfügbarer Lauf: {latest:%d.%m.%Y · %H} UTC.")
    branded_png=_rebrand_archived_png(result.read_bytes())
    st.image(branded_png,use_container_width=True)
    st.download_button("Ensemble als PNG herunterladen",branded_png,
                       file_name=result.name,mime="image/png",key="ens_archive_download")
    st.caption("NOAA/ECMWF-Originaldaten · vorab berechnete MeteoOST-V2-Grafik. "
               "Neue Modellläufe erscheinen automatisch nach der Veröffentlichung durch den GitHub-Workflow.")

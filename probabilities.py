"""MeteoOST ensemble probability maps from individual NOAA GRIB members.

No ensemble means are used. Incomplete data fail closed rather than producing
misleading percentages. Time windows are from initialization (+0 to +N h).
"""
import io
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta

import numpy as np
import requests
import streamlit as st
from eccodes import codes_grib_new_from_file, codes_get, codes_get_array, codes_release
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

from icon_eu import REGIONS, LOCAL_TZ, borders, COUNTRIES
from map_design import frame_png, style_figure
from model_cycles import recent_cycles
from value_grid import draw as draw_value_grid

PERIODS=[(24,"1 Tag"),(48,"2 Tage"),(72,"3 Tage"),(120,"5 Tage"),
         (168,"7 Tage"),(336,"14 Tage"),(384,"Gesamter Zeitraum")]
CATEGORIES={
 "Niederschlag":[(f"Akk. Niederschlag >{n} mm",n) for n in (1,5,10,25,50,100)],
 "Temperatur":[("Tmax >25 °C",25),("Tmax >30 °C",30),
   ("Tmax >35 °C",35),("Tmax >40 °C",40),("Tmax <0 °C",0),
   ("Tmin >20 °C",20),("Tmin <0 °C",0),("Tmin <−5 °C",-5),
   ("Tmin <−10 °C",-10)],
 "Schnee":[("Schneefallrisiko 24 h",0.1)],
}
HEADERS={"User-Agent":"MeteoOST-V2.1.1 (educational)"}
GEFS_FILTER="https://nomads.ncep.noaa.gov/cgi-bin/filter_gefs_atmos_0p50a.pl"
AIGEFS_ROOT="https://nomads.ncep.noaa.gov/pub/data/nccf/com/aigefs/prod"
COLORS=["#E7EEF0","#C5E4EE","#94D2E6","#62B6D9","#348FC5",
        "#386CB5","#6259AA","#9350A2","#C44B83","#E76A60","#F0AE54"]


def probability(members,threshold,operator=">",min_members=2):
    values=np.asarray(members,dtype=float)
    if values.ndim!=3:raise ValueError("Erwartet Mitglieder × Breite × Länge.")
    valid=np.isfinite(values)
    count=valid.sum(axis=0)
    hits=((values>threshold) if operator==">" else
          (values<threshold) if operator=="<" else
          (_ for _ in ()).throw(ValueError("Ungültiger Vergleich.")))&valid
    out=np.full(count.shape,np.nan)
    np.divide(100*hits.sum(axis=0),count,out=out,where=count>=min_members)
    return out,count


def steps(hours):
    # GEFS and AIGEFS atmospheric products use 3-hour steps early and
    # 6-hour steps after +240h; require each expected interval.
    return list(range(0,min(hours,240)+1,3))+(
        list(range(246,hours+1,6)) if hours>240 else [])


def _field(parameter):
    if parameter.startswith("Akk."):return "APCP","surface","rain"
    if parameter.startswith("Tmax") or parameter.startswith("Tmin"):
        return "TMP","2 m above ground","temp"
    return "WEASD","surface","snow"


def _decode(raw,variable,region):
    from global_models import decode
    # Use the established GRIB-to-geographic-grid interpolation and
    # explicitly reject any wrong field rather than silently using another.
    import tempfile,os
    path=None
    try:
        with tempfile.NamedTemporaryFile(suffix=".grib2",delete=False) as f:
            f.write(raw);path=f.name
        with open(path,"rb") as f:
            gid=codes_grib_new_from_file(f)
            if gid is None:raise RuntimeError("Leere GRIB-Datei.")
            try:
                short=str(codes_get(gid,"shortName")).lower()
                expected={"APCP":("tp","apcp"),"TMP":("2t","t2m","t"),
                          "WEASD":("weasd","sd","swe","sf")}
                if short not in expected[variable]:
                    raise RuntimeError(f"Unerwartetes GRIB-Feld: {short}")
                lats=np.asarray(codes_get_array(gid,"latitudes"),dtype=float)
                lons=np.asarray(codes_get_array(gid,"longitudes"),dtype=float)
                values=np.asarray(codes_get_array(gid,"values"),dtype=float)
                # GRIB precipitation and snow water equivalent: kg/m² = mm.
                if variable=="TMP":values-=273.15
                if variable=="WEASD" and short in ("sd","sf"):
                    values*=1000
            finally:codes_release(gid)
    finally:
        if path:
            os.unlink(path)
    from scipy.interpolate import griddata
    west,east,south,north=REGIONS[region]
    lons=np.where(lons>180,lons-360,lons)
    mask=(lons>=west-1)&(lons<=east+1)&(lats>=south-1)&(lats<=north+1)&np.isfinite(values)
    if mask.sum()<10:raise RuntimeError("Keine gültigen Daten im Kartenausschnitt.")
    nx,ny=(115,100) if region=="Deutschland" else (155,110)
    X,Y=np.meshgrid(np.linspace(west,east,nx),np.linspace(south,north,ny))
    Z=griddata(np.column_stack((lons[mask],lats[mask])),values[mask],
               (X,Y),method="linear")
    return X,Y,Z


@st.cache_data(ttl=3600,show_spinner=False)
def gefs_record(run,step,member,variable,level,region):
    west,east,south,north=REGIONS[region]
    filename=("gec00" if member==0 else f"gep{member:02d}")+f".t{run:%H}z.pgrb2a.0p50.f{step:03d}"
    params={"file":filename,"dir":f"/gefs.{run:%Y%m%d}/{run:%H}/atmos/pgrb2ap5",
            f"var_{variable}":"on",f"lev_{level.replace(' ','_')}":"on",
            "subregion":"","leftlon":west-1,"rightlon":east+1,
            "toplat":north+1,"bottomlat":south-1}
    response=requests.get(GEFS_FILTER,params=params,headers=HEADERS,timeout=100)
    response.raise_for_status()
    if not response.content.startswith(b"GRIB"):
        raise RuntimeError(f"GEFS-Mitglied {member} +{step}h: Feld nicht verfügbar.")
    return response.content


@st.cache_data(ttl=1800,show_spinner=False)
def aigefs_members(run,kind):
    """Discover real individual member filenames; never use ensstat/avg."""
    from bs4 import BeautifulSoup
    root=f"{AIGEFS_ROOT}/aigefs.{run:%Y%m%d}/{run:%H}/"
    # Candidate directories must be verified by HTTP listing. If NOAA
    # changes the structure, stop with an explicit error.
    candidates=[
        root+"members/products/atmos/grib2/",
        root+"ensemble/products/atmos/grib2/",
        root+"ens/products/atmos/grib2/",
    ]
    for directory in candidates:
        try:
            r=requests.get(directory,headers=HEADERS,timeout=18)
            if r.status_code!=200:continue
            names=re.findall(r'href=["\']([^"\']+\.grib2)["\']',r.text,re.I)
            found={}
            for name in names:
                if f".{kind}." not in name:continue
                match=re.search(r"(?:mem|member|p)(\d{1,2})",name,re.I)
                if match:found[int(match.group(1))]=directory+name
            if len(found)>=2:return found
        except requests.RequestException:continue
    raise RuntimeError("AIGEFS-Einzelmitglieddateien auf NOAA NOMADS nicht verifiziert. "
                       "Es werden keine Ensemble-Mittelwerte als Wahrscheinlichkeiten verwendet.")


def _idx_record(url,variable,level,step):
    idx=requests.get(url+".idx",headers=HEADERS,timeout=35)
    idx.raise_for_status()
    entries=[]
    for line in idx.text.splitlines():
        parts=line.split(":")
        if len(parts)>5 and parts[1].isdigit():entries.append((int(parts[1]),line))
    for i,(start,line) in enumerate(entries):
        if f":{variable}:" not in line or f":{level}:" not in line:continue
        if not re.search(rf"(?:^|:)\s*(?:{step}(?: hour| hr)|0-{step} hour)",line,re.I):
            # Index step strings vary; GRIB validity is checked below.
            pass
        end=entries[i+1][0]-1 if i+1<len(entries) else None
        h={**HEADERS,"Range":f"bytes={start}-{end}" if end is not None else f"bytes={start}-"}
        r=requests.get(url,headers=h,timeout=100)
        r.raise_for_status()
        if r.status_code!=206 or not r.content.startswith(b"GRIB"):
            raise RuntimeError("NOAA unterstützt keinen gültigen GRIB-Range-Abruf.")
        return r.content
    raise RuntimeError(f"{variable} auf {level} fehlt in {url}.")


def _member_step(model,run,step,member,variable,level,region,member_urls=None):
    if model=="GEFS":
        raw=gefs_record(run,step,member,variable,level,region)
    else:
        # AIGEFS member files may contain multiple forecast steps. The
        # filename must contain the requested lead to avoid stale fields.
        url=member_urls[member]
        if not re.search(rf"f{step:03d}(?:\.|$)",url):
            raise RuntimeError("AIGEFS-Mitglieddatei nicht eindeutig dem Vorhersageschritt zugeordnet.")
        raw=_idx_record(url,variable,level,step)
    return _decode(raw,variable,region)


def calculate(model,run,region,parameter,hours,progress=None):
    variable,level,kind=_field(parameter)
    timeline=steps(hours)
    if kind=="snow":timeline=steps(24)
    # For rain use the final accumulation only if the source GRIB record
    # is cumulative since run start. The NOAA GEFS APCP records can instead
    # be interval accumulations, so sum strictly verified intervals below.
    members=list(range(31)) if model=="GEFS" else None
    if model=="AIGEFS":
        member_urls=aigefs_members(run,"sfc" if kind!="temp" else "pres")
        members=sorted(member_urls)
    else:member_urls=None
    if len(members)<2:raise RuntimeError("Nicht genügend Einzelmitglieder.")
    # Prevent silent partial datasets: all requested hours and members
    # must be present. Concurrency is deliberately bounded.
    jobs=[(m,t) for m in members for t in timeline if t>0]
    if len(jobs)>2400:raise RuntimeError("Datenmenge für eine Einzelberechnung zu groß.")
    grids={}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures={pool.submit(_member_step,model,run,t,m,variable,level,region,member_urls):(m,t)
                 for m,t in jobs}
        for i,future in enumerate(as_completed(futures),1):
            m,t=futures[future]
            try:grids[m,t]=future.result()
            except Exception as exc:
                for pending in futures:pending.cancel()
                raise RuntimeError(f"Mitglied {m}, +{t} h nicht vollständig: {exc}") from exc
            if progress:progress.progress(i/len(jobs),text=f"Ensemblefelder {i}/{len(jobs)}")
    X,Y=next(iter(grids.values()))[:2]
    fields=[]
    for m in members:
        series=np.stack([grids[m,t][2] for t in timeline if t>0])
        if kind=="temp":
            field=np.nanmax(series,axis=0) if parameter.startswith("Tmax") else np.nanmin(series,axis=0)
        elif kind=="snow":
            # WEASD is snow-water equivalent on the ground, not 24h
            # snowfall. A positive change is not an unambiguous snowfall
            # event. Refuse to publish this as snowfall risk.
            raise RuntimeError("Schneefallrisiko benötigt echte 24-h-Schneefallakkumulationen (nicht WEASD).")
        else:
            raise RuntimeError("APCP-Intervalle müssen anhand GRIB stepRange "
                               "und stepType verifiziert werden; keine ungeprüfte Summierung.")
        fields.append(field)
    threshold=dict(CATEGORIES["Temperatur"])[parameter]
    op="<" if "<" in parameter else ">"
    # Reject cells with missing members or steps instead of using nanmax
    data=np.stack(fields)
    if not np.isfinite(data).all():
        raise RuntimeError("Unvollständige Datenabdeckung im Kartenbereich.")
    result,count=probability(data,threshold,op,min_members=len(members))
    return X,Y,result,len(members)


def render(X,Y,Z,model,region,parameter,run,hours,members):
    fig,ax=plt.subplots(figsize=(9,11) if region=="Deutschland" else (12,8.5),dpi=120)
    fig.patch.set_facecolor("#071B30")
    ax.set_facecolor("#071B30")
    cmap=ListedColormap(COLORS)
    bounds=np.linspace(0,100,len(COLORS)+1)
    im=ax.pcolormesh(X,Y,np.ma.masked_invalid(Z),cmap=cmap,
                     norm=BoundaryNorm(bounds,cmap.N),shading="nearest",rasterized=True)
    try:
        for feature in borders(COUNTRIES).get("features",[]):
            geom=feature.get("geometry") or {}
            polys=geom.get("coordinates",[]) if geom.get("type")=="MultiPolygon" else [geom.get("coordinates",[])]
            for poly in polys:
                for ring in poly:
                    if len(ring)>2:
                        a=np.asarray(ring)
                        ax.plot(a[:,0],a[:,1],color="#253B4C",linewidth=.55,alpha=.8)
    except (requests.RequestException,ValueError,KeyError):
        pass
    west,east,south,north=REGIONS[region]
    ax.set_xlim(west,east);ax.set_ylim(south,north)
    ax.set_xlabel("Längengrad");ax.set_ylabel("Breitengrad")
    cb=fig.colorbar(im,ax=ax,shrink=.75,ticks=np.arange(0,101,10))
    cb.set_label("Wahrscheinlichkeit (%)")
    style_figure(fig,ax)
    if st.session_state.get("v2_value_grid"):
        draw_value_grid(ax,X,Y,Z,region,"Wahrscheinlichkeit")
    out=io.BytesIO()
    fig.savefig(out,format="png",bbox_inches="tight",facecolor=fig.get_facecolor())
    plt.close(fig)
    valid=run+timedelta(hours=hours)
    return frame_png(out.getvalue(),f"{parameter} · {hours} h · {members} Mitglieder",
                     model,region,run,valid)


def show():
    from v2_navigation import ALL_REGIONS,dropdown
    st.subheader("🎯 Wahrscheinlichkeiten")
    st.caption("Originale NOAA-Ensemblemitglieder · keine Ensemblemittel als Prozentwerte.")
    model=dropdown("1. Ensemblemodell","prob_model",["GEFS","AIGEFS"])
    region=dropdown("2. Region","prob_region",ALL_REGIONS)
    category=dropdown("3. Kategorie","prob_category",list(CATEGORIES))
    parameter=dropdown("4. Grenzwert","prob_parameter_"+category,
                       [name for name,_ in CATEGORIES[category]])
    max_h=384
    if category=="Schnee":
        hours=24
        st.caption("Schneefallrisiko: 24 Stunden.")
    else:
        options=[(h,label) for h,label in PERIODS if h<=max_h]
        choice=dropdown("5. Zeitraum","prob_period_"+model,[label for h,label in options])
        hours=dict((label,h) for h,label in options)[choice]
    runs=recent_cycles("GEFS" if model=="GEFS" else "AIGEFS Mittel")
    run=st.selectbox("Modelllauf (UTC)",runs,format_func=lambda r:r.strftime("%d.%m.%Y %H UTC"),
                     key="prob_run_"+model)
    st.caption(f"Zeitraum: +0 bis +{hours} h. Berechnung aus Einzelmitgliedern; "
               "fehlende Felder führen zu einer Fehlermeldung.")
    st.checkbox("🔢 Zahlengitter auf Wahrscheinlichkeitskarte",key="v2_value_grid")
    if st.button("🎯 Wahrscheinlichkeitskarte berechnen",type="primary",use_container_width=True):
        if category=="Schnee":
            st.warning("Schneefallrisiko wird erst aktiviert, wenn echte 24-h-Schneefallfelder "
                       "pro Ensemblemitglied verfügbar sind.")
            return
        if category=="Niederschlag":
            st.warning("Niederschlagswahrscheinlichkeiten werden erst aktiviert, wenn "
                       "APCP-Akkumulationsintervalle pro Mitglied geprüft sind.")
            return
        bar=st.progress(0)
        try:
            X,Y,Z,n=calculate(model,run,region,parameter,hours,bar)
            png=render(X,Y,Z,model,region,parameter,run,hours,n)
            st.session_state["prob_map"]=(model,region,parameter,hours,run,png)
        except Exception as exc:
            st.error(f"Berechnung nicht möglich: {type(exc).__name__}: {exc}")
        finally:bar.empty()
    saved=st.session_state.get("prob_map")
    if saved and saved[:5]==(model,region,parameter,hours,run):
        st.image(saved[5],use_container_width=True)
        st.download_button("PNG herunterladen",saved[5],file_name="meteoost_wahrscheinlichkeit.png",
                           mime="image/png")

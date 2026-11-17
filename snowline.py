"""GFS wet-bulb-height diagnostic for a potential snowline over Germany.

This is a meteorological estimate, not a precipitation-phase forecast.
It does not imply that precipitation is present or that snow accumulates.
"""
import bz2
import io
import os
import tempfile
from datetime import timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
import streamlit as st
from eccodes import codes_grib_new_from_file, codes_get, codes_get_array, codes_release
from scipy.interpolate import griddata

from icon_eu import REGIONS, LOCAL_TZ, HEADERS, borders, COUNTRIES
from map_design import style_figure, frame_png
from v2_model_runs import run_picker, candidates, probe

REGION="Deutschland"
BOUNDS=REGIONS[REGION]
FILTER="https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
# Obtain every variable in a single geographically filtered GRIB request.
EXPECTED=("t2","td2","ps","orog","t850","rh850","h850","t700","rh700","h700")


def wetbulb(temperature, vapor_pressure, pressure_hpa):
    """Solve the ventilated-psychrometer approximation for T_w in Celsius.

    This relation is approximate in cloud/precipitating air, especially aloft.
    It explicitly accounts for pressure, unlike a fixed dry-bulb lapse rate.
    """
    temperature=np.asarray(temperature,dtype=float)
    vapor_pressure=np.asarray(vapor_pressure,dtype=float)
    pressure_hpa=np.asarray(pressure_hpa,dtype=float)
    lo=np.full_like(temperature,-75.0)
    hi=temperature.copy()
    for _ in range(30):
        mid=(lo+hi)/2
        es=6.112*np.exp(17.62*mid/(243.12+mid))
        gamma=0.00066*(1+0.00115*mid)*pressure_hpa
        f=es-gamma*(temperature-mid)-vapor_pressure
        hi=np.where(f>0,mid,hi)
        lo=np.where(f>0,lo,mid)
    return (hi+lo)/2


def saturation_vapor_pressure(temperature):
    temperature=np.asarray(temperature,dtype=float)
    return 6.112*np.exp(17.62*temperature/(243.12+temperature))


def identify(gid):
    """Map exact GRIB level and variable to required pressure-profile fields."""
    short=str(codes_get(gid,"shortName")).lower()
    level=int(codes_get(gid,"level"))
    kind=str(codes_get(gid,"typeOfLevel")).lower()
    if kind=="heightaboveground" and level==2:
        if short in ("2t","t","tmp"):return "t2"
        if short in ("2d","dpt"):return "td2"
    if kind=="surface":
        if short in ("sp","pres"):return "ps"
        if short in ("gh","hgt","orog"):return "orog"
    if kind in ("isobaricinhpa","isobaricinpa"):
        if kind=="isobaricinpa":level=level//100
        if level in (850,700):
            if short in ("t","tmp"):return f"t{level}"
            if short in ("r","rh"):return f"rh{level}"
            if short in ("gh","hgt"):return f"h{level}"
    return None


@st.cache_data(ttl=1800,show_spinner=False)
def gfs_profile(run,lead):
    params={
        "file":f"gfs.t{run:%H}z.pgrb2.0p25.f{lead:03d}",
        "dir":f"/gfs.{run:%Y%m%d}/{run:%H}/atmos",
        "subregion":"",
        "leftlon":str(BOUNDS[0]-.5),"rightlon":str(BOUNDS[1]+.5),
        "bottomlat":str(BOUNDS[2]-.5),"toplat":str(BOUNDS[3]+.5),
        "var_TMP":"on","var_DPT":"on","var_RH":"on",
        "var_HGT":"on","var_PRES":"on",
        "lev_2_m_above_ground":"on","lev_surface":"on",
        "lev_850_mb":"on","lev_700_mb":"on",
    }
    r=requests.get(FILTER,params=params,headers=HEADERS,timeout=(10,100))
    r.raise_for_status()
    if not r.content.startswith(b"GRIB"):
        raise RuntimeError("NOAA hat für diesen GFS-Lauf noch keine passenden GRIB-Daten bereitgestellt.")
    path=None
    fields={}
    try:
        with tempfile.NamedTemporaryFile(suffix=".grib2",delete=False) as tmp:
            tmp.write(r.content)
            path=tmp.name
        with open(path,"rb") as stream:
            while True:
                gid=codes_grib_new_from_file(stream)
                if gid is None:break
                try:
                    key=identify(gid)
                    if key is None or key in fields:continue
                    lat=np.asarray(codes_get_array(gid,"latitudes"),dtype=float)
                    lon=np.asarray(codes_get_array(gid,"longitudes"),dtype=float)
                    values=np.asarray(codes_get_array(gid,"values"),dtype=float)
                    lon=np.where(lon>180,lon-360,lon)
                    mask=(lon>=BOUNDS[0]-.5)&(lon<=BOUNDS[1]+.5)&(lat>=BOUNDS[2]-.5)&(lat<=BOUNDS[3]+.5)&np.isfinite(values)
                    if mask.sum()<100:continue
                    fields[key]=(lat[mask],lon[mask],values[mask])
                finally:
                    codes_release(gid)
    finally:
        if path is not None:os.unlink(path)
    missing=set(EXPECTED)-set(fields)
    if missing:
        raise RuntimeError("Für die Schneefallgrenze fehlen GFS-Profilfelder: "+", ".join(sorted(missing)))
    nx,ny=225,180
    X,Y=np.meshgrid(np.linspace(BOUNDS[0],BOUNDS[1],nx),
                    np.linspace(BOUNDS[2],BOUNDS[3],ny))
    values={}
    for name,(lat,lon,data) in fields.items():
        grid=griddata(np.column_stack((lon,lat)),data,(X,Y),method="linear")
        if not np.isfinite(grid).all():
            near=griddata(np.column_stack((lon,lat)),data,(X,Y),method="nearest")
            grid=np.where(np.isfinite(grid),grid,near)
        values[name]=grid
    return X,Y,values


def potential_snowline(fields):
    """Find the first upward crossing of 1°C wet bulb, in metres above MSL.

    Crossing below terrain is reported as reaching the local ground.
    A warm 700-hPa layer remains unclassified rather than extrapolated.
    """
    t2=fields["t2"]-273.15
    td2=fields["td2"]-273.15
    t850=fields["t850"]-273.15
    t700=fields["t700"]-273.15
    rh850=np.clip(fields["rh850"]/100,0,1)
    rh700=np.clip(fields["rh700"]/100,0,1)
    surface_p=fields["ps"]/100
    z0=fields["orog"]
    z850=fields["h850"]
    z700=fields["h700"]
    if not 700<float(np.nanmedian(surface_p))<1100:
        raise RuntimeError("GFS-Oberflächendruck besitzt keine plausiblen Einheiten.")
    if not 900<float(np.nanmedian(z850))<2200 or not 2300<float(np.nanmedian(z700))<4000:
        raise RuntimeError("GFS-Druckflächenhöhen besitzen keine plausiblen Einheiten.")
    ws=wetbulb(t2,saturation_vapor_pressure(td2),surface_p)
    w850=wetbulb(t850,saturation_vapor_pressure(t850)*rh850,850)
    w700=wetbulb(t700,saturation_vapor_pressure(t700)*rh700,700)
    threshold=1.0
    # A pressure surface below the terrain must not be used for interpolation.
    lower_ok=(z850>z0+20)&(ws>threshold)&(w850<=threshold)
    high_ok=(z700>np.maximum(z850,z0)+20)&(w850>threshold)&(w700<=threshold)
    low_cross=z0+(z850-z0)*(ws-threshold)/np.maximum(ws-w850,0.01)
    high_cross=z850+(z700-z850)*(w850-threshold)/np.maximum(w850-w700,0.01)
    result=np.full(t2.shape,np.nan)
    result=np.where(ws<=threshold,z0,result)
    result=np.where(lower_ok,low_cross,result)
    result=np.where(high_ok,high_cross,result)
    # If 850 hPa is underground, interpolate ground directly to 700 hPa.
    mountain=(z850<=z0+20)&(ws>threshold)&(w700<=threshold)&(z700>z0+20)
    mountain_cross=z0+(z700-z0)*(ws-threshold)/np.maximum(ws-w700,0.01)
    result=np.where(mountain,mountain_cross,result)
    result=np.where(np.isfinite(result),np.clip(result,0,3500),np.nan)
    return result


def render(X,Y,snowline,run,valid):
    from matplotlib.colors import BoundaryNorm,ListedColormap
    from matplotlib.patches import Patch
    levels=[0,100,200,300,400,600,800,1000,1250,1500,1800,2200,2600,3000,3500]
    colors=["#E9F8FF","#CEEFFF","#A3DDF8","#75C6EE","#43A9DF",
            "#2588CE","#2666B3","#384F9B","#59439C","#8250A9",
            "#B865AD","#D886A1","#E6AE80","#F1D575"]
    cmap=ListedColormap(colors,name="elbtal_snowline")
    cmap.set_bad("#687789")
    norm=BoundaryNorm(levels,cmap.N)
    fig,ax=plt.subplots(figsize=(10,10),dpi=120)
    im=ax.pcolormesh(X,Y,np.ma.masked_invalid(snowline),
                     cmap=cmap,norm=norm,shading="auto",rasterized=True)
    for feature in borders(COUNTRIES).get("features",[]):
        geom=feature.get("geometry",{})
        polygons=(geom.get("coordinates",[]) if geom.get("type")=="MultiPolygon"
                  else [geom.get("coordinates",[])] if geom.get("type")=="Polygon" else [])
        for polygon in polygons:
            for ring in polygon:
                if len(ring)>2:
                    border=np.asarray(ring,dtype=float)
                    ax.plot(border[:,0],border[:,1],color="#1B2F44",linewidth=.55,alpha=.8,zorder=4)
    ax.set_xlim(BOUNDS[0],BOUNDS[1]);ax.set_ylim(BOUNDS[2],BOUNDS[3])
    ax.set_xlabel("Längengrad");ax.set_ylabel("Breitengrad")
    bar=fig.colorbar(im,ax=ax,shrink=.78,ticks=[0,200,400,800,1200,1800,2600,3500])
    bar.set_label("Potenzielles Schneefallgrenzen-Niveau (m ü. NHN)")
    ax.set_title("")
    style_figure(fig,ax)
    bar.outline.set_edgecolor("#B5C8DC")
    out=io.BytesIO()
    fig.savefig(out,format="png",bbox_inches="tight")
    plt.close(fig)
    return frame_png(out.getvalue(),"Schneefallgrenze (Schätzung)","GFS",REGION,run,valid,source="NOAA GFS")


def show(region="Deutschland"):
    if region!=REGION:
        st.warning("Die Schneefallgrenzen-Schätzung ist zunächst nur für Deutschland verfügbar.")
        return
    st.subheader("❄️ GFS · Potenzielle Schneefallgrenze")
    st.caption("Modellbasierte Näherung: Höhe des 1-°C-Nasskugelniveaus aus GFS-Temperatur, "
               "Feuchte und Druckflächenhöhen (Boden, 850 und 700 hPa). "
               "Kein gemessener Wert und keine garantierte Grenze zwischen Regen und Schnee.")
    run_choice=run_picker("GFS","snowline_gfs_run",allow_auto=True)
    runs=candidates("GFS")
    reference=run_choice or next((r for r in runs if probe("GFS",r) is True),runs[0])
    lead=st.select_slider("Prognosestunde",options=list(range(0,241,3)),
                          value=12,
                          format_func=lambda h:f"+{h} h · {(reference+timedelta(hours=h)).astimezone(LOCAL_TZ):%d.%m.%Y %H:%M %Z}",
                          key="snowline_gfs_lead")
    if st.button("Schneefallgrenze berechnen",type="primary",key="snowline_generate"):
        choices=[run_choice] if run_choice else [r for r in runs if probe("GFS",r) is True] or runs
        failures=[]
        for run in choices:
            try:
                with st.spinner("Originale NOAA-GFS-Profilfelder laden und Schneefallgrenze schätzen …"):
                    X,Y,fields=gfs_profile(run,lead)
                    result=potential_snowline(fields)
                    if np.isfinite(result).mean()<.1:
                        raise RuntimeError("Kein 1-°C-Nasskugelniveau innerhalb der betrachteten Luftsäule.")
                    valid=run+timedelta(hours=lead)
                    png=render(X,Y,result,run,valid)
                st.session_state["snowline_map"]=(run,lead,png,valid)
                break
            except Exception as exc:
                failures.append(f"{run:%d.%m. %H UTC}: {type(exc).__name__}: {exc}")
        else:
            st.error("Keine vollständigen GFS-Profilfelder verfügbar. "+"; ".join(failures[:2]))
    result=st.session_state.get("snowline_map")
    if result and result[1]==lead and (run_choice is None or result[0]==run_choice):
        run,hours,png,valid=result
        st.caption(f"GFS-Lauf {run:%d.%m.%Y %H} UTC · gültig {valid.astimezone(LOCAL_TZ):%d.%m.%Y %H:%M %Z}")
        st.image(png,use_container_width=True)
        st.download_button("PNG herunterladen",png,
                           file_name=f"ELBTAL_GFS_Schneefallgrenze_{run:%Y%m%d%H}_{hours:03d}.png",
                           mime="image/png",key="snowline_download")
    st.caption("Nur ein thermodynamischer Indikator: Wo kein Niederschlag fällt, gibt es keinen Schneefall. "
               "Lokale Schmelzschichten, Niederschlagsintensität und bodennahe Effekte "
               "können die tatsächliche Schnee-Regen-Grenze deutlich verschieben. "
               "Grau: kein entsprechendes Niveau bis 700 hPa aufgelöst.")

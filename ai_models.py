"""NOAA AIGFS deterministic and AIGEFS ensemble-mean weather maps.
Official NOMADS GRIB2/IDX range downloads; never Open-Meteo.
"""
import io
import os
import re
import tempfile
from datetime import timedelta
import numpy as np
import requests
import streamlit as st
from eccodes import codes_grib_new_from_file, codes_get, codes_get_array, codes_release
from icon_eu import REGIONS, LOCAL_TZ, render
from global_models import decode, climatology_on_grid, anomaly_map
from model_cycles import recent_cycles

ROOT = "https://nomads.ncep.noaa.gov/pub/data/nccf/com"
HEADERS = {"User-Agent": "ELBTAL-Wetterkarten/1.20 (educational)"}
FIELDS = {
    "MSLP": ("sfc", "PRMSL", "mean sea level", "msl"),
    "T850": ("pres", "TMP", "850 mb", "t"),
    "T2": ("sfc", "TMP", "2 m above ground", "2t"),
    "Z500 + MSLP": ("pres", "HGT", "500 mb", "z"),
    "T850 + MSLP": ("pres", "TMP", "850 mb", "t"),
    "Niederschlagssumme": ("sfc", "APCP", "surface", "prec"),
}
def file_url(model, run, lead, kind):
    cycle=f"{run:%H}"
    day=f"{run:%Y%m%d}"
    if model=="AIGEFS Mittel":
        return (f"{ROOT}/aigefs/prod/aigefs.{day}/{cycle}/ensstat/products/atmos/grib2/"
                f"aigefs.t{cycle}z.{kind}.avg.f{lead:03d}.grib2")
    return (f"{ROOT}/aigfs/prod/aigfs.{day}/{cycle}/model/atmos/grib2/"
            f"aigfs.t{cycle}z.{kind}.f{lead:03d}.grib2")

@st.cache_data(ttl=1800,show_spinner=False)
def get_record(model,run,lead,kind,variable,level):
    url=file_url(model,run,lead,kind)
    response=requests.get(url+".idx",headers=HEADERS,timeout=35)
    response.raise_for_status()
    records=[]
    for line in response.text.splitlines():
        parts=line.split(":")
        if len(parts)<6:continue
        try:offset=int(parts[1])
        except ValueError:continue
        records.append((offset,line))
    selected=None
    for i,(offset,line) in enumerate(records):
        # Match the exact GRIB field and vertical level, not a similarly named field.
        if re.search(r":"+re.escape(variable)+r":",line) and re.search(r":"+re.escape(level)+r":",line,re.I):
            end=records[i+1][0]-1 if i+1<len(records) else None
            selected=(offset,end)
            break
    if selected is None:
        raise RuntimeError(f"{variable} auf {level} ist in der NOAA-Indexdatei nicht vorhanden.")
    start,end=selected
    headers={**HEADERS,"Range":f"bytes={start}-{end}" if end is not None else f"bytes={start}-"}
    result=requests.get(url,headers=headers,timeout=100)
    result.raise_for_status()
    if result.status_code!=206:
        raise RuntimeError("NOAA hat den GRIB-Bytebereich nicht separat ausgeliefert.")
    if not result.content.startswith(b"GRIB"):
        raise RuntimeError("NOAA lieferte keinen gültigen GRIB-Datensatz.")
    return result.content

def field(model,run,lead,parameter,region):
    kind,var,level,code=FIELDS[parameter]
    raw=get_record(model,run,lead,kind,var,level)
    if code=="prec":
        # Precipitation is an accumulation in mm; GRIB field is already mm.
        code="apcp"
    return decode(raw,run+timedelta(hours=lead),REGIONS[region],code)

def show(region,forced_model=None,forced_parameter=None):
    if forced_model is None:
        st.subheader(f"NOAA KI-Wettermodelle · {region}")
    model=forced_model or st.segmented_control("Modell",["AIGFS","AIGEFS Mittel"],default="AIGFS",key="ai_model")
    st.caption("Originaldaten NOAA NOMADS · AIGEFS Mittel = räumliche Karten des Ensemblemittels, keine Diagramme. Kein Open-Meteo.")
    runs=recent_cycles(model,count=9)
    run=st.selectbox("Modelllauf (UTC)",runs,format_func=lambda x:f"{x:%d.%m.%Y %H} UTC",key="ai_run_"+model)
    lead=st.select_slider("Vorhersagestunde",options=list(range(0,385,6)),value=24,
        format_func=lambda h:f"+{h} h · {(run+timedelta(hours=h)).astimezone(LOCAL_TZ):%d.%m.%Y %H:%M %Z}",
        key="ai_lead_"+model)
    parameters=["MSLP","T850","T2"] if model=="AIGEFS Mittel" else [
        "MSLP","T850","T850 + MSLP","T2","Z500 + MSLP","MSLP-Anomalie","Niederschlagssumme"]
    parameter=forced_parameter if forced_parameter in parameters else st.radio("Parameter",parameters,horizontal=True,key="ai_param_"+model)
    if st.button("KI-Wetterkarte berechnen",type="primary",key="ai_generate_"+model):
        try:
            with st.spinner("NOAA-Originaldaten laden und Wetterkarte erstellen …"):
                target="MSLP" if parameter=="MSLP-Anomalie" else parameter
                X,Y,Z=field(model,run,lead,target,region)
                valid=run+timedelta(hours=lead)
                if parameter=="MSLP-Anomalie":
                    baseline=climatology_on_grid(X,Y,valid.month)
                    png=anomaly_map(X,Y,Z,baseline,region,model,run,valid)
                else:
                    pressure=field(model,run,lead,"MSLP",region)[2] if parameter in ("T850 + MSLP","Z500 + MSLP") else None
                    name="2-m-Temperatur" if parameter=="T2" else parameter
                    png=render(X,Y,Z,region,name,run,valid,model=model,mslp=pressure)
                st.session_state["ai_map"]=(model,region,run,lead,parameter,valid,png)
        except Exception as exc:
            st.error(f"NOAA-Abruf fehlgeschlagen: {type(exc).__name__}: {exc}")
    result=st.session_state.get("ai_map")
    if result and result[:5]==(model,region,run,lead,parameter):
        st.image(result[6],use_container_width=True)
        st.download_button("PNG herunterladen",result[6],file_name=f"ELBTAL_{model.replace(' ','_')}_{parameter}_{result[5]:%Y%m%d%H}.png",mime="image/png")

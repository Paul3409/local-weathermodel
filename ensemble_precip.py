"""Accumulated precipitation from NOAA ensemble-mean APCP GRIB records.

Only actual ensemble-mean products (GEFS geavg / AIGEFS ensstat avg).
GRIB startStep/endStep are verified; no assumption that APCP is cumulative.
"""
from datetime import timedelta
import io
import tempfile
import os
import numpy as np
import streamlit as st
from eccodes import codes_grib_new_from_file, codes_get, codes_release
from icon_eu import REGIONS
from global_models import decode, gefs_bytes, HEADERS
from ai_models import get_record

@st.cache_data(ttl=1800,show_spinner=False)
def mean_apcp_record(model,run,lead,region):
    if model=="GEFS Mittel":
        import requests
        west,east,south,north=REGIONS[region]
        params={"file":f"geavg.t{run:%H}z.pgrb2a.0p50.f{lead:03d}",
                "dir":f"/gefs.{run:%Y%m%d}/{run:%H}/atmos/pgrb2ap5",
                "var_APCP":"on","lev_surface":"on","subregion":"",
                "leftlon":west-1,"rightlon":east+1,
                "toplat":north+1,"bottomlat":south-1}
        r=requests.get("https://nomads.ncep.noaa.gov/cgi-bin/filter_gefs_atmos_0p50a.pl",
                       params=params,headers=HEADERS,timeout=110)
        r.raise_for_status()
        raw=r.content
    else:
        raw=get_record("AIGEFS Mittel",run,lead,"sfc","APCP","surface")
    if not raw.startswith(b"GRIB"):
        raise RuntimeError(f"Kein gültiges APCP-Mittelwertfeld bei +{lead} h.")
    return raw

def interval(raw,lead):
    path=None
    try:
        with tempfile.NamedTemporaryFile(suffix=".grib2",delete=False) as f:
            f.write(raw)
            path=f.name
        with open(path,"rb") as f:
            gid=codes_grib_new_from_file(f)
            if gid is None:raise RuntimeError("GRIB-Datensatz leer.")
            try:
                short=str(codes_get(gid,"shortName")).lower()
                kind=str(codes_get(gid,"stepType")).lower()
                start=int(codes_get(gid,"startStep"))
                end=int(codes_get(gid,"endStep"))
                if short not in ("tp","apcp") or kind!="accum" or not (0<=start<end==lead):
                    raise RuntimeError(f"APCP-Intervall ungültig: {short}, {kind}, +{start} bis +{end} h.")
                return start,end
            finally:codes_release(gid)
    finally:
        if path and os.path.exists(path):os.unlink(path)

def _leads(model,lead):
    if model=="GEFS Mittel":
        return [h for h in range(3,min(lead,240)+1,3)]+(
            list(range(246,lead+1,6)) if lead>240 else [])
    return list(range(6,lead+1,6))

@st.cache_data(ttl=1800,show_spinner=False)
def accumulated_mean(model,run,lead,region):
    if lead==0:
        # A genuine zero accumulation at model initialization.
        from global_models import get_field
        X,Y,ref=get_field(model,run,0,"msl",REGIONS[region])
        return X,Y,np.zeros_like(ref)
    candidates={}
    errors=[]
    for h in _leads(model,lead):
        try:
            raw=mean_apcp_record(model,run,h,region)
            start,end=interval(raw,h)
            candidates.setdefault(end,[]).append((start,raw))
        except Exception as exc:
            errors.append(f"+{h}h: {exc}")
    # Find a continuous partition ending at the requested forecast lead.
    # Prefer the shortest interval at each endpoint to avoid double counting.
    chosen=[]
    cursor=lead
    while cursor>0:
        options=candidates.get(cursor,[])
        if not options:
            raise RuntimeError(
                f"Keine lückenlose APCP-Akkumulation bis +{lead} h: "
                f"Intervall vor +{cursor} h fehlt. "
                f"Erster Abruffehler: {errors[0] if errors else 'keiner'}")
        start,raw=max(options,key=lambda item:item[0])
        chosen.append((cursor,raw))
        cursor=start
    X=Y=total=None
    for h,raw in chosen:
        x,y,z=decode(raw,run+timedelta(hours=h),REGIONS[region],"apcp")
        if total is None:
            X,Y=x,y
            total=np.zeros_like(z)
        if z.shape!=total.shape or not np.isfinite(z).all():
            raise RuntimeError(f"APCP-Gitter bei +{h} h unvollständig.")
        total+=z
    if np.nanmin(total)<-0.01:
        raise RuntimeError("Negative akkumulierte Niederschlagsmenge.")
    return X,Y,np.maximum(total,0)

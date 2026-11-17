"""ELBTAL V2: conservative, cached model-run availability checks.

A positive result verifies a published source artifact, not that every lead
and parameter is complete. Never confuse a nominal cycle with verified data.
"""
from datetime import datetime, timedelta, timezone
import requests
import streamlit as st
from model_cycles import recent_cycles

HEADERS={"User-Agent":"ELBTAL-Wetterkarten/2.0"}
TIMEOUT=7

def candidates(model):
    if model=="ICON-D2-RUC":
        cutoff=datetime.now(timezone.utc)-timedelta(minutes=55)
        latest=cutoff.replace(minute=0,second=0,microsecond=0)
        return [latest-timedelta(hours=i) for i in range(49)]
    return recent_cycles(model,count=9)

@st.cache_data(ttl=900,show_spinner=False)
def probe(model,run):
    """Return True/False/None. None means unverified, not unavailable."""
    try:
        if model in ("ICON-EU","ICON-D2"):
            from icon_eu import discover, discover_d2
            fn=discover_d2 if model=="ICON-D2" else discover
            return bool(fn(run,"pmsl"))
        if model in ("AIGFS","AIGEFS Mittel"):
            from ai_models import file_url
            # IDX is a small, real NOAA product index; no GRIB downloads.
            for lead in (6,12,24):
                url=file_url(model,run,lead,"sfc")+".idx"
                resp=requests.get(url,headers=HEADERS,timeout=TIMEOUT)
                if resp.status_code==200 and ":PRMSL:" in resp.text:
                    return True
            return False
        if model=="GFS":
            url=(f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/"
                 f"gfs.{run:%Y%m%d}/{run:%H}/atmos/gfs.t{run:%H}z.pgrb2.0p25.f006.idx")
            resp=requests.get(url,headers=HEADERS,timeout=TIMEOUT)
            return resp.status_code==200 and ":PRMSL:" in resp.text
        if model=="GEFS Mittel":
            url=(f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/gens/prod/"
                 f"gefs.{run:%Y%m%d}/{run:%H}/atmos/pgrb2ap5/"
                 f"geavg.t{run:%H}z.pgrb2a.0p50.f006.idx")
            resp=requests.get(url,headers=HEADERS,timeout=TIMEOUT)
            return resp.status_code==200 and ":PRMSL:" in resp.text
        if model=="ICON-D2-RUC":
            from icon_d2_ruc import url
            response=requests.head(url(run,6,"PMSL"),headers=HEADERS,timeout=TIMEOUT,allow_redirects=True)
            return response.status_code==200
        # ECMWF IFS uses a dedicated Open Data client. Do not claim
        # availability based on an unrelated endpoint or nominal cycle.
        return None
    except (requests.RequestException,RuntimeError,ValueError,KeyError):
        return None

def run_picker(model,key,allow_auto=False):
    """Only verified runs are selectable when a probe succeeds.

    If source checks are inconclusive, keep cycles accessible but label
    them unverified rather than falsely declaring them available.
    """
    all_runs=candidates(model)
    with st.spinner("Modellläufe prüfen …"):
        checks=[probe(model,run) for run in all_runs]
    verified=[run for run,ok in zip(all_runs,checks) if ok is True]
    # Never hide older candidates: DWD/NOAA may remove an index while
    # individual archived forecast files remain retrievable.
    runs=all_runs
    if verified:
        st.caption(f"{len(verified)} von {len(all_runs)} Läufen durch Quelldaten bestätigt. Alle Laufkandidaten bleiben auswählbar; einzelne Prognoseschritte werden beim Abruf geprüft.")
    else:
        st.caption("Keine Laufbestätigung durch die Vorprüfung. Alle älteren UTC-Zyklen bleiben auswählbar; Verfügbarkeit wird beim Kartenabruf geprüft.")
    options=([None]+runs) if allow_auto else runs
    def label(run):
        if run is None:
            return "Automatisch · jüngster abrufbarer Lauf"
        suffix=" ✓ bestätigt" if run in verified else " · nicht vorgeprüft"
        return f"{run:%d.%m.%Y %H} UTC"+suffix
    return st.selectbox("Modelllauf auswählen (UTC)",options,
                        format_func=label,key=key)

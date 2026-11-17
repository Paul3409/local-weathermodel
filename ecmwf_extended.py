"""ECMWF extended-range weekly anomaly charts via official OpenCharts API.

These are original ECMWF forecast graphics (not locally calculated anomaly
grids). No Open-Meteo, fabricated values or unlicensed data sources.
"""
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse
import requests
import streamlit as st

API="https://charts.ecmwf.int/opencharts-api/v1/"
HEADERS={"User-Agent":"ELBTAL-Wetterkarten/2.0"}
PARAMETERS={
    "Temperaturanomalie (2 m)":("temperature","2m","2-m","2 metre","2 metre","2mt"),
    "Niederschlagsanomalie":("precipitation","rainfall","precip"),
    "Luftdruckanomalie (MSLP)":("mslp","mean sea level pressure","pressure"),
}
WEEK_STEPS=[168,336,504,672,840,1008]

def _request_json(url,params=None):
    r=requests.get(url,params=params,headers=HEADERS,timeout=(6,22))
    r.raise_for_status()
    result=r.json()
    if isinstance(result,dict) and result.get("error"):
        raise RuntimeError(str(result["error"]))
    return result

@st.cache_data(ttl=3600,show_spinner=False)
def catalogue():
    result=_request_json(API+"products/")
    if isinstance(result,dict):
        result=result.get("products",result.get("results",[]))
    if not isinstance(result,list):
        raise RuntimeError("ECMWF hat einen unerwarteten Produktkatalog zurückgegeben.")
    return result

def _name(entry):
    if not isinstance(entry,dict):return ""
    return str(entry.get("name") or entry.get("id") or entry.get("product") or "")

def _description(entry):
    if not isinstance(entry,dict):return ""
    return " ".join(str(entry.get(k,"")) for k in ("name","title","description","label")).lower()

def discover_products(parameter):
    """Show only actual ECMWF catalog products; never invent product IDs."""
    keywords=PARAMETERS[parameter]
    found=[]
    for item in catalogue():
        name=_name(item)
        desc=_description(item)
        if not name or not any(word in desc for word in keywords):
            continue
        # Extended-range weekly anomalies only. Prevent accidentally
        # presenting a daily or medium-range chart as an EC46 forecast.
        if not any(word in desc for word in ("extended","weekly","week","46d","46-day","subseason","sub-season","monthly")):
            continue
        if not any(word in desc for word in ("anomal","departure","deviation")):
            continue
        found.append((name,str(item.get("title") or item.get("description") or name)))
    return found

@st.cache_data(ttl=1800,show_spinner=False)
def chart_metadata(product,step,projection):
    params={"step":step}
    if projection:
        params["projection"]=projection
    data=_request_json(API+f"products/{product}/",params=params)
    if not isinstance(data,dict):
        raise RuntimeError("Ungültige Antwort der ECMWF-Kartenschnittstelle.")
    record=data.get("data") or {}
    url=(record.get("link") or {}).get("href")
    attributes=record.get("attributes") or {}
    if not url or urlparse(url).hostname not in ("charts.ecmwf.int","www.ecmwf.int"):
        raise RuntimeError("ECMWF hat keine gültige Kartenadresse geliefert.")
    return url,str(attributes.get("description") or ""),str(attributes.get("title") or product)

@st.cache_data(ttl=1800,show_spinner=False)
def download_chart(url):
    if urlparse(url).hostname not in ("charts.ecmwf.int","www.ecmwf.int"):
        raise ValueError("Unerwarteter Kartenserver.")
    r=requests.get(url,headers=HEADERS,timeout=(6,35))
    r.raise_for_status()
    if not r.content.startswith(b"\x89PNG\r\n\x1a\n"):
        raise RuntimeError("Der ECMWF-Server hat kein PNG-Bild geliefert.")
    if len(r.content)>20_000_000:
        raise RuntimeError("Die Bilddatei ist unerwartet groß.")
    return r.content

def show():
    from v2_navigation import dropdown
    st.subheader("🗓️ ECMWF-Langfristkarten")
    st.caption("Offizielle ECMWF-OpenCharts · Wochenanomalien bis zu 46 Tage · keine künstlich berechneten Daten.")
    parameter=dropdown("1. Parameter","v2_ec46_parameter",list(PARAMETERS))
    try:
        with st.spinner("Offizielle ECMWF-Kartenprodukte suchen …"):
            products=discover_products(parameter)
    except (requests.RequestException,ValueError,RuntimeError) as exc:
        st.warning(f"Der ECMWF-Produktkatalog ist gerade nicht erreichbar: {exc}")
        st.link_button("Offizielle ECMWF-Karten öffnen","https://charts.ecmwf.int/")
        return
    if not products:
        st.info("Für diesen Parameter wurde im öffentlichen ECMWF-Katalog kein eindeutig zuordenbares Wochenanomalie-Produkt gefunden. Es werden keine Ersatzdaten oder erfundenen Karten angezeigt.")
        st.link_button("ECMWF-Langfristkarten ansehen","https://charts.ecmwf.int/")
        return
    names=[name for name,_ in products]
    labels=dict(products)
    product=dropdown("2. ECMWF-Kartenprodukt","v2_ec46_product",names)
    st.caption(labels[product])
    week=dropdown("3. Prognosewoche","v2_ec46_week",
                  [f"Woche {i} · Tage {7*i-6}–{7*i}" for i in range(1,7)])
    index=int(week.split()[1])
    # ECMWF API accepts 'step' as a forecast lead. The response metadata
    # is shown so users can verify the actual period provided by ECMWF.
    step=WEEK_STEPS[index-1]
    projection=dropdown("4. Kartengebiet","v2_ec46_area",
                        ["Europa","Mitteleuropa"])
    projection_id={"Europa":"opencharts_europe","Mitteleuropa":"opencharts_central_europe"}[projection]
    if st.button("ECMWF-Wochenkarte laden",type="primary",use_container_width=True):
        try:
            with st.spinner("ECMWF-Wochenkarte laden …"):
                url,description,title=chart_metadata(product,step,projection_id)
                png=download_chart(url)
            st.session_state["v2_ec46_result"]=(product,index,projection,png,description,title)
        except (requests.RequestException,RuntimeError,ValueError) as exc:
            st.error(f"Die ausgewählte ECMWF-Wochenkarte ist nicht verfügbar: {exc}")
    result=st.session_state.get("v2_ec46_result")
    if result and result[:3]==(product,index,projection):
        _,_,_,png,description,title=result
        st.image(png,caption=f"ECMWF · {title}",use_container_width=True)
        st.download_button("ECMWF-Karte als PNG herunterladen",png,
                           file_name=f"ELBTAL_ECMWF_EC46_Woche_{index}.png",
                           mime="image/png")
        if description:st.caption("Originalangaben ECMWF: "+description)
        st.caption("Quelle: ECMWF OpenCharts (CC BY 4.0). Grafik und Farbskala stammen unverändert von ECMWF.")
    with st.expander("ℹ️ Hinweise zu Langfristprognosen",expanded=False):
        st.write("Wochenanomalien vergleichen die prognostizierten Wochenmittel mit der ECMWF-Modellklimatologie. Die Prognoseunsicherheit nimmt mit dem Vorhersagezeitraum zu.")
        st.write("Die öffentlichen ECMWF-Karten sind fertige Originalgrafiken. Eigene ELBTAL-Rohdatenkarten mit frei gestaltbarer Skala benötigen einen geeigneten Zugang zu den zugrunde liegenden Rasterdaten.")
        st.write("Einzelne Wochen und Regionen können bei ECMWF fehlen. Die tatsächlich gelieferte Periode steht in den Originalangaben unter der Karte.")

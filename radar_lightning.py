"""Optional Xweather Lightning Enterprise overlay; credentials stay server-side."""
from datetime import timedelta
import os
import requests
import streamlit as st

def credentials():
    try:
        return (os.getenv("XWEATHER_CLIENT_ID") or st.secrets.get("XWEATHER_CLIENT_ID"),
                os.getenv("XWEATHER_CLIENT_SECRET") or st.secrets.get("XWEATHER_CLIENT_SECRET"))
    except Exception:
        return os.getenv("XWEATHER_CLIENT_ID"), os.getenv("XWEATHER_CLIENT_SECRET")

@st.cache_data(ttl=60,show_spinner=False,max_entries=24)
def load(bounds, stamp):
    client, secret = credentials()
    if not client or not secret:
        raise RuntimeError("Xweather-Zugang fehlt (Streamlit-Secrets).")
    w,e,s,n=bounds
    params=dict(p=f"{s},{w},{n},{e}",client_id=client,client_secret=secret,
                **{"from":(stamp-timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M:%S UTC"),
                   "to":stamp.strftime("%Y-%m-%d %H:%M:%S UTC"),"limit":50000})
    try:
        response=requests.get("https://data.api.xweather.com/lightning/within",params=params,timeout=(4,12))
        if response.status_code!=200:
            raise RuntimeError(f"Blitzdaten nicht verfügbar (HTTP {response.status_code}).")
        data=response.json()
    except (requests.RequestException,ValueError):
        raise RuntimeError("Blitzdatenserver nicht erreichbar oder ungültige Antwort.") from None
    if not data.get("success") or not isinstance(data.get("response"),list):
        raise RuntimeError("Blitzdatenabfrage fehlgeschlagen; Enterprise-Zugang prüfen.")
    rows=data["response"]
    if len(rows)>=50000:
        raise RuntimeError("Blitzdatenlimit erreicht; vollständige Darstellung nicht möglich.")
    points=[]
    for row in rows:
        lon=float(row["loc"]["long"]); lat=float(row["loc"]["lat"])
        age=(stamp.timestamp()-float(row["ob"]["timestamp"]))/60
        if w<=lon<=e and s<=lat<=n and 0<=age<=30:
            points.append((lon,lat,age))
    return points

def draw(ax, points):
    for lo,hi,color,label in reversed([(0,5,"#ffffff","0–5 min"),(5,10,"#ffe500","5–10 min"),(10,20,"#ff9100","10–20 min"),(20,30.00001,"#ff2020","20–30 min")]):
        selected=[(x,y) for x,y,a in points if lo<=a<hi]
        ax.scatter([p[0] for p in selected],[p[1] for p in selected],marker="+",s=24,
                   linewidths=1.1,color=color,label=label,zorder=40+(30-lo)/10)
    legend=ax.legend(title="Blitzalter · Xweather",loc="lower left",facecolor="#10273e",
                     edgecolor="#7895ab",labelcolor="white",fontsize=8)
    legend.get_title().set_color("white")

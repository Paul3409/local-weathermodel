"""Interactive, deliberately smoothed personal synoptic outlook editor.

Tap the map to choose a location, enter a value, then add a control point.
The result is a subjective illustration, not a numerical forecast.
"""
import io
import json
from datetime import datetime, timezone
import numpy as np
import streamlit as st
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist
from icon_eu import REGIONS, borders, COUNTRIES, render

KINDS={"T850":"Temperatur T850 (°C)","H":"Hochdruckzentrum (hPa)","T":"Tiefdruckzentrum (hPa)"}


def _outline(fig, geometry):
    if not geometry:
        return
    typ=geometry.get("type")
    polys=[geometry["coordinates"]] if typ=="Polygon" else geometry.get("coordinates",[]) if typ=="MultiPolygon" else []
    import plotly.graph_objects as go
    for polygon in polys:
        for ring in polygon:
            if len(ring)<2:
                continue
            arr=np.asarray(ring)
            fig.add_trace(go.Scatter(x=arr[:,0],y=arr[:,1],mode="lines",
                line=dict(color="#647C8C",width=0.8),hoverinfo="skip",showlegend=False))


def _map(region, points):
    import plotly.graph_objects as go
    west,east,south,north=REGIONS[region]
    fig=go.Figure()
    try:
        for feature in borders(COUNTRIES).get("features",[]):
            _outline(fig,feature.get("geometry"))
    except Exception:
        st.caption("Grenzdaten derzeit nicht verfügbar; Punkte können trotzdem gesetzt werden.")
    # A regular, nearly invisible hit grid makes mobile taps work anywhere
    # in the map area (Plotly selection requires a selectable trace).
    xs=np.linspace(west,east,48)
    ys=np.linspace(south,north,32)
    gx,gy=np.meshgrid(xs,ys)
    fig.add_trace(go.Scatter(x=gx.ravel(),y=gy.ravel(),mode="markers",
        marker=dict(size=12,color="rgba(90,180,220,0.025)"),
        customdata=["editor_hit"]*gx.size,
        selected=dict(marker=dict(opacity=0.12)),
        unselected=dict(marker=dict(opacity=0.025)),
        name="Kartenposition",hovertemplate="%{x:.2f}° / %{y:.2f}°<extra></extra>"))
    for kind,marker,color in [("H","H","#49C2FF"),("T","T","#FF7171"),("T850","°","#FFFFFF")]:
        selected=[p for p in points if p["kind"]==kind]
        if selected:
            fig.add_trace(go.Scatter(x=[p["lon"] for p in selected],y=[p["lat"] for p in selected],
                mode="markers+text",text=[f'{marker} {p["value"]:g}' for p in selected],
                textposition="top center",textfont=dict(color=color,size=13),
                marker=dict(size=10,color=color,line=dict(color="#071B30",width=2)),
                hoverinfo="skip",showlegend=False))
    fig.update_layout(height=500,margin=dict(l=10,r=10,t=10,b=10),
        paper_bgcolor="#071B30",plot_bgcolor="#102A43",showlegend=False,
        dragmode="select",selectdirection="any",
        xaxis=dict(range=[west,east],showgrid=False,zeroline=False,title="Längengrad",
                   color="#B8CDDF",fixedrange=True),
        yaxis=dict(range=[south,north],showgrid=False,zeroline=False,title="Breitengrad",
                   color="#B8CDDF",fixedrange=True,scaleanchor="x",scaleratio=1))
    return fig


def _field(X,Y,points,kind,base,smoothing):
    """Broad Gaussian control-point influence; no invented fine structure."""
    chosen=[p for p in points if p["kind"]==kind]
    if not chosen:
        return np.full_like(X,base,dtype=float)
    lat_ref=float(np.mean(Y))
    scale=np.cos(np.deg2rad(lat_ref))
    loc=np.column_stack((X.ravel()*scale,Y.ravel()))
    centers=np.array([[p["lon"]*scale,p["lat"]] for p in chosen])
    dist=cdist(loc,centers).reshape(X.shape+(len(chosen),))
    west,east,south,north=X.min(),X.max(),Y.min(),Y.max()
    sigma=max((east-west)*scale,north-south)*(0.12+0.20*smoothing/100)
    weights=np.exp(-0.5*(dist/max(sigma,0.05))**2)
    # The base value is a weak, broad anchor to prevent extreme extrapolation.
    baseline_weight=0.12 if kind=="T850" else 0.45
    vals=np.array([p["value"] for p in chosen])
    result=(np.sum(weights*vals,axis=2)+baseline_weight*base)/(weights.sum(axis=2)+baseline_weight)
    return gaussian_filter(result,sigma=1.5+3*smoothing/100)


def _pressure(X,Y,points,smoothing):
    # Pressure systems act as broad anomalies on a neutral background.
    pressure=np.full_like(X,1015.,dtype=float)
    lat_ref=float(np.mean(Y))
    scale=np.cos(np.deg2rad(lat_ref))
    span=max((X.max()-X.min())*scale,Y.max()-Y.min())
    radius=span*(0.12+0.12*smoothing/100)
    for p in points:
        if p["kind"] not in ("H","T"):
            continue
        d2=((X-p["lon"])*scale)**2+(Y-p["lat"])**2
        pressure+=(p["value"]-1015)*np.exp(-d2/(2*radius**2))
    return gaussian_filter(pressure,sigma=1.5+3*smoothing/100)


def _render(region,points,smoothing,valid):
    from matplotlib import pyplot as plt
    from matplotlib import patheffects
    west,east,south,north=REGIONS[region]
    X,Y=np.meshgrid(np.linspace(west,east,240),np.linspace(south,north,190))
    temp=_field(X,Y,points,"T850",0.,smoothing)
    pressure=_pressure(X,Y,points,smoothing)
    # Reuse the established MeteoOST temperature palette and 4:3 frame.
    run=valid
    png=render(X,Y,temp,region,"T850 + MSLP",run,valid,
               model="Eigene Prognose",mslp=pressure)
    from PIL import Image,ImageDraw,ImageFont
    image=Image.open(io.BytesIO(png)).convert("RGB")
    # Explicit and prominent attribution on the finished export.
    draw=ImageDraw.Draw(image)
    draw.rectangle((0,0,image.width,35),fill="#071B30")
    draw.text((12,9),"EIGENE PROGNOSE · SUBJEKTIVE, STARK GEGLÄTTETE EINSCHÄTZUNG",
              fill="white")
    out=io.BytesIO()
    image.save(out,format="PNG")
    return out.getvalue()


def show():
    st.subheader("✏️ Eigene Prognosen · Karteneditor")
    st.caption("Tippe auf die Karte, setze Temperaturpunkte und Hoch-/Tiefdruckzentren. "
               "Das Ergebnis ist eine grobe persönliche Einschätzung, keine Modellvorhersage.")
    region=st.selectbox("Region",list(REGIONS),key="editor_region")
    kind=st.selectbox("Werkzeug",list(KINDS),format_func=lambda k:KINDS[k],key="editor_tool")
    smoothing=st.slider("Glättung",0,100,80,10,key="editor_smoothing")
    valid_date=st.date_input("Gültig am",value=datetime.now(timezone.utc).date(),key="editor_date")
    valid_hour=st.selectbox("Gültig um (UTC)",[0,6,12,18],index=2,key="editor_hour")
    points=st.session_state.setdefault("editor_points",{}).setdefault(region,[])
    st.info("Karte antippen bzw. anklicken: Ein Rasterpunkt wird ausgewählt. "
            "Danach den Wert eingeben und auf „Punkt hinzufügen“ tippen.")
    event=st.plotly_chart(_map(region,points),use_container_width=True,
                          key="editor_canvas_"+region,on_select="rerun",selection_mode="points")
    selected=event.selection.points if event is not None else []
    # Only hit-grid points (curve_number may vary with the country outlines).
    selected=[p for p in selected if "x" in p and "y" in p
              and p.get("customdata")=="editor_hit"]
    if selected:
        point=selected[-1]
        st.session_state["editor_click_"+region]=(float(point["x"]),float(point["y"]))
    loc=st.session_state.get("editor_click_"+region)
    if loc is None:
        st.caption("Noch keine Kartenposition ausgewählt.")
    else:
        st.caption(f"Gewählte Position: {loc[1]:.2f}° N, {loc[0]:.2f}° E")
    default= -5. if kind=="T850" else (1030. if kind=="H" else 990.)
    value=st.number_input("Wert (°C oder hPa)",value=default,step=1.,key="editor_value_"+kind)
    if st.button("➕ Punkt hinzufügen",type="primary",disabled=loc is None):
        if kind in ("H","T") and not 870<=value<=1080:
            st.error("Bitte einen plausiblen Luftdruck zwischen 870 und 1080 hPa wählen.")
        elif kind=="T850" and not -50<=value<=40:
            st.error("Bitte T850 zwischen −50 und +40 °C wählen.")
        else:
            points.append(dict(kind=kind,lon=loc[0],lat=loc[1],value=float(value)))
            st.rerun()
    if points:
        st.markdown("**Deine Stützpunkte**")
        for i,p in enumerate(points):
            c1,c2=st.columns([5,1])
            c1.write(f'{i+1}. {KINDS[p["kind"]]} · {p["value"]:g} · {p["lat"]:.2f}°N / {p["lon"]:.2f}°E')
            if c2.button("✕",key=f"editor_delete_{region}_{i}"):
                points.pop(i)
                st.rerun()
        if st.button("↶ Letzten Punkt rückgängig machen"):
            points.pop()
            st.rerun()
        if st.button("🗑 Alle Punkte dieser Region löschen"):
            points.clear()
            st.rerun()
    if st.button("🗺 Eigene Wetterkarte erstellen",disabled=not points):
        try:
            valid=datetime.combine(valid_date,datetime.min.time(),tzinfo=timezone.utc).replace(hour=valid_hour)
            with st.spinner("Großräumige Wetterfelder glätten und Karte zeichnen …"):
                png=_render(region,points,smoothing,valid)
            st.session_state["editor_result"]=(region,valid,tuple((p["kind"],p["lon"],p["lat"],p["value"]) for p in points),png)
        except Exception as exc:
            st.error(f"Karte konnte nicht erstellt werden: {type(exc).__name__}: {exc}")
    result=st.session_state.get("editor_result")
    current_signature=tuple((p["kind"],p["lon"],p["lat"],p["value"]) for p in points)
    if result and result[0]==region and result[2]==current_signature:
        st.image(result[3],use_container_width=True)
        st.download_button("PNG herunterladen",result[3],
            file_name=f'MeteoOST_Eigene_Prognose_{region.replace(" ","_")}.png',mime="image/png")
    st.download_button("💾 Entwurf als JSON speichern",
        json.dumps({"region":region,"points":points,"smoothing":smoothing},ensure_ascii=False,indent=2),
        file_name="MeteoOST_Prognoseentwurf.json",mime="application/json")
    uploaded=st.file_uploader("Gespeicherten Entwurf laden",type=["json"],key="editor_import")
    if uploaded and st.button("Entwurf importieren"):
        try:
            data=json.load(uploaded)
            loaded=data["points"]
            if not isinstance(loaded,list) or len(loaded)>250:
                raise ValueError("Ungültige Punktliste.")
            for p in loaded:
                if p["kind"] not in KINDS or not np.isfinite([p["lon"],p["lat"],p["value"]]).all():
                    raise ValueError("Ungültiger Stützpunkt.")
                w,e,so,n=REGIONS[region]
                if not w<=p["lon"]<=e or not so<=p["lat"]<=n:
                    raise ValueError("Punkt liegt außerhalb der gewählten Region.")
            st.session_state["editor_points"][region]=loaded
            st.rerun()
        except Exception as exc:
            st.error(f"Entwurf ungültig: {exc}")

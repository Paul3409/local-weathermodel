"""MeteoOST V2 probability maps: menu and ensemble calculation foundations.

Only member-resolved fields may be used for probabilities. Ensemble means
(geavg and AIGEFS ensstat avg) are deliberately never treated as members.
"""
import numpy as np
import streamlit as st

PERIODS = [(24,"1 Tag"),(48,"2 Tage"),(72,"3 Tage"),(120,"5 Tage"),
           (168,"7 Tage"),(336,"14 Tage"),(384,"Gesamter Zeitraum")]
CATEGORIES = {
    "Niederschlag": [
        ("Akk. Niederschlag >1 mm",1),("Akk. Niederschlag >5 mm",5),
        ("Akk. Niederschlag >10 mm",10),("Akk. Niederschlag >25 mm",25),
        ("Akk. Niederschlag >50 mm",50),("Akk. Niederschlag >100 mm",100)],
    "Temperatur": [
        ("Tmax >25 °C",25),("Tmax >30 °C",30),
        ("Tmax >35 °C",35),("Tmax >40 °C",40),
        ("Tmax <0 °C",0),("Tmin >20 °C",20),
        ("Tmin <0 °C",0),("Tmin <−5 °C",-5),
        ("Tmin <−10 °C",-10)],
    "Schnee": [("Schneefallrisiko 24 h",0)],
}
MODELS = ["GEFS", "AIGEFS"]


def probability(members, threshold, operator=">", min_members=2):
    """Percentage of finite ensemble members meeting a strict threshold.

    Input shape: (members, latitude, longitude). No missing members are
    counted as failures; grid cells with insufficient coverage stay NaN.
    """
    values=np.asarray(members,dtype=float)
    if values.ndim!=3:
        raise ValueError("Erwartet Ensemblemitglieder × Breite × Länge.")
    valid=np.isfinite(values)
    count=valid.sum(axis=0)
    if operator==">":
        hits=(values>threshold)&valid
    elif operator=="<":
        hits=(values<threshold)&valid
    else:
        raise ValueError("Nur > und < sind erlaubt.")
    result=np.full(count.shape,np.nan,dtype=float)
    np.divide(100.0*hits.sum(axis=0),count,out=result,where=count>=min_members)
    return result,count


def window_extreme(series,kind):
    """Daily maximum/minimum from hourly member data, excluding missing cells.

    Input shape: (members, time, latitude, longitude). Caller must ensure
    all required valid hours are present before accepting a 24h extreme.
    """
    a=np.asarray(series,dtype=float)
    if a.ndim!=4:
        raise ValueError("Erwartet Mitglieder × Zeit × Breite × Länge.")
    if not np.isfinite(a).all():
        raise ValueError("Unvollständige Temperaturzeitreihe; keine Tmax/Tmin-Wahrscheinlichkeit.")
    if kind=="max":return np.max(a,axis=1)
    if kind=="min":return np.min(a,axis=1)
    raise ValueError("Unbekanntes Extremum.")


def show():
    from v2_navigation import ALL_REGIONS, dropdown
    st.subheader("🎯 Wahrscheinlichkeiten")
    st.caption("Ensemblebasierte Wahrscheinlichkeit in % · ausschließlich GEFS und AIGEFS.")
    model=dropdown("1. Ensemblemodell","prob_model",MODELS)
    region=dropdown("2. Region","prob_region",ALL_REGIONS)
    category=dropdown("3. Kategorie","prob_category",list(CATEGORIES))
    parameter=dropdown("4. Grenzwert","prob_parameter_"+category,
                       [name for name,_ in CATEGORIES[category]])
    if category=="Schnee":
        hours=24
        st.caption("Schneefallrisiko: fester Zeitraum von 24 Stunden.")
    else:
        # Model availability is determined from actual member files, not
        # inferred from the availability of an ensemble-mean map.
        hours=dropdown("5. Zeitraum","prob_period",
                       [label for _,label in PERIODS])
        hours=dict((label,h) for h,label in PERIODS)[hours]
    st.info(
        f"Auswahl: {model} · {region} · {parameter} · "
        f"{hours if isinstance(hours,int) else 24} Stunden. "
        "Die Kartenberechnung ist noch nicht freigeschaltet: "
        "Zuerst müssen die verfügbaren Einzelmitglieder und die zeitliche "
        "Abdeckung der Originaldaten verifiziert werden. "
        "Ensemblemittel werden ausdrücklich nicht als Wahrscheinlichkeiten angezeigt."
    )
    st.caption("Nächster Schritt: Mitgliederdaten-Abruf, Zeitraumprüfung, "
               "Tmax/Tmin-Aggregation und Prozentkarten. Kein Open-Meteo.")

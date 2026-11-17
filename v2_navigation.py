"""ELBTAL V2 navigation: mobile-friendly popover menus without search input.

Only valid model/region/parameter combinations are offered. Existing
radar, ensemble meteograms and local products remain separate.
"""
import streamlit as st
from contextlib import contextmanager

ALL_REGIONS = ["Deutschland", "Mitteleuropa", "Europa & Nordatlantik"]
SYNOPTIC = ["MSLP", "T850", "T850 + MSLP", "Z500 + MSLP"]
LOCAL_FIELDS = ["2-m-Temperatur", "Taupunkt", "Windböen", "Niederschlagsrate",
                "Niederschlagssumme", "Signifikantes Wetter", "Neuschneemenge",
                "Schneehöhe", "Windchill"]
CAPABILITIES = {
    "ICON-D2": {"Deutschland": SYNOPTIC + LOCAL_FIELDS},
    "ICON-D2-RUC": {"Deutschland": SYNOPTIC + ["T2", "Schneehöhe", "Windböen",
        "Niederschlagssumme", "Niederschlagsrate", "CAPE",
        "Simulierte Radarreflektivität", "MSLP-Anomalie"]},
    "ICON-EU": {
        "Deutschland": SYNOPTIC + LOCAL_FIELDS,
        "Mitteleuropa": SYNOPTIC + ["Neuschneemenge", "Schneehöhe", "Windchill"],
        "Europa & Nordatlantik": SYNOPTIC + ["Neuschneemenge", "Schneehöhe", "Windchill"],
    },
    "GFS": {
        "Deutschland": SYNOPTIC + ["2-m-Temperatur", "Taupunkt", "Windböen",
            "Neuschneemenge", "Schneehöhe", "Windchill", "Schneefallgrenze (Schätzung)"],
        "Mitteleuropa": SYNOPTIC + ["Neuschneemenge", "Schneehöhe", "Windchill"],
        "Europa & Nordatlantik": SYNOPTIC + ["Neuschneemenge", "Schneehöhe", "Windchill"],
    },
    "ECMWF IFS": {region: SYNOPTIC + ["MSLP-Anomalie", "T2", "Schneehöhe"] for region in ALL_REGIONS},
    "GEFS Mittel": {region: SYNOPTIC + ["MSLP-Anomalie"] for region in ALL_REGIONS},
    "AIGFS": {region: ["MSLP", "T850", "T850 + MSLP", "T2",
                            "Z500 + MSLP", "MSLP-Anomalie", "Niederschlagssumme"]
              for region in ALL_REGIONS},
    "AIGEFS Mittel": {region: ["MSLP", "T850", "T2"] for region in ALL_REGIONS},
}

def dropdown(title, key, options):
    """Popover and buttons: no searchable combobox and no mobile keyboard."""
    if st.session_state.get(key) not in options:
        st.session_state[key] = options[0]
    with st.popover(f"{title} · {st.session_state[key]}", use_container_width=True):
        for item in options:
            if st.button(("✓  " if item == st.session_state[key] else "    ") + item,
                         key=f"v2_option_{key}_{item}", use_container_width=True):
                st.session_state[key] = item
                st.rerun()
    return st.session_state[key]

@contextmanager
def collect_explanations():
    """Place informational captions below the weather map."""
    original = st.caption
    notes = []
    def collect(message, *args, **kwargs):
        notes.append(str(message))
    st.caption = collect
    try:
        yield notes
    finally:
        st.caption = original


def show_model_maps():
    st.subheader("🌍 Wettermodelle")
    model=dropdown("1. Modell", "v2_model", list(CAPABILITIES))
    regions=list(CAPABILITIES[model])
    region=dropdown("2. Region", "v2_region", regions)
    parameter=dropdown("3. Parameter", "v2_parameter", CAPABILITIES[model][region])
    with collect_explanations() as notes:
        if model=="ICON-D2-RUC":
            from icon_d2_ruc import show
            show(region, forced_parameter=parameter)
        elif model=="GFS" and parameter=="Schneefallgrenze (Schätzung)":
            from snowline import show
            show(region)
        elif model in ("AIGFS","AIGEFS Mittel"):
            from ai_models import show
            show(region, forced_model=model, forced_parameter=parameter)
        elif model in ("ECMWF IFS","GEFS Mittel"):
            from global_models import show
            show(region, forced_model=model, forced_parameter=parameter)
        else:
            from icon_eu import show
            show(region, forced_model=model, forced_parameter=parameter)
    with st.expander("ℹ️ Modellinformationen & Hinweise", expanded=False):
        st.caption("Regionen und Parameter sind modellabhängig. Die tatsächliche Dateiverfügbarkeit wird beim Abruf geprüft.")
        for note in dict.fromkeys(notes):
            st.caption(note)

def show():
    st.caption("ELBTAL V2 · zentrale Navigation")
    section=dropdown("Bereich", "v2_section",
                     ["Modelle", "Modellvergleich", "Ensembles", "Radar"])
    if section=="Modelle":
        show_model_maps()
        return True
    if section=="Modellvergleich":
        from v2_comparison import show as show_comparison
        show_comparison()
        return True
    if section=="Ensembles":
        from ensembles import show as show_ensembles
        show_ensembles()
        return True
    if section=="Radar":
        from radar import show as show_radar
        show_radar()
        return True
    return False

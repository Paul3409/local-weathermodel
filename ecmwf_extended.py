"""Official ECMWF extended-range anomaly maps via public ecCharts WMS.

Only layers actually advertised by ECMWF's public GetCapabilities document
are selectable. No invented forecast data or unverified layer identifiers.
"""
from datetime import datetime, timezone
import xml.etree.ElementTree as ET

import requests
import streamlit as st

WMS = "https://eccharts.ecmwf.int/wms/"
HEADERS = {"User-Agent": "ELBTAL-Wetterkarten/2.0"}
PARAMETERS = {
    "Temperaturanomalie (2 m)": ("2m", "2 m", "2 metre", "2 meter", "temperature", "temperatur"),
    "Niederschlagsanomalie": ("precip", "rainfall", "niederschlag"),
    "Luftdruckanomalie (MSLP)": ("mslp", "mean sea level", "pressure", "luftdruck"),
}
EXTENDED = ("extended", "weekly", "week", "subseason", "sub-season", "monthly", "46-day", "46 day", "ec46")
ANOMALY = ("anomal", "departure", "deviation")


def _tag(element):
    return element.tag.rsplit("}", 1)[-1]


def _child(element, name):
    return next((node for node in element if _tag(node) == name), None)


def _value(element, name):
    child = _child(element, name)
    return (child.text or "").strip() if child is not None else ""


def _dimension(element, name):
    for node in element:
        if _tag(node) in ("Dimension", "Extent") and node.attrib.get("name", "").lower() == name:
            return (node.text or "").strip()
    return ""


def _times(value):
    """Use only explicitly advertised timestamps; never fabricate forecast dates."""
    if not value:
        return []
    result = []
    for item in value.split(","):
        item = item.strip()
        if not item or "/" in item:
            continue  # ISO interval expansion needs a separate verified implementation.
        try:
            parsed = datetime.fromisoformat(item.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            result.append((parsed, item))
        except ValueError:
            continue
    return sorted(set(result))


@st.cache_data(ttl=3600, show_spinner=False)
def capabilities():
    response = requests.get(
        WMS,
        params={"service": "WMS", "request": "GetCapabilities", "version": "1.3.0", "token": "public"},
        headers=HEADERS, timeout=(8, 35),
    )
    response.raise_for_status()
    try:
        root = ET.fromstring(response.content)
    except ET.ParseError as exc:
        raise RuntimeError("ECMWF liefert kein gültiges WMS-Fähigkeitsdokument.") from exc
    if _tag(root) not in ("WMS_Capabilities", "WMT_MS_Capabilities"):
        raise RuntimeError("ECMWF liefert kein WMS-Fähigkeitsdokument.")
    layers = []
    def walk(node, inherited_time=""):
        current_time = _dimension(node, "time") or inherited_time
        name = _value(node, "Name")
        if name:
            title = _value(node, "Title") or name
            abstract = _value(node, "Abstract")
            layers.append({"name": name, "title": title, "abstract": abstract, "time": current_time})
        for child in node:
            if _tag(child) == "Layer":
                walk(child, current_time)
    for node in root.iter():
        if _tag(node) == "Capability":
            top = _child(node, "Layer")
            if top is not None:
                walk(top)
            break
    return layers


def matching_layers(parameter, layers):
    found = []
    for layer in layers:
        description = " ".join(str(layer.get(k, "")) for k in ("name", "title", "abstract")).lower()
        if not any(term in description for term in PARAMETERS[parameter]):
            continue
        if not any(term in description for term in EXTENDED):
            continue
        if not any(term in description for term in ANOMALY):
            continue
        found.append(layer)
    return found


@st.cache_data(ttl=1800, show_spinner=False)
def get_map(layer_name, valid_time, area):
    # WMS 1.3.0 EPSG:4326 axis order is south,west,north,east.
    bbox = {"Europa": "30,-25,72,45", "Mitteleuropa": "45,3,57,22"}[area]
    params = {
        "service": "WMS", "request": "GetMap", "version": "1.3.0",
        "token": "public", "layers": layer_name, "styles": "",
        "crs": "EPSG:4326", "bbox": bbox, "width": 1100, "height": 800,
        "format": "image/png", "transparent": "false", "time": valid_time,
    }
    response = requests.get(WMS, params=params, headers=HEADERS, timeout=(8, 50))
    response.raise_for_status()
    if not response.content.startswith(b"\x89PNG\r\n\x1a\n"):
        snippet = response.text[:250] if "xml" in response.headers.get("Content-Type", "").lower() else ""
        raise RuntimeError("ECMWF liefert für diese Ebene und Zeit keine PNG-Karte. " + snippet)
    if len(response.content) > 20_000_000:
        raise RuntimeError("Die ECMWF-Karte ist unerwartet groß.")
    return response.content


def show():
    from v2_navigation import dropdown
    st.subheader("🗓️ ECMWF-Langfristkarten")
    st.caption("ECMWF-Langfrist · Prüfung der öffentlich verfügbaren ecCharts-WMS-Karten. Eigene ELBTAL-Rasterkarten benötigen einen separaten Datenzugang.")
    parameter = dropdown("1. Parameter", "v2_ec46_parameter", list(PARAMETERS))
    try:
        with st.spinner("ECMWF-WMS-Kartenebenen abrufen …"):
            available_layers = capabilities()
            layers = matching_layers(parameter, available_layers)
    except (requests.RequestException, RuntimeError) as exc:
        st.warning(f"Der öffentliche ECMWF-WMS-Dienst ist derzeit nicht erreichbar: {exc}")
        st.link_button("ECMWF-Kartenbrowser öffnen", "https://charts.ecmwf.int/")
        return
    if not layers:
        st.info("Im öffentlichen ECMWF-WMS wurde keine passende Wochenanomalie-Ebene gefunden. Das bedeutet nicht, dass ECMWF keine Langfristprognosen berechnet.")
        with st.expander("🔎 ECMWF-Datenzugriff prüfen", expanded=False):
            st.write(f"Öffentlich gemeldete WMS-Ebenen: {len(available_layers)}")
            st.write("Davon eindeutig passende Wochenanomalie-Ebenen: 0")
            st.caption("WMS liefert fertige Kartenbilder, keine Rasterwerte für eigene ELBTAL-Karten. Für eigene Karten bis Tag 46 wird ein geeigneter ECMWF-Rasterdatenzugang einschließlich Modellklimatologie benötigt.")
        st.link_button("Offizielle ECMWF-Langfristkarten öffnen", "https://charts.ecmwf.int/")
        return
    names = [layer["name"] for layer in layers]
    selected = dropdown("2. ECMWF-Kartenebene", "v2_ec46_layer", names)
    layer = next(item for item in layers if item["name"] == selected)
    st.caption(layer["title"])
    times = _times(layer["time"])
    if not times:
        st.info("Für diese Ebene meldet ECMWF keine einzeln auswählbaren Gültigkeitszeiten. Es wird keine Prognosewoche geraten.")
        st.link_button("Karte bei ECMWF ansehen", "https://charts.ecmwf.int/")
        return
    choices = {f"{dt:%d.%m.%Y %H:%M} UTC": raw for dt, raw in times}
    chosen = dropdown("3. Gültigkeitszeit", "v2_ec46_valid_time", list(choices))
    area = dropdown("4. Kartengebiet", "v2_ec46_area", ["Europa", "Mitteleuropa"])
    if st.button("ECMWF-Wochenkarte laden", type="primary", use_container_width=True):
        try:
            with st.spinner("ECMWF-WMS-Karte laden …"):
                png = get_map(selected, choices[chosen], area)
            st.session_state["v2_ec46_wms_result"] = (selected, choices[chosen], area, png)
        except (requests.RequestException, RuntimeError, ValueError) as exc:
            st.error(f"Die ECMWF-Karte konnte nicht geladen werden: {exc}")
    result = st.session_state.get("v2_ec46_wms_result")
    if result and result[:3] == (selected, choices[chosen], area):
        png = result[3]
        st.image(png, caption=f"ECMWF ecCharts · {layer['title']} · {chosen}", use_container_width=True)
        st.download_button("ECMWF-Karte als PNG herunterladen", png,
                           file_name="ELBTAL_ECMWF_Langfrist.png", mime="image/png")
        st.caption("Quelle: ECMWF ecCharts WMS. Originalkarte, keine berechneten Ersatzwerte.")
    with st.expander("ℹ️ Hinweise zu Langfristprognosen"):
        st.write("Wochenanomalien zeigen Abweichungen gegenüber der Modellklimatologie. Die Unsicherheit steigt mit dem Vorhersagezeitraum.")
        st.write("Der öffentliche WMS-Zugang kann weniger Produkte anbieten als der ECMWF-Kartenbrowser.")

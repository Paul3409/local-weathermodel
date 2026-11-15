# ELBTAL-T2M v0.4

Manuell gestartete, statische Temperaturkarten aus DWD ICON-D2 für Belgern–Nünchritz.

## Neu
- Experimentelle kleinräumige Geländekorrektur mit echten Höhenrastern (AWS Terrain Tiles / Terrarium)
- Original-ICON-D2 und lokales Ergebnis separat auswählbar
- Prognosekarten einmalig berechnen und während der Streamlit-Sitzung zwischen PNGs wechseln
- Karten als PNG herunterladen

## Installation
`app.py` und `requirements.txt` im GitHub-Repository ersetzen, dann Streamlit neu starten.

## Grenzen
Die Geländekorrektur basiert auf einer pauschalen Temperaturabnahme von 0,65 °C pro 100 Höhenmeter, angewendet nur auf die Abweichung zur geglätteten Geländehöhe (~2,2 km Skala). Sie ist keine validierte Verbesserung gegenüber ICON-D2, modelliert weder nächtliche Kaltluftseen noch Bebauung oder Elbeinfluss und ersetzt nicht die native Modellorografie. 100 m ist die ungefähre Ausgabeauflösung, nicht die meteorologische Vorhersagegenauigkeit. Bei Ausfall der Höhendaten werden Originalkarten erzeugt.

Die PNGs liegen nur in der aktuellen Streamlit-Session im Speicher; nach Neustart gehen sie verloren. Keine automatischen Hintergrundläufe.

Quelle Wetterdaten: DWD OpenData ICON-D2. Geländedaten: AWS Terrain Tiles (Terrarium).

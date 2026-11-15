# ELBTAL-T2M v0.5.2 – Landnutzung offline

## Einmalige Vorbereitung
1. `app.py` und `requirements.txt` auf GitHub aktualisieren.
2. Streamlit öffnen, Bereich **Landnutzung einmalig vorbereiten** ausklappen.
3. **OSM-Landnutzung einmalig abrufen** anklicken. Dieser Schritt benötigt einmalig einen erreichbaren Overpass-Server.
4. **landuse.npz herunterladen**.
5. Im GitHub-Repository Ordner `data` anlegen und Datei als `data/landuse.npz` hochladen, Commit speichern.
6. Nach dem Streamlit-Neustart muss **Gespeichertes Landnutzungsraster gefunden** erscheinen.

Danach greifen normale Modellläufe nur noch auf DWD-Daten, Geländedaten und die lokale NPZ-Datei zu. Es findet **kein Overpass-Abruf beim Modellstart** statt.

**Hinweis:** Das OSM-Raster kann unvollständig sein; unbekannte Pixel werden nicht korrigiert. Landnutzungskoeffizienten sind experimentell. Die Datei ist an die aktuellen Koordinatengrenzen und das Rasterformat gebunden. GitHub-Uploads sind bei öffentlichen Repositories öffentlich sichtbar. Streamlit-Sitzungsspeicher ist nicht dauerhaft.

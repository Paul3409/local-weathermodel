# ELBTAL-T2M v0.3

Manuell gestartete, **statische PNG-Temperaturkarten** aus den frei zugänglichen DWD-ICON-D2-Vorhersagen. Für Streamlit Community Cloud.

## Installation

1. `app.py` und `requirements.txt` im GitHub-Repository ersetzen.
2. In Streamlit Community Cloud die App neu starten bzw. auf das automatische Redeploy warten.
3. Auf **DWD-Prognose manuell berechnen** klicken.

## Technische Hinweise

- Die App liest DWD-Verzeichnislisten, sucht einen verfügbaren Modelllauf und lädt nur `T_2M`-Dateien.
- Die DWD-Dateien werden als GRIB2/BZip2 geladen und mit `eccodes` dekodiert.
- Der Modelllauf ist auf Stundenprognosen 0 bis +6 beschränkt, um Ressourcen zu sparen.
- Feine PNG-Pixel sind **nur räumliche Interpolation**, keine neue physikalische 100-m-Vorhersage.
- Das östliche Gebiet wurde bis 13.40°O erweitert, damit Nünchritz und Glaubitz auf der Karte liegen.
- DWD OpenData ist ein Live-Dienst: Dateinamen, Dateiformat und Verfügbarkeit können sich ändern. Bei Änderungen muss der Downloader angepasst werden.
- DWD-Quelle bitte bei Weiterverbreitung nennen; DWD-Daten unter CC BY 4.0 gemäß geltenden Bedingungen.

Datenquelle: https://opendata.dwd.de/weather/nwp/icon-d2/grib/

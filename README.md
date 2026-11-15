# ELBTAL-T2M v0.5.3 – ESA WorldCover statt Overpass

Die ICON-D2-Prognose, Geländekorrektur, vier PNG-Kartentypen und das
Streamlit-Kartenarchiv bleiben erhalten.

## Einmalige Vorbereitung
1. Den Branch `feature/worldcover-v053` testen bzw. in `main` übernehmen.
2. Streamlit öffnen und **Landnutzung einmalig vorbereiten** aufklappen.
3. Den ESA-WorldCover-Abruf ausführen. Er lädt nur einen Ausschnitt der
   öffentlich zugänglichen Cloud-Optimized GeoTIFF-Kachel N51E012.
4. Die generierte `landuse.npz` herunterladen.
5. Auf GitHub als `data/landuse.npz` committen.
6. Streamlit neu starten; die Datei wird danach offline gelesen.

**Wichtig:** Die Rasterdatei ist noch nicht Bestandteil dieses Branches.
Der einmalige ESA-Abruf benötigt Internetzugang und kann scheitern.
Normale Prognosen fragen weder Overpass noch WorldCover ab.

Datenquelle: ESA WorldCover 10 m 2021 v200 (CC BY 4.0).
© ESA WorldCover project 2021 / Contains modified Copernicus Sentinel
data (2021) processed by ESA WorldCover consortium.
https://esa-worldcover.org/en/data-access

Die Zuordnung der WorldCover-Klassen zu den Modellklassen ist vereinfacht;
insbesondere sind WorldCover-Bebauungsklassen keine Unterscheidung zwischen
Wohn- und Industriegebieten. Die Temperaturkorrektur ist experimentell und
nicht durch unabhängige Messungen validiert. Die 100-m-Kartenauflösung
bedeutet nicht 100-m-Prognosegenauigkeit.

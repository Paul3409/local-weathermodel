# ELBTAL-T2M

Experimentelles Lokalwetterprojekt für die Region Belgern–Nünchritz–Glaubitz.

## Version 0.1

- Manuell gestarteter Abruf echter stündlicher 2-m-Temperaturprognosen über Open-Meteo
- Grobe Abtastung im Rechteck 51.28–51.50° N, 13.10–13.37° E
- Interaktive Temperaturdarstellung in Streamlit für die nächsten 6 Stunden
- **Noch kein** eigenes Downscaling, keine 100-m-Modellauflösung, keine 15-Minuten-Prognose

## Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Auf Streamlit Community Cloud: GitHub-Repository auswählen und `app.py` als Einstiegspunkt angeben.

## Hinweise

Die Forecast API kann bei dicht beieinanderliegenden Punkten dieselben zugrunde liegenden Modellwerte liefern. Räumliche Glättung oder Darstellung erzeugt keine zusätzliche Prognoseinformation. Die Verwendung der Wetterdaten unterliegt den Bedingungen von Open-Meteo und der jeweiligen Datenquellen.

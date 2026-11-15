# ELBTAL-T2M v0.5

Manually triggered DWD ICON-D2 T2M maps for Belgern–Nünchritz.

## New
- Optional OSM land-use polygons via public Overpass endpoint
- Experimental nocturnal temperature adjustment for forest, farmland, industry, residential areas, water
- Difference PNG: experimental local forecast minus interpolated ICON-D2
- All forecast images generated first and kept in Streamlit session state for instant switching
- Existing elevation-only and ICON-D2 comparison remain available

## Install
Replace `app.py` and `requirements.txt` in your GitHub repository, redeploy Streamlit, click **DWD-Prognose manuell berechnen**.

## Scientific caveats
Land-use coefficients are **illustrative heuristics**. They are not derived from observations, and no cloud or wind dependence is included. OSM polygon coverage is incomplete and may be unavailable due to Overpass rate limits. No artificial data are substituted if the land-use request fails; elevation-only maps remain available. Terrain and land-use are not yet validated against independent stations. Grid resolution is a visualization grid, not true 100 m predictive skill. Downloaded PNGs and Streamlit session-state images are not durable across app restarts.


## v0.5.1 Bugfix

OSM-Polygonflächen werden ausschließlich nach ihrer Fläche sortiert (`key=lambda item: item[0]`). Bei gleichen Flächen wird nicht mehr versucht, NumPy-Koordinatenarrays miteinander zu vergleichen.

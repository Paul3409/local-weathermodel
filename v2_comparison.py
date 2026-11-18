"""ELBTAL V2: side-by-side-in-time weather model comparison.

Both panels are generated for the exact same UTC validity time.
Model data are downloaded only after pressing the render button.
"""
from datetime import timedelta
import streamlit as st
from v2_navigation import CAPABILITIES, dropdown
from v2_model_runs import candidates, probe
from icon_eu import REGIONS, LOCAL_TZ

COMPARE_FIELDS=("MSLP","T850","T850 + MSLP","Z500 + MSLP")

def leads(model,run):
    if model=="ICON-D2-RUC":return list(range(28))
    if model=="ICON-D2":return list(range(49))
    if model=="ICON-EU":return list(range(79))+list(range(81,121,3))
    if model=="GFS":return list(range(0,385,3))
    if model in ("AIGFS","AIGEFS Mittel"):return list(range(0,385,6))
    from global_models import steps
    return steps(model,run)

def get_map(model,region,parameter,run,lead):
    from icon_eu import render
    valid=run+timedelta(hours=lead)
    bounds=REGIONS[region]
    pressure=None
    if model in ("ICON-D2","ICON-EU"):
        from icon_eu import discover, discover_d2, load_grid
        lookup=discover_d2 if model=="ICON-D2" else discover
        variable="pmsl" if parameter=="MSLP" else ("fi" if parameter=="Z500 + MSLP" else "t")
        files=lookup(run,variable)
        if lead not in files:raise RuntimeError(f"{variable} +{lead} h ist für diesen Lauf nicht veröffentlicht.")
        X,Y,Z=load_grid(files[lead],valid,bounds,variable)
        if parameter in ("T850 + MSLP","Z500 + MSLP"):
            pressure_files=lookup(run,"pmsl")
            if lead not in pressure_files:raise RuntimeError("MSLP-Isobaren für diese Prognosestunde fehlen.")
            pressure=load_grid(pressure_files[lead],valid,bounds,"pmsl")[2]
    elif model=="GFS":
        from icon_eu import discover_gfs,load_gfs
        X,Y,Z=load_gfs(discover_gfs(run,parameter,bounds)[lead],valid,bounds,parameter)
        if parameter in ("T850 + MSLP","Z500 + MSLP"):
            pressure=load_gfs(discover_gfs(run,"MSLP",bounds)[lead],valid,bounds,"MSLP")[2]
    elif model in ("ECMWF IFS","GEFS Mittel"):
        from global_models import get_field,PARAMS
        X,Y,Z=get_field(model,run,lead,PARAMS[parameter],bounds)
        if parameter in ("T850 + MSLP","Z500 + MSLP"):
            pressure=get_field(model,run,lead,"msl",bounds)[2]
    elif model in ("AIGFS","AIGEFS Mittel"):
        from ai_models import field
        X,Y,Z=field(model,run,lead,parameter,region)
        if parameter in ("T850 + MSLP","Z500 + MSLP"):
            pressure=field(model,run,lead,"MSLP",region)[2]
    elif model=="ICON-D2-RUC":
        from icon_d2_ruc import field
        X,Y,Z=field(run,lead,parameter,region)
        if parameter in ("T850 + MSLP","Z500 + MSLP"):
            pressure=field(run,lead,"MSLP",region)[2]
    else:raise ValueError("Modell wird im Vergleich nicht unterstützt.")
    return render(X,Y,Z,region,parameter,run,valid,model=model,mslp=pressure)

def show():
    st.subheader("⚖️ Modellvergleich")
    st.caption("Zwei Originalmodelle · gleiche Region, gleicher Parameter und exakt gleiche UTC-Gültigkeit.")
    models=list(CAPABILITIES)
    left=dropdown("1. Modell A","v2_cmp_model_a",models)
    right_options=[m for m in models if m!=left and set(CAPABILITIES[m]).intersection(CAPABILITIES[left])]
    right=dropdown("2. Modell B","v2_cmp_model_b",right_options)
    regions=[r for r in CAPABILITIES[left] if r in CAPABILITIES[right]]
    region=dropdown("3. Region","v2_cmp_region",regions)
    common=[p for p in COMPARE_FIELDS if p in CAPABILITIES[left][region] and p in CAPABILITIES[right][region]]
    if not common:
        st.warning("Für diese Kombination gibt es derzeit keinen gemeinsamen Vergleichsparameter.")
        return
    parameter=dropdown("4. Parameter","v2_cmp_parameter",common)
    # Keep older, unverified cycles accessible; the user can compare archived runs.
    def choose_run(model,key):
        runs=candidates(model)
        return st.selectbox(f"Modelllauf {model} (UTC)",runs,
            format_func=lambda d:f"{d:%d.%m.%Y %H} UTC"+(" ✓" if probe(model,d) is True else ""),
            key=key)
    run_a=choose_run(left,"v2_cmp_run_a")
    run_b=choose_run(right,"v2_cmp_run_b")
    valid_a={run_a+timedelta(hours=h):h for h in leads(left,run_a)}
    valid_b={run_b+timedelta(hours=h):h for h in leads(right,run_b)}
    common_times=sorted(set(valid_a).intersection(valid_b))
    if not common_times:
        st.warning("Die gewählten Modellläufe haben keine gemeinsame Vorhersagezeit. Bitte andere Läufe wählen.")
        return
    # A small selection keeps the control usable even with long-range models.
    target=st.select_slider("Gemeinsamer Gültigkeitszeitpunkt (MEZ/MESZ)",
        options=common_times,value=common_times[min(2,len(common_times)-1)],
        format_func=lambda d:f"{d.astimezone(LOCAL_TZ):%d.%m.%Y %H:%M %Z}",
        key="v2_cmp_valid")
    st.caption(f"{left}: +{valid_a[target]} h · {right}: +{valid_b[target]} h")
    if st.button("Beide Wetterkarten vergleichen",type="primary",key="v2_cmp_generate"):
        images=[]
        for model,run,lead in ((left,run_a,valid_a[target]),(right,run_b,valid_b[target])):
            try:
                with st.spinner(f"{model} wird geladen …"):
                    images.append((model,run,lead,get_map(model,region,parameter,run,lead),None))
            except Exception as exc:
                images.append((model,run,lead,None,f"{type(exc).__name__}: {exc}"))
        st.session_state["v2_cmp_result"]=((left,right,region,parameter,run_a,run_b,target),images)
    result=st.session_state.get("v2_cmp_result")
    if result and result[0]==(left,right,region,parameter,run_a,run_b,target):
        # Stacked maps preserve the 4:3/3:4 card and legibility on mobile.
        for model,run,lead,png,error in result[1]:
            st.markdown(f"**{model} · +{lead} h**")
            if error:st.error(f"{model}: {error}")
            else:
                st.image(png,use_container_width=True)
                st.download_button(f"{model} · PNG herunterladen",png,
                    file_name=f"MeteoOST_V2_Vergleich_{model.replace(' ','_')}_{target:%Y%m%d%H}.png",
                    mime="image/png",key=f"v2_cmp_download_{model}")

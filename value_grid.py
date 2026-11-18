"""Optional, legible MeteoOST V2.1.1 numeric values over forecast maps."""
import numpy as np
import streamlit as st


def enabled():
    return bool(st.session_state.get("v2_value_grid", False))


def draw(ax, X, Y, values, region, parameter):
    """Sample the actual plotted field; never infer values from color pixels."""
    if not enabled() or parameter=="Signifikantes Wetter":
        return
    x=np.asarray(X,dtype=float)
    y=np.asarray(Y,dtype=float)
    z=np.asarray(np.ma.filled(np.ma.masked_invalid(values),np.nan),dtype=float)
    if x.shape!=y.shape or x.shape!=z.shape or x.ndim!=2:
        return
    from matplotlib import patheffects
    from matplotlib.transforms import Bbox
    # Keep a compact, nearly uniform geographic sampling, independent of
    # model resolution. Each label corresponds to an actual model grid point.
    ncols,nrows={"Deutschland":(7,9),"Mitteleuropa":(9,7),
                 "Europa & Nordatlantik":(12,6)}.get(region,(9,7))
    west,east=ax.get_xlim()
    south,north=ax.get_ylim()
    if east<=west or north<=south:
        return
    xs=np.linspace(west+(east-west)*.07,east-(east-west)*.07,ncols)
    ys=np.linspace(south+(north-south)*.09,north-(north-south)*.09,nrows)
    valid=np.isfinite(x)&np.isfinite(y)&np.isfinite(z)
    if not np.any(valid):
        return
    coords=np.column_stack((x[valid],y[valid]))
    vals=z[valid]
    # Limit search to the nearest genuine grid point; never extrapolate
    # values into an area with no source data.
    from scipy.spatial import cKDTree
    latitude_scale=max(.25,np.cos(np.deg2rad((south+north)/2)))
    tree=cKDTree(np.column_stack((coords[:,0]*latitude_scale,coords[:,1])))
    dx=np.nanmedian(np.abs(np.diff(x,axis=1))) if x.shape[1]>1 else 0
    dy=np.nanmedian(np.abs(np.diff(y,axis=0))) if y.shape[0]>1 else 0
    tolerance=max(float(np.hypot(dx*latitude_scale,dy))*2, .2)
    unit=("hPa" if parameter in ("MSLP","MSLP-Anomalie") else
          "°C" if parameter in ("T850","T850 + MSLP","T850-Anomalie",
                                "2-m-Temperatur","Taupunkt","Windchill") else
          "km/h" if parameter=="Windböen" else
          "cm" if parameter in ("Schneehöhe","Neuschneemenge") else
          "mm")
    if parameter=="Z500 + MSLP":
        unit="gpdam"
    for yy in ys:
        for xx in xs:
            distance,index=tree.query([xx*latitude_scale,yy])
            if distance>tolerance:
                continue
            value=float(vals[index])
            label=f"{value:.0f}" if unit in ("hPa","gpdam","km/h") else f"{value:.1f}"
            text=ax.text(xx,yy,label,fontsize=7.2,ha="center",va="center",
                         color="#F8FCFF",fontweight="bold",zorder=30,
                         bbox={"boxstyle":"round,pad=0.22","facecolor":"#071D34",
                               "edgecolor":"#72A9C7","linewidth":.55,"alpha":.84})
            text.set_path_effects([patheffects.withStroke(linewidth=.6,foreground="#071D34")])
    ax.text(.985,.012,f"Zahlen: {unit} · Modellgitterwerte",
            transform=ax.transAxes,ha="right",va="bottom",fontsize=7,
            color="#F8FCFF",zorder=31,
            bbox={"boxstyle":"round,pad=.25","facecolor":"#071D34",
                  "edgecolor":"none","alpha":.86})

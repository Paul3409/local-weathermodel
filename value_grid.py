"""Subtle, optional MeteoOST numeric overlay for model forecast maps."""
import numpy as np
import streamlit as st


def enabled():
    return bool(st.session_state.get("v2_value_grid", False))


def draw(ax, X, Y, values, region, parameter):
    """Display actual model-grid values with a lightweight cartographic halo."""
    if not enabled() or parameter=="Signifikantes Wetter":
        return
    x=np.asarray(X,dtype=float)
    y=np.asarray(Y,dtype=float)
    z=np.asarray(np.ma.filled(np.ma.masked_invalid(values),np.nan),dtype=float)
    if x.shape!=y.shape or x.shape!=z.shape or x.ndim!=2:
        return

    from matplotlib import patheffects
    from scipy.spatial import cKDTree

    # More detail than the original 7x9 labels, but still room for the
    # geographic outlines and the underlying colored forecast field.
    ncols,nrows={"Deutschland":(11,15),"Mitteleuropa":(13,10),
                 "Europa & Nordatlantik":(16,8)}.get(region,(12,10))
    west,east=ax.get_xlim()
    south,north=ax.get_ylim()
    if east<=west or north<=south:
        return
    xs=np.linspace(west+(east-west)*.055,east-(east-west)*.055,ncols)
    ys=np.linspace(south+(north-south)*.065,north-(north-south)*.065,nrows)

    valid=np.isfinite(x)&np.isfinite(y)&np.isfinite(z)
    water=None
    if parameter in ("Schneehöhe","Neuschneemenge"):
        # GRIB missing-value sentinels (e.g. 999900 cm) are not measurements.
        valid &= (z>=-0.1)&(z<9000)
        # Use the same land/sea geometry as the snow map itself. Do not
        # place numbers over the North Sea, Baltic Sea, or other ocean pixels.
        from icon_eu import snow_water_mask
        water=snow_water_mask(x,y)
        valid &= ~water
    if not np.any(valid):
        return
    coords=np.column_stack((x[valid],y[valid]))
    vals=z[valid]
    latitude_scale=max(.25,np.cos(np.deg2rad((south+north)/2)))
    tree=cKDTree(np.column_stack((coords[:,0]*latitude_scale,coords[:,1])))
    dx=np.nanmedian(np.abs(np.diff(x,axis=1))) if x.shape[1]>1 else 0
    dy=np.nanmedian(np.abs(np.diff(y,axis=0))) if y.shape[0]>1 else 0
    tolerance=max(float(np.hypot(dx*latitude_scale,dy))*2,.2)

    unit=("hPa" if parameter in ("MSLP","MSLP-Anomalie") else
          "°C" if parameter in ("T850","T850 + MSLP","T850-Anomalie",
                                "2-m-Temperatur","Taupunkt","Windchill") else
          "km/h" if parameter=="Windböen" else
          "cm" if parameter in ("Schneehöhe","Neuschneemenge") else
          "gpdam" if parameter=="Z500 + MSLP" else
          "%" if parameter=="Wahrscheinlichkeit" else "mm")
    # A slim dark stroke remains readable on bright yellows and deep reds,
    # without the heavy rectangular badges that previously obscured the map.
    halo=[patheffects.Stroke(linewidth=2.2,foreground="#071B30",alpha=.96),
          patheffects.Normal()]
    for yy in ys:
        for xx in xs:
            distance,index=tree.query([xx*latitude_scale,yy])
            if distance>tolerance:
                continue
            # Do not let the nearest land point label an offshore target.
            if water is not None:
                if water[int(np.abs(y[:,0]-yy).argmin()),int(np.abs(x[0,:]-xx).argmin())]:
                    continue
            value=float(vals[index])
            label=f"{value:.0f}" if unit in ("hPa","gpdam","km/h","%") else f"{value:.1f}"
            artist=ax.text(xx,yy,label,fontsize=6.9,ha="center",va="center",
                           color="#FFFFFF",fontweight="semibold",zorder=30,
                           clip_on=True)
            artist.set_path_effects(halo)
    # The map's colorbar already names the quantity and unit; no extra
    # dark label in the lower-right corner is necessary.

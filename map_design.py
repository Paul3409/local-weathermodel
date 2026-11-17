"""Shared navy-blue presentation for MeteoOst PNG maps.

Only presentation is changed. The rendered model field, colormaps and
meteorological classification remain untouched.
"""
import io
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager
from significant_weather import CATEGORIES

NAVY = "#071B30"
PANEL = "#102A43"
CYAN = "#39C8EE"
WHITE = "#F2F7FF"
MUTED = "#A9C0D4"


def style_figure(fig, ax):
    """Style the canvas, annotations, legend and colorbars without touching map data."""
    fig.patch.set_facecolor(NAVY)
    ax.set_facecolor(NAVY)
    ax.tick_params(axis="both", colors=WHITE, labelcolor=WHITE)
    ax.xaxis.label.set_color(WHITE)
    ax.yaxis.label.set_color(WHITE)
    for spine in ax.spines.values():
        spine.set_color(MUTED)
    ax.title.set_color(WHITE)
    for t in fig.texts:
        t.set_color(WHITE)
    for axis in fig.axes:
        if axis is ax:
            continue
        axis.tick_params(colors=WHITE, labelcolor=WHITE)
        axis.xaxis.label.set_color(WHITE)
        axis.yaxis.label.set_color(WHITE)
        for spine in axis.spines.values():
            spine.set_color(MUTED)
    for legend in fig.legends:
        for t in legend.get_texts():
            t.set_color(WHITE)


def _font(size, bold=False):
    candidates = (
        ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
         "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"]
        if bold else
        ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
         "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"]
    )
    try:
        candidates.insert(0, font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold" if bold else "normal")))
    except Exception:
        pass
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def frame_png(data, parameter, model, region, run, valid, source=None, portrait_override=None):
    """Professional navy frame. All weather values and palette colors stay intact."""
    from zoneinfo import ZoneInfo
    raw = Image.open(io.BytesIO(data)).convert("RGB")
    # Every exported weather map uses a genuine 4:3 canvas.
    # Contain the complete map image (never crop or stretch geographic data).
    portrait = (region == "Deutschland") if portrait_override is None else portrait_override
    width = 1200 if portrait else 1600
    height = 1600 if portrait else 1200
    scale = width / 1100.0
    u = lambda n: max(1, round(n * scale))
    margin = u(11)
    top = u(153)
    is_weather = "Signifikantes Wetter" in str(parameter)
    legend_h = u(145) if is_weather else 0
    footer_h = u(43)
    map_bottom = height - footer_h - legend_h - u(3)
    max_map_w = width - 2 * margin
    max_map_h = map_bottom - top
    if max_map_h <= 0:
        raise RuntimeError("Ungültiges 4:3-Kartenlayout.")
    if raw.width and raw.height:
        # Trim only the uniform Matplotlib background on all map regions.
        # Preserve all colored map pixels, labels, ticks and colorbar.
        from PIL import ImageChops
        background = Image.new("RGB", raw.size, raw.getpixel((0, 0)))
        difference = ImageChops.difference(raw, background).convert("L")
        occupied = difference.point(lambda value: 255 if value > 18 else 0).getbbox()
        if occupied:
            pad = 18
            raw = raw.crop((
                max(0, occupied[0] - pad),
                max(0, occupied[1] - pad),
                min(raw.width, occupied[2] + pad),
                min(raw.height, occupied[3] + pad),
            ))
    raw.thumbnail((max_map_w,max_map_h),Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), NAVY)
    d = ImageDraw.Draw(canvas)
    # Premium Dark: clean outer canvas, no competing double border.
    def text(x, y, value, size, color=WHITE, bold=False):
        d.text((u(x),u(y)),str(value),font=_font(u(size),bold),fill=color)
    def panel(x0,y0,x1,y1,r=8):
        d.rounded_rectangle((u(x0),u(y0),u(x1),u(y1)),radius=u(r),fill=PANEL,outline="#244660",width=max(1,u(1)))
    base_width = width / scale
    # Logo strip, matching the reference design: no oversized empty box.
    d.rectangle((0,0,width,u(79)),fill="#0B253D")
    d.rectangle((0,0,u(5),u(79)),fill=CYAN)
    d.line((0,u(78),width,u(78)),fill="#244C66",width=max(1,u(1)))
    # Cloud outline with simple vector primitives (no image asset or external request).
    cx,cy=u(46),u(37)
    lw=max(2,u(4))
    for xy in [(cx-u(31),cy-u(12),cx+u(10),cy+u(28)),
               (cx-u(7),cy-u(34),cx+u(43),cy+u(20)),
               (cx+u(21),cy-u(11),cx+u(64),cy+u(31))]:
        d.arc(xy,180,355,fill=CYAN,width=lw)
    d.line((cx-u(28),cy+u(27),cx+u(55),cy+u(27)),fill=CYAN,width=lw)
    text(125,7,"MeteoOst",32,bold=True)
    text(128,49,"W E T T E R K A R T E N",9,CYAN,True)
    d.line((u(335),u(13),u(335),u(65)),fill="#2C8DB3",width=max(1,u(1)))
    title = str(parameter)
    for symbol in ("🌦️ ","🌡️ ","🌬️ ","🌧️ ","💧 ","💦 ","🌫️ "):
        title=title.replace(symbol,"")
    title_size=26 if len(title)<23 else 20
    text(360,7,title,title_size,bold=True)
    text(362,47,f"{model}  ·  {region}",15,MUTED)
    # Three status panels with icons and larger, readable typography.
    xcoords=[11,376,741]
    widths=[354,354,base_width-752]
    valid_local=valid.astimezone(ZoneInfo("Europe/Berlin"))
    lead=round((valid-run).total_seconds()/3600)
    cells=[("MODELLLAUF",run.strftime("%d.%m.%Y · %H:%M UTC")),
           ("GÜLTIG",valid_local.strftime("%d.%m.%Y · %H:%M %Z")),
           ("PROGNOSE",f"+{lead} h")]
    for i,((label,value),x,cw) in enumerate(zip(cells,xcoords,widths)):
        panel(x,88,x+cw,145)
        ix=u(x+14); iy=u(105)
        if i==0:
            d.rounded_rectangle((ix,iy,ix+u(29),iy+u(26)),radius=u(3),outline=CYAN,width=max(1,u(2)))
            d.line((ix,iy+u(8),ix+u(29),iy+u(8)),fill=CYAN,width=max(1,u(2)))
        elif i==1:
            d.ellipse((ix,iy,ix+u(28),iy+u(28)),outline=CYAN,width=max(1,u(2)))
            d.line((ix+u(14),iy+u(5),ix+u(14),iy+u(14),ix+u(21),iy+u(18)),fill=CYAN,width=max(1,u(2)))
        else:
            d.polygon([(ix,iy),(ix+u(26),iy+u(14)),(ix,iy+u(28))],outline=CYAN)
        text(x+50,93,label,11,MUTED,True)
        text(x+50,113,value,15 if i!=2 else 19,WHITE,True)
    image_x=(width-raw.width)//2
    image_y=top+(max_map_h-raw.height)//2
    # Let the meteorological plot remain the visual focus; no decorative frame.
    canvas.paste(raw,(image_x,image_y))
    if is_weather:
        # Full-width grouped legend; same exact category colors as the DWD WW raster.
        ly=(height-footer_h-legend_h)/scale
        panel(16,ly,base_width-16,ly+140)
        groups=[
            ("Bewölkung",list(range(0,4))),
            ("Niederschlag",list(range(5,8))),
            ("Schneefall",list(range(10,13))),
            ("Mischformen",[8,9]),
            ("Gefrierender Regen",[13,14]),
            ("Nebel / Gewitter",[4,15,16]),
        ]
        available=base_width-56
        cell=available/6
        for gi,(group,indices) in enumerate(groups):
            gx=29+gi*cell
            if gi:
                d.line((u(gx-10),u(ly+17),u(gx-10),u(ly+128)),fill="#245E83",width=max(1,u(1)))
            text(gx,ly+13,group,13,WHITE,True)
            for j,idx in enumerate(indices):
                label,color=CATEGORIES[idx]
                yy=ly+44+j*34
                d.rounded_rectangle((u(gx),u(yy),u(gx+24),u(yy+18)),radius=u(2),fill=color)
                # Wrap longer entries rather than overflowing the next group.
                words={"Schneeregen mäßig/stark":"Schneeregen\nmäßig/stark",
                       "Gefr. Regen mäßig/stark":"Gefr. Regen\nmäßig/stark",
                       "Gewitter leicht/mäßig":"Gewitter\nleicht/mäßig"}
                name=words.get(label,label)
                d.multiline_text((u(gx+30),u(yy-2)),name,font=_font(u(11)),fill=WHITE,spacing=u(1))
    fy=(height-footer_h)/scale+3
    d.line((u(11),u(fy-3),width-u(11),u(fy-3)),fill="#244660",width=max(1,u(1)))
    src=source or ("DWD" if model.startswith("ICON") else "NOAA")
    source_label = "DWD RADOLAN" if model == "DWD RADOLAN" else f"{src} ({model})"
    text(19,fy+9,f"Daten: {source_label}    |    Region: {region}",12,MUTED)
    from app_version import VERSION
    brand=f"MeteoOst V{VERSION}"
    bb=d.textbbox((0,0),brand,font=_font(u(13),True))
    d.text((width-u(19)-(bb[2]-bb[0]),u(fy+9)),brand,font=_font(u(13),True),fill=MUTED)
    output=io.BytesIO()
    canvas.save(output,format="PNG")
    return output.getvalue()

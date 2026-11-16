"""Shared navy-blue presentation for ELBTAL PNG maps.

Only presentation is changed. The rendered model field, colormaps and
meteorological classification remain untouched.
"""
import io
from PIL import Image, ImageDraw, ImageFont

NAVY = "#061C35"
PANEL = "#092844"
CYAN = "#35C9F4"
WHITE = "#F2F7FF"
MUTED = "#B5C8DC"


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
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def frame_png(data, parameter, model, region, run, valid, source=None):
    """Wrap the existing PNG in a consistent dark navy information frame."""
    image = Image.open(io.BytesIO(data)).convert("RGB")
    width = image.width
    scale = width / 1100.0
    u = lambda n: round(n * scale)
    top = u(205)
    bottom = u(77)
    margin = u(20)
    result = Image.new("RGB", (width + 2 * margin, image.height + top + bottom), NAVY)
    d = ImageDraw.Draw(result)
    w = result.width
    d.rounded_rectangle((margin, u(12), w-margin, u(106)), radius=u(12), fill=PANEL, outline=CYAN, width=max(1,u(2)))
    d.text((u(42),u(23)), "ELBTAL", font=_font(u(42),True), fill=WHITE)
    d.text((u(44),u(76)), "WETTERKARTEN", font=_font(u(14),True), fill=CYAN)
    # Keep the parameter name on one line if possible.
    title = str(parameter).replace("🌦️ ","").replace("🌡️ ","").replace("🌬️ ","").replace("🌧️ ","").replace("💧 ","").replace("💦 ","").replace("🌫️ ","")
    size = 27 if len(title)<23 else 22
    d.text((u(350),u(22)), title, font=_font(u(size),True), fill=WHITE)
    d.text((u(352),u(69)), f"{model}  |  {region}", font=_font(u(18)), fill=MUTED)
    valid_local = valid.astimezone(__import__("zoneinfo").ZoneInfo("Europe/Berlin"))
    lead = int(round((valid-run).total_seconds()/3600))
    cells = [
        ("MODELLLAUF",run.strftime("%d.%m.%Y  %H:%M UTC")),
        ("GÜLTIG",valid_local.strftime("%d.%m.%Y  %H:%M %Z")),
        ("PROGNOSE",f"+{lead} h"),
    ]
    gap = u(12)
    cellw = (w - 2*margin - 2*gap)//3
    for i,(label,value) in enumerate(cells):
        x = margin+i*(cellw+gap)
        d.rounded_rectangle((x,u(118),x+cellw,u(189)),radius=u(10),fill=PANEL,outline="#236F94",width=max(1,u(1)))
        d.text((x+u(14),u(127)),label,font=_font(u(13),True),fill=CYAN)
        font_size = 17 if i<2 else 22
        d.text((x+u(14),u(151)),value,font=_font(u(font_size),True),fill=WHITE)
    result.paste(image,(margin,top))
    foot_y = top + image.height + u(12)
    d.rounded_rectangle((margin,foot_y,w-margin,foot_y+u(51)),radius=u(9),fill=PANEL,outline="#236F94",width=max(1,u(1)))
    d.text((u(36),foot_y+u(15)),f"Daten: {source or ('DWD' if model.startswith('ICON') else 'NOAA')}  |  {model}  |  {region}",font=_font(u(14)),fill=MUTED)
    brand="ELBTAL  v1.0"
    bb=d.textbbox((0,0),brand,font=_font(u(17),True))
    d.text((w-u(35)-(bb[2]-bb[0]),foot_y+u(13)),brand,font=_font(u(17),True),fill=CYAN)
    output=io.BytesIO()
    result.save(output,format="PNG",optimize=True)
    return output.getvalue()

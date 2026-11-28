"""Build a static Pages artifact from the existing GEFS PNG archive."""
from pathlib import Path
import json
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "_site"
OUT.mkdir(exist_ok=True)
for name in ("index.html", "style.css", "app.js"):
    shutil.copy2(ROOT / "docs" / name, OUT / name)
products = []
for path in sorted((ROOT / "ensemble_png" / "gefs").glob("*/m-hlberg-elbe/*.png")):
    run = path.parent.parent.name
    match = re.fullmatch(r"([a-z0-9-]+)_(\d+)\.png", path.name)
    if not match or not re.fullmatch(r"\d{10}", run):
        continue
    relative = path.relative_to(ROOT)
    dest = OUT / relative
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dest)
    products.append(dict(run=run, param=match[1], hours=int(match[2]), path=relative.as_posix()))
(OUT / "products.json").write_text(json.dumps({"products": products}), encoding="utf-8")
print(f"Indexed {len(products)} GEFS products")

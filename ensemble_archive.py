"""Stable paths and offline generation of ELBTAL ensemble PNG files."""
import argparse
import json
import os
import re
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

ROOT=Path(__file__).resolve().parent/"ensemble_png"
HORIZONS={"GEFS":384,"ECMWF ENS":360}
PARAMETERS=("MSLP","T850","T2","T2 + Niederschlag","Neuschnee")

def image_path(model,place,parameter,run,horizon):
    slug=lambda text:re.sub(r"[^a-z0-9]+","-",text.lower()).strip("-")
    return ROOT/slug(model)/run.strftime("%Y%m%d%H")/slug(place)/f"{slug(parameter)}_{horizon:03d}.png"

def available_images(model,place,parameter):
    """Newest locally published images first; never infer availability from model cycles."""
    from datetime import datetime
    reference=image_path(model,place,parameter,datetime(2000,1,1,tzinfo=timezone.utc),48)
    model_dir=ROOT/reference.relative_to(ROOT).parts[0]
    place_slug=reference.parent.name
    prefix=reference.stem.rsplit("_",1)[0]
    found=[]
    for file in (ROOT/model_dir).glob(f"*/{place_slug}/{prefix}_*.png"):
        try:
            run=datetime.strptime(file.parent.parent.name,"%Y%m%d%H").replace(tzinfo=timezone.utc)
            horizon=int(file.stem.rsplit("_",1)[-1])
            if file.is_file() and file.stat().st_size>1000:
                found.append((run,horizon,file))
        except (ValueError,OSError):
            continue
    return sorted(found,key=lambda item:(item[0],item[1]),reverse=True)


def generate(model,place,run,horizon,parameter):
    # Imported only by the scheduled GitHub worker, never on a Streamlit page view.
    from ensembles import PLACES,FIELDS,fetch,draw
    lat,lon=PLACES[place]
    steps=list(range(0,horizon+1,6))
    field=FIELDS[parameter]
    data,errors=fetch(model,run,field,lat,lon,steps)
    if parameter=="Neuschnee":
        data=np.maximum(np.diff(data,axis=1,prepend=np.nan),0)
    precipitation=None
    if parameter=="T2 + Niederschlag":
        precipitation,other_errors=fetch(model,run,"precip",lat,lon,steps)
        errors+=other_errors
    # A single missing GRIB value must not discard an otherwise useful ensemble.
    # Keep missing points as NaN (visible gaps); never interpolate model values.
    def complete(values):
        usable = values[:, 1:] if parameter=="Neuschnee" else values
        finite = np.isfinite(usable)
        return (finite.mean() >= 0.85 and
                (finite.mean(axis=0) >= 0.70).all() and
                (finite.mean(axis=1) >= 0.60).all())
    if not complete(data) or (precipitation is not None and not complete(precipitation)):
        valid=int(np.isfinite(data).sum())
        print(f"NOT READY {model} {place} {parameter}: {valid}/{data.size} values; errors={len(errors)}")
        return False
    path=image_path(model,place,parameter,run,horizon)
    path.parent.mkdir(parents=True,exist_ok=True)
    main=None
    if model=="GEFS":
        from ensembles import fetch_gfs_main
        main,main_errors=fetch_gfs_main(run,field,lat,lon,steps)
        errors+=main_errors
        if parameter=="Neuschnee":
            main=np.maximum(np.diff(main,prepend=np.nan),0)
    path.write_bytes(draw(model,place,run,field,steps,data,precipitation,main=main))
    print(f"CREATED {path} | completeness {np.isfinite(data).mean():.1%} | errors {len(errors)}")
    return True

def main():
    from ensembles import PLACES
    from model_cycles import recent_cycles
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",default="GEFS",choices=list(HORIZONS))
    parser.add_argument("--place",default="Mühlberg (Elbe)",choices=list(PLACES))
    parser.add_argument("--run",help="UTC YYYYMMDDHH; default: latest eligible run")
    parser.add_argument("--parameters",nargs="+",default=list(PARAMETERS),choices=list(PARAMETERS))
    args=parser.parse_args()
    run=(datetime.strptime(args.run,"%Y%m%d%H").replace(tzinfo=timezone.utc)
         if args.run else recent_cycles(args.model)[0])
    done=0
    for parameter in args.parameters:
        path=image_path(args.model,args.place,parameter,run,HORIZONS[args.model])
        if path.exists():
            print(f"EXISTS {path}")
            continue
        if generate(args.model,args.place,run,HORIZONS[args.model],parameter):done+=1
    print(f"Generated {done} new PNG(s)")

if __name__=="__main__":
    main()

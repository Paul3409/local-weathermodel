"""Stable paths and offline generation of MeteoOst ensemble PNG files."""
import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
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


CACHE_ROOT=Path(__file__).resolve().parent/"ensemble_cache"


def _cache_file(model,place,field,run):
    slug=lambda text:re.sub(r"[^a-z0-9]+","-",text.lower()).strip("-")
    return CACHE_ROOT/slug(model)/run.strftime("%Y%m%d%H")/slug(place)/f"{field}.npz"



def _persist_gefs_cache(cache_path,steps,data):
    """Merge completed points atomically, including older/longer lead times."""
    known={}
    if cache_path.is_file():
        try:
            with np.load(cache_path,allow_pickle=False) as saved:
                old_steps=saved["steps"].astype(int)
                old_data=saved["data"]
                if old_data.shape==(31,len(old_steps)):
                    known={int(h):old_data[:,j].copy() for j,h in enumerate(old_steps)}
        except (ValueError,KeyError,OSError,EOFError) as exc:
            print(f"CACHE RESET {cache_path}: {exc}")
    for j,hour in enumerate(steps):
        if int(hour) not in known:
            known[int(hour)]=np.full(31,np.nan,dtype=np.float32)
        mask=np.isfinite(data[:,j])
        known[int(hour)][mask]=data[mask,j]
    ordered=sorted(known)
    values=np.column_stack([known[hour] for hour in ordered]).astype(np.float32)
    cache_path.parent.mkdir(parents=True,exist_ok=True)
    temporary=cache_path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary,steps=np.array(ordered,dtype=np.int16),data=values)
    temporary.replace(cache_path)


def fetch_cached(model,place,run,field,lat,lon,steps):
    """Incrementally cache real GEFS point values across hourly workflow runs.

    Each batch is saved even when NOAA misses some fields. Missing data remain
    NaN and will be retried; no made-up values or interpolation.
    """
    if model!="GEFS":
        from ensembles import fetch
        return fetch(model,run,field,lat,lon,steps)
    from ensembles import gefs_point
    cache_path=_cache_file(model,place,field,run)
    data=np.full((31,len(steps)),np.nan)
    if cache_path.is_file():
        try:
            with np.load(cache_path,allow_pickle=False) as saved:
                old_steps=saved["steps"].astype(int)
                old_data=saved["data"]
                if old_data.shape==(31,len(old_steps)):
                    lookup={int(hour):index for index,hour in enumerate(old_steps)}
                    for j,hour in enumerate(steps):
                        if hour in lookup:
                            data[:,j]=old_data[:,lookup[hour]]
        except (ValueError,KeyError,OSError,EOFError) as exc:
            print(f"CACHE RESET {cache_path}: {exc}")
    missing=[(i,j) for j in range(len(steps)) for i in range(31)
             if not np.isfinite(data[i,j])]
    errors=[]
    if missing:
        print(f"FETCH {model} {field} +{steps[-1]}h: {len(missing)} missing of {data.size}",flush=True)
        # Persist regularly so a worker time limit does not waste all progress.
        for start in range(0,len(missing),96):
            batch=missing[start:start+96]
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures={pool.submit(gefs_point,run,steps[j],i,field,lat,lon):(i,j)
                         for i,j in batch}
                for future in as_completed(futures):
                    i,j=futures[future]
                    try:data[i,j]=future.result()
                    except Exception as exc:
                        errors.append(f"+{steps[j]}h P{i:02d}: {exc}")
            _persist_gefs_cache(cache_path,steps,data)
            print(f"CACHED {field} {run:%Y%m%d%H}: {np.isfinite(data).mean():.1%} "
                  f"({min(start+len(batch),len(missing))}/{len(missing)} checked)",flush=True)
    return data,errors


def generate(model,place,run,horizon,parameter):
    # Imported only by the scheduled GitHub worker, never on a Streamlit page view.
    from ensembles import PLACES,FIELDS,fetch,draw
    lat,lon=PLACES[place]
    steps=list(range(0,horizon+1,6))
    field=FIELDS[parameter]
    data,errors=fetch_cached(model,place,run,field,lat,lon,steps)
    if parameter=="Neuschnee":
        data=np.maximum(np.diff(data,axis=1,prepend=np.nan),0)
    precipitation=None
    if parameter=="T2 + Niederschlag":
        precipitation,other_errors=fetch_cached(model,place,run,"precip",lat,lon,steps)
        errors+=other_errors
    # A single missing GRIB value must not discard an otherwise useful ensemble.
    # Keep missing points as NaN (visible gaps); never interpolate model values.
    def complete(values, ignore_initial=False):
        # Accumulations (APCP and snow changes) have no preceding six-hour
        # interval at forecast step zero; that timestamp is not required.
        usable = values[:, 1:] if ignore_initial else values
        finite = np.isfinite(usable)
        return (finite.size > 0 and finite.mean() >= 0.85 and
                (finite.mean(axis=0) >= 0.70).all() and
                (finite.mean(axis=1) >= 0.60).all())
    temperature_ok=complete(data, ignore_initial=parameter=="Neuschnee")
    precipitation_ok=(precipitation is None or complete(precipitation, ignore_initial=True))
    if not temperature_ok or not precipitation_ok:
        valid=int(np.isfinite(data).sum())
        precip_valid=(f", precipitation {int(np.isfinite(precipitation[:, 1:]).sum())}/"
                      f"{precipitation[:, 1:].size}" if precipitation is not None else "")
        print(f"NOT READY {model} {place} {parameter}: {valid}/{data.size} values"
              f"{precip_valid}; errors={len(errors)}")
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

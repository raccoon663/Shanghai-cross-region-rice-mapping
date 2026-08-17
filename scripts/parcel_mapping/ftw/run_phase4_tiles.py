from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

import ee
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds
from shapely.geometry import box


ROOT = Path(__file__).resolve().parents[3]
REPO = ROOT / "work/field_delineation/ftw/ftw-baselines-fa86d4a3d766b96932521d92a41d43c9e4fc980e"
sys.path.insert(0, str(REPO))
from ftw_tools.inference.inference import run  # noqa: E402
from ftw_tools.postprocess.polygonize import polygonize  # noqa: E402


BASE = ROOT / "outputs/phase4_chongming_staged"
TILE_INDEX = BASE / "tiles/tile_index.csv"
CHECKPOINT = ROOT / "work/field_delineation/ftw/prue_efnetb5_ccby_checkpoint.ckpt"
LOG = BASE / "processing.log"
NAMES = {0: "water", 1: "trees", 2: "grass", 3: "flooded_vegetation", 4: "crops", 5: "shrub_and_scrub", 6: "built", 7: "bare", 8: "snow_and_ice"}
BANDS = ("B04_t1", "B03_t1", "B02_t1", "B08_t1", "B04_t2", "B03_t2", "B02_t2", "B08_t2")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8"); tmp.replace(path)


def log(message: str) -> None:
    line = f"{datetime.now().astimezone().isoformat(timespec='seconds')} {message}"
    with LOG.open("a", encoding="utf-8") as f: f.write(line + "\n")
    print(line, flush=True)


def cloud_mask(image: ee.Image) -> ee.Image:
    scl = image.select("SCL")
    valid = (scl.neq(0).And(scl.neq(1)).And(scl.neq(3)).And(scl.neq(7)).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11)))
    return image.updateMask(valid)


def window(region: ee.Geometry, start: str, end: str, suffix: str) -> tuple[ee.Image, int]:
    collection = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(region).filterDate(start, end).filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", 60)).map(cloud_mask))
    image = collection.select(["B4", "B3", "B2", "B8"]).median().rename([f"B04_{suffix}", f"B03_{suffix}", f"B02_{suffix}", f"B08_{suffix}"]).unmask(0).toUint16()
    return image, int(collection.size().getInfo())


def download(image: ee.Image, name: str, path: Path, b: dict) -> None:
    width = int(round((b["halo_east"] - b["halo_west"]) / 10)); height = int(round((b["halo_north"] - b["halo_south"]) / 10))
    region = ee.Geometry.Rectangle([b["halo_west"], b["halo_south"], b["halo_east"], b["halo_north"]], proj="EPSG:32651", geodesic=False)
    params = {"name": name, "crs": "EPSG:32651", "crs_transform": [10, 0, b["halo_west"], 0, -10, b["halo_north"]], "dimensions": [width, height], "format": "GEO_TIFF", "filePerBand": False}
    response = requests.get(image.clip(region).getDownloadURL(params), timeout=300); response.raise_for_status(); path.write_bytes(response.content)
    if zipfile.is_zipfile(path):
        z = path.with_suffix(".zip"); path.replace(z)
        with zipfile.ZipFile(z) as archive: archive.extractall(path.parent)
        candidates = sorted(path.parent.glob(f"{name}*.tif"))
        if len(candidates) != 1: raise RuntimeError(f"Expected one TIFF for {name}, got {candidates}")
        candidates[0].replace(path)


def landcover_fractions(geom, ds, arr: np.ndarray) -> dict[str, float]:
    win = from_bounds(*geom.bounds, transform=ds.transform).round_offsets().round_lengths().intersection(rasterio.windows.Window(0, 0, ds.width, ds.height))
    z = arr[int(win.row_off):int(win.row_off+win.height), int(win.col_off):int(win.col_off+win.width)]
    inside = geometry_mask([geom], out_shape=z.shape, transform=ds.window_transform(win), invert=True)
    values = z[inside]
    return {f"dw_{name}_fraction": float((values == code).sum()/max(len(values),1)) for code,name in NAMES.items()}


def pad_input_to_patch(source: Path, padded: Path, patch_size: int = 256) -> tuple[Path, dict]:
    with rasterio.open(source) as ds:
        original_shape = (ds.height, ds.width)
        target_shape = (max(ds.height, patch_size), max(ds.width, patch_size))
        if target_shape == original_shape:
            return source, {"applied": False, "original_shape": list(original_shape), "inference_shape": list(original_shape), "crop_back": False}
        data = ds.read(); profile = ds.profile.copy(); descriptions = ds.descriptions; tags = ds.tags()
    padded_data = np.zeros((data.shape[0], target_shape[0], target_shape[1]), dtype=data.dtype)
    padded_data[:, :original_shape[0], :original_shape[1]] = data
    profile.update(height=target_shape[0], width=target_shape[1])
    with rasterio.open(padded, "w", **profile) as dst:
        dst.write(padded_data); dst.descriptions = descriptions; dst.update_tags(**tags)
    return padded, {"applied": True, "original_shape": list(original_shape), "inference_shape": list(target_shape), "padding_location": "bottom/right outside frozen halo", "padding_value": 0, "crop_back": True}


def crop_native(source: Path, destination: Path, shape: tuple[int, int]) -> None:
    with rasterio.open(source) as ds:
        data = ds.read(window=rasterio.windows.Window(0, 0, shape[1], shape[0])); profile = ds.profile.copy(); descriptions = ds.descriptions; tags = ds.tags()
    profile.update(height=shape[0], width=shape[1])
    with rasterio.open(destination, "w", **profile) as dst:
        dst.write(data); dst.descriptions = descriptions; dst.update_tags(**tags)


def process_tile(row: pd.Series) -> None:
    tile_id = row.tile_id; td = BASE / "tiles" / tile_id; td.mkdir(parents=True, exist_ok=True)
    manifest_path = td / "tile_manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("status") == "complete": log(f"{tile_id} SKIP complete"); return
    b = {k: float(row[k]) for k in ["core_west","core_south","core_east","core_north","halo_west","halo_south","halo_east","halo_north"]}
    manifest = {"tile_id":tile_id,"status":"started","started_at":datetime.now().astimezone().isoformat(timespec="seconds"),"core_extent":[b["core_west"],b["core_south"],b["core_east"],b["core_north"]],"halo_extent":[b["halo_west"],b["halo_south"],b["halo_east"],b["halo_north"]]}; atomic_json(manifest_path,manifest)
    region = ee.Geometry.Rectangle([b["halo_west"],b["halo_south"],b["halo_east"],b["halo_north"]],proj="EPSG:32651",geodesic=False)
    early,n1=window(region,"2022-04-01","2022-06-30","t1"); late,n2=window(region,"2022-08-15","2022-11-01","t2")
    s2=(ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(region).filterDate("2022-04-01","2022-11-01").filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE",60)).map(cloud_mask))
    dwc=ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterBounds(region).filterDate("2022-04-01","2022-11-01")
    ftw, rgb, dw = td/"ftw_input_8band.tif", td/"rgb_input.tif", td/"dynamic_world_mode.tif"
    download(early.addBands(late),f"{tile_id}_ftw",ftw,b); download(s2.select(["B4","B3","B2"]).median().rename(["B04_red","B03_green","B02_blue"]).unmask(0).toUint16(),f"{tile_id}_rgb",rgb,b); download(dwc.select("label").mode().rename("dynamic_world_label").toUint8(),f"{tile_id}_dw",dw,b)
    with rasterio.open(ftw,"r+") as ds: ds.nodata=0; ds.descriptions=BANDS; ds.update_tags(t1="2022-04-01/2022-06-30",t2="2022-08-15/2022-11-01",model_input_contract="B04,B03,B02,B08 per window")
    with rasterio.open(rgb,"r+") as ds: ds.nodata=0; ds.descriptions=("B04_red","B03_green","B02_blue")
    manifest.update(status="inputs_complete",source_imagery_hashes={"ftw":sha256(ftw),"rgb":sha256(rgb),"dynamic_world":sha256(dw)},source_image_counts={"t1":n1,"t2":n2,"rgb":int(s2.size().getInfo()),"dynamic_world":int(dwc.size().getInfo())}); atomic_json(manifest_path,manifest); log(f"{tile_id} inputs complete")
    inference_input,padding_meta=pad_input_to_patch(ftw,td/"ftw_input_inference_padded.tif",256)
    common=dict(input=str(inference_input),model=str(CHECKPOINT),resize_factor=2,gpu=0,patch_size=256,batch_size=2,num_workers=0,padding=16,overwrite=True,mps_mode=False,nan_fill_value=0.0)
    score_target = td/("ftw_scores_native_padded.tif" if padding_meta["applied"] else "ftw_scores_native.tif")
    semantic_target = td/("ftw_semantic_native_padded.tif" if padding_meta["applied"] else "ftw_semantic_native.tif")
    t=time.perf_counter(); run(out=str(score_target),save_scores=True,compute_consensus=False,**common); score_s=time.perf_counter()-t
    t=time.perf_counter(); run(out=str(semantic_target),save_scores=False,compute_consensus=True,**common); sem_s=time.perf_counter()-t
    if padding_meta["applied"]:
        original_shape=tuple(padding_meta["original_shape"])
        padded_disagreement=semantic_target.with_name(semantic_target.stem+"_disagreements.tif")
        crop_native(score_target,td/"ftw_scores_native.tif",original_shape)
        crop_native(semantic_target,td/"ftw_semantic_native.tif",original_shape)
        crop_native(padded_disagreement,td/"ftw_semantic_native_disagreements.tif",original_shape)
    poly_common=dict(input=str(td/"ftw_semantic_native.tif"),simplify=True,max_size=None,overwrite=True,close_interiors=False,polygonization_stride=2048,softmax_threshold=None,merge_adjacent=None,erode_dilate=0,dilate_erode=0,erode_dilate_raster=0,dilate_erode_raster=0,thin_boundaries=False)
    polygonize(out=str(td/"ftw_polygons_raw.gpkg"),min_size=500,**poly_common); polygonize(out=str(td/"ftw_polygons_accepted_halo.gpkg"),min_size=2500,**poly_common)
    raw=gpd.read_file(td/"ftw_polygons_raw.gpkg"); accepted=gpd.read_file(td/"ftw_polygons_accepted_halo.gpkg")
    core=box(b["core_west"],b["core_south"],b["core_east"],b["core_north"]); halo=box(b["halo_west"],b["halo_south"],b["halo_east"],b["halo_north"])
    with rasterio.open(dw) as ds:
        arr=ds.read(1)
        for idx,geom in enumerate(accepted.geometry):
            for key,value in landcover_fractions(geom,ds,arr).items(): accepted.loc[idx,key]=value
    cols=[f"dw_{n}_fraction" for n in NAMES.values()]; accepted["dw_dominant"]=accepted[cols].idxmax(axis=1).str.replace("dw_","",regex=False).str.replace("_fraction","",regex=False)
    accepted["source_tile_id"]=tile_id; accepted["owner_core_flag"]=[core.covers(g.representative_point()) for g in accepted.geometry]; accepted["core_crossing_flag"]=[g.intersects(core.boundary) for g in accepted.geometry]; accepted["source_halo_edge_flag"]=[g.distance(halo.boundary)<=1e-6 for g in accepted.geometry]; accepted["area_m2"]=accepted.area
    (td/"ftw_polygons_accepted_halo.gpkg").unlink(); accepted.to_file(td/"ftw_polygons_accepted_halo.gpkg",layer="fields",driver="GPKG")
    owner=accepted[accepted.owner_core_flag].copy(); owner.to_file(td/"ftw_polygons_owner.gpkg",layer="fields",driver="GPKG")
    with rasterio.open(td/"ftw_semantic_native_disagreements.tif") as ds: disagreement=ds.read(1)>0
    qa={"raw_count":len(raw),"accepted_halo_count":len(accepted),"owner_count":len(owner),"accepted_over_20ha":int((owner.area>200000).sum()),"accepted_over_50ha":int((owner.area>500000).sum()),"owner_built_dominant":int((owner.dw_dominant=="built").sum()),"owner_tree_dominant":int((owner.dw_dominant=="trees").sum()),"owner_water_dominant":int((owner.dw_dominant=="water").sum()),"core_crossing_owner_count":int(owner.core_crossing_flag.sum()),"source_halo_edge_owner_count":int(owner.source_halo_edge_flag.sum()),"disagreement_pixel_fraction_halo":float(disagreement.mean())}
    manifest.update(status="complete",completed_at=datetime.now().astimezone().isoformat(timespec="seconds"),inference_status="complete",inference_padding=padding_meta,elapsed_seconds={"scores":score_s,"semantic":sem_s},output_counts=qa,qa_metrics=qa,output_sha256={p.name:sha256(p) for p in td.iterdir() if p.is_file() and p.name!="tile_manifest.json"}); atomic_json(manifest_path,manifest); log(f"{tile_id} COMPLETE owners={len(owner)} raw={len(raw)} padding={padding_meta['applied']}")


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--tile"); args=parser.parse_args()
    ee.Initialize(); index=pd.read_csv(TILE_INDEX); failures=[]
    if args.tile: index=index[index.tile_id==args.tile]
    for _,row in index.iterrows():
        try: process_tile(row)
        except Exception as exc:
            failures.append((row.tile_id,repr(exc))); log(f"{row.tile_id} FAILED {exc!r}")
    if failures: raise RuntimeError(f"Failed tiles: {failures}")


if __name__ == "__main__": main()

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import ee
import numpy as np
import rasterio
import requests


ROOT = Path(__file__).resolve().parents[3]
AOI = ROOT / "outputs/field_delineation/ftw_test_aoi_2/aoi_2.geojson"
BASE = ROOT / "work/field_delineation/ftw/input_data_aoi2"
FTW = BASE / "chongming_phase3b_aoi2_ftw_prue_s2_l2a_8band_2022_v1.tif"
RGB = BASE / "chongming_phase3b_aoi2_s2_l2a_rgb_2022_v1.tif"
DW = BASE / "chongming_phase3b_aoi2_dynamic_world_mode_2022_v1.tif"
META = BASE / "input_metadata.json"
BANDS = ("B04_t1", "B03_t1", "B02_t1", "B08_t1", "B04_t2", "B03_t2", "B02_t2", "B08_t2")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def cloud_mask(image: ee.Image) -> ee.Image:
    scl = image.select("SCL")
    valid = (scl.neq(0).And(scl.neq(1)).And(scl.neq(3)).And(scl.neq(7))
             .And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11)))
    return image.updateMask(valid)


def s2_window(region: ee.Geometry, start: str, end: str, suffix: str) -> tuple[ee.Image, int]:
    collection = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                  .filterBounds(region).filterDate(start, end)
                  .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", 60)).map(cloud_mask))
    return (collection.select(["B4", "B3", "B2", "B8"]).median()
            .rename([f"B04_{suffix}", f"B03_{suffix}", f"B02_{suffix}", f"B08_{suffix}"])
            .unmask(0).toUint16(), int(collection.size().getInfo()))


def download(image: ee.Image, name: str, path: Path, props: dict) -> None:
    region = ee.Geometry.Rectangle([props["west"], props["south"], props["east"], props["north"]], proj="EPSG:32651", geodesic=False)
    params = {"name": name, "crs": "EPSG:32651", "crs_transform": [10, 0, props["west"], 0, -10, props["north"]], "dimensions": [500, 500], "format": "GEO_TIFF", "filePerBand": False}
    response = requests.get(image.clip(region).getDownloadURL(params), timeout=300)
    response.raise_for_status()
    path.write_bytes(response.content)
    if zipfile.is_zipfile(path):
        archive_path = path.with_suffix(".zip")
        path.replace(archive_path)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(path.parent)
        candidates = sorted(path.parent.glob(f"{name}*.tif"))
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one TIFF for {name}; found {candidates}")
        candidates[0].replace(path)


def main() -> None:
    ee.Initialize()
    BASE.mkdir(parents=True, exist_ok=True)
    props = json.loads(AOI.read_text(encoding="utf-8"))["features"][0]["properties"]
    region = ee.Geometry.Rectangle([props["west"], props["south"], props["east"], props["north"]], proj="EPSG:32651", geodesic=False)
    early, early_count = s2_window(region, "2022-04-01", "2022-06-30", "t1")
    late, late_count = s2_window(region, "2022-08-15", "2022-11-01", "t2")
    full_s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(region)
               .filterDate("2022-04-01", "2022-11-01")
               .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", 60)).map(cloud_mask))
    dw_collection = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterBounds(region).filterDate("2022-04-01", "2022-11-01")
    ftw_image = early.addBands(late)
    rgb_image = full_s2.select(["B4", "B3", "B2"]).median().rename(["B04_red", "B03_green", "B02_blue"]).unmask(0).toUint16()
    dw_image = dw_collection.select("label").mode().rename("dynamic_world_label").toUint8()
    download(ftw_image, FTW.stem, FTW, props)
    download(rgb_image, RGB.stem, RGB, props)
    download(dw_image, DW.stem, DW, props)

    with rasterio.open(FTW, "r+") as ds:
        if ds.count != 8 or ds.shape != (500, 500) or ds.crs.to_epsg() != 32651:
            raise ValueError(f"Unexpected FTW input contract: {ds.profile}")
        ds.nodata = 0; ds.descriptions = BANDS
        ds.update_tags(source_collection="COPERNICUS/S2_SR_HARMONIZED", model_input_contract="FTW two windows: B04,B03,B02,B08 per window", t1="2022-04-01/2022-06-30", t2="2022-08-15/2022-11-01")
        ftw_data = ds.read()
        ftw_profile = {"crs": str(ds.crs), "transform": list(ds.transform)[:6], "shape": list(ds.shape), "count": ds.count, "dtype": ds.dtypes[0], "nodata": ds.nodata, "descriptions": list(ds.descriptions), "valid_all_bands_fraction": float(np.all(ftw_data > 0, axis=0).mean())}
    with rasterio.open(RGB, "r+") as ds:
        ds.nodata = 0; ds.descriptions = ("B04_red", "B03_green", "B02_blue")
        ds.update_tags(source_collection="COPERNICUS/S2_SR_HARMONIZED", date_start="2022-04-01", date_end_exclusive="2022-11-01", composite="cloud-masked median", band_order="B04,B03,B02")
        rgb_profile = {"crs": str(ds.crs), "transform": list(ds.transform)[:6], "shape": list(ds.shape), "count": ds.count, "dtype": ds.dtypes[0], "nodata": ds.nodata, "descriptions": list(ds.descriptions)}
    with rasterio.open(DW) as ds:
        dw_profile = {"crs": str(ds.crs), "transform": list(ds.transform)[:6], "shape": list(ds.shape), "dtype": ds.dtypes[0], "nodata_tag_warning": "value 0 is valid water class even if exporter tags it nodata"}
    metadata = {
        "version": "phase3b_aoi2_v1_20260814", "aoi": str(AOI.relative_to(ROOT)), "aoi_sha256": sha256(AOI),
        "ftw": {"path": str(FTW.relative_to(ROOT)), "sha256": sha256(FTW), "band_order": list(BANDS), "t1": {"start": "2022-04-01", "end_exclusive": "2022-06-30", "image_count": early_count}, "t2": {"start": "2022-08-15", "end_exclusive": "2022-11-01", "image_count": late_count}, **ftw_profile},
        "rgb": {"path": str(RGB.relative_to(ROOT)), "sha256": sha256(RGB), "date_start": "2022-04-01", "date_end_exclusive": "2022-11-01", "source_image_count": int(full_s2.size().getInfo()), **rgb_profile},
        "dynamic_world": {"path": str(DW.relative_to(ROOT)), "sha256": sha256(DW), "collection": "GOOGLE/DYNAMICWORLD/V1", "composite": "temporal mode", "source_image_count": int(dw_collection.size().getInfo()), **dw_profile},
        "scene_cloud_max_percent": 60, "pixel_cloud_mask": "SCL excludes 0,1,3,7,8,9,10,11",
        "rice_labels_used": False, "rice_prediction_values_used": False,
    }
    META.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

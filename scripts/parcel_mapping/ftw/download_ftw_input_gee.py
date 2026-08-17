from __future__ import annotations

import json
import zipfile
from pathlib import Path

import ee
import requests


ROOT = Path(__file__).resolve().parents[3]
AOI = ROOT / "data/field_delineation_test_aoi.geojson"
OUT_DIR = ROOT / "work/field_delineation/ftw/input_data"
NAME = "chongming_aoi_v2_label_free_ftw_prue_s2_l2a_8band_2022_v1"
OUT = OUT_DIR / f"{NAME}.tif"


def cloud_mask(image: ee.Image) -> ee.Image:
    scl = image.select("SCL")
    valid = (scl.neq(0).And(scl.neq(1)).And(scl.neq(3)).And(scl.neq(7))
             .And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11)))
    return image.updateMask(valid)


def window(region: ee.Geometry, start: str, end: str, suffix: str) -> tuple[ee.Image, int]:
    collection = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                  .filterBounds(region).filterDate(start, end)
                  .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", 60)).map(cloud_mask))
    bands = ["B4", "B3", "B2", "B8"]
    renamed = [f"B04_{suffix}", f"B03_{suffix}", f"B02_{suffix}", f"B08_{suffix}"]
    return collection.select(bands).median().rename(renamed).unmask(0).toUint16(), int(collection.size().getInfo())


def main() -> None:
    ee.Initialize()
    props = json.loads(AOI.read_text(encoding="utf-8"))["features"][0]["properties"]
    region = ee.Geometry.Rectangle([props["west"], props["south"], props["east"], props["north"]], proj="EPSG:32651", geodesic=False)
    # Predeclared non-label seasonal windows spanning the same April-November 2022 period as DAv2.
    early, early_count = window(region, "2022-04-01", "2022-06-30", "t1")
    late, late_count = window(region, "2022-08-15", "2022-11-01", "t2")
    image = early.addBands(late)
    params = {"name": NAME, "crs": "EPSG:32651", "crs_transform": [10, 0, props["west"], 0, -10, props["north"]], "dimensions": [500, 500], "format": "GEO_TIFF", "filePerBand": False}
    response = requests.get(image.clip(region).getDownloadURL(params), timeout=300)
    response.raise_for_status()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(response.content)
    if zipfile.is_zipfile(OUT):
        archive_path = OUT.with_suffix(".zip")
        OUT.replace(archive_path)
        with zipfile.ZipFile(archive_path) as archive:
            names = archive.namelist()
            archive.extractall(OUT_DIR)
        candidates = sorted(OUT_DIR.glob(f"{NAME}*.tif"))
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one TIFF; archive contained {names}")
        candidates[0].replace(OUT)
    metadata = {"version": "v1_20260814", "path": str(OUT.relative_to(ROOT)), "collection": "COPERNICUS/S2_SR_HARMONIZED", "product": "Sentinel-2 L2A surface reflectance", "band_order": ["B04_t1", "B03_t1", "B02_t1", "B08_t1", "B04_t2", "B03_t2", "B02_t2", "B08_t2"], "t1": {"start": "2022-04-01", "end_exclusive": "2022-06-30", "image_count": early_count}, "t2": {"start": "2022-08-15", "end_exclusive": "2022-11-01", "image_count": late_count}, "scene_cloud_max_percent": 60, "pixel_cloud_mask": "SCL excludes 0,1,3,7,8,9,10,11", "composite": "per-window per-band median", "crs": "EPSG:32651", "resolution_m": 10, "dtype": "uint16", "nodata": 0, "rice_labels_used": False, "rice_prediction_values_used": False}
    (OUT_DIR / "input_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import ee
import requests


ROOT = Path(__file__).resolve().parents[2]
AOI = ROOT / "data/field_delineation_test_aoi.geojson"
OUT = ROOT / "work/field_delineation/input_data"
RGB_NAME = "chongming_field_test_aoi_v2_label_free_s2_l2a_rgb_2022"
DW_NAME = "chongming_field_test_aoi_v2_label_free_dynamic_world_mode_2022"


def cloud_mask(image: ee.Image) -> ee.Image:
    scl = image.select("SCL")
    valid = (scl.neq(0).And(scl.neq(1)).And(scl.neq(3)).And(scl.neq(7))
             .And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11)))
    return image.updateMask(valid)


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
            names = archive.namelist()
            archive.extractall(path.parent)
        candidates = sorted(path.parent.glob(f"{name}*.tif"))
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one TIFF, got {names}")
        candidates[0].replace(path)


def main() -> None:
    ee.Initialize()
    OUT.mkdir(parents=True, exist_ok=True)
    props = json.loads(AOI.read_text(encoding="utf-8"))["features"][0]["properties"]
    region = ee.Geometry.Rectangle([props["west"], props["south"], props["east"], props["north"]], proj="EPSG:32651", geodesic=False)
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(region)
          .filterDate("2022-04-01", "2022-11-01").filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", 60)).map(cloud_mask))
    dw = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterBounds(region).filterDate("2022-04-01", "2022-11-01")
    rgb = s2.select(["B4", "B3", "B2"]).median().rename(["B04_red", "B03_green", "B02_blue"]).unmask(0).toUint16()
    dw_mode = dw.select("label").mode().rename("dynamic_world_label").toUint8()
    rgb_path, dw_path = OUT / f"{RGB_NAME}.tif", OUT / f"{DW_NAME}.tif"
    download(rgb, RGB_NAME, rgb_path, props)
    download(dw_mode, DW_NAME, dw_path, props)
    metadata = {
        "version": "v2_label_free_20260814", "aoi": str(AOI.relative_to(ROOT)),
        "rgb": {"path": str(rgb_path.relative_to(ROOT)), "collection": "COPERNICUS/S2_SR_HARMONIZED", "product": "Sentinel-2 L2A surface reflectance", "bands": ["B04", "B03", "B02"], "band_order": "RGB", "date_start": "2022-04-01", "date_end_exclusive": "2022-11-01", "scene_cloud_max_percent": 60, "pixel_cloud_mask": "SCL excludes 0,1,3,7,8,9,10,11", "composite": "per-band median of valid observations", "source_image_count": int(s2.size().getInfo()), "resolution_m": 10, "crs": "EPSG:32651", "dtype": "uint16", "nodata": 0},
        "qa_landcover": {"path": str(dw_path.relative_to(ROOT)), "collection": "GOOGLE/DYNAMICWORLD/V1", "band": "label", "composite": "temporal mode", "source_image_count": int(dw.size().getInfo()), "used_for_model_input": False, "used_for_parameter_tuning": False},
        "rice_labels_used": False, "rice_prediction_values_used": False,
    }
    (OUT / "input_metadata_v2_label_free.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import ee
import requests


ROOT = Path(__file__).resolve().parents[2]
AOI = ROOT / "data/field_delineation_test_aoi.geojson"
OUT = ROOT / "work/field_delineation/input_data"


def cloud_mask(image: ee.Image) -> ee.Image:
    scl = image.select("SCL")
    valid = (
        scl.neq(0)
        .And(scl.neq(1))
        .And(scl.neq(3))
        .And(scl.neq(7))
        .And(scl.neq(8))
        .And(scl.neq(9))
        .And(scl.neq(10))
        .And(scl.neq(11))
    )
    return image.updateMask(valid)


def download(image: ee.Image, name: str, out_path: Path, *, dtype: str) -> dict:
    feature = json.loads(AOI.read_text(encoding="utf-8"))["features"][0]
    p = feature["properties"]
    region = ee.Geometry.Rectangle(
        [p["west"], p["south"], p["east"], p["north"]],
        proj="EPSG:32651",
        geodesic=False,
    )
    if dtype == "uint16":
        image = image.toUint16()
    elif dtype == "uint8":
        image = image.toUint8()
    params = {
        "name": name,
        "crs": "EPSG:32651",
        "crs_transform": [10, 0, p["west"], 0, -10, p["north"]],
        "dimensions": [500, 500],
        "format": "GEO_TIFF",
        "filePerBand": False,
    }
    url = image.clip(region).getDownloadURL(params)
    response = requests.get(url, timeout=180)
    response.raise_for_status()
    out_path.write_bytes(response.content)
    if zipfile.is_zipfile(out_path):
        zip_path = out_path.with_suffix(".zip")
        out_path.replace(zip_path)
        with zipfile.ZipFile(zip_path) as archive:
            names = archive.namelist()
            archive.extractall(out_path.parent)
        tif_files = sorted(out_path.parent.glob(f"{name}*.tif"))
        if len(tif_files) != 1:
            raise RuntimeError(f"Expected one GeoTIFF in {zip_path}, found {names}")
        tif_files[0].replace(out_path)
    return {"url_generated": True, "bytes": out_path.stat().st_size}


def main() -> None:
    ee.Initialize()
    OUT.mkdir(parents=True, exist_ok=True)
    feature = json.loads(AOI.read_text(encoding="utf-8"))["features"][0]
    p = feature["properties"]
    region = ee.Geometry.Rectangle(
        [p["west"], p["south"], p["east"], p["north"]],
        proj="EPSG:32651",
        geodesic=False,
    )

    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate("2022-04-01", "2022-11-01")
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", 60))
        .map(cloud_mask)
    )
    s2_count = int(s2.size().getInfo())
    rgb = s2.select(["B4", "B3", "B2"]).median().rename(["red", "green", "blue"])

    dw = (
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        .filterBounds(region)
        .filterDate("2022-04-01", "2022-11-01")
    )
    dw_count = int(dw.size().getInfo())
    dw_mode = dw.select("label").mode().rename("dynamic_world_label")

    rgb_path = OUT / "chongming_field_test_5km_rgb_s2_2022.tif"
    dw_path = OUT / "chongming_field_test_5km_dynamic_world_mode_2022.tif"
    rgb_meta = download(rgb, "chongming_field_test_5km_rgb_s2_2022", rgb_path, dtype="uint16")
    dw_meta = download(dw_mode, "chongming_field_test_5km_dynamic_world_mode_2022", dw_path, dtype="uint8")
    metadata = {
        "aoi": str(AOI.relative_to(ROOT)),
        "rgb": {
            "path": str(rgb_path.relative_to(ROOT)),
            "collection": "COPERNICUS/S2_SR_HARMONIZED",
            "bands": ["B4", "B3", "B2"],
            "date_start": "2022-04-01",
            "date_end_exclusive": "2022-11-01",
            "scene_cloud_property_max_percent": 60,
            "pixel_cloud_mask": "SCL excludes 0,1,3,7,8,9,10,11",
            "composite": "per-band median of valid observations",
            "source_image_count": s2_count,
            "dtype": "uint16",
            **rgb_meta,
        },
        "qa_landcover": {
            "path": str(dw_path.relative_to(ROOT)),
            "collection": "GOOGLE/DYNAMICWORLD/V1",
            "band": "label",
            "composite": "temporal mode",
            "source_image_count": dw_count,
            "used_for_model_input": False,
            **dw_meta,
        },
    }
    (OUT / "input_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

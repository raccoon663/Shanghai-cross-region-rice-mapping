from __future__ import annotations

import json
import zipfile
from pathlib import Path

import ee
import requests


ROOT = Path(__file__).resolve().parents[2]
PROBABILITY = ROOT / "outputs/experiments/rf_baseline_m2_v1/shanghai_source_only_probability_s1_s2_fusion.tif"
OUT = ROOT / "work/field_delineation/input_data/chongming_dynamic_world_mode_2022_20m.tif"
META = ROOT / "work/field_delineation/input_data/chongming_dynamic_world_mode_2022_20m_metadata.json"


def main() -> None:
    # These values were read from the retained probability raster's spatial
    # metadata in the provenance check. Prediction pixels are never read here.
    left, bottom, right, top = 362320.0, 3488900.0, 391060.0, 3510320.0
    width, height = 1437, 1071
    transform = [20.0, 0.0, left, 0.0, -20.0, top]

    ee.Initialize()
    region = ee.Geometry.Rectangle([left, bottom, right, top], proj="EPSG:32651", geodesic=False)
    dw = (
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        .filterBounds(region)
        .filterDate("2022-04-01", "2022-11-01")
    )
    count = int(dw.size().getInfo())
    image = dw.select("label").mode().rename("dynamic_world_label").toUint8()
    params = {
        "name": "chongming_dynamic_world_mode_2022_20m",
        "crs": "EPSG:32651",
        "crs_transform": transform,
        "dimensions": [width, height],
        "format": "GEO_TIFF",
        "filePerBand": False,
    }
    response = requests.get(image.clip(region).getDownloadURL(params), timeout=300)
    response.raise_for_status()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(response.content)
    if zipfile.is_zipfile(OUT):
        zip_path = OUT.with_suffix(".zip")
        OUT.replace(zip_path)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(OUT.parent)
        candidates = sorted(OUT.parent.glob("chongming_dynamic_world_mode_2022_20m*.tif"))
        if len(candidates) != 1:
            raise RuntimeError(f"Unexpected archive members: {archive.namelist()}")
        candidates[0].replace(OUT)

    metadata = {
        "selection_source": "GOOGLE/DYNAMICWORLD/V1 temporal label mode",
        "date_start": "2022-04-01",
        "date_end_exclusive": "2022-11-01",
        "source_image_count": count,
        "crs": "EPSG:32651",
        "transform": transform,
        "shape": [height, width],
        "resolution_m": 20,
        "rice_labels_read": False,
        "rice_prediction_pixels_read": False,
        "probability_raster_use": "extent/grid metadata only",
    }
    META.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

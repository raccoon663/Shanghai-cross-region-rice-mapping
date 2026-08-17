from __future__ import annotations

import hashlib
import json
from pathlib import Path

import rasterio


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "work/field_delineation/input_data"
RGB = BASE / "chongming_field_test_aoi_v2_label_free_s2_l2a_rgb_2022.tif"
DW = BASE / "chongming_field_test_aoi_v2_label_free_dynamic_world_mode_2022.tif"
META = BASE / "input_metadata_v2_label_free.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    with rasterio.open(RGB, "r+") as ds:
        ds.nodata = 0
        ds.descriptions = ("B04_red", "B03_green", "B02_blue")
        ds.update_tags(source_collection="COPERNICUS/S2_SR_HARMONIZED", date_start="2022-04-01", date_end_exclusive="2022-11-01", composite="cloud-masked median", band_order="B04,B03,B02")
        rgb_profile = {"crs": str(ds.crs), "transform": list(ds.transform)[:6], "shape": [ds.height, ds.width], "count": ds.count, "dtype": ds.dtypes[0], "nodata": ds.nodata, "descriptions": list(ds.descriptions)}
    with rasterio.open(DW) as ds:
        dw_profile = {"crs": str(ds.crs), "transform": list(ds.transform)[:6], "shape": [ds.height, ds.width], "dtype": ds.dtypes[0], "nodata_tag_warning": "value 0 is valid water class even if exporter tags it nodata"}
    metadata = json.loads(META.read_text(encoding="utf-8"))
    metadata["rgb"].update(rgb_profile)
    metadata["rgb"]["sha256"] = sha256(RGB)
    metadata["qa_landcover"].update(dw_profile)
    metadata["qa_landcover"]["sha256"] = sha256(DW)
    META.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

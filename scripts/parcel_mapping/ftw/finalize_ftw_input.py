from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import rasterio


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "work/field_delineation/ftw/input_data"
INPUT = BASE / "chongming_aoi_v2_label_free_ftw_prue_s2_l2a_8band_2022_v1.tif"
META = BASE / "input_metadata.json"
BANDS = ("B04_t1", "B03_t1", "B02_t1", "B08_t1", "B04_t2", "B03_t2", "B02_t2", "B08_t2")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    with rasterio.open(INPUT, "r+") as ds:
        if ds.count != 8 or ds.shape != (500, 500) or ds.crs.to_epsg() != 32651:
            raise ValueError(f"Unexpected FTW input contract: {ds.profile}")
        ds.nodata = 0
        ds.descriptions = BANDS
        ds.update_tags(source_collection="COPERNICUS/S2_SR_HARMONIZED", model_input_contract="FTW two windows: B04,B03,B02,B08 per window", t1="2022-04-01/2022-06-30", t2="2022-08-15/2022-11-01")
        data = ds.read()
        profile = {"crs": str(ds.crs), "transform": list(ds.transform)[:6], "shape": list(ds.shape), "count": ds.count, "dtype": ds.dtypes[0], "nodata": ds.nodata, "descriptions": list(ds.descriptions), "valid_all_bands_fraction": float(np.all(data > 0, axis=0).mean()), "per_band_zero_fraction": [float((data[i] == 0).mean()) for i in range(8)]}
    metadata = json.loads(META.read_text(encoding="utf-8"))
    metadata.update(profile)
    metadata["sha256"] = sha256(INPUT)
    META.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

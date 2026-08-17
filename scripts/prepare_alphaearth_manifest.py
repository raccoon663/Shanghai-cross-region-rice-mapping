from pathlib import Path

import pandas as pd
from rasterio.warp import transform


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/experiments/rf_baseline_m2_v1/source_splits.csv"
TARGET = ROOT / "outputs/experiments/weak_label_budget_m4_v1/target_weak_samples.csv"
OUTPUT = ROOT / "data_metadata/alphaearth_sample_manifest.csv"


def main() -> None:
    source = pd.read_csv(SOURCE)
    source = source[["sample_id", "region", "class_id", "label_source", "longitude", "latitude", "spatial_block", "split"]]

    target = pd.read_csv(TARGET, usecols=["sample_id", "region", "class_id", "label_source", "x", "y", "spatial_block", "split"])
    longitude, latitude = transform("EPSG:32651", "EPSG:4326", target.x.to_list(), target.y.to_list())
    target["longitude"], target["latitude"] = longitude, latitude
    target = target[["sample_id", "region", "class_id", "label_source", "longitude", "latitude", "spatial_block", "split"]]

    manifest = pd.concat([source, target], ignore_index=True)
    manifest.insert(1, "year", 2022)
    if manifest.duplicated(["region", "sample_id"]).any():
        raise ValueError("Duplicate region/sample_id keys")
    if not manifest.longitude.between(115, 123).all() or not manifest.latitude.between(27, 33).all():
        raise ValueError("Unexpected coordinate range after target reprojection")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(OUTPUT, index=False)
    print(manifest.groupby(["region", "split", "class_id"]).size().to_string())
    print(f"Wrote {len(manifest):,} rows to {OUTPUT}")


if __name__ == "__main__":
    main()

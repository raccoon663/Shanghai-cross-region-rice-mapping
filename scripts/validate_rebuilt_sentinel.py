"""Validate reconstructed Sentinel exports before any experiment rerun."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from src.data.common import feature_groups, parse_gee_points


ORIGINAL_SOURCE_HASH = "042425e9a1d754191c6626379d43b5ee812656ea5b5b937ca1c8c9c1dc0b00dc"
ORIGINAL_TARGET_HASH = "16efb56adf0853bdd49fa0ed1e623ae8650ad32c92292d5c7521a3456ff7cd19"


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/raw/jiangxi_official_2022_binary_features_gee_s1_s2.csv"))
    parser.add_argument("--target", type=Path, default=Path("data/raw/shanghai_light_2022_fusion_92_20m.tif"))
    parser.add_argument("--report", type=Path, default=Path("data_metadata/rebuilt_sentinel_validation.json"))
    args = parser.parse_args()
    if not args.source.is_file() or not args.target.is_file():
        raise FileNotFoundError("Rebuilt source CSV and target GeoTIFF are both required")

    source = pd.read_csv(args.source)
    if ".geo" in source:
        source = parse_gee_points(source)
    groups = feature_groups(list(source.columns))
    features = groups["s1_s2_fusion"]
    if source.sample_id.duplicated().any():
        raise ValueError("Duplicate source sample IDs")
    if not {0, 1}.issubset(set(source.class_id)):
        raise ValueError("Source export must contain binary class_id values 0 and 1")
    values = source[features].replace(-9999, np.nan).to_numpy(float)

    with rasterio.open(args.target) as target:
        descriptions = list(target.descriptions)
        if descriptions != features:
            raise ValueError("Target band descriptions/order do not match source features")
        raster = {
            "count": target.count,
            "height": target.height,
            "width": target.width,
            "crs": str(target.crs),
            "resolution": [abs(target.transform.a), abs(target.transform.e)],
            "bounds": list(target.bounds),
            "nodata": target.nodata,
        }

    source_hash, target_hash = sha256(args.source), sha256(args.target)
    report = {
        "data_version": "historical_recovery" if source_hash == ORIGINAL_SOURCE_HASH and target_hash == ORIGINAL_TARGET_HASH else "reconstructed",
        "source": {
            "path": str(args.source), "sha256": source_hash,
            "matches_original_hash": source_hash == ORIGINAL_SOURCE_HASH,
            "rows": len(source), "features": len(features),
            "duplicate_ids": int(source.sample_id.duplicated().sum()),
            "missing_feature_values": int(np.isnan(values).sum()),
            "class_counts": {str(k): int(v) for k, v in source.class_id.value_counts().items()},
        },
        "target": {
            "path": str(args.target), "sha256": target_hash,
            "matches_original_hash": target_hash == ORIGINAL_TARGET_HASH,
            **raster,
        },
        "warning": "Do not append reconstructed-data results to historical tables without a new experiment version and full frozen rerun.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

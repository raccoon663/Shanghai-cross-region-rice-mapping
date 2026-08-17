from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from .common import feature_groups, load_config, parse_gee_points, resolve_data_path, resolve_project_path, sha256


def inspect(config_file: str | Path) -> dict:
    config, config_path = load_config(config_file)
    source_path = resolve_data_path(config["data"]["source_features"], config_path)
    target_path = resolve_data_path(config["data"]["target_features"], config_path)
    if not source_path.is_file() or not target_path.is_file():
        raise FileNotFoundError(f"Required input missing: source={source_path}, target={target_path}")

    source = parse_gee_points(pd.read_csv(source_path))
    groups = feature_groups(list(source.columns))
    feature_columns = groups["s1_s2_fusion"]
    invalid = ~np.isfinite(source[feature_columns].to_numpy(dtype="float64"))

    with rasterio.open(target_path) as raster:
        raster_names = list(raster.descriptions)
        if len(raster_names) != len(set(raster_names)) or set(raster_names) != set(feature_columns):
            raise ValueError("Target raster descriptions do not match the 92 source features")
        target_info = {
            "path": str(target_path),
            "sha256": sha256(target_path),
            "width": raster.width,
            "height": raster.height,
            "bands": raster.count,
            "crs": str(raster.crs),
            "resolution": [abs(raster.transform.a), abs(raster.transform.e)],
            "nodata": raster.nodata,
            "bounds": list(raster.bounds),
            "band_descriptions": raster_names,
        }

    result = {
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "git_commit": None,
        },
        "source": {
            "path": str(source_path),
            "sha256": sha256(source_path),
            "rows": len(source),
            "columns": len(source.columns),
            "class_counts": {str(k): int(v) for k, v in source.class_id.value_counts().items()},
            "reference_class_counts": (
                {str(k): int(v) for k, v in source.reference_class.value_counts().items()}
                if "reference_class" in source else None
            ),
            "duplicate_sample_ids": int(source.sample_id.duplicated().sum()),
            "duplicate_coordinates": int(source[["longitude", "latitude"]].duplicated().sum()),
            "missing_feature_values": int(invalid.sum()),
            "feature_dimensions": {name: len(values) for name, values in groups.items()},
            "longitude_range": [float(source.longitude.min()), float(source.longitude.max())],
            "latitude_range": [float(source.latitude.min()), float(source.latitude.max())],
        },
        "target": target_info,
        "limitations": [
            "target_reference is unavailable; no independent or weak-label target metrics are run",
            "git_commit is unavailable because the current directory is not a Git repository",
        ],
    }
    output_dir = resolve_project_path(config["project"]["output_dir"], config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dataset_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    print(json.dumps(inspect(args.config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

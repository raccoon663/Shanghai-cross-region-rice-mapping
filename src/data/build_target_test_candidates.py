from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform


def build(input_dir: Path, output_dir: Path, per_stratum: int = 50, seed: int = 20260803):
    names = ("s2_only", "s1_only", "s1_s2_fusion")
    arrays = {}
    profile = None
    for name in names:
        path = input_dir / f"shanghai_source_only_probability_{name}.tif"
        with rasterio.open(path) as source:
            arrays[name] = source.read(1)
            if profile is None:
                profile = {"transform": source.transform, "crs": source.crs, "shape": source.shape}
    fusion = arrays["s1_s2_fusion"]
    valid = np.isfinite(fusion) & (fusion >= 0)
    map_class = fusion >= 0.5
    confidence = np.maximum(fusion, 1 - fusion)
    confidence_group = np.full(fusion.shape, "high", dtype=object)
    confidence_group[confidence < 0.8] = "medium"
    confidence_group[confidence < 0.6] = "uncertain"
    stack = np.stack([arrays[name] for name in names])
    disagreement = stack.max(axis=0) - stack.min(axis=0)
    rng = np.random.default_rng(seed)
    selected = []
    for label in (0, 1):
        for confidence_name in ("uncertain", "medium", "high"):
            rows, cols = np.where(valid & (map_class == label) & (confidence_group == confidence_name))
            candidates = pd.DataFrame({"row": rows, "col": cols})
            candidates["spatial_block"] = (candidates.row // 100).astype(str) + "_" + (candidates.col // 100).astype(str)
            candidates["random"] = rng.random(len(candidates))
            candidates = candidates.sort_values(["random"]).drop_duplicates("spatial_block")
            if len(candidates) < per_stratum:
                remaining = pd.DataFrame({"row": rows, "col": cols})
                remaining["spatial_block"] = (remaining.row // 100).astype(str) + "_" + (remaining.col // 100).astype(str)
                remaining = remaining.merge(candidates[["row", "col"]], how="left", indicator=True).query("_merge == 'left_only'").drop(columns="_merge")
                remaining["random"] = rng.random(len(remaining))
                candidates = pd.concat([candidates, remaining.sort_values("random")], ignore_index=True)
            chosen = candidates.head(per_stratum).copy()
            chosen["map_class"] = label
            chosen["confidence_stratum"] = confidence_name
            selected.append(chosen)
    result = pd.concat(selected, ignore_index=True).drop_duplicates(["row", "col"])
    rr, cc = result.row.to_numpy(), result.col.to_numpy()
    xs, ys = rasterio.transform.xy(profile["transform"], rr, cc)
    lon, lat = transform(profile["crs"], "EPSG:4326", list(xs), list(ys))
    result.insert(0, "sample_id", [f"SH_TEST_{index:04d}" for index in range(1, len(result) + 1)])
    result["x_utm51n"], result["y_utm51n"] = xs, ys
    result["longitude"], result["latitude"] = lon, lat
    for name in names:
        result[f"probability_{name}"] = arrays[name][rr, cc]
    result["model_disagreement"] = disagreement[rr, cc]
    result["predicted_class_fusion"] = map_class[rr, cc].astype("uint8")
    result["label_source"] = "manual_pending"
    result["reference_class"] = ""
    result["landscape_type"] = "pending_visual_review"
    result["selection_reason"] = "class_confidence_stratum_with_spatial_dispersion"
    result["reserved_split"] = "target_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    result.drop(columns="random").to_csv(output_dir / "shanghai_independent_test_sample.csv", index=False)
    features = []
    for row in result.drop(columns="random").to_dict("records"):
        longitude, latitude = row["longitude"], row["latitude"]
        features.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [longitude, latitude]}, "properties": row})
    geojson = {"type": "FeatureCollection", "name": "shanghai_independent_test_sample", "features": features}
    (output_dir / "shanghai_independent_test_sample.geojson").write_text(json.dumps(geojson, ensure_ascii=False), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="outputs/experiments/rf_baseline_m2_v1")
    parser.add_argument("--output-dir", default="outputs/labeling")
    parser.add_argument("--per-stratum", type=int, default=50)
    args = parser.parse_args()
    result = build(Path(args.input_dir), Path(args.output_dir), args.per_stratum)
    print(result.groupby(["map_class", "confidence_stratum"]).size())
    print(f"Total candidates: {len(result)}")


if __name__ == "__main__":
    main()


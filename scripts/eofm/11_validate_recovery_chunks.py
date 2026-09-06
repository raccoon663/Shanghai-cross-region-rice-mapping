"""Validate available frozen recovery CSVs; never delete or submit cloud data."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_helper(filename):
    spec = importlib.util.spec_from_file_location("recovery_helper", Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_frame(frame, columns, manifest, start, stop, patch):
    if frame.columns.tolist() != columns:
        raise ValueError("Schema/band ordering mismatch")
    keys = ["manifest_row", "patch_row", "patch_col"] if patch else ["manifest_row"]
    ordered = frame.sort_values(keys).reset_index(drop=True)
    expected = np.array(
        [(c, r, k) for c in range(start, stop) for r in range(3) for k in range(3)]
        if patch else [(c,) for c in range(start, stop)]
    )
    actual = ordered[keys].to_numpy()
    if not np.array_equal(actual, expected) or frame.duplicated(keys).any():
        raise ValueError("Chunk/sample keys, row count, or row-major patch membership mismatch")
    expected_regions = manifest.iloc[start:stop].region.str.lower().to_numpy()
    if patch:
        expected_regions = np.repeat(expected_regions, 9)
    if not np.array_equal(ordered.region.str.lower().to_numpy(), expected_regions):
        raise ValueError("Region identity mismatch")
    if not np.isfinite(frame.drop(columns=["region"]).to_numpy(dtype=np.float64)).all():
        raise ValueError("Unreadable or non-finite numeric value")
    return {
        "rows": len(frame), "columns": len(columns), "centers": stop - start,
        "schema_and_band_order_exact": True, "chunk_sample_keys_exact": True,
        "region_identity_all_rows_exact": True, "duplicate_keys": 0,
        "row_major_keys_exact": True if patch else None,
        "readable_and_finite": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representation", choices=["galileo", "presto_primary"], required=True)
    parser.add_argument("--chunk-dir", type=Path, action="append", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    patch = args.representation == "galileo"
    helper = load_helper("09_prepare_galileo_smoke_patches.py" if patch else "08_prepare_presto_monthly_inputs.py")
    config_path = ROOT / ("configs/eofm_input_reconstruction.yaml" if patch else "configs/eofm_presto_monthly.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manifest_path = ROOT / config["sample_manifest"]
    grid_path = ROOT / config["temporal_grid_manifest"]
    manifest = pd.read_csv(manifest_path)
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    tags = [w["tag"] for w in grid["windows"]]
    columns = (helper.expected_columns if patch else helper.columns_for)(config["earth_engine"]["sentinel_2_bands"], tags)
    prefix = "eofm_galileo_patch3_2022" if patch else "eofm_presto_monthly_2022"
    old = json.loads(args.report.read_text(encoding="utf-8")) if args.report.exists() else {}
    for key, value in {
        "representation": args.representation,
        "sample_manifest_sha256": helper.sha256(manifest_path),
        "temporal_grid_sha256": helper.sha256(grid_path),
    }.items():
        if old and old.get(key) != value:
            raise ValueError(f"Recovery report frozen contract changed: {key}")
    old_hashes = {x["file"]: x["sha256"] for x in old.get("chunks", [])}
    files = [p for folder in args.chunk_dir for p in folder.glob(prefix + "_c*.csv")]
    planned = {f"{prefix}_c{i:03d}_{i*500:05d}_{min((i+1)*500,len(manifest)):05d}.csv": i for i in range(27)}
    if any(p.name not in planned for p in files) or len({p.name for p in files}) != len(files):
        raise ValueError("Unexpected or duplicate chunk files")
    records = []
    for path in sorted(files, key=lambda p: p.name):
        i = planned[path.name]
        digest = helper.sha256(path)
        if path.name in old_hashes and old_hashes[path.name] != digest:
            raise ValueError(f"Previously recorded checksum changed: {path.name}")
        checks = validate_frame(pd.read_csv(path), columns, manifest, i*500, min((i+1)*500,len(manifest)), patch)
        records.append({"chunk": i, "file": path.name, "sha256": digest, **checks})
    if set(old_hashes) - {p.name for p in files}:
        raise ValueError("Previously validated local file is missing")
    report = {
        "representation": args.representation,
        "status": "all_local_inputs_validated" if len(records) == 27 else "partial_local_inputs_validated",
        "sample_manifest_sha256": helper.sha256(manifest_path),
        "temporal_grid_sha256": helper.sha256(grid_path),
        "completed_chunks": len(records), "planned_chunks": 27,
        "local_centers": sum(x["centers"] for x in records),
        "local_rows": sum(x["rows"] for x in records),
        "missing_chunks": sorted(set(range(27)) - {x["chunk"] for x in records}),
        "cloud_cleanup_status": "not_asserted_by_local_validation",
        "chunks": records,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "chunks"}))


if __name__ == "__main__":
    main()

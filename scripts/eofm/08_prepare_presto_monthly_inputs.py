"""Validate monthly GEE chunks and build canonical Presto primary inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/eofm_presto_monthly.yaml"
DEFAULT_CHUNKS = ROOT / "outputs/eofm/ee/presto_monthly_downloaded"
DEFAULT_OUTPUT = ROOT / "outputs/eofm/presto_primary_inputs_2022_monthly.npz"
DEFAULT_REPORT = ROOT / "results/manifests/eofm_presto_primary_input_validation.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def columns_for(s2_bands: list[str], tags: list[str]) -> list[str]:
    columns = ["manifest_row", "region"]
    for tag in tags:
        columns += [f"s2_{band.lower()}_{tag}" for band in s2_bands]
        columns += [f"s2_valid_{tag}", f"s2_count_{tag}", f"s1_vv_{tag}", f"s1_vh_{tag}", f"s1_valid_{tag}", f"s1_count_{tag}"]
    return columns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--chunk-dir", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    grid_path = ROOT / config["temporal_grid_manifest"]
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    manifest_path = ROOT / config["sample_manifest"]
    manifest = pd.read_csv(manifest_path)
    tags = [w["tag"] for w in grid["windows"]]
    s2_bands = config["presto_input"]["s2_band_order"]
    expected_columns = columns_for(s2_bands, tags)
    paths = sorted(args.chunk_dir.resolve().glob("eofm_presto_monthly_2022_c*.csv"))
    if len(paths) != 27:
        raise ValueError(f"Expected 27 monthly CSV chunks, found {len(paths)}")
    frames, chunks = [], []
    for chunk_id, path in enumerate(paths):
        if f"_c{chunk_id:03d}_" not in path.name:
            raise ValueError(f"Chunk order/name mismatch at {path.name}")
        frame = pd.read_csv(path)
        if frame.columns.tolist() != expected_columns:
            raise ValueError(f"Schema mismatch in {path.name}")
        frames.append(frame)
        chunks.append({"chunk_id": chunk_id, "file": path.name, "rows": len(frame), "sha256": sha256(path)})
    frame = pd.concat(frames, ignore_index=True)
    expected_rows = np.arange(len(manifest))
    if len(frame) != len(manifest) or not np.array_equal(frame.manifest_row.to_numpy(), expected_rows):
        raise ValueError("Merged rows are not the exact ordered frozen manifest range")
    regions = manifest.region.str.lower().to_numpy()
    if not np.array_equal(frame.region.str.lower().to_numpy(), regions):
        raise ValueError("Export regions do not match the frozen manifest")
    missing = float(config["earth_engine"]["missing_value"])
    s1 = np.stack([frame[[f"s1_vv_{tag}", f"s1_vh_{tag}"]].to_numpy(np.float32) for tag in tags], axis=1)
    s2 = np.stack([frame[[f"s2_{band.lower()}_{tag}" for band in s2_bands]].to_numpy(np.float32) for tag in tags], axis=1)
    s1_explicit = np.stack([frame[f"s1_valid_{tag}"].to_numpy(np.uint8) for tag in tags], axis=1)
    s2_explicit = np.stack([frame[f"s2_valid_{tag}"].to_numpy(np.uint8) for tag in tags], axis=1)
    s1_valid = (s1_explicit & (s1 != missing).all(axis=2)).astype(np.uint8)
    s2_valid = (s2_explicit & (s2 != missing).all(axis=2)).astype(np.uint8)
    if not (np.isfinite(s1).all() and np.isfinite(s2).all()):
        raise ValueError("Non-finite native values found")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, s1=s1, s2=s2, s1_valid=s1_valid, s2_valid=s2_valid,
        manifest_row=expected_rows.astype(np.int32), sample_id=manifest.sample_id.astype(str).to_numpy(dtype="U"),
        region=regions.astype("U"), class_id=manifest.class_id.to_numpy(np.int8), split=manifest.split.astype(str).to_numpy(dtype="U"),
        latlons=manifest[["latitude", "longitude"]].to_numpy(np.float32),
        month_zero_based=np.int8(0), month_zero_based_by_timestep=np.arange(12, dtype=np.int8),
        s1_band_order=np.array(config["presto_input"]["s1_band_order"]), s2_band_order=np.array(s2_bands), missing_value=np.float32(missing),
    )
    report = {
        "manifest_version": "1.0", "status": "validated_ready_for_presto_primary_constructor", "representation": "presto_primary",
        "frozen_contract": {"sample_manifest": config["sample_manifest"], "sample_manifest_sha256": sha256(manifest_path), "temporal_grid": config["temporal_grid_manifest"], "temporal_grid_sha256": sha256(grid_path)},
        "source": {"directory": "gitignored runtime output", "chunks": chunks},
        "arrays": {"rows": len(manifest), "timesteps": 12, "s1_shape": list(s1.shape), "s2_shape": list(s2.shape), "s1_missing_timesteps": int((s1_valid == 0).sum()), "s2_missing_timesteps": int((s2_valid == 0).sum())},
        "canonical_npz": {"path": "gitignored runtime output", "sha256": sha256(output)},
        "presto_month_adaptation": {"official_input": "zero-based first month followed by internally generated consecutive months", "value": 0, "meaning": "January through December 2022"},
    }
    args.report.resolve().write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "sha256": sha256(output), **report["arrays"]}))


if __name__ == "__main__":
    main()

"""Validate Earth Engine chunks and build canonical raw Presto inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/eofm_input_reconstruction.yaml"
DEFAULT_CHUNKS = ROOT / "outputs/eofm/ee/downloaded_20260823"
DEFAULT_OUTPUT = ROOT / "outputs/eofm/presto_inputs_2022.npz"
DEFAULT_REPORT = ROOT / "results/manifests/eofm_presto_input_validation.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_columns(s2_bands: list[str], tags: list[str]) -> list[str]:
    columns = ["manifest_row", "region"]
    for tag in tags:
        columns.extend(f"s2_{band.lower()}_{tag}" for band in s2_bands)
        columns.extend([f"s2_ndvi_{tag}", f"s2_valid_{tag}", f"s2_count_{tag}"])
        columns.extend(
            [f"s1_vv_{tag}", f"s1_vh_{tag}", f"s1_rvi_{tag}",
             f"s1_valid_{tag}", f"s1_count_{tag}"]
        )
    return columns


def load_chunks(chunk_dir: Path, columns: list[str]) -> tuple[pd.DataFrame, list[dict]]:
    paths = sorted(chunk_dir.glob("eofm_presto_points_2022_c*.csv"))
    if len(paths) != 27:
        raise ValueError(f"Expected 27 CSV chunks, found {len(paths)} in {chunk_dir}")
    frames, records = [], []
    for chunk_id, path in enumerate(paths):
        expected_token = f"_c{chunk_id:03d}_"
        if expected_token not in path.name:
            raise ValueError(f"Chunk order/name mismatch at {path.name}")
        frame = pd.read_csv(path)
        if frame.columns.tolist() != columns:
            raise ValueError(f"Schema mismatch in {path.name}")
        frames.append(frame)
        records.append({"chunk_id": chunk_id, "file": path.name,
                        "rows": len(frame), "sha256": sha256(path)})
    return pd.concat(frames, ignore_index=True), records


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
    tags = [item["tag"] for item in grid["windows"]]
    s2_bands = config["presto_input"]["s2_band_order"]
    frame, chunks = load_chunks(args.chunk_dir.resolve(), expected_columns(s2_bands, tags))

    rows = len(manifest)
    expected_rows = np.arange(rows)
    if len(frame) != rows or not np.array_equal(frame.manifest_row.to_numpy(), expected_rows):
        raise ValueError("Merged rows are not the exact ordered frozen manifest range")
    expected_regions = manifest.region.str.lower().to_numpy()
    if not np.array_equal(frame.region.str.lower().to_numpy(), expected_regions):
        raise ValueError("Export regions do not match the frozen manifest")

    missing = float(config["earth_engine"]["missing_value"])
    s1 = np.stack([
        frame[[f"s1_vv_{tag}", f"s1_vh_{tag}"]].to_numpy(dtype=np.float32)
        for tag in tags
    ], axis=1)
    s2 = np.stack([
        frame[[f"s2_{band.lower()}_{tag}" for band in s2_bands]].to_numpy(dtype=np.float32)
        for tag in tags
    ], axis=1)
    s1_valid = np.stack([frame[f"s1_valid_{tag}"].to_numpy(dtype=np.uint8) for tag in tags], axis=1)
    s2_valid = np.stack([frame[f"s2_valid_{tag}"].to_numpy(dtype=np.uint8) for tag in tags], axis=1)
    s1_count = np.stack([frame[f"s1_count_{tag}"].to_numpy(dtype=np.uint16) for tag in tags], axis=1)
    s2_count = np.stack([frame[f"s2_count_{tag}"].to_numpy(dtype=np.uint16) for tag in tags], axis=1)
    s1_sentinel_valid = (s1 != missing).all(axis=2).astype(np.uint8)
    s2_sentinel_valid = (s2 != missing).all(axis=2).astype(np.uint8)
    s1_valid_disagreements = int((s1_valid != s1_sentinel_valid).sum())
    s2_valid_disagreements = int((s2_valid != s2_sentinel_valid).sum())
    # The explicitly exported masks and unmasked values can differ at pixel
    # boundaries because Earth Engine resamples them independently. Be
    # conservative: a timestep is usable only when both signals say it is.
    s1_valid = (s1_valid & s1_sentinel_valid).astype(np.uint8)
    s2_valid = (s2_valid & s2_sentinel_valid).astype(np.uint8)
    if not (np.isfinite(s1).all() and np.isfinite(s2).all()):
        raise ValueError("Non-finite native values found")

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, s1=s1, s2=s2, s1_valid=s1_valid, s2_valid=s2_valid,
        s1_count=s1_count, s2_count=s2_count,
        manifest_row=expected_rows.astype(np.int32),
        sample_id=manifest.sample_id.astype(str).to_numpy(dtype="U"),
        region=expected_regions.astype("U"), class_id=manifest.class_id.to_numpy(dtype=np.int8),
        split=manifest.split.astype(str).to_numpy(dtype="U"),
        latlons=manifest[["latitude", "longitude"]].to_numpy(dtype=np.float32),
        anchor_month=np.array([item["month"] for item in grid["windows"]], dtype=np.int8),
        anchor_date=np.array([item["anchor"] for item in grid["windows"]]),
        s1_band_order=np.array(config["presto_input"]["s1_band_order"]),
        s2_band_order=np.array(s2_bands), missing_value=np.float32(missing),
    )
    report = {
        "manifest_version": "1.0", "status": "validated_ready_for_presto_constructor",
        "source": {"directory": "gitignored runtime output", "chunks": chunks},
        "frozen_contract": {"sample_manifest": config["sample_manifest"],
                            "sample_manifest_sha256": sha256(manifest_path),
                            "temporal_grid": config["temporal_grid_manifest"],
                            "temporal_grid_sha256": sha256(grid_path)},
        "arrays": {"rows": rows, "timesteps": len(tags), "s1_shape": list(s1.shape),
                   "s2_shape": list(s2.shape), "s1_missing_timesteps": int((s1_valid == 0).sum()),
                   "s2_missing_timesteps": int((s2_valid == 0).sum()),
                   "s1_exported_mask_sentinel_disagreements": s1_valid_disagreements,
                   "s2_exported_mask_sentinel_disagreements": s2_valid_disagreements,
                   "jiangxi_rows": int((expected_regions == "jiangxi").sum()),
                   "shanghai_rows": int((expected_regions == "shanghai").sum())},
        "canonical_npz": {"path": "gitignored runtime output", "sha256": sha256(output)},
        "presto_month_adaptation": {
            "official_input": "zero-based first month followed by internally generated consecutive months",
            "value": 2,
            "meaning": "March; preserves the frozen 23 observations but not their irregular anchor-month spacing",
            "actual_anchor_months_retained_in_npz": True,
        },
    }
    args.report.resolve().write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "sha256": sha256(output), **report["arrays"]}))


if __name__ == "__main__":
    main()

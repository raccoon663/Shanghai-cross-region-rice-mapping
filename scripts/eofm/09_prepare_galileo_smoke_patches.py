"""Validate frozen Galileo smoke exports and build canonical patch tensors."""
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
DEFAULT_CHUNKS = ROOT / "outputs/eofm/galileo_smoke_downloaded"
DEFAULT_OUTPUT = ROOT / "outputs/eofm/galileo_smoke_patches_2022.npz"
DEFAULT_REPORT = ROOT / "results/manifests/eofm_galileo_smoke_patch_validation.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_columns(s2_bands: list[str], tags: list[str]) -> list[str]:
    columns = ["manifest_row", "region", "patch_row", "patch_col"]
    for tag in tags:
        columns.extend(f"s2_{band.lower()}_{tag}" for band in s2_bands)
        columns.extend([f"s2_ndvi_{tag}", f"s2_valid_{tag}", f"s2_count_{tag}"])
        columns.extend(
            [f"s1_vv_{tag}", f"s1_vh_{tag}", f"s1_rvi_{tag}",
             f"s1_valid_{tag}", f"s1_count_{tag}"]
        )
    return columns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--chunk-dir", type=Path, action="append",
                        help="Repeat for chunks retained in separate runtime directories")
    parser.add_argument("--scope", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    galileo = config["galileo_input"]
    if galileo["patch_pixels"] != 3 or galileo["encoder_patch_size"] != 1:
        raise ValueError("Frozen Galileo primary must be a 3x3 input with encoder patch_size=1")
    grid_path = ROOT / config["temporal_grid_manifest"]
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    manifest_path = ROOT / config["sample_manifest"]
    manifest = pd.read_csv(manifest_path)
    tags = [item["tag"] for item in grid["windows"]]
    s2_bands = list(config["earth_engine"]["sentinel_2_bands"])
    columns = expected_columns(s2_bands, tags)
    paths = sorted(
        [path for folder in (args.chunk_dir or [DEFAULT_CHUNKS])
         for path in folder.resolve().glob("eofm_galileo_patch3_2022_c*.csv")],
        key=lambda path: path.name,
    )
    chunk_ids = [0, 3] if args.scope == "smoke" else list(range(27))
    expected_names = [
        f"eofm_galileo_patch3_2022_c{chunk:03d}_{chunk * 500:05d}_"
        f"{min((chunk + 1) * 500, len(manifest)):05d}.csv"
        for chunk in chunk_ids
    ]
    if [path.name for path in paths] != expected_names:
        raise ValueError(f"Expected the frozen {args.scope} chunk inventory, found {[p.name for p in paths]}")

    frames, chunk_records = [], []
    for path in paths:
        frame = pd.read_csv(path)
        if frame.columns.tolist() != columns:
            raise ValueError(f"Schema or band ordering mismatch in {path.name}")
        frames.append(frame)
        chunk_records.append({"file": path.name, "rows": len(frame), "sha256": sha256(path)})
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.sort_values(["manifest_row", "patch_row", "patch_col"]).reset_index(drop=True)
    expected_centers = (
        np.r_[np.arange(500), np.arange(1500, 2000)]
        if args.scope == "smoke" else np.arange(len(manifest))
    )
    center_rows = frame.manifest_row.drop_duplicates().to_numpy(dtype=np.int32)
    if not np.array_equal(center_rows, expected_centers):
        raise ValueError("Smoke center membership differs from frozen c000/c003 design")
    expected_keys = np.array(
        [(center, row, col) for center in expected_centers for row in range(3) for col in range(3)]
    )
    actual_keys = frame[["manifest_row", "patch_row", "patch_col"]].to_numpy(dtype=np.int32)
    if len(frame) != len(expected_centers) * 9 or not np.array_equal(actual_keys, expected_keys):
        raise ValueError("Patch ordering is not exact row-major 3x3 for every center")
    expected_regions = manifest.iloc[expected_centers].region.str.lower().to_numpy()
    regions = frame.groupby("manifest_row", sort=True).region.first().str.lower().to_numpy()
    if not np.array_equal(regions, expected_regions):
        raise ValueError("Smoke export region labels disagree with frozen manifest")

    missing = float(config["earth_engine"]["missing_value"])
    n, height, width, timesteps, bands = len(expected_centers), 3, 3, len(tags), 13
    space_time_x = np.empty((n, height, width, timesteps, bands), dtype=np.float32)
    space_time_mask = np.ones((n, height, width, timesteps, 7), dtype=np.uint8)
    s1_disagreements = s2_disagreements = 0
    for t, tag in enumerate(tags):
        s1 = frame[[f"s1_vv_{tag}", f"s1_vh_{tag}"]].to_numpy(dtype=np.float32)
        s2 = frame[[f"s2_{band.lower()}_{tag}" for band in s2_bands]].to_numpy(dtype=np.float32)
        ndvi = frame[[f"s2_ndvi_{tag}"]].to_numpy(dtype=np.float32)
        s1_exported = frame[f"s1_valid_{tag}"].to_numpy(dtype=np.uint8).astype(bool)
        s2_exported = frame[f"s2_valid_{tag}"].to_numpy(dtype=np.uint8).astype(bool)
        s1_numeric = (s1 != missing).all(axis=1) & np.isfinite(s1).all(axis=1)
        s2_numeric = (s2 != missing).all(axis=1) & (ndvi[:, 0] != missing)
        s2_numeric &= np.isfinite(s2).all(axis=1) & np.isfinite(ndvi[:, 0])
        s1_disagreements += int((s1_exported != s1_numeric).sum())
        s2_disagreements += int((s2_exported != s2_numeric).sum())
        s1_valid = s1_exported & s1_numeric
        s2_valid = s2_exported & s2_numeric
        values = np.concatenate([s1, s2, ndvi], axis=1).reshape(n, height, width, bands)
        space_time_x[:, :, :, t, :] = values
        space_time_mask[:, :, :, t, 0] = (~s1_valid).reshape(n, height, width)
        space_time_mask[:, :, :, t, 1:] = np.repeat(
            (~s2_valid).reshape(n, height, width, 1), 6, axis=3
        )
    if not np.isfinite(space_time_x).all():
        raise ValueError("Non-finite values found in Galileo raw patch tensor")

    months = np.array([item["month"] - 1 for item in grid["windows"]], dtype=np.int64)
    output = (args.output or (
        DEFAULT_OUTPUT if args.scope == "smoke"
        else ROOT / "outputs/eofm/galileo_full_patches_2022.npz"
    )).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        space_time_x=space_time_x,
        space_time_mask=space_time_mask,
        months=months,
        manifest_row=expected_centers,
        region=expected_regions.astype("U"),
        band_order=np.array(galileo["space_time_band_order"]),
        patch_order=np.array([(r, c) for r in range(3) for c in range(3)], dtype=np.int8),
        missing_value=np.float32(missing),
    )
    report = {
        "manifest_version": "1.0",
        "status": "validated_ready_for_official_galileo_encoder",
        "frozen_primary": {
            "input_patch_pixels": [3, 3], "encoder_patch_size": 1,
            "resolution_m": 10, "footprint_m": [30, 30], "timesteps": timesteps,
            "tensor_order": ["sample", "height", "width", "timestep", "band"],
            "band_order": list(galileo["space_time_band_order"]),
            "months_zero_based": months.tolist(),
        },
        "source": {"chunks": chunk_records, "center_rows": n, "patch_pixel_rows": len(frame)},
        "validation": {
            "center_membership_exact": True, "row_major_patch_order_exact": True,
            "schema_and_band_order_exact": True, "region_membership_exact": True,
            "duplicate_patch_keys": int(frame.duplicated(
                ["manifest_row", "patch_row", "patch_col"]
            ).sum()),
            "s1_mask_sentinel_disagreements": s1_disagreements,
            "s2_mask_sentinel_disagreements": s2_disagreements,
            "masked_s1_tokens": int(space_time_mask[..., 0].sum()),
            "masked_s2_group_tokens": int(space_time_mask[..., 1:].sum()),
        },
        "frozen_inputs": {
            "sample_manifest_sha256": sha256(manifest_path),
            "temporal_grid_sha256": sha256(grid_path),
        },
        "canonical_npz": {"path": "gitignored runtime output", "sha256": sha256(output)},
    }
    report_path = (args.report or (
        DEFAULT_REPORT if args.scope == "smoke"
        else ROOT / "results/manifests/eofm_galileo_full_patch_validation.json"
    )).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "sha256": sha256(output), **report["validation"]}))


if __name__ == "__main__":
    main()

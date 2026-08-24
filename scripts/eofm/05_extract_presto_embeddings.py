"""Extract frozen Presto embeddings for the complete common sample manifest."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "outputs/eofm/presto_inputs_2022.npz"
DEFAULT_OUTPUT = ROOT / "outputs/eofm/presto_embeddings_2022.npz"
DEFAULT_REPORT = ROOT / "results/manifests/presto_embedding_freeze.json"
SMOKE_SCRIPT = ROOT / "scripts/eofm/04_smoke_test_presto.py"
PINNED_COMMIT = "11e207a668a34336ced1d8e492a1bd5849b96c4a"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_smoke_helpers():
    spec = importlib.util.spec_from_file_location("presto_smoke_helpers", SMOKE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def construct_batch(data, indices: np.ndarray, constructor, pipeline, torch):
    s1 = torch.from_numpy(data["s1"][indices].copy())
    s2 = torch.from_numpy(data["s2"][indices].copy())
    s1_valid = torch.from_numpy(data["s1_valid"][indices].astype(bool))
    s2_valid = torch.from_numpy(data["s2_valid"][indices].astype(bool))
    s1[~s1_valid] = 0
    s2[~s2_valid] = 0
    x, mask, dynamic_world = constructor.construct_batch_presto_input(
        s1=s1,
        s1_bands=data["s1_band_order"].astype(str).tolist(),
        s2=s2,
        s2_bands=data["s2_band_order"].astype(str).tolist(),
        normalize=True,
    )
    for band in ("VV", "VH"):
        band_idx = pipeline.NORMED_BANDS.index(band)
        mask[:, :, band_idx].masked_fill_(~s1_valid, 1)
        x[:, :, band_idx].masked_fill_(~s1_valid, 0)
    for band in data["s2_band_order"].astype(str).tolist() + ["NDVI"]:
        band_idx = pipeline.NORMED_BANDS.index(band)
        mask[:, :, band_idx].masked_fill_(~s2_valid, 1)
        x[:, :, band_idx].masked_fill_(~s2_valid, 0)
    if not torch.isfinite(x).all():
        raise ValueError("Non-finite official Presto input after normalization")
    return x, mask, dynamic_world.long()


def encode_all(data, batch_size: int, month: int, constructor, pipeline, model_module, torch) -> np.ndarray:
    device = model_module.device
    model = model_module.Presto.load_pretrained().to(device).eval()
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(data["manifest_row"]), batch_size):
            indices = np.arange(start, min(start + batch_size, len(data["manifest_row"])))
            x, mask, dynamic_world = construct_batch(
                data, indices, constructor, pipeline, torch
            )
            latlons = torch.from_numpy(data["latlons"][indices].copy())
            encoded = model.encoder(
                x.to(device), dynamic_world.to(device), latlons.to(device),
                mask.to(device), month=month, eval_task=True,
            )
            outputs.append(encoded.cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(outputs, axis=0)


def logical_hash(embeddings: np.ndarray, manifest_row: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(manifest_row.astype("<i4", copy=False).tobytes())
    digest.update(embeddings.astype("<f4", copy=False).tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--presto-source", type=Path, required=True)
    parser.add_argument("--extra-site-packages", type=Path)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--month", type=int, default=2,
                        help="Zero-based first month; use 0 for the monthly primary input")
    parser.add_argument("--representation", default="presto_legacy_cadence_sensitivity")
    parser.add_argument("--verify-repeat", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")

    source = args.presto_source.resolve()
    if PINNED_COMMIT not in source.name:
        raise ValueError(f"Expected official Presto commit {PINNED_COMMIT}")
    helpers = load_smoke_helpers()
    constructor, pipeline, model_module = helpers.load_official_modules(
        source, args.extra_site_packages
    )
    import torch

    input_path = args.input.resolve()
    data = np.load(input_path)
    manifest_row = data["manifest_row"]
    timesteps = int(data["s1"].shape[1])
    if not 0 <= args.month < 12:
        raise ValueError("month must be a zero-based value in [0, 11]")
    if timesteps != 23 and (args.output.resolve() == DEFAULT_OUTPUT.resolve()
                            or args.report.resolve() == DEFAULT_REPORT.resolve()):
        raise ValueError("Non-legacy inputs require explicit non-legacy --output and --report")
    if not np.array_equal(manifest_row, np.arange(13429, dtype=np.int32)):
        raise ValueError("Canonical input is not the exact frozen 13,429-row order")
    embeddings = encode_all(
        data, args.batch_size, args.month, constructor, pipeline, model_module, torch
    )
    if embeddings.shape != (13429, 128) or not np.isfinite(embeddings).all():
        raise ValueError("Invalid complete Presto embedding matrix")
    representation_hash = logical_hash(embeddings, manifest_row)

    repeat_hash = None
    if args.verify_repeat:
        repeated = encode_all(
            data, args.batch_size, args.month, constructor, pipeline, model_module, torch
        )
        repeat_hash = logical_hash(repeated, manifest_row)
        if not np.array_equal(embeddings, repeated):
            raise ValueError("Complete repeated extraction was not bitwise deterministic")

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        embeddings=embeddings,
        manifest_row=manifest_row.astype(np.int32),
        sample_id=data["sample_id"],
        region=data["region"],
        class_id=data["class_id"],
        split=data["split"],
        feature_names=np.array([f"P{i:03d}" for i in range(128)]),
    )
    region = data["region"].astype(str)
    norms = np.linalg.norm(embeddings, axis=1)
    report = {
        "manifest_version": "1.0",
        "status": "complete_and_frozen",
        "representation": args.representation,
        "presto": {
            "repository": "nasaharvest/presto",
            "commit": PINNED_COMMIT,
            "checkpoint_sha256": sha256(source / "data/default_model.pt"),
            "embedding_dimensions": 128,
        },
        "input": {
            "canonical_npz_sha256": sha256(input_path),
            "rows": 13429,
            "timesteps": timesteps,
            "month_zero_based": args.month,
            "temporal_semantics": "official consecutive months from supplied zero-based first month",
            "latlon_use": "encoder only; excluded from all downstream feature tables",
        },
        "extraction": {
            "batch_size": args.batch_size,
            "device": str(model_module.device),
            "output_shape": list(embeddings.shape),
            "all_finite": True,
            "logical_sha256_manifest_rows_plus_float32_embeddings": representation_hash,
            "repeat_logical_sha256": repeat_hash,
            "repeat_bitwise_equal": bool(args.verify_repeat),
            "runtime_output": "gitignored outputs/eofm/presto_embeddings_2022.npz",
            "runtime_file_sha256": sha256(output),
        },
        "population": {
            "jiangxi_rows": int((region == "jiangxi").sum()),
            "shanghai_rows": int((region == "shanghai").sum()),
            "missing_embeddings": 0,
        },
        "embedding_norm": {
            "minimum": float(norms.min()),
            "mean": float(norms.mean()),
            "maximum": float(norms.max()),
        },
        "leakage_contract": [
            "sample_id, region, class_id, split, and manifest_row are join metadata only",
            "downstream predictors receive only P000-P127",
            "longitude and latitude are not stored in the embedding artifact",
            "no Shanghai labels or weak-reference metrics were used during extraction",
        ],
    }
    args.report.resolve().write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output), "shape": list(embeddings.shape),
        "logical_sha256": representation_hash,
        "repeat_bitwise_equal": bool(args.verify_repeat),
    }))


if __name__ == "__main__":
    main()

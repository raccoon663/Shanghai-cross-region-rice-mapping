"""Run the frozen official Galileo encoder on validated smoke patches."""
from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "outputs/eofm/galileo_smoke_patches_2022.npz"
DEFAULT_OUTPUT = ROOT / "outputs/eofm/galileo_smoke_embeddings_2022.npz"
DEFAULT_REPORT = ROOT / "results/manifests/eofm_galileo_smoke_embedding_validation.json"
EXPECTED_COMMIT = "0f0b5b95ac81acef4b74cf4686dc877202f4541b"
EXPECTED_ENCODER_SHA256 = "eab6acacf527886a508a555bc24efe4b4a2cf91b162e84cd1c4a1807e942180c"
EXPECTED_CONFIG_SHA256 = "8ec17cb83cb78d987b4b920b9c1c35aa245ec0f80e1abf0c22d6f77d828f431c"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def logical_embedding_sha256(rows: np.ndarray, embeddings: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(rows.astype("<i4", copy=False).tobytes(order="C"))
    digest.update(embeddings.astype("<f4", copy=False).tobytes(order="C"))
    return digest.hexdigest()


def stub_optional_imports() -> None:
    """Stub data-export-only dependencies not needed for encoder inference."""
    for name in ["h5py", "rioxarray", "xarray"]:
        if importlib.util.find_spec(name) is None:
            module = types.ModuleType(name)
            module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
            sys.modules[name] = module
    if importlib.util.find_spec("ee") is None:
        module = types.ModuleType("ee")
        module.__spec__ = importlib.machinery.ModuleSpec("ee", loader=None)
        placeholder = type("EarthEngineImportPlaceholder", (), {})
        for name in ["Geometry", "Image", "ImageCollection"]:
            setattr(module, name, placeholder)
        sys.modules["ee"] = module


def load_normalizing_dict(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {int(key) if key.isdigit() else key: value for key, value in raw.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--galileo-root", type=Path, required=True)
    parser.add_argument("--scope", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    source_root = args.galileo_root.resolve()
    model_folder = source_root / "data/models/nano"
    encoder_path = model_folder / "encoder.pt"
    config_path = model_folder / "config.json"
    normalization_path = source_root / "config/normalization.json"
    if sha256(encoder_path) != EXPECTED_ENCODER_SHA256:
        raise ValueError("Galileo nano encoder checkpoint hash mismatch")
    if sha256(config_path) != EXPECTED_CONFIG_SHA256:
        raise ValueError("Galileo nano model config hash mismatch")

    runtime_deps = ROOT / "outputs/eofm/galileo_runtime_deps_clean"
    if runtime_deps.exists():
        sys.path.insert(0, str(runtime_deps))
    sys.path.insert(0, str(source_root))
    stub_optional_imports()
    from src.data.dataset import (  # pylint: disable=import-outside-toplevel
        Normalizer, SPACE_BANDS, SPACE_TIME_BANDS, SPACE_TIME_BANDS_GROUPS_IDX,
        SPACE_BAND_GROUPS_IDX, STATIC_BANDS, STATIC_BAND_GROUPS_IDX,
        TIME_BANDS, TIME_BAND_GROUPS_IDX,
    )
    from src.galileo import Encoder  # pylint: disable=import-outside-toplevel

    input_path = (args.input or (
        DEFAULT_INPUT if args.scope == "smoke"
        else ROOT / "outputs/eofm/galileo_full_patches_2022.npz"
    )).resolve()
    with np.load(input_path) as data:
        raw = data["space_time_x"].astype(np.float32)
        s_t_m_np = data["space_time_mask"].astype(np.float32)
        months_np = data["months"].astype(np.int64)
        rows = data["manifest_row"].astype(np.int32)
        regions = data["region"]
        band_order = data["band_order"].tolist()
    expected_rows = 1000 if args.scope == "smoke" else 13429
    if raw.shape != (expected_rows, 3, 3, 23, 13) or s_t_m_np.shape != (
        expected_rows, 3, 3, 23, 7
    ):
        raise ValueError("Unexpected frozen smoke tensor shapes")
    if band_order != list(SPACE_TIME_BANDS):
        raise ValueError("Prepared band order differs from official Galileo SPACE_TIME_BANDS")
    expected_months = [2, 2, 3, 3, 3, 4, 4, 5, 6, 7, 7, 7, 7, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10]
    if months_np.tolist() != expected_months:
        raise ValueError("True zero-based anchor months differ from the frozen temporal grid")

    normalizer = Normalizer(std=True, normalizing_dicts=load_normalizing_dict(normalization_path))
    normalized = normalizer(raw).astype(np.float32)
    for group_idx, channel_indices in enumerate(SPACE_TIME_BANDS_GROUPS_IDX.values()):
        masked = s_t_m_np[..., group_idx].astype(bool)
        for channel_idx in channel_indices:
            normalized[..., channel_idx][masked] = 0.0
    if not np.isfinite(normalized).all():
        raise ValueError("Official-normalized Galileo input contains non-finite values")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    model = Encoder.load_from_folder(model_folder).to(device).eval()

    def extract() -> tuple[np.ndarray, float]:
        started = time.perf_counter()
        batches: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(raw), args.batch_size):
                stop = min(start + args.batch_size, len(raw))
                batch = stop - start
                s_t_x = torch.from_numpy(normalized[start:stop]).to(device)
                s_t_m = torch.from_numpy(s_t_m_np[start:stop]).to(device)
                sp_x = torch.zeros((batch, 3, 3, len(SPACE_BANDS)), device=device)
                sp_m = torch.ones((batch, 3, 3, len(SPACE_BAND_GROUPS_IDX)), device=device)
                t_x = torch.zeros((batch, 23, len(TIME_BANDS)), device=device)
                t_m = torch.ones((batch, 23, len(TIME_BAND_GROUPS_IDX)), device=device)
                st_x = torch.zeros((batch, len(STATIC_BANDS)), device=device)
                st_m = torch.ones((batch, len(STATIC_BAND_GROUPS_IDX)), device=device)
                months = torch.from_numpy(np.repeat(months_np[None, :], batch, axis=0)).to(device)
                encoded = model(
                    s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m,
                    months.long(), patch_size=1, input_resolution_m=10,
                )
                pooled = Encoder.average_tokens(*encoded[:-1])
                batches.append(pooled.cpu().numpy().astype(np.float32))
        return np.concatenate(batches), time.perf_counter() - started

    first, first_seconds = extract()
    second, second_seconds = extract()
    if first.shape != (expected_rows, 128):
        raise ValueError(f"Unexpected Galileo embedding shape {first.shape}")
    if not np.isfinite(first).all():
        raise ValueError("Galileo embeddings contain non-finite values")
    if not np.array_equal(first, second):
        raise ValueError("Repeated Galileo extraction was not bitwise deterministic")
    if np.unique(rows).size != len(rows):
        raise ValueError("Duplicate manifest rows in smoke embeddings")

    output = (args.output or (
        DEFAULT_OUTPUT if args.scope == "smoke"
        else ROOT / "outputs/eofm/galileo_full_embeddings_2022.npz"
    )).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, embeddings=first, manifest_row=rows, region=regions)
    report = {
        "manifest_version": "1.0", "status": "passed",
        "upstream": {
            "repository": "nasaharvest/galileo", "commit": EXPECTED_COMMIT,
            "model": "nano", "encoder_checkpoint_sha256": sha256(encoder_path),
            "model_config_sha256": sha256(config_path),
            "normalization_json_sha256": sha256(normalization_path),
            "normalizer": "official src.data.dataset.Normalizer(std=True)",
        },
        "input": {
            "samples": len(rows), "jiangxi": int((regions == "jiangxi").sum()),
            "shanghai": int((regions == "shanghai").sum()), "shape": list(raw.shape),
            "encoder_patch_size": 1, "input_resolution_m": 10,
            "months_zero_based": months_np.tolist(),
            "space_time_band_order": list(SPACE_TIME_BANDS),
            "missing_modalities": ["space_x", "time_x", "static_x"],
            "missing_modality_policy": "zero tensors with official masks set to 1; no fabricated values",
            "normalized_finite": True,
        },
        "embedding": {
            "shape": list(first.shape), "dtype": str(first.dtype), "finite": True,
            "pooling": "official Encoder.average_tokens after Transformer attention",
            "repeated_runs_bitwise_equal": True,
            "logical_sha256": logical_embedding_sha256(rows, first),
            "runtime_npz_sha256": sha256(output),
            "first_run_seconds": first_seconds, "second_run_seconds": second_seconds,
            "device": str(device),
        },
    }
    report_path = (args.report or (
        DEFAULT_REPORT if args.scope == "smoke"
        else ROOT / "results/manifests/eofm_galileo_full_embedding_freeze.json"
    )).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["embedding"]))


if __name__ == "__main__":
    main()

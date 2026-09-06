"""Run a deterministic 50+50 smoke test with the pinned official Presto code."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "outputs/eofm/presto_inputs_2022.npz"
DEFAULT_REPORT = ROOT / "results/manifests/eofm_presto_smoke_test.json"
PINNED_COMMIT = "11e207a668a34336ced1d8e492a1bd5849b96c4a"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def install_import_only_stubs() -> None:
    """Stub optional extraction dependencies unused by construction/inference."""
    ee = types.ModuleType("ee")
    ee.Image = type("Image", (), {})
    ee.Geometry = type("Geometry", (), {"Polygon": type("Polygon", (), {})})
    sys.modules["ee"] = ee
    xr = types.ModuleType("xarray")
    xr.DataArray = type("DataArray", (), {})
    sys.modules["xarray"] = xr
    google = types.ModuleType("google")
    google.__path__ = []
    cloud = types.ModuleType("google.cloud")
    cloud.__path__ = []
    storage = types.ModuleType("google.cloud.storage")
    storage.Client = type("Client", (), {})
    storage.Blob = type("Blob", (), {})
    sys.modules.update({"google": google, "google.cloud": cloud,
                        "google.cloud.storage": storage})
    openmapflow = types.ModuleType("openmapflow")
    openmapflow.__path__ = []
    exporter = types.ModuleType("openmapflow.ee_exporter")
    exporter.create_ee_image = lambda *args, **kwargs: None
    exporter.ee_safe_str = lambda value: str(value)
    exporter.get_ee_task_list = lambda: []
    engineer = types.ModuleType("openmapflow.engineer")
    engineer.calculate_ndvi = lambda value: value
    engineer.load_tif = lambda *args, **kwargs: None
    engineer.remove_bands = lambda value, *args, **kwargs: value
    sys.modules.update({"openmapflow": openmapflow,
                        "openmapflow.ee_exporter": exporter,
                        "openmapflow.engineer": engineer})


def load_official_modules(source: Path, extra_site: Path | None):
    if extra_site is not None:
        sys.path.insert(0, str(extra_site.resolve()))
    install_import_only_stubs()
    package = types.ModuleType("presto")
    package.__path__ = [str(source / "presto")]
    dataops = types.ModuleType("presto.dataops")
    dataops.__path__ = [str(source / "presto" / "dataops")]
    sys.modules["presto"] = package
    sys.modules["presto.dataops"] = dataops
    constructor = importlib.import_module("presto.dataops.utils")
    pipeline = importlib.import_module("presto.dataops.pipelines.s1_s2_era5_srtm")
    model_module = importlib.import_module("presto.presto")
    return constructor, pipeline, model_module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--presto-source", type=Path, required=True)
    parser.add_argument("--extra-site-packages", type=Path)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--month", type=int, default=2,
                        help="Zero-based first month; use 0 for the monthly primary input")
    args = parser.parse_args()

    source = args.presto_source.resolve()
    if PINNED_COMMIT not in source.name:
        raise ValueError(f"Official source directory must identify pinned commit {PINNED_COMMIT}")
    constructor, pipeline, model_module = load_official_modules(source, args.extra_site_packages)
    import torch

    data = np.load(args.input.resolve())
    region = data["region"].astype(str)
    selected = np.concatenate([np.flatnonzero(region == name)[:50]
                               for name in ("jiangxi", "shanghai")])
    s1 = torch.from_numpy(data["s1"][selected].copy())
    s2 = torch.from_numpy(data["s2"][selected].copy())
    s1_valid = torch.from_numpy(data["s1_valid"][selected].astype(bool))
    s2_valid = torch.from_numpy(data["s2_valid"][selected].astype(bool))
    s1[~s1_valid] = 0
    s2[~s2_valid] = 0
    x, mask, dynamic_world = constructor.construct_batch_presto_input(
        s1=s1, s1_bands=data["s1_band_order"].astype(str).tolist(),
        s2=s2, s2_bands=data["s2_band_order"].astype(str).tolist(), normalize=True,
    )
    for band in ("VV", "VH"):
        idx = pipeline.NORMED_BANDS.index(band)
        mask[:, :, idx].masked_fill_(~s1_valid, 1)
        x[:, :, idx].masked_fill_(~s1_valid, 0)
    for band in data["s2_band_order"].astype(str).tolist() + ["NDVI"]:
        idx = pipeline.NORMED_BANDS.index(band)
        mask[:, :, idx].masked_fill_(~s2_valid, 1)
        x[:, :, idx].masked_fill_(~s2_valid, 0)

    timesteps = int(data["s1"].shape[1])
    if not 0 <= args.month < 12:
        raise ValueError("month must be a zero-based value in [0, 11]")
    if timesteps != 23 and args.report.resolve() == DEFAULT_REPORT.resolve():
        raise ValueError("Non-legacy inputs require an explicit, non-legacy --report")
    assert tuple(x.shape) == (100, timesteps, 17)
    assert tuple(mask.shape) == tuple(x.shape)
    assert tuple(dynamic_world.shape) == (100, timesteps)
    assert torch.isfinite(x).all()
    assert torch.equal(dynamic_world, torch.full_like(dynamic_world, 9))
    dynamic_world = dynamic_world.long()
    latlons = torch.from_numpy(data["latlons"][selected].copy())
    device = model_module.device
    x, mask = x.to(device), mask.to(device)
    dynamic_world, latlons = dynamic_world.to(device), latlons.to(device)
    model = model_module.Presto.load_pretrained().to(device).eval()
    with torch.inference_mode():
        first = model.encoder(x, dynamic_world, latlons, mask, month=args.month, eval_task=True)
        second = model.encoder(x, dynamic_world, latlons, mask, month=args.month, eval_task=True)
    if not torch.equal(first, second):
        raise ValueError("Repeated encoder inference was not bitwise deterministic")
    if first.shape[0] != 100 or not torch.isfinite(first).all():
        raise ValueError("Invalid smoke-test embeddings")

    report = {
        "manifest_version": "1.0", "status": "passed",
        "presto": {"repository": "nasaharvest/presto", "commit": PINNED_COMMIT,
                   "official_constructor": "construct_batch_presto_input",
                   "checkpoint_sha256": sha256(source / "data/default_model.pt")},
        "input": {"canonical_npz_sha256": sha256(args.input.resolve()),
                  "rows": 100, "jiangxi_rows": 50, "shanghai_rows": 50,
                  "selection": "first 50 frozen manifest rows within each region",
                  "selected_manifest_rows_sha256": hashlib.sha256(
                      data["manifest_row"][selected].astype("<i4").tobytes()).hexdigest()},
        "contract": {"x_shape": list(x.shape), "mask_shape": list(mask.shape),
                     "dynamic_world_shape": list(dynamic_world.shape),
                     "latlons_shape": list(latlons.shape), "missing_mask_value": 1,
                     "dynamic_world_ignored_value": 9, "month_zero_based": args.month,
                     "timesteps": timesteps,
                     "temporal_semantics": "official consecutive months from supplied zero-based first month"},
        "result": {"embedding_shape": list(first.shape), "all_finite": True,
                   "device": str(device), "repeat_bitwise_equal": True,
                   "embedding_sha256": hashlib.sha256(first.cpu().numpy().tobytes()).hexdigest()},
        "environment_note": "Optional Earth Engine/data-export imports were stubbed; official constructor, normalization, model, and checkpoint were unmodified.",
    }
    args.report.resolve().write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["result"]))


if __name__ == "__main__":
    main()

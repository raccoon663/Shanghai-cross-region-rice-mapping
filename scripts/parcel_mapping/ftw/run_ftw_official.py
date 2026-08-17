from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
REPO = ROOT / "work/field_delineation/ftw/ftw-baselines-fa86d4a3d766b96932521d92a41d43c9e4fc980e"
sys.path.insert(0, str(REPO))

from ftw_tools.inference.inference import run  # noqa: E402


INPUT = ROOT / "work/field_delineation/ftw/input_data/chongming_aoi_v2_label_free_ftw_prue_s2_l2a_8band_2022_v1.tif"
CHECKPOINT = ROOT / "work/field_delineation/ftw/prue_efnetb5_ccby_checkpoint.ckpt"
AOI = ROOT / "data/field_delineation_test_aoi.geojson"
OUT = ROOT / "outputs/field_delineation/ftw_test_aoi"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(INPUT, OUT / "ftw_input_8band.tif")
    shutil.copy2(AOI, OUT / "aoi.geojson")
    common = dict(input=str(INPUT), model=str(CHECKPOINT), resize_factor=2, gpu=0, patch_size=256, batch_size=2, num_workers=0, padding=16, overwrite=True, mps_mode=False, nan_fill_value=0.0)
    started = time.perf_counter()
    run(out=str(OUT / "ftw_scores_native.tif"), save_scores=True, compute_consensus=False, **common)
    scores_seconds = time.perf_counter() - started
    started = time.perf_counter()
    run(out=str(OUT / "ftw_semantic_native.tif"), save_scores=False, compute_consensus=True, **common)
    semantic_seconds = time.perf_counter() - started
    execution = {"official_function": "ftw_tools.inference.inference.run", "input": str(INPUT.relative_to(ROOT)), "checkpoint": str(CHECKPOINT.relative_to(ROOT)), "parameters": {"resize_factor": 2, "gpu": 0, "patch_size": 256, "batch_size": 2, "num_workers": 0, "padding": 16, "stride": 224, "save_scores_pass": True, "semantic_consensus_pass": True, "nan_fill_value": 0.0}, "elapsed_seconds": {"scores_pass": scores_seconds, "semantic_consensus_pass": semantic_seconds}}
    (OUT / "official_execution.json").write_text(json.dumps(execution, indent=2), encoding="utf-8")
    print(json.dumps(execution, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
REPO = ROOT / "work/field_delineation/ftw/ftw-baselines-fa86d4a3d766b96932521d92a41d43c9e4fc980e"
sys.path.insert(0, str(REPO))

from ftw_tools.postprocess.polygonize import polygonize  # noqa: E402


OUT = ROOT / "outputs/field_delineation/ftw_test_aoi"
SEMANTIC = OUT / "ftw_semantic_native.tif"


def main() -> None:
    common = dict(input=str(SEMANTIC), simplify=True, max_size=None, overwrite=True, close_interiors=False, polygonization_stride=2048, softmax_threshold=None, merge_adjacent=None, erode_dilate=0, dilate_erode=0, erode_dilate_raster=0, dilate_erode_raster=0, thin_boundaries=False)
    polygonize(out=str(OUT / "ftw_polygons_raw_default.gpkg"), min_size=500, **common)
    polygonize(out=str(OUT / "ftw_polygons_clean.gpkg"), min_size=2500, **common)
    metadata = {"official_function": "ftw_tools.postprocess.polygonize.polygonize", "native_semantic_input": str(SEMANTIC.relative_to(ROOT)), "class_contract": {"0": "background/neither", "1": "field interior", "2": "boundary"}, "raw_polygonization": {"min_size_m2": 500, "simplify_m": 1, "other_morphology": False}, "comparison_clean_polygonization": {"min_size_m2": 2500, "simplify_m": 1, "other_morphology": False}, "threshold_tuning": False, "rice_labels_used": False}
    (OUT / "polygonization_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

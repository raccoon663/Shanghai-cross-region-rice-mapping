from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.features import geometry_mask, rasterize
from rasterio.windows import from_bounds


ROOT = Path(__file__).resolve().parents[3]
REPO = ROOT / "work/field_delineation/ftw/ftw-baselines-fa86d4a3d766b96932521d92a41d43c9e4fc980e"
sys.path.insert(0, str(REPO))
from ftw_tools.inference.inference import run  # noqa: E402
from ftw_tools.postprocess.polygonize import polygonize  # noqa: E402


INPUT = ROOT / "work/field_delineation/ftw/input_data_aoi2/chongming_phase3b_aoi2_ftw_prue_s2_l2a_8band_2022_v1.tif"
RGB = ROOT / "work/field_delineation/ftw/input_data_aoi2/chongming_phase3b_aoi2_s2_l2a_rgb_2022_v1.tif"
DW = ROOT / "work/field_delineation/ftw/input_data_aoi2/chongming_phase3b_aoi2_dynamic_world_mode_2022_v1.tif"
INPUT_META = ROOT / "work/field_delineation/ftw/input_data_aoi2/input_metadata.json"
CHECKPOINT = ROOT / "work/field_delineation/ftw/prue_efnetb5_ccby_checkpoint.ckpt"
AOI = ROOT / "outputs/field_delineation/ftw_test_aoi_2/aoi_2.geojson"
OUT = ROOT / "outputs/field_delineation/ftw_test_aoi_2"
NAMES = {0: "water", 1: "trees", 2: "grass", 3: "flooded_vegetation", 4: "crops", 5: "shrub_and_scrub", 6: "built", 7: "bare", 8: "snow_and_ice"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def landcover_fractions(geom, ds, array: np.ndarray) -> dict[str, float]:
    window = from_bounds(*geom.bounds, transform=ds.transform).round_offsets().round_lengths().intersection(rasterio.windows.Window(0, 0, ds.width, ds.height))
    subset = array[int(window.row_off):int(window.row_off + window.height), int(window.col_off):int(window.col_off + window.width)]
    inside = geometry_mask([geom], out_shape=subset.shape, transform=ds.window_transform(window), invert=True)
    values = subset[inside]
    return {f"dw_{name}_fraction": float((values == code).sum() / max(len(values), 1)) for code, name in NAMES.items()}


def overlap_stats(gdf: gpd.GeoDataFrame) -> dict:
    pairs, area = 0, 0.0
    for i in range(len(gdf)):
        for j in gdf.sindex.query(gdf.geometry.iloc[i], predicate="intersects"):
            if j <= i:
                continue
            value = gdf.geometry.iloc[i].intersection(gdf.geometry.iloc[j]).area
            if value > 0:
                pairs += 1; area += float(value)
    union_area = float(gdf.geometry.union_all().area) if len(gdf) else 0.0
    return {"pair_count": pairs, "overlap_area_m2_sum": area, "rate_vs_union": area / union_area if union_area else 0.0}


def area_stats(gdf: gpd.GeoDataFrame) -> dict:
    a = gdf.area.to_numpy()
    return {"count": len(gdf), "min_m2": float(a.min()), "p10_m2": float(np.quantile(a, .1)), "median_m2": float(np.median(a)), "mean_m2": float(a.mean()), "p90_m2": float(np.quantile(a, .9)), "max_m2": float(a.max()), "over_20ha_count": int((a > 200000).sum()), "over_50ha_count": int((a > 500000).sum())}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(INPUT, OUT / "ftw_input_8band.tif")
    shutil.copy2(RGB, OUT / "rgb_input.tif")
    common = dict(input=str(INPUT), model=str(CHECKPOINT), resize_factor=2, gpu=0, patch_size=256, batch_size=2, num_workers=0, padding=16, overwrite=True, mps_mode=False, nan_fill_value=0.0)
    started = time.perf_counter(); run(out=str(OUT / "ftw_scores_native.tif"), save_scores=True, compute_consensus=False, **common); scores_seconds = time.perf_counter() - started
    started = time.perf_counter(); run(out=str(OUT / "ftw_semantic_native.tif"), save_scores=False, compute_consensus=True, **common); semantic_seconds = time.perf_counter() - started
    execution = {"official_function": "ftw_tools.inference.inference.run", "input": str(INPUT.relative_to(ROOT)), "checkpoint": str(CHECKPOINT.relative_to(ROOT)), "parameters": {"resize_factor": 2, "gpu": 0, "patch_size": 256, "batch_size": 2, "num_workers": 0, "padding": 16, "stride": 224, "save_scores_pass": True, "semantic_consensus_pass": True, "nan_fill_value": 0.0}, "elapsed_seconds": {"scores_pass": scores_seconds, "semantic_consensus_pass": semantic_seconds}}
    (OUT / "official_execution.json").write_text(json.dumps(execution, indent=2), encoding="utf-8")
    common_poly = dict(input=str(OUT / "ftw_semantic_native.tif"), simplify=True, max_size=None, overwrite=True, close_interiors=False, polygonization_stride=2048, softmax_threshold=None, merge_adjacent=None, erode_dilate=0, dilate_erode=0, erode_dilate_raster=0, dilate_erode_raster=0, thin_boundaries=False)
    polygonize(out=str(OUT / "ftw_polygons_raw_default.gpkg"), min_size=500, **common_poly)
    polygonize(out=str(OUT / "ftw_polygons_clean.gpkg"), min_size=2500, **common_poly)
    polygonization = {"official_function": "ftw_tools.postprocess.polygonize.polygonize", "native_semantic_input": str((OUT / "ftw_semantic_native.tif").relative_to(ROOT)), "class_contract": {"0": "background/neither", "1": "field interior", "2": "boundary"}, "raw_polygonization": {"min_size_m2": 500, "simplify_m": 1, "other_morphology": False}, "comparison_clean_polygonization": {"min_size_m2": 2500, "simplify_m": 1, "other_morphology": False}, "threshold_tuning": False, "rice_labels_used": False}
    (OUT / "polygonization_metadata.json").write_text(json.dumps(polygonization, indent=2), encoding="utf-8")

    raw = gpd.read_file(OUT / "ftw_polygons_raw_default.gpkg")
    ftw = gpd.read_file(OUT / "ftw_polygons_clean.gpkg")
    with rasterio.open(DW) as dw_ds:
        dw = dw_ds.read(1)
        for idx, geom in enumerate(ftw.geometry):
            for key, value in landcover_fractions(geom, dw_ds, dw).items():
                ftw.loc[idx, key] = value
    ftw["field_id"] = [f"FTW2_{i:05d}" for i in range(1, len(ftw) + 1)]
    ftw["area_m2"] = ftw.area; ftw["area_ha"] = ftw.area / 10000
    ftw["source_model"] = "FTW_PRUE_EFNET_B5_CCBY_v3.1"
    dw_cols = [f"dw_{name}_fraction" for name in NAMES.values()]
    ftw["dw_dominant"] = ftw[dw_cols].idxmax(axis=1).str.replace("dw_", "", regex=False).str.replace("_fraction", "", regex=False)
    ftw["noncrop_dominant_qa"] = (ftw.dw_crops_fraction < .2) & (ftw[["dw_water_fraction", "dw_trees_fraction", "dw_built_fraction"]].max(axis=1) >= .5)
    (OUT / "ftw_polygons_clean.gpkg").unlink()
    ftw.to_file(OUT / "ftw_polygons_clean.gpkg", layer="fields", driver="GPKG")
    ftw.to_file(OUT / "ftw_polygons_clean.geojson", driver="GeoJSON")

    with rasterio.open(OUT / "ftw_semantic_native.tif") as ds:
        semantic = ds.read(1); profile = ds.profile.copy(); transform = ds.transform; bounds = ds.bounds
    instance = rasterize(((geom, i) for i, geom in enumerate(ftw.geometry, 1)), out_shape=semantic.shape, transform=transform, fill=0, dtype="uint32")
    field_mask = (semantic == 1).astype("uint8")
    p = profile.copy(); p.update(count=1, dtype="uint32", nodata=None, compress="deflate")
    with rasterio.open(OUT / "ftw_field_instance.tif", "w", **p) as dst: dst.write(instance, 1)
    p = profile.copy(); p.update(count=1, dtype="uint8", nodata=255, compress="deflate")
    with rasterio.open(OUT / "ftw_field_mask.tif", "w", **p) as dst: dst.write(field_mask, 1)
    with rasterio.open(RGB) as ds: arr = ds.read([1, 2, 3]).astype("float32")
    valid = np.all(arr > 0, axis=0); display = np.zeros((500, 500, 3), dtype="uint8")
    for b in range(3):
        lo, hi = np.percentile(arr[b][valid], [1, 99]); display[..., b] = np.clip((arr[b] - lo) * 255 / max(hi - lo, 1), 0, 255).astype("uint8")
    extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)
    fig, ax = plt.subplots(figsize=(8, 8)); ax.imshow(display, extent=extent); ftw.boundary.plot(ax=ax, color="#00e5ff", linewidth=.7); ftw[ftw.noncrop_dominant_qa].boundary.plot(ax=ax, color="#ff3d00", linewidth=1.1); ax.set_title("Phase 3B AOI 2: FTW fields (cyan); non-crop-dominant QA (orange)"); ax.ticklabel_format(style="plain", useOffset=False); fig.tight_layout(); fig.savefig(OUT / "ftw_overlay_preview.png", dpi=220, bbox_inches="tight"); plt.close(fig)
    disagreement_path = OUT / "ftw_semantic_native_disagreements.tif"
    with rasterio.open(disagreement_path) as ds: disagreement = ds.read(1)
    qa = {"model": "FTW_PRUE_EFNET_B5_CCBY", "checkpoint_worked": True, "native_semantic_class_pixels": {str(k): int(v) for k, v in zip(*np.unique(semantic, return_counts=True))}, "raw_default_polygon_count_min500m2": len(raw), "clean_polygon_count_min2500m2": len(ftw), "tiny_raw_500_to_under2500_count": int((raw.area < 2500).sum()), "tiny_raw_fraction": float((raw.area < 2500).mean()), "invalid_clean_geometries": int((~ftw.is_valid).sum()), "overlap": overlap_stats(ftw), "area": area_stats(ftw), "cross_tile_disagreement_pixels": int((disagreement > 0).sum()), "cross_tile_disagreement_fraction_of_aoi": float((disagreement > 0).mean()), "dynamic_world_dominant_counts": ftw.dw_dominant.value_counts().to_dict(), "noncrop_dominant_count": int(ftw.noncrop_dominant_qa.sum())}
    (OUT / "qa_summary.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    input_meta = json.loads(INPUT_META.read_text(encoding="utf-8"))
    import kornia, segmentation_models_pytorch, torch, torchgeo
    manifest = {"phase": "3B", "model_repo": "https://github.com/fieldsoftheworld/ftw-baselines", "model_commit": "fa86d4a3d766b96932521d92a41d43c9e4fc980e", "checkpoint_release": "v3.1", "checkpoint": CHECKPOINT.name, "checkpoint_sha256": sha256(CHECKPOINT), "checkpoint_license": "CC-BY-4.0", "architecture": "3-class U-Net with EfficientNet-B5 encoder; 8 input channels", "aoi_sha256": sha256(AOI), "input_metadata": input_meta, "official_inference": execution, "official_polygonization": polygonization, "software": {"python": sys.version, "torch": torch.__version__, "cuda_runtime": torch.version.cuda, "segmentation_models_pytorch": segmentation_models_pytorch.__version__, "kornia": kornia.__version__, "torchgeo": torchgeo.__version__, "rasterio": rasterio.__version__, "geopandas": gpd.__version__, "numpy": np.__version__}, "runner_sha256": sha256(Path(__file__)), "output_sha256": {p.name: sha256(p) for p in OUT.iterdir() if p.is_file() and p.name != "run_manifest.json"}, "rice_labels_used": False, "rice_prediction_values_used_for_aoi_selection": False, "full_scene_inference": False}
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"execution": execution, "qa": qa}, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
from shapely.geometry import box


ROOT = Path(__file__).resolve().parents[3]
PROB = ROOT / "outputs/experiments/rf_baseline_m2_v1/shanghai_source_only_probability_s1_s2_fusion.tif"
CHECKPOINT = ROOT / "work/field_delineation/ftw/prue_efnetb5_ccby_checkpoint.ckpt"
REPO_ARCHIVE = ROOT / "work/field_delineation/ftw/ftw-baselines-fa86d4a3d766b96932521d92a41d43c9e4fc980e.zip"
OUT = ROOT / "outputs/phase4_chongming_staged"
TILES = OUT / "tiles"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); TILES.mkdir(parents=True, exist_ok=True)
    with rasterio.open(PROB) as ds:
        bounds = ds.bounds; crs = ds.crs
        profile = {"crs": str(ds.crs), "bounds": list(ds.bounds), "transform": list(ds.transform)[:6], "shape": list(ds.shape), "resolution": list(ds.res), "nodata": ds.nodata, "dtype": ds.dtypes[0]}
    core_size, halo = 5000, 320
    xs = list(range(int(bounds.left), int(bounds.right), core_size))
    y_tops = list(range(int(bounds.top), int(bounds.bottom), -core_size))
    rows = []
    for row, north in enumerate(y_tops):
        south = max(int(bounds.bottom), north - core_size)
        for col, west in enumerate(xs):
            east = min(int(bounds.right), west + core_size)
            tile_id = f"r{row:02d}c{col:02d}"
            rows.append({
                "tile_id": tile_id, "row": row, "col": col,
                "core_west": west, "core_south": south, "core_east": east, "core_north": north,
                "halo_west": max(int(bounds.left), west - halo), "halo_south": max(int(bounds.bottom), south - halo),
                "halo_east": min(int(bounds.right), east + halo), "halo_north": min(int(bounds.top), north + halo),
                "core_width_m": east-west, "core_height_m": north-south,
                "status": "frozen_pending", "source_imagery_hashes": "pending", "inference_status": "pending",
                "output_counts": "pending", "qa_metrics": "pending",
                "geometry": box(west, south, east, north),
            })
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)
    gdf.to_file(TILES / "tile_index.geojson", driver="GeoJSON")
    gdf.drop(columns="geometry").to_csv(TILES / "tile_index.csv", index=False)
    thresholds = {
        "basis": "predeclared before Phase 4 tile inference; reference maxima from frozen AOI 1/AOI 2 multiplied by 1.5 where meaningful",
        "ftw_retained_fraction_warning_gt": 0.68116,
        "built_dominant_parcel_fraction_warning_gt": 0.13637,
        "tree_dominant_parcel_fraction_warning_gt": 0.09091,
        "water_dominant_parcel_fraction_warning_gt": 0.01040,
        "over_20ha_parcel_fraction_warning_gt": 0.09741,
        "over_50ha_warning": "count >= 2 OR parcel fraction > 0.03247",
        "tile_disagreement_parcel_fraction_warning_gt": 0.27296,
        "summary_disagreement_parcel_fraction_warning_gt": 0.02080,
        "cross_tile_reconciliation_warning": "post-run robust outlier: count > median + 3 * max(MAD, 1); declared before results",
        "notes": "warnings trigger review only; no tile or parcel is automatically deleted",
    }
    frozen_at = datetime.now().astimezone().isoformat(timespec="seconds")
    manifest = {
        "phase": "4", "name": "staged central/eastern Chongming prototype parcel-mapping pilot", "frozen_at": frozen_at,
        "deployment_extent_source": str(PROB.relative_to(ROOT)), "deployment_extent_source_sha256": sha256(PROB), "deployment_grid": profile,
        "scope_language": "existing central/eastern Chongming prototype; not all Shanghai",
        "model": {"name": "FTW PRUE EfficientNet-B5 CC-BY", "repo_commit": "fa86d4a3d766b96932521d92a41d43c9e4fc980e", "repo_archive_sha256": sha256(REPO_ARCHIVE), "checkpoint": str(CHECKPOINT.relative_to(ROOT)), "checkpoint_sha256": sha256(CHECKPOINT), "license": "CC-BY-4.0", "architecture": "3-class U-Net with EfficientNet-B5 encoder; 8 input channels"},
        "imagery": {"collection": "COPERNICUS/S2_SR_HARMONIZED", "product": "Sentinel-2 L2A surface reflectance", "t1": ["2022-04-01", "2022-06-30"], "t2": ["2022-08-15", "2022-11-01"], "bands_per_window": ["B04", "B03", "B02", "B08"], "cloud_scene_max_percent": 60, "cloud_mask": "SCL excludes 0,1,3,7,8,9,10,11", "composite": "per-window per-band median", "resolution_m": 10, "nodata": 0, "dtype": "uint16"},
        "ftw_inference": {"resize_factor": 2, "gpu": 0, "patch_size": 256, "batch_size": 2, "num_workers": 0, "padding": 16, "stride": 224, "save_scores_pass": True, "semantic_consensus_pass": True, "nan_fill_value": 0.0},
        "polygonization": {"official_function": "ftw_tools.postprocess.polygonize.polygonize", "raw_min_size_m2": 500, "accepted_min_size_m2": 2500, "simplify_m": 1, "morphology": False},
        "tiling": {"core_size_m": core_size, "halo_m": halo, "tile_count": len(gdf), "tile_index": str((TILES / "tile_index.geojson").relative_to(ROOT)), "deterministic_order": "north-to-south rows, west-to-east columns", "independently_resumable": True},
        "cross_tile_reconciliation": {"algorithm": "representative-point core ownership of full halo polygons; residual inter-tile duplicates with intersection/min-area >=0.50 are resolved by retaining the version whose representative point is farther from its source halo edge; no dissolve for low-overlap adjacent parcels", "cross_tile_flag": "true for polygons crossing a core edge, participating in residual duplicate resolution, or touching a source halo edge", "deployment_edge_flag": "geometry touches retained probability extent"},
        "rice": {"classifier": "existing retained source-only Jiangxi-to-Shanghai probability raster", "threshold": 0.50, "high_probability_diagnostic": 0.80, "retraining": False, "target_tuning": False},
        "dynamic_world": {"collection": "GOOGLE/DYNAMICWORLD/V1", "date_window": ["2022-04-01", "2022-11-01"], "composite": "temporal label mode", "rule": "inside accepted FTW mask AND at least 2 of 4 underlying 10m pixels are crops(4) or flooded_vegetation(3)"},
        "parcel_qa": {"large_parcel_flag": "area > 20 ha", "very_large_parcel_flag": "area > 50 ha", "low_pixel_count_flag": "n_valid_pixels < 4", "built_dominant_flag": "Dynamic World built fraction >= 0.5", "tree_dominant_flag": "Dynamic World tree fraction >= 0.5", "water_dominant_flag": "Dynamic World water fraction >= 0.5", "tile_disagreement_flag": "official disagreement covers >=5% of parcel", "cross_tile_reconciliation_flag": "defined by frozen reconciliation algorithm", "deployment_edge_flag": "touches prototype boundary", "summary_disagreement_flag": "mean/median/positive-fraction threshold classes are not unanimous", "diagnostic_class": "Uncertain / QA-risk if any QA flag; otherwise unanimous Rice/Non-rice"},
        "tile_warning_thresholds": thresholds,
        "claim_limits": {"target_rice_truth_used": False, "accuracy_claims_allowed": False, "removed_pixels_called_false_positives": False, "production_ready_claim_allowed": False},
        "stop_point": "complete staged prototype, write Phase 4 report, stop before all-Shanghai extension",
    }
    (OUT / "phase4_freeze_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"frozen_at": frozen_at, "tile_count": len(gdf), "extent": list(bounds), "manifest_sha256": sha256(OUT / "phase4_freeze_manifest.json")}, indent=2))


if __name__ == "__main__":
    main()

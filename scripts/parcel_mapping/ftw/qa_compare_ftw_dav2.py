from __future__ import annotations

import hashlib
import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.features import geometry_mask, rasterize
from rasterio.windows import from_bounds


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs/field_delineation/ftw_test_aoi"
DAV2 = ROOT / "outputs/field_delineation/test_aoi"
DW = ROOT / "work/field_delineation/input_data/chongming_field_test_aoi_v2_label_free_dynamic_world_mode_2022.tif"
RGB = DAV2 / "rgb_input.tif"
PROB = ROOT / "outputs/experiments/rf_baseline_m2_v1/shanghai_source_only_probability_s1_s2_fusion.tif"
CHECKPOINT = ROOT / "work/field_delineation/ftw/prue_efnetb5_ccby_checkpoint.ckpt"
INPUT = ROOT / "work/field_delineation/ftw/input_data/chongming_aoi_v2_label_free_ftw_prue_s2_l2a_8band_2022_v1.tif"
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
                pairs += 1
                area += float(value)
    union_area = float(gdf.geometry.union_all().area) if len(gdf) else 0
    return {"pair_count": pairs, "overlap_area_m2_sum": area, "rate_vs_union": area / union_area if union_area else 0}


def area_stats(gdf: gpd.GeoDataFrame) -> dict:
    a = gdf.area.to_numpy()
    return {"count": len(gdf), "min_m2": float(a.min()), "p10_m2": float(np.quantile(a, .1)), "median_m2": float(np.median(a)), "mean_m2": float(a.mean()), "p90_m2": float(np.quantile(a, .9)), "max_m2": float(a.max()), "over_20ha_count": int((a > 200000).sum()), "over_50ha_count": int((a > 500000).sum())}


def main() -> None:
    raw = gpd.read_file(OUT / "ftw_polygons_raw_default.gpkg")
    ftw = gpd.read_file(OUT / "ftw_polygons_clean.gpkg")
    dav2 = gpd.read_file(DAV2 / "field_polygons.gpkg")
    with rasterio.open(DW) as dw_ds:
        dw = dw_ds.read(1)
        for idx, geom in enumerate(ftw.geometry):
            for key, value in landcover_fractions(geom, dw_ds, dw).items():
                ftw.loc[idx, key] = value
    ftw["field_id"] = [f"FTW{i:05d}" for i in range(1, len(ftw) + 1)]
    ftw["area_m2"] = ftw.area
    ftw["area_ha"] = ftw.area / 10000
    ftw["source_model"] = "FTW_PRUE_EFNET_B5_CCBY_v3.1"
    dw_cols = [f"dw_{name}_fraction" for name in NAMES.values()]
    ftw["dw_dominant"] = ftw[dw_cols].idxmax(axis=1).str.replace("dw_", "", regex=False).str.replace("_fraction", "", regex=False)
    ftw["noncrop_dominant_qa"] = (ftw.dw_crops_fraction < .2) & (ftw[["dw_water_fraction", "dw_trees_fraction", "dw_built_fraction"]].max(axis=1) >= .5)
    # Replace the official polygonizer's default layer with one enriched,
    # unambiguous cleaned layer; the native semantic raster remains preserved.
    (OUT / "ftw_polygons_clean.gpkg").unlink()
    ftw.to_file(OUT / "ftw_polygons_clean.gpkg", layer="fields", driver="GPKG")
    ftw.to_file(OUT / "ftw_polygons_clean.geojson", driver="GeoJSON")

    with rasterio.open(OUT / "ftw_semantic_native.tif") as sem_ds:
        semantic = sem_ds.read(1)
        profile = sem_ds.profile.copy()
        transform, bounds = sem_ds.transform, sem_ds.bounds
    instance = rasterize(((geom, i) for i, geom in enumerate(ftw.geometry, 1)), out_shape=semantic.shape, transform=transform, fill=0, dtype="uint32")
    field_mask = (semantic == 1).astype("uint8")
    p = profile.copy(); p.update(count=1, dtype="uint32", nodata=None, compress="deflate")
    with rasterio.open(OUT / "ftw_field_instance.tif", "w", **p) as dst: dst.write(instance, 1)
    p = profile.copy(); p.update(count=1, dtype="uint8", nodata=255, compress="deflate")
    with rasterio.open(OUT / "ftw_field_mask.tif", "w", **p) as dst: dst.write(field_mask, 1)

    with rasterio.open(RGB) as rgb_ds:
        arr = rgb_ds.read([1, 2, 3]).astype("float32")
    valid = np.all(arr > 0, axis=0)
    display = np.zeros((arr.shape[1], arr.shape[2], 3), dtype="uint8")
    for b in range(3):
        lo, hi = np.percentile(arr[b][valid], [1, 99]); display[..., b] = np.clip((arr[b] - lo) * 255 / max(hi - lo, 1), 0, 255).astype("uint8")
    extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5), sharex=True, sharey=True)
    for ax, title, polygons, color in [(axes[0], "DAv2 cleaned instances (n=116)", dav2, "#ffeb3b"), (axes[1], f"FTW PRUE B5 CC-BY (n={len(ftw)})", ftw, "#00e5ff")]:
        ax.imshow(display, extent=extent); polygons.boundary.plot(ax=ax, color=color, linewidth=.65); ax.set_title(title); ax.ticklabel_format(style="plain", useOffset=False); ax.set_xlabel("Easting, EPSG:32651")
    axes[0].set_ylabel("Northing, EPSG:32651")
    fig.suptitle("Same label-free 5 x 5 km Chongming AOI; no manual feature deletion")
    fig.tight_layout(); fig.savefig(OUT / "ftw_vs_dav2_side_by_side.png", dpi=220, bbox_inches="tight"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 8)); ax.imshow(display, extent=extent); ftw.boundary.plot(ax=ax, color="#00e5ff", linewidth=.7); flagged=ftw[ftw.noncrop_dominant_qa]; flagged.boundary.plot(ax=ax, color="#ff3d00", linewidth=1.1); ax.set_title("FTW PRUE fields (cyan); non-crop-dominant QA (orange)"); ax.ticklabel_format(style="plain", useOffset=False); fig.tight_layout(); fig.savefig(OUT / "ftw_overlay_preview.png", dpi=220, bbox_inches="tight"); plt.close(fig)

    with rasterio.open(OUT / "ftw_semantic_native_disagreements.tif") as ds: disagreement = ds.read(1)
    with rasterio.open(PROB) as prob_ds:
        alignment = {"probability_crs": str(prob_ds.crs), "probability_extent": list(prob_ds.bounds), "probability_transform": list(prob_ds.transform)[:6], "probability_resolution": list(prob_ds.res), "probability_nodata": prob_ds.nodata, "ftw_crs": str(profile["crs"]), "ftw_extent": list(bounds), "ftw_transform": list(transform)[:6], "ftw_resolution": [10,10], "crs_match": prob_ds.crs == profile["crs"], "aoi_within_probability_extent": prob_ds.bounds.left <= bounds.left and prob_ds.bounds.right >= bounds.right and prob_ds.bounds.bottom <= bounds.bottom and prob_ds.bounds.top >= bounds.top, "origin_aligned_to_20m_grid": (bounds.left-prob_ds.bounds.left)%20 == 0 and (prob_ds.bounds.top-bounds.top)%20 == 0, "probability_modified": False}
    ftw_capture = {name: float(((dw == code) & (field_mask == 1)).sum() / max((dw == code).sum(), 1)) for code, name in NAMES.items()}
    with rasterio.open(DAV2 / "field_mask.tif") as ds: dav2_mask = ds.read(1) == 1
    dav2_capture = {name: float(((dw == code) & dav2_mask).sum() / max((dw == code).sum(), 1)) for code, name in NAMES.items()}
    qa = {"model": "FTW_PRUE_EFNET_B5_CCBY", "checkpoint_worked": True, "native_semantic_class_pixels": {str(k): int(v) for k,v in zip(*np.unique(semantic, return_counts=True))}, "raw_default_polygon_count_min500m2": len(raw), "clean_polygon_count_min2500m2": len(ftw), "tiny_raw_500_to_under2500_count": int((raw.area < 2500).sum()), "tiny_raw_fraction": float((raw.area < 2500).mean()), "invalid_clean_geometries": int((~ftw.is_valid).sum()), "overlap": overlap_stats(ftw), "area": area_stats(ftw), "cross_tile_disagreement_pixels": int(disagreement.sum()), "cross_tile_disagreement_fraction_of_aoi": float((disagreement > 0).mean()), "official_overlap_consistency": {"agreements": 54463, "total_overlap_pixels": 68096, "fraction": 54463/68096}, "dynamic_world_dominant_counts": ftw.dw_dominant.value_counts().to_dict(), "noncrop_dominant_count": int(ftw.noncrop_dominant_qa.sum()), "ftw_landcover_capture": ftw_capture, "dav2_landcover_capture_same_aoi": dav2_capture, "dav2_area_for_comparison": area_stats(dav2), "alignment": alignment}
    (OUT / "qa_summary.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    import kornia, segmentation_models_pytorch, torch, torchgeo
    manifest = {"model_repo": "https://github.com/fieldsoftheworld/ftw-baselines", "model_commit": "fa86d4a3d766b96932521d92a41d43c9e4fc980e", "repo_archive_sha256": "692f9b55988cdd30b2f441d226939bc8e1ba471cc065e9d4e8dccf22ad9dc0f1", "checkpoint_release": "v3.1", "checkpoint": CHECKPOINT.name, "checkpoint_sha256": sha256(CHECKPOINT), "checkpoint_license": "CC-BY-4.0", "architecture": "3-class U-Net with EfficientNet-B5 encoder; 8 input channels", "input_sha256": sha256(INPUT), "input_contract": ["B04_t1","B03_t1","B02_t1","B08_t1","B04_t2","B03_t2","B02_t2","B08_t2"], "official_inference": json.loads((OUT / "official_execution.json").read_text()), "official_polygonization": json.loads((OUT / "polygonization_metadata.json").read_text()), "software": {"python": __import__("sys").version, "torch": torch.__version__, "cuda_runtime": torch.version.cuda, "segmentation_models_pytorch": segmentation_models_pytorch.__version__, "kornia": kornia.__version__, "torchgeo": torchgeo.__version__, "rasterio": rasterio.__version__, "geopandas": gpd.__version__, "numpy": np.__version__}, "runner_sha256": {"official_inference_wrapper": sha256(ROOT / "work/field_delineation/ftw/run_ftw_official.py"), "polygonization_wrapper": sha256(ROOT / "work/field_delineation/ftw/polygonize_ftw.py"), "qa_comparison": sha256(Path(__file__).resolve())}, "output_sha256": {p.name: sha256(p) for p in OUT.iterdir() if p.is_file() and p.name != "run_manifest.json"}, "rice_labels_used": False, "rice_prediction_values_used": False, "full_scene_inference": False}
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()

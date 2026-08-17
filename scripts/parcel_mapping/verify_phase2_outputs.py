from __future__ import annotations

import hashlib
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import box


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/field_delineation/test_aoi"
PROB = ROOT / "outputs/experiments/rf_baseline_m2_v1/shanghai_source_only_probability_s1_s2_fusion.tif"
DW = ROOT / "work/field_delineation/input_data/chongming_field_test_aoi_v2_label_free_dynamic_world_mode_2022.tif"
NAMES = ["water", "trees", "grass", "flooded_vegetation", "crops", "shrub_and_scrub", "built", "bare", "snow_and_ice"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def raster_profile(path: Path) -> dict:
    with rasterio.open(path) as ds:
        return {"crs": str(ds.crs), "extent": list(ds.bounds), "transform": list(ds.transform)[:6], "resolution": list(ds.res), "shape": list(ds.shape), "dtype": ds.dtypes[0], "nodata": ds.nodata, "count": ds.count}


def main() -> None:
    required = ["aoi.geojson", "rgb_input.tif", "field_polygons_raw.geojson", "field_polygons.gpkg", "field_instance.tif", "field_mask.tif", "field_overlay_preview.png", "run_metadata.json", "qa_summary.json"]
    gdf = gpd.read_file(OUT / "field_polygons.gpkg")
    raw = gpd.read_file(OUT / "field_polygons_raw.geojson")
    with rasterio.open(OUT / "field_mask.tif") as mask_ds:
        mask = mask_ds.read(1) == 1
        mask_bounds, mask_crs = box(*mask_ds.bounds), mask_ds.crs
        mask_transform, mask_shape = mask_ds.transform, mask_ds.shape
    with rasterio.open(OUT / "field_instance.tif") as inst_ds:
        instance = inst_ds.read(1)
        derived_alignment = {"same_crs": inst_ds.crs == mask_crs, "same_transform": inst_ds.transform == mask_transform, "same_shape": inst_ds.shape == mask_shape}
    with rasterio.open(DW) as dw_ds:
        dw = dw_ds.read(1)
        derived_alignment["dynamic_world_same_grid"] = dw_ds.crs == mask_crs and dw_ds.transform == mask_transform and dw_ds.shape == mask_shape
    with rasterio.open(PROB) as prob_ds:
        probability = raster_profile(PROB)
        probability_overlap = {"crs_match": prob_ds.crs == mask_crs, "aoi_fully_within_extent": box(*prob_ds.bounds).covers(mask_bounds), "vector_union_fully_within_extent": box(*prob_ds.bounds).covers(gdf.geometry.union_all()), "aoi_origin_on_20m_grid": ((mask_bounds.bounds[0] - prob_ds.bounds.left) % 20 == 0 and (prob_ds.bounds.top - mask_bounds.bounds[3]) % 20 == 0)}

    cols = [f"dw_{name}_fraction" for name in NAMES]
    dominant = gdf[cols].idxmax(axis=1).str.replace("dw_", "", regex=False).str.replace("_fraction", "", regex=False)
    examples = {}
    for name in ["water", "built", "trees", "crops"]:
        subset = gdf.loc[dominant == name, ["field_id", "area_m2", "confidence", f"dw_{name}_fraction", "geometry"]].copy()
        subset["bounds"] = subset.geometry.apply(lambda geom: list(geom.bounds))
        examples[name] = subset.drop(columns="geometry").sort_values(f"dw_{name}_fraction", ascending=False).head(3).to_dict("records")
    capture = {NAMES[code]: {"pixels_inside_field_mask": int(((dw == code) & mask).sum()), "total_class_pixels": int((dw == code).sum()), "capture_fraction": float(((dw == code) & mask).sum() / max((dw == code).sum(), 1))} for code in range(9)}
    result = {
        "all_required_outputs_exist": all((OUT / name).exists() for name in required),
        "required_outputs": {name: {"exists": (OUT / name).exists(), "sha256": sha256(OUT / name) if (OUT / name).exists() else None} for name in required},
        "raw_polygon_count": len(raw), "clean_polygon_count": len(gdf),
        "invalid_clean_geometries": int((~gdf.is_valid).sum()),
        "derived_raster_alignment": derived_alignment,
        "rasterized_instance_id_count": int(len(np.unique(instance[instance > 0]))),
        "field_mask_fraction_of_aoi": float(mask.mean()),
        "rgb": raster_profile(OUT / "rgb_input.tif"), "field_mask": raster_profile(OUT / "field_mask.tif"), "field_instance": raster_profile(OUT / "field_instance.tif"),
        "probability": probability, "probability_overlap": probability_overlap,
        "dominant_dynamic_world_class_counts": dominant.value_counts().to_dict(),
        "dynamic_world_capture_by_field_mask": capture,
        "noncrop_examples": examples,
        "probability_pixel_values_read": False,
    }
    (OUT / "technical_validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
from pathlib import Path

import cv2
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
from huggingface_hub import hf_hub_download
from rasterio.features import geometry_mask, rasterize, shapes
from rasterio.windows import from_bounds
from shapely import make_valid
from shapely.geometry import box, shape
from shapely.ops import unary_union
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "work/field_delineation/input_data/chongming_field_test_aoi_v2_label_free_s2_l2a_rgb_2022.tif"
LANDCOVER = ROOT / "work/field_delineation/input_data/chongming_field_test_aoi_v2_label_free_dynamic_world_mode_2022.tif"
PROBABILITY = ROOT / "outputs/experiments/rf_baseline_m2_v1/shanghai_source_only_probability_s1_s2_fusion.tif"
AOI = ROOT / "data/field_delineation_test_aoi.geojson"
OUT = ROOT / "outputs/field_delineation/test_aoi"

CONFIDENCE = 0.15
TILE_SIZE_SOURCE = 256
TILE_SIZE_MODEL = 512
TILE_STEP = 128
MINIMUM_AREA_M2 = 2500.0  # official DAv2 sample config at 10 m
DW_NAMES = {
    0: "water",
    1: "trees",
    2: "grass",
    3: "flooded_vegetation",
    4: "crops",
    5: "shrub_and_scrub",
    6: "built",
    7: "bare",
    8: "snow_and_ice",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def positions(length: int) -> list[int]:
    if length <= TILE_SIZE_SOURCE:
        return [0]
    values = list(range(0, length - TILE_SIZE_SOURCE + 1, TILE_STEP))
    final = length - TILE_SIZE_SOURCE
    if values[-1] != final:
        values.append(final)
    return values


def preprocess() -> tuple[np.ndarray, np.ndarray, dict]:
    with rasterio.open(INPUT) as src:
        raw = src.read([1, 2, 3]).astype("float32")
        profile = src.profile.copy()
        descriptions = list(src.descriptions)
    valid = np.all(raw > 0, axis=0) & np.all(np.isfinite(raw), axis=0)
    mins, maxs = [], []
    rgb = np.zeros_like(raw, dtype="uint8")
    for band in range(3):
        values = raw[band][valid]
        lo, hi = np.percentile(values, [1, 99])
        mins.append(float(lo)); maxs.append(float(hi))
        rgb[band] = np.clip(255 * (raw[band] - lo) / max(hi - lo, 1e-6), 0, 255).astype("uint8")
    rgb[:, ~valid] = 0
    return np.moveaxis(rgb, 0, -1), valid, {
        "profile": profile,
        "band_descriptions": descriptions,
        "normalization_percentiles": [1, 99],
        "normalization_min_rgb": mins,
        "normalization_max_rgb": maxs,
        "valid_fraction": float(valid.mean()),
    }


def mask_to_geometry(mask: np.ndarray, transform: rasterio.Affine):
    parts = [shape(geom) for geom, value in shapes(mask.astype("uint8"), mask=mask, transform=transform) if value == 1]
    if not parts:
        return None
    geom = make_valid(unary_union(parts))
    if geom.is_empty:
        return None
    if geom.geom_type == "GeometryCollection":
        polys = [item for item in geom.geoms if item.geom_type in {"Polygon", "MultiPolygon"}]
        geom = unary_union(polys) if polys else None
    return geom


def inference(rgb: np.ndarray, transform, model_path: Path) -> tuple[list[dict], dict]:
    row_positions = positions(rgb.shape[0])
    col_positions = positions(rgb.shape[1])
    images, tile_records = [], []
    for row0 in row_positions:
        for col0 in col_positions:
            tile_rgb = rgb[row0:row0 + TILE_SIZE_SOURCE, col0:col0 + TILE_SIZE_SOURCE]
            tile_bgr = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2BGR)
            images.append(cv2.resize(tile_bgr, (TILE_SIZE_MODEL, TILE_SIZE_MODEL), interpolation=cv2.INTER_CUBIC))
            tile_records.append((row0, col0))

    model = YOLO(str(model_path))
    device = 0 if torch.cuda.is_available() else "cpu"
    start = time.perf_counter()
    results = model.predict(
        images,
        conf=CONFIDENCE,
        half=torch.cuda.is_available(),
        device=device,
        retina_masks=True,
        imgsz=TILE_SIZE_MODEL,
        batch=4,
        max_det=300,
        verbose=False,
    )
    elapsed = time.perf_counter() - start
    detections: list[dict] = []
    for tile_index, (result, (row0, col0)) in enumerate(zip(results, tile_records)):
        if result.masks is None:
            continue
        masks = result.masks.data.float()
        with torch.no_grad():
            masks = -torch.nn.functional.max_pool2d(-masks, 3, stride=1, padding=1)
            masks = torch.nn.functional.max_pool2d(masks, 3, stride=1, padding=1)
            masks = torch.nn.functional.max_pool2d(masks, 3, stride=1, padding=1)
            masks = -torch.nn.functional.max_pool2d(-masks, 3, stride=1, padding=1)
        masks = masks.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        tile_transform = transform * rasterio.Affine.translation(col0, row0)
        for local_index, (mask_model, confidence) in enumerate(zip(masks, confidences)):
            mask = cv2.resize(mask_model, (TILE_SIZE_SOURCE, TILE_SIZE_SOURCE), interpolation=cv2.INTER_NEAREST) >= 0.5
            geom = mask_to_geometry(mask, tile_transform)
            if geom is None:
                continue
            detections.append({
                "geometry": geom,
                "confidence": float(confidence),
                "tile_index": tile_index,
                "tile_row": row0,
                "tile_col": col0,
                "local_index": local_index,
                "raw_area_m2": float(geom.area),
            })
    return detections, {
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "tile_size_source_px": TILE_SIZE_SOURCE,
        "tile_size_model_px": TILE_SIZE_MODEL,
        "tile_step_source_px": TILE_STEP,
        "tile_overlap_fraction": 0.5,
        "n_tiles": len(images),
        "inference_seconds": elapsed,
        "raw_detections": len(detections),
    }


def deduplicate(detections: list[dict], aoi_geom) -> tuple[list[dict], int]:
    accepted: list[dict] = []
    duplicates = 0
    for item in sorted(detections, key=lambda row: row["confidence"], reverse=True):
        geom = make_valid(item["geometry"].intersection(aoi_geom))
        if geom.is_empty:
            continue
        is_duplicate = False
        for kept in accepted:
            if not box(*geom.bounds).intersects(box(*kept["geometry"].bounds)):
                continue
            intersection = geom.intersection(kept["geometry"]).area
            if intersection <= 0:
                continue
            union_area = geom.area + kept["geometry"].area - intersection
            iou = intersection / union_area
            containment = intersection / min(geom.area, kept["geometry"].area)
            if iou >= 0.60 or containment >= 0.80:
                # Retain the spatial support from both overlapping-tile detections.
                # Keeping only the higher-confidence partial mask creates visible
                # straight truncations at tile boundaries.
                kept["geometry"] = make_valid(kept["geometry"].union(geom))
                kept["raw_area_m2"] = float(kept["geometry"].area)
                kept["n_merged_detections"] += 1
                is_duplicate = True
                duplicates += 1
                break
        if not is_duplicate:
            copy = dict(item)
            copy["geometry"] = geom
            copy["n_merged_detections"] = 1
            accepted.append(copy)
    return accepted, duplicates


def overlap_stats(items: list[dict]) -> dict:
    pair_count, area_sum = 0, 0.0
    geoms = [item["geometry"] for item in items]
    for i in range(len(geoms)):
        for j in range(i + 1, len(geoms)):
            intersection_area = geoms[i].intersection(geoms[j]).area
            if intersection_area > 0:
                pair_count += 1
                area_sum += float(intersection_area)
    union_area = float(unary_union(geoms).area) if geoms else 0.0
    return {"pair_count": pair_count, "area_m2_sum": area_sum, "rate_vs_union_area": area_sum / union_area if union_area else 0.0}


def resolve_conflicting_instances(items: list[dict]) -> tuple[list[dict], int]:
    """Give overlapping pixels to the higher-confidence instance deterministically."""
    occupied = None
    resolved: list[dict] = []
    removed_slivers = 0
    for item in sorted(items, key=lambda row: row["confidence"], reverse=True):
        remaining = item["geometry"] if occupied is None else make_valid(item["geometry"].difference(occupied))
        if remaining.is_empty:
            removed_slivers += 1
            continue
        parts = list(remaining.geoms) if remaining.geom_type == "MultiPolygon" else [remaining]
        kept_parts = [part for part in parts if part.geom_type == "Polygon" and part.area >= MINIMUM_AREA_M2]
        removed_slivers += len(parts) - len(kept_parts)
        for part in kept_parts:
            copy = dict(item)
            copy["geometry"] = part
            copy["overlap_conflict_resolved"] = True
            resolved.append(copy)
            occupied = part if occupied is None else make_valid(occupied.union(part))
    return resolved, removed_slivers


def landcover_fractions(geom, ds, array: np.ndarray) -> dict[str, float]:
    window = from_bounds(*geom.bounds, transform=ds.transform).round_offsets().round_lengths()
    full = rasterio.windows.Window(0, 0, ds.width, ds.height)
    window = window.intersection(full)
    if window.width <= 0 or window.height <= 0:
        return {f"dw_{name}_fraction": 0.0 for name in DW_NAMES.values()}
    subset = array[
        int(window.row_off):int(window.row_off + window.height),
        int(window.col_off):int(window.col_off + window.width),
    ]
    local_transform = ds.window_transform(window)
    inside = geometry_mask([geom], out_shape=subset.shape, transform=local_transform, invert=True)
    values = subset[inside]
    total = max(len(values), 1)
    return {f"dw_{name}_fraction": float((values == code).sum() / total) for code, name in DW_NAMES.items()}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(AOI, OUT / "aoi.geojson")
    shutil.copy2(INPUT, OUT / "rgb_input.tif")
    model_path = Path(hf_hub_download(
        repo_id="MykolaL/DelineateAnything",
        filename="DelineateAnythingv2.pt",
        local_files_only=True,
    ))
    rgb, valid, input_meta = preprocess()
    profile = input_meta.pop("profile")
    transform = profile["transform"]
    crs = profile["crs"]
    aoi_geom = box(*rasterio.transform.array_bounds(profile["height"], profile["width"], transform))

    # Input preview is a visualization only; model input is the percentile-scaled RGB array.
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rgb)
    ax.set_title("Sentinel-2 SR median RGB, 2022-04-01 to 2022-10-31\nDAv2 AOI input preview")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "input_preview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    raw, infer_meta = inference(rgb, transform, model_path)
    for raw_index, item in enumerate(raw, start=1):
        item["raw_instance_id"] = raw_index
    raw_gdf = gpd.GeoDataFrame(raw, geometry="geometry", crs=crs)
    raw_gdf.to_file(OUT / "field_polygons_raw.geojson", driver="GeoJSON")

    accepted, duplicate_count = deduplicate(raw, aoi_geom)
    prefilter_count = len(accepted)
    tiny_count = sum(item["geometry"].area < MINIMUM_AREA_M2 for item in accepted)
    accepted = [item for item in accepted if item["geometry"].area >= MINIMUM_AREA_M2]
    pre_conflict_overlap = overlap_stats(accepted)
    gpd.GeoDataFrame(accepted, geometry="geometry", crs=crs).to_file(
        OUT / "field_polygons_deduplicated_pre_overlap.geojson", driver="GeoJSON"
    )
    accepted, conflict_sliver_count = resolve_conflicting_instances(accepted)

    with rasterio.open(LANDCOVER) as landcover_ds:
        landcover = landcover_ds.read(1)
        if landcover_ds.crs != crs or landcover_ds.transform != transform or landcover_ds.shape != (profile["height"], profile["width"]):
            raise ValueError("Dynamic World QA raster is not aligned with RGB input")
        for item in accepted:
            item.update(landcover_fractions(item["geometry"], landcover_ds, landcover))

    west, south, east, north = aoi_geom.bounds
    internal_x = [west + 10 * value for value in positions(profile["width"])[1:]]
    internal_y = [north - 10 * value for value in positions(profile["height"])[1:]]
    for index, item in enumerate(sorted(accepted, key=lambda row: (row["geometry"].centroid.y, row["geometry"].centroid.x))):
        geom = item["geometry"]
        boundary_buffer = 10.1
        item["field_id"] = f"FD{index + 1:05d}"
        item["area_m2"] = float(geom.area)
        item["area_ha"] = float(geom.area / 10000)
        item["source_model"] = "DelineateAnythingv2"
        item["checkpoint_revision"] = "cff6ad11e3a0a7ccdcf1261fefb148309b5d6a8a"
        item["touches_aoi_edge"] = bool(
            geom.distance(box(west, south, east, north).boundary) <= boundary_buffer
        )
        near_internal = any(abs(x - geom.bounds[0]) <= boundary_buffer or abs(x - geom.bounds[2]) <= boundary_buffer for x in internal_x)
        near_internal |= any(abs(y - geom.bounds[1]) <= boundary_buffer or abs(y - geom.bounds[3]) <= boundary_buffer for y in internal_y)
        item["possible_tile_edge_artifact"] = bool(near_internal)
        item["noncrop_dominant_qa"] = bool(
            item["dw_crops_fraction"] < 0.2
            and max(item["dw_water_fraction"], item["dw_trees_fraction"], item["dw_built_fraction"]) >= 0.5
        )

    gdf = gpd.GeoDataFrame(accepted, geometry="geometry", crs=crs)
    preferred = [
        "field_id", "area_m2", "area_ha", "source_model", "checkpoint_revision", "confidence",
        "touches_aoi_edge", "possible_tile_edge_artifact", "noncrop_dominant_qa",
    ]
    remaining = [column for column in gdf.columns if column not in preferred + ["geometry"]]
    gdf = gdf[preferred + remaining + ["geometry"]]
    gdf.to_file(OUT / "field_polygons.geojson", driver="GeoJSON")
    gdf.to_file(OUT / "field_polygons.gpkg", layer="fields", driver="GPKG")

    # Derived project-compatible rasters. The raw per-tile native instances
    # above remain untouched; these are produced only from cleaned polygons.
    out_shape = (profile["height"], profile["width"])
    instance_values = rasterize(
        ((geom, idx) for idx, geom in enumerate(gdf.geometry, start=1)),
        out_shape=out_shape, transform=transform, fill=0, dtype="uint32",
    )
    mask_values = (instance_values > 0).astype("uint8")
    instance_profile = profile.copy()
    instance_profile.update(count=1, dtype="uint32", nodata=None, compress="deflate")
    with rasterio.open(OUT / "field_instance.tif", "w", **instance_profile) as dst:
        dst.write(instance_values, 1)
        dst.set_band_description(1, "field_instance_id; 0=background")
    mask_profile = profile.copy()
    mask_profile.update(count=1, dtype="uint8", nodata=255, compress="deflate")
    with rasterio.open(OUT / "field_mask.tif", "w", **mask_profile) as dst:
        dst.write(mask_values, 1)
        dst.set_band_description(1, "field_candidate_mask; 0=background,1=field,255=nodata")

    # Overlay preview: no geometry is manually removed for display.
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(rgb, extent=(west, east, south, north))
    if len(gdf):
        gdf.boundary.plot(ax=ax, color="#ffeb3b", linewidth=0.8)
        flagged = gdf[gdf.noncrop_dominant_qa]
        if len(flagged):
            flagged.boundary.plot(ax=ax, color="#f44336", linewidth=1.2)
    ax.set_title("DAv2 field instances (yellow); Dynamic World non-crop-dominant QA flags (red)")
    ax.set_xlabel("Easting (m), EPSG:32651")
    ax.set_ylabel("Northing (m), EPSG:32651")
    ax.ticklabel_format(style="plain", useOffset=False)
    fig.tight_layout()
    fig.savefig(OUT / "field_overlay_preview.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Alignment audit; no probability values are modified.
    with rasterio.open(PROBABILITY) as probability_ds:
        probability_bounds = box(*probability_ds.bounds)
        vector_union = unary_union(gdf.geometry.tolist()) if len(gdf) else None
        alignment = {
            "probability_path": str(PROBABILITY.relative_to(ROOT)),
            "probability_crs": str(probability_ds.crs),
            "probability_shape": list(probability_ds.shape),
            "probability_resolution": list(probability_ds.res),
            "probability_transform": list(probability_ds.transform)[:6],
            "probability_nodata": probability_ds.nodata,
            "field_crs": str(crs),
            "field_aoi_bounds": list(aoi_geom.bounds),
            "crs_match": probability_ds.crs == crs,
            "aoi_within_probability_extent": probability_bounds.covers(aoi_geom),
            "aoi_aligned_to_probability_grid": bool(
                math.isclose((west - probability_ds.bounds.left) / probability_ds.res[0], round((west - probability_ds.bounds.left) / probability_ds.res[0]))
                and math.isclose((probability_ds.bounds.top - north) / abs(probability_ds.res[1]), round((probability_ds.bounds.top - north) / abs(probability_ds.res[1])))
            ),
            "expected_probability_window_shape": [250, 250],
            "all_vectors_overlap_probability_extent": bool(vector_union is None or probability_bounds.covers(vector_union)),
            "probability_modified": False,
        }

    areas = gdf.area.to_numpy() if len(gdf) else np.array([])
    post_conflict_overlap = overlap_stats(gdf.to_dict("records"))
    qa = {
        "selected_model": "Delineate Anything v2",
        "checkpoint_worked": True,
        "n_parcels": int(len(gdf)),
        "raw_tile_detections": int(len(raw)),
        "deduplicated_prefilter_count": int(prefilter_count),
        "duplicate_detections_removed": int(duplicate_count),
        "raw_duplicate_fraction": float(duplicate_count / len(raw)) if raw else 0.0,
        "pre_conflict_overlap": pre_conflict_overlap,
        "overlap_conflict_slivers_removed": int(conflict_sliver_count),
        "remaining_overlap_pair_count": post_conflict_overlap["pair_count"],
        "remaining_overlap_area_m2_sum": post_conflict_overlap["area_m2_sum"],
        "remaining_overlap_rate_vs_union_area": post_conflict_overlap["rate_vs_union_area"],
        "tiny_polygon_threshold_m2": MINIMUM_AREA_M2,
        "tiny_polygon_count_before_default_filter": int(tiny_count),
        "tiny_polygon_fraction_before_default_filter": float(tiny_count / prefilter_count) if prefilter_count else 0.0,
        "invalid_geometry_count": int((~gdf.is_valid).sum()) if len(gdf) else 0,
        "empty_geometry_count": int(gdf.is_empty.sum()) if len(gdf) else 0,
        "touches_aoi_edge_count": int(gdf.touches_aoi_edge.sum()) if len(gdf) else 0,
        "possible_tile_edge_artifact_count": int(gdf.possible_tile_edge_artifact.sum()) if len(gdf) else 0,
        "possible_tile_edge_artifact_fraction": float(gdf.possible_tile_edge_artifact.mean()) if len(gdf) else 0.0,
        "dynamic_world_noncrop_dominant_count": int(gdf.noncrop_dominant_qa.sum()) if len(gdf) else 0,
        "area_m2": {
            "min": float(areas.min()) if len(areas) else None,
            "p10": float(np.quantile(areas, 0.10)) if len(areas) else None,
            "median": float(np.median(areas)) if len(areas) else None,
            "mean": float(areas.mean()) if len(areas) else None,
            "p90": float(np.quantile(areas, 0.90)) if len(areas) else None,
            "max": float(areas.max()) if len(areas) else None,
        },
        "dynamic_world_note": "Weak land-cover QA only; not parcel ground truth and not used as model input or filtering.",
        "alignment": alignment,
    }
    (OUT / "qa_summary.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    gdf.drop(columns="geometry").to_csv(OUT / "parcel_attributes.csv", index=False)

    run_metadata = {
        "model_repo": "https://github.com/Lavreniuk/Delineate-Anything",
        "model_repo_commit": "278a3d91e78535174e2a80865a67c77e855d3a91",
        "model_repo_pinned_archive": "work/field_delineation/Delineate-Anything-278a3d91e78535174e2a80865a67c77e855d3a91.zip",
        "model_repo_pinned_archive_sha256": file_sha256(ROOT / "work/field_delineation/Delineate-Anything-278a3d91e78535174e2a80865a67c77e855d3a91.zip"),
        "checkpoint_repo": "MykolaL/DelineateAnything",
        "checkpoint_filename": model_path.name,
        "checkpoint_revision": model_path.parent.name,
        "checkpoint_sha256": file_sha256(model_path),
        "checkpoint_bytes": model_path.stat().st_size,
        "architecture": "Ultralytics YOLO instance-segmentation model; loaded task=segment, one class=field",
        "execution_path": "Pinned official checkpoint loaded through Ultralytics; local geospatial tiled runner because osgeo.gdal was unavailable. Official repository entry point was not run end-to-end.",
        "raw_native_output": "outputs/field_delineation/test_aoi/field_polygons_raw.geojson (per-tile instance masks polygonized before deduplication, area filtering, or land-cover QA)",
        "aoi_path": str(AOI.relative_to(ROOT)),
        "aoi_sha256": file_sha256(AOI),
        "aoi_selection": "2022 Dynamic World land-cover mixture; no rice label or rice prediction values",
        "input_path": str(INPUT.relative_to(ROOT)),
        "input_sha256": file_sha256(INPUT),
        "input_crs": str(crs),
        "input_shape": [profile["height"], profile["width"], 3],
        "input_resolution_m": [abs(transform.a), abs(transform.e)],
        "input_metadata": input_meta,
        "inference": infer_meta,
        "thresholds": {
            "confidence": CONFIDENCE,
            "duplicate_iou": 0.60,
            "duplicate_containment": 0.80,
            "minimum_area_m2": MINIMUM_AREA_M2,
            "parameter_tuning_against_rice_labels": False,
        },
        "software": {
            "python": __import__("sys").version,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "ultralytics": __import__("ultralytics").__version__,
            "rasterio": rasterio.__version__,
            "geopandas": gpd.__version__,
            "shapely": __import__("shapely").__version__,
        },
        "derived_outputs": {
            "field_instance_tif": "uint32; 0=background; no nodata tag",
            "field_mask_tif": "uint8; 0=background, 1=field, 255=nodata",
        },
        "runner_path": str(Path(__file__).resolve().relative_to(ROOT)),
        "runner_sha256": file_sha256(Path(__file__).resolve()),
        "output_sha256": {
            name: file_sha256(OUT / name) for name in [
                "aoi.geojson", "rgb_input.tif", "field_polygons_raw.geojson",
                "field_polygons.gpkg", "field_instance.tif", "field_mask.tif",
                "field_overlay_preview.png",
            ]
        },
    }
    (OUT / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()

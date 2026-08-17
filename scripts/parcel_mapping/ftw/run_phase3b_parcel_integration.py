from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask, rasterize
from rasterio.windows import from_bounds
from shapely.geometry import box


ROOT = Path(__file__).resolve().parents[3]
FTW = ROOT / "outputs/field_delineation/ftw_test_aoi_2/ftw_polygons_clean.gpkg"
DISAGREE = ROOT / "outputs/field_delineation/ftw_test_aoi_2/ftw_semantic_native_disagreements.tif"
AOI = ROOT / "outputs/field_delineation/ftw_test_aoi_2/aoi_2.geojson"
RGB = ROOT / "outputs/field_delineation/ftw_test_aoi_2/rgb_input.tif"
DW = ROOT / "work/field_delineation/ftw/input_data_aoi2/chongming_phase3b_aoi2_dynamic_world_mode_2022_v1.tif"
PROB = ROOT / "outputs/experiments/rf_baseline_m2_v1/shanghai_source_only_probability_s1_s2_fusion.tif"
OUT = ROOT / "outputs/field_parcel_integration/test_aoi_2"
TABLES = ROOT / "outputs/tables"
AOI1_STATS = ROOT / "outputs/field_parcel_integration/test_aoi/phase3a_diagnostics.json"
THRESHOLD = 0.50
HIGH_PROBABILITY = 0.80
LOW_PIXEL_COUNT = 4
TILE_DISAGREEMENT_FRACTION = 0.05
DW_NAMES = {0: "water", 1: "trees", 2: "grass", 3: "flooded_vegetation", 4: "crops", 5: "shrub_and_scrub", 6: "built", 7: "bare", 8: "snow_and_ice"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_raster(path: Path, data: np.ndarray, profile: dict, dtype: str, nodata, description: str, tags: dict | None = None) -> None:
    p = profile.copy(); p.update(driver="GTiff", count=1, dtype=dtype, nodata=nodata, compress="deflate", tiled=True, blockxsize=128, blockysize=128)
    with rasterio.open(path, "w", **p) as dst:
        dst.write(data.astype(dtype), 1); dst.set_band_description(1, description)
        if tags: dst.update_tags(**{k: str(v) for k, v in tags.items()})


def masked_probability(array: np.ndarray, nodata: float) -> np.ma.MaskedArray:
    return np.ma.masked_where((array == nodata) | ~np.isfinite(array), array)


def map_probability(path: Path, data: np.ndarray, nodata: float, extent, title: str, polygons=None) -> None:
    fig, ax = plt.subplots(figsize=(8, 7.5)); image = ax.imshow(masked_probability(data, nodata), extent=extent, cmap="YlGn", vmin=0, vmax=1)
    if polygons is not None: polygons.boundary.plot(ax=ax, color="#37474f", linewidth=.25, alpha=.55)
    ax.set_title(title); ax.set_xlabel("Easting, EPSG:32651"); ax.set_ylabel("Northing, EPSG:32651"); ax.ticklabel_format(style="plain", useOffset=False)
    fig.colorbar(image, ax=ax, fraction=.046, pad=.04, label="Rice probability"); fig.tight_layout(); fig.savefig(path, dpi=210, bbox_inches="tight"); plt.close(fig)


def neighbor_disagreement(binary: np.ndarray, valid: np.ndarray) -> float:
    h = valid[:, :-1] & valid[:, 1:]; v = valid[:-1, :] & valid[1:, :]
    numerator = int(((binary[:, :-1] != binary[:, 1:]) & h).sum() + ((binary[:-1, :] != binary[1:, :]) & v).sum())
    denominator = int(h.sum() + v.sum())
    return numerator / denominator if denominator else 0.0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); TABLES.mkdir(parents=True, exist_ok=True)
    fields = gpd.read_file(FTW, layer="fields")
    bounds = tuple(float(v) for v in fields.total_bounds); aoi_geom = box(*bounds); aoi_boundary = aoi_geom.boundary
    expected_bounds = (385320.0, 3491320.0, 390320.0, 3496320.0)
    if bounds != expected_bounds:
        # Official polygonization can omit exact AOI edges; use frozen AOI bounds.
        bounds = expected_bounds; aoi_geom = box(*bounds); aoi_boundary = aoi_geom.boundary
    with rasterio.open(PROB) as src:
        window = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
        probability = src.read(1, window=window).astype("float32"); transform = src.window_transform(window)
        profile = src.profile.copy(); profile.update(width=probability.shape[1], height=probability.shape[0], transform=transform)
        nodata = float(src.nodata); source_profile = {"crs": str(src.crs), "extent": list(src.bounds), "transform": list(src.transform)[:6], "resolution": list(src.res), "nodata": src.nodata}
    if probability.shape != (250, 250) or list(transform)[:6] != [20.0, 0.0, 385320.0, 0.0, -20.0, 3496320.0]:
        raise ValueError(f"Unexpected AOI 2 probability grid: {probability.shape}, {transform}")
    valid = np.isfinite(probability) & (probability != nodata)
    raw_binary = np.zeros(probability.shape, dtype="uint8"); raw_binary[valid] = probability[valid] >= THRESHOLD
    binary_out = raw_binary.copy(); binary_out[~valid] = 255
    write_raster(OUT / "M0_raw_probability.tif", probability, profile, "float32", nodata, "M0 retained source-only rice probability", {"threshold_inherited": THRESHOLD, "source_sha256": sha256(PROB)})
    write_raster(OUT / "M0_raw_binary.tif", binary_out, profile, "uint8", 255, "M0 binary; 0=non-rice,1=rice,255=nodata", {"threshold_inherited": THRESHOLD})

    ftw_mask = rasterize(((geom, 1) for geom in fields.geometry), out_shape=probability.shape, transform=transform, fill=0, all_touched=False, dtype="uint8").astype(bool)
    m1_mask = ftw_mask & valid; m1_probability = np.full(probability.shape, nodata, dtype="float32"); m1_probability[m1_mask] = probability[m1_mask]
    m1_binary = np.zeros(probability.shape, dtype="uint8"); m1_binary[m1_mask] = probability[m1_mask] >= THRESHOLD; m1_binary[~valid] = 255
    m1_mask_out = ftw_mask.astype("uint8"); m1_mask_out[~valid] = 255
    write_raster(OUT / "M1_ftw_field_mask_20m.tif", m1_mask_out, profile, "uint8", 255, "FTW accepted-field center-pixel mask; 0=outside,1=inside,255=nodata", {"polygon_sha256": sha256(FTW), "rasterization": "pixel center; all_touched=False"})
    write_raster(OUT / "M1_ftw_gated_probability.tif", m1_probability, profile, "float32", nodata, "M1 FTW-gated rice probability")
    write_raster(OUT / "M1_ftw_gated_binary.tif", m1_binary, profile, "uint8", 255, "M1 FTW-gated binary", {"threshold_inherited": THRESHOLD})

    with rasterio.open(DW) as ds:
        dw = ds.read(1)
        if ds.shape != (500, 500) or ds.crs != profile["crs"] or list(ds.transform)[:6] != [10.0, 0.0, 385320.0, 0.0, -10.0, 3496320.0]: raise ValueError("AOI 2 Dynamic World raster is not the exact 2x grid")
    allowed10 = np.isin(dw, [3, 4]); allowed_count20 = allowed10.reshape(250, 2, 250, 2).sum(axis=(1, 3)); landcover_gate = allowed_count20 >= 2
    class_counts = np.stack([(dw == code).reshape(250, 2, 250, 2).sum(axis=(1, 3)) for code in range(9)]); dw20_majority = class_counts.argmax(axis=0)
    m1b_mask = m1_mask & landcover_gate; m1b_probability = np.full(probability.shape, nodata, dtype="float32"); m1b_probability[m1b_mask] = probability[m1b_mask]
    m1b_binary = np.zeros(probability.shape, dtype="uint8"); m1b_binary[m1b_mask] = probability[m1b_mask] >= THRESHOLD; m1b_binary[~valid] = 255
    m1b_mask_out = m1b_mask.astype("uint8"); m1b_mask_out[~valid] = 255
    rule = "FTW mask AND at least 2 of 4 underlying 10m Dynamic World pixels are crops(4) or flooded_vegetation(3)"
    tags = {"rule": rule, "dynamic_world_sha256": sha256(DW), "not_baked_into_ftw_polygons": True}
    write_raster(OUT / "M1b_ftw_plus_landcover_mask_20m.tif", m1b_mask_out, profile, "uint8", 255, "Independent FTW plus land-cover gate", tags)
    write_raster(OUT / "M1b_ftw_plus_landcover_gated_probability.tif", m1b_probability, profile, "float32", nodata, "M1b gated probability", tags)
    write_raster(OUT / "M1b_ftw_plus_landcover_gated_binary.tif", m1b_binary, profile, "uint8", 255, "M1b gated binary", {**tags, "threshold_inherited": THRESHOLD})

    with rasterio.open(DISAGREE) as ds: disagreement = ds.read(1) > 0; disagreement_transform = ds.transform
    rows = []
    for _, feature in fields.iterrows():
        geom = feature.geometry
        inside = geometry_mask([geom], out_shape=probability.shape, transform=transform, invert=True, all_touched=False); values = probability[inside & valid]
        dis_inside = geometry_mask([geom], out_shape=disagreement.shape, transform=disagreement_transform, invert=True, all_touched=False)
        dis_fraction = float(disagreement[dis_inside].mean()) if dis_inside.any() else 0.0
        record = feature.drop(labels="geometry").to_dict(); record.update({"geometry": geom, "n_valid_pixels": int(len(values)), "rice_prob_mean": float(np.mean(values)) if len(values) else np.nan, "rice_prob_median": float(np.median(values)) if len(values) else np.nan, "rice_prob_std": float(np.std(values)) if len(values) else np.nan, "rice_positive_fraction": float((values >= THRESHOLD).mean()) if len(values) else np.nan, "large_parcel_flag": bool(geom.area > 200000), "very_large_parcel_flag": bool(geom.area > 500000), "low_pixel_count_flag": bool(len(values) < LOW_PIXEL_COUNT), "built_dominant_flag": bool(feature.get("dw_built_fraction", 0) >= .5), "tree_dominant_flag": bool(feature.get("dw_trees_fraction", 0) >= .5), "AOI_edge_flag": bool(geom.distance(aoi_boundary) <= 1e-6), "FTW_tile_disagreement_fraction": dis_fraction, "FTW_tile_disagreement_flag": bool(dis_fraction >= TILE_DISAGREEMENT_FRACTION)})
        rows.append(record)
    parcels = gpd.GeoDataFrame(rows, geometry="geometry", crs=fields.crs)
    for summary in ["mean", "median"]: parcels[f"class_by_{summary}"] = np.where(parcels[f"rice_prob_{summary}"] >= THRESHOLD, "Rice", "Non-rice")
    parcels["class_by_positive_fraction"] = np.where(parcels.rice_positive_fraction >= THRESHOLD, "Rice", "Non-rice")
    base_flags = ["large_parcel_flag", "very_large_parcel_flag", "low_pixel_count_flag", "built_dominant_flag", "tree_dominant_flag", "AOI_edge_flag", "FTW_tile_disagreement_flag"]
    parcels["summary_disagreement_flag"] = parcels[["class_by_mean", "class_by_median", "class_by_positive_fraction"]].nunique(axis=1) > 1
    parcels["qa_flag_count"] = parcels[base_flags].sum(axis=1).astype(int) + parcels.summary_disagreement_flag.astype(int)
    parcels["parcel_class"] = np.where(parcels.qa_flag_count > 0, "Uncertain / QA-risk", parcels.class_by_mean)
    for path in [OUT / "M2_parcel_diagnostics.gpkg"]:
        if path.exists(): path.unlink()
    parcels.to_file(OUT / "M2_parcel_diagnostics.gpkg", layer="parcels", driver="GPKG"); parcels.to_file(OUT / "M2_parcel_diagnostics.geojson", driver="GeoJSON"); parcels.drop(columns="geometry").to_csv(OUT / "M2_parcel_diagnostics.csv", index=False)
    failure_cols = ["field_id", "area_ha", "n_valid_pixels", "dw_dominant", *base_flags, "summary_disagreement_flag", "FTW_tile_disagreement_fraction", "qa_flag_count", "parcel_class"]
    parcels[failure_cols].to_csv(TABLES / "phase3b_ftw_failure_modes.csv", index=False)
    parcel_mean_class = rasterize(((geom, {"Non-rice": 0, "Rice": 1}.get(cls, 255)) for geom, cls in zip(parcels.geometry, parcels.class_by_mean)), out_shape=probability.shape, transform=transform, fill=255, all_touched=False, dtype="uint8")
    parcel_final = rasterize(((geom, {"Non-rice": 0, "Rice": 1, "Uncertain / QA-risk": 2}[cls]) for geom, cls in zip(parcels.geometry, parcels.parcel_class)), out_shape=probability.shape, transform=transform, fill=255, all_touched=False, dtype="uint8")
    write_raster(OUT / "M2_parcel_mean_class_20m.tif", parcel_mean_class, profile, "uint8", 255, "Parcel class from mean probability")
    write_raster(OUT / "M2_parcel_diagnostic_class_20m.tif", parcel_final, profile, "uint8", 255, "Diagnostic parcel class")

    valid_count = int(valid.sum()); raw_pos = valid & (probability >= THRESHOLD); high = valid & (probability >= HIGH_PROBABILITY)
    def stage(mask: np.ndarray) -> dict:
        return {"fraction_aoi_retained": float(mask.sum() / valid_count), "raw_predicted_rice_pixels_retained_fraction": float((raw_pos & mask).sum() / max(raw_pos.sum(), 1)), "high_probability_pixels_removed_fraction": float((high & ~mask).sum() / max(high.sum(), 1)), "predicted_rice_pixels": int((raw_pos & mask).sum()), "predicted_rice_area_ha": float((raw_pos & mask).sum() * .04)}
    m0 = {"predicted_rice_pixels": int(raw_pos.sum()), "predicted_rice_area_ha": float(raw_pos.sum() * .04), "high_probability_pixel_count": int(high.sum()), "valid_pixels": valid_count}
    removed_positive_m1b = raw_pos & m1_mask & ~m1b_mask; removed_classes = {DW_NAMES[code]: int((removed_positive_m1b & (dw20_majority == code)).sum()) for code in range(9)}
    mean_median_abs = np.abs(parcels.rice_prob_mean - parcels.rice_prob_median); very_large = parcels[parcels.very_large_parcel_flag]; normal = parcels[~parcels.very_large_parcel_flag]
    raw_noise = neighbor_disagreement(raw_binary, m1_mask); parcel_noise = neighbor_disagreement(parcel_mean_class, parcel_mean_class != 255)
    stats = {"frozen_inputs": {"aoi": str(AOI.relative_to(ROOT)), "aoi_sha256": sha256(AOI), "ftw_polygon": str(FTW.relative_to(ROOT)), "ftw_polygon_sha256": sha256(FTW), "probability": str(PROB.relative_to(ROOT)), "probability_sha256": sha256(PROB), "dynamic_world_sha256": sha256(DW), "decision_threshold": THRESHOLD, "high_probability_definition": HIGH_PROBABILITY}, "grid": {"aoi_bounds": list(bounds), "crs": str(profile["crs"]), "shape": list(probability.shape), "transform": list(transform)[:6], "resolution_m": 20, "source_probability": source_profile}, "M0": m0, "M1_FTW_only": stage(m1_mask), "M1b_FTW_plus_landcover": {**stage(m1b_mask), "landcover_rule": rule, "positive_pixels_removed_from_M1_by_majority_landcover_class": removed_classes}, "M2": {"parcel_count": len(parcels), "mean_median_probability_pearson": float(parcels[["rice_prob_mean", "rice_prob_median"]].corr().iloc[0, 1]), "mean_median_mean_absolute_difference": float(mean_median_abs.mean()), "mean_median_class_agreement_fraction": float((parcels.class_by_mean == parcels.class_by_median).mean()), "all_three_summary_class_agreement_fraction": float((~parcels.summary_disagreement_flag).mean()), "low_pixel_count_count": int(parcels.low_pixel_count_flag.sum()), "large_over_20ha_count": int(parcels.large_parcel_flag.sum()), "very_large_over_50ha_count": int(parcels.very_large_parcel_flag.sum()), "built_dominant_count": int(parcels.built_dominant_flag.sum()), "tree_dominant_count": int(parcels.tree_dominant_flag.sum()), "AOI_edge_count": int(parcels.AOI_edge_flag.sum()), "tile_disagreement_flag_count": int(parcels.FTW_tile_disagreement_flag.sum()), "summary_disagreement_count": int(parcels.summary_disagreement_flag.sum()), "diagnostic_class_counts": parcels.parcel_class.value_counts().to_dict(), "qa_risk_count": int((parcels.parcel_class == "Uncertain / QA-risk").sum()), "qa_risk_fraction": float((parcels.parcel_class == "Uncertain / QA-risk").mean()), "very_large_mean_within_parcel_std": float(very_large.rice_prob_std.mean()) if len(very_large) else None, "other_parcels_mean_within_parcel_std": float(normal.rice_prob_std.mean()), "raw_pixel_neighbor_disagreement_within_ftw": raw_noise, "parcel_mean_class_neighbor_disagreement": parcel_noise, "neighbor_disagreement_reduction_fraction": float((raw_noise - parcel_noise) / raw_noise) if raw_noise else 0.0}}
    (OUT / "phase3b_diagnostics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    a1 = json.loads(AOI1_STATS.read_text(encoding="utf-8")); a2 = stats
    def removal(s): return 1 - s["raw_predicted_rice_pixels_retained_fraction"]
    add_removed_a1 = a1["M1_FTW_only"]["predicted_rice_area_ha"] - a1["M1b_FTW_plus_landcover"]["predicted_rice_area_ha"]
    add_removed_a2 = a2["M1_FTW_only"]["predicted_rice_area_ha"] - a2["M1b_FTW_plus_landcover"]["predicted_rice_area_ha"]
    def built_share(s):
        d = s["M1b_FTW_plus_landcover"]["positive_pixels_removed_from_M1_by_majority_landcover_class"]; return d.get("built", 0) / max(sum(d.values()), 1)
    comparison = [
        ("FTW parcel count", a1["M2"]["parcel_count"], a2["M2"]["parcel_count"]), ("FTW AOI retained %", 100*a1["M1_FTW_only"]["fraction_aoi_retained"], 100*a2["M1_FTW_only"]["fraction_aoi_retained"]),
        ("M1 positive removal %", 100*removal(a1["M1_FTW_only"]), 100*removal(a2["M1_FTW_only"])), ("M1 predicted area removed ha", a1["M0"]["predicted_rice_area_ha"]-a1["M1_FTW_only"]["predicted_rice_area_ha"], a2["M0"]["predicted_rice_area_ha"]-a2["M1_FTW_only"]["predicted_rice_area_ha"]),
        ("M1b additional removal ha", add_removed_a1, add_removed_a2), ("Built share of M1b removal %", 100*built_share(a1), 100*built_share(a2)),
        ("Mean-median correlation", a1["M2"]["mean_median_probability_pearson"], a2["M2"]["mean_median_probability_pearson"]), ("Mean-median class agreement %", 100*a1["M2"]["mean_median_class_agreement_fraction"], 100*a2["M2"]["mean_median_class_agreement_fraction"]),
        ("Three-summary agreement %", 100*a1["M2"]["all_three_summary_class_agreement_fraction"], 100*a2["M2"]["all_three_summary_class_agreement_fraction"]), ("Salt-and-pepper reduction %", 100*a1["M2"]["neighbor_disagreement_reduction_fraction"], 100*a2["M2"]["neighbor_disagreement_reduction_fraction"]),
        ("Low-pixel parcels", a1["M2"]["low_pixel_count_count"], a2["M2"]["low_pixel_count_count"]), (">20 ha parcels", a1["M2"]["large_over_20ha_count"], a2["M2"]["large_over_20ha_count"]), (">50 ha parcels", a1["M2"]["very_large_over_50ha_count"], a2["M2"]["very_large_over_50ha_count"]),
        ("Built-dominant parcels", a1["M2"]["built_dominant_count"], a2["M2"]["built_dominant_count"]), ("Tree-dominant parcels", a1["M2"]["tree_dominant_count"], a2["M2"]["tree_dominant_count"]),
        ("QA-risk parcels", sum(v for k,v in a1["M2"]["diagnostic_class_counts"].items() if "QA-risk" in k), a2["M2"]["qa_risk_count"]), ("QA-risk fraction %", 100*sum(v for k,v in a1["M2"]["diagnostic_class_counts"].items() if "QA-risk" in k)/a1["M2"]["parcel_count"], 100*a2["M2"]["qa_risk_fraction"]),
    ]
    pd.DataFrame(comparison, columns=["Metric", "AOI 1", "AOI 2"]).to_csv(TABLES / "phase3a_vs_phase3b_robustness.csv", index=False)

    extent = (bounds[0], bounds[2], bounds[1], bounds[3]);
    with rasterio.open(RGB) as ds: rgb = ds.read([1, 2, 3]).astype("float32")
    rgb_valid = np.all(rgb > 0, axis=0); display = np.zeros((500, 500, 3), dtype="uint8")
    for b in range(3): lo, hi = np.percentile(rgb[b][rgb_valid], [1, 99]); display[..., b] = np.clip((rgb[b]-lo)*255/max(hi-lo, 1), 0, 255).astype("uint8")
    fig, ax = plt.subplots(figsize=(8, 7.5)); ax.imshow(display, extent=extent); fields.boundary.plot(ax=ax, color="#00e5ff", linewidth=.55); ax.set_title("A. RGB + frozen FTW boundaries"); ax.ticklabel_format(style="plain", useOffset=False); fig.tight_layout(); fig.savefig(OUT / "A_RGB_FTW_boundaries.png", dpi=210, bbox_inches="tight"); plt.close(fig)
    map_probability(OUT/"B_M0_raw_rice_probability.png", probability, nodata, extent, "B. M0 raw rice probability")
    map_probability(OUT/"C_M1_FTW_gated_probability.png", m1_probability, nodata, extent, "C. M1 FTW-gated probability", fields)
    map_probability(OUT/"D_M1b_FTW_landcover_gated_probability.png", m1b_probability, nodata, extent, "D. M1b FTW + independent land-cover gate", fields)
    for column, name, title in [("rice_prob_mean", "E_parcel_mean_probability.png", "E. Parcel mean rice probability"), ("rice_prob_median", "F_parcel_median_probability.png", "F. Parcel median rice probability")]:
        fig, ax = plt.subplots(figsize=(8, 7.5)); parcels.plot(ax=ax, column=column, cmap="YlGn", vmin=0, vmax=1, edgecolor="#455a64", linewidth=.25, legend=True, legend_kwds={"label": "Rice probability"}); ax.set_title(title); ax.ticklabel_format(style="plain", useOffset=False); fig.tight_layout(); fig.savefig(OUT/name, dpi=210, bbox_inches="tight"); plt.close(fig)
    colors = {"Rice": "#2e7d32", "Non-rice": "#eceff1", "Uncertain / QA-risk": "#ef6c00"}
    fig, ax = plt.subplots(figsize=(8, 7.5))
    for cls, color in colors.items():
        subset = parcels[parcels.parcel_class == cls]
        if len(subset): subset.plot(ax=ax, color=color, edgecolor="#37474f", linewidth=.3)
    ax.legend(handles=[mpatches.Patch(facecolor=c, edgecolor="#37474f", label=k) for k,c in colors.items()]); ax.set_title("G. Diagnostic parcel class map"); ax.ticklabel_format(style="plain", useOffset=False); fig.tight_layout(); fig.savefig(OUT/"G_parcel_class_map.png", dpi=210, bbox_inches="tight"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 7.5)); parcels.plot(ax=ax, column="qa_flag_count", cmap="YlOrRd", vmin=0, vmax=max(int(parcels.qa_flag_count.max()),1), edgecolor="#37474f", linewidth=.25, legend=True, legend_kwds={"label":"Number of QA-risk flags"}); ax.set_title("H. Parcel QA-risk map"); ax.ticklabel_format(style="plain", useOffset=False); fig.tight_layout(); fig.savefig(OUT/"H_parcel_QA_risk_map.png", dpi=210, bbox_inches="tight"); plt.close(fig)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharex=True, sharey=True)
    axes[0,0].imshow(display, extent=extent); fields.boundary.plot(ax=axes[0,0], color="#00e5ff", linewidth=.3); axes[0,0].set_title("A RGB + FTW")
    for ax, data, title in [(axes[0,1], probability, "B M0 raw"), (axes[0,2], m1_probability, "C M1 FTW"), (axes[0,3], m1b_probability, "D M1b + land cover")]: ax.imshow(masked_probability(data,nodata), extent=extent, cmap="YlGn", vmin=0, vmax=1); ax.set_title(title)
    parcels.plot(ax=axes[1,0], column="rice_prob_mean", cmap="YlGn", vmin=0, vmax=1, edgecolor="#455a64", linewidth=.15); axes[1,0].set_title("E Parcel mean")
    parcels.plot(ax=axes[1,1], column="rice_prob_median", cmap="YlGn", vmin=0, vmax=1, edgecolor="#455a64", linewidth=.15); axes[1,1].set_title("F Parcel median")
    for cls,color in colors.items():
        subset=parcels[parcels.parcel_class==cls]
        if len(subset): subset.plot(ax=axes[1,2], color=color, edgecolor="#37474f", linewidth=.15)
    axes[1,2].set_title("G Parcel class"); axes[1,2].legend(handles=[mpatches.Patch(facecolor=c,edgecolor="#37474f",label=k) for k,c in colors.items()], fontsize=7, loc="upper right")
    parcels.plot(ax=axes[1,3], column="qa_flag_count", cmap="YlOrRd", vmin=0, vmax=max(int(parcels.qa_flag_count.max()),1), edgecolor="#37474f", linewidth=.15); axes[1,3].set_title("H QA-risk")
    for ax in axes.ravel(): ax.ticklabel_format(style="plain", useOffset=False); ax.tick_params(labelsize=7)
    fig.suptitle("Phase 3B: independent second-AOI robustness test\nFrozen FTW-structured deployment; no retraining or target-label tuning"); fig.tight_layout(); fig.savefig(OUT/"phase3b_aoi2_publication_comparison.png", dpi=220, bbox_inches="tight"); plt.close(fig)
    manifest = {"phase":"3B", "stop_after_second_aoi":True, "script":{"path":str(Path(__file__).relative_to(ROOT)),"sha256":sha256(Path(__file__))}, "frozen_inputs":stats["frozen_inputs"], "rules":{"FTW_rasterization":"20m pixel center, all_touched=False", "landcover_gate":rule, "decision_threshold":THRESHOLD, "high_probability_diagnostic":HIGH_PROBABILITY, "low_pixel_count":"n_valid_pixels < 4", "large_parcel":"area > 20 ha", "very_large_parcel":"area > 50 ha", "built_dominant":"Dynamic World built fraction >= 0.5", "tree_dominant":"Dynamic World tree fraction >= 0.5", "AOI_edge":"geometry touches exact AOI boundary", "tile_disagreement":"official FTW disagreement fraction >= 0.05", "final_class":"Uncertain/QA-risk if any QA flag or summaries disagree; otherwise unanimous inherited-threshold class"}, "outputs_sha256":{p.name:sha256(p) for p in OUT.iterdir() if p.is_file() and p.name != "run_manifest.json"}, "classifier_retrained":False, "probability_raster_modified":False, "target_labels_used":False, "full_scene_inference":False}
    (OUT/"run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

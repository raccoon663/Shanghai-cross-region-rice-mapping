from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from scipy import ndimage
from shapely.geometry import box, mapping


ROOT = Path(__file__).resolve().parents[2]
DW = ROOT / "work/field_delineation/input_data/chongming_dynamic_world_mode_2022_20m.tif"
OUTPUT = ROOT / "data/field_delineation_test_aoi.geojson"
METADATA = ROOT / "work/field_delineation/aoi_selection_label_free_v2.json"


def main() -> None:
    window_px, stride_px = 250, 25  # 5 km window, 500 m stride
    with rasterio.open(DW) as src:
        labels = src.read(1)
        transform, crs = src.transform, src.crs
    # Dynamic World class 0 is water. Earth Engine's uint8 export tags 0 as
    # nodata even though it is a valid class, so use the complete clipped grid.
    valid = np.ones(labels.shape, dtype=bool)

    rows = []
    for row0 in range(0, labels.shape[0] - window_px + 1, stride_px):
        for col0 in range(0, labels.shape[1] - window_px + 1, stride_px):
            z = labels[row0:row0 + window_px, col0:col0 + window_px]
            v = valid[row0:row0 + window_px, col0:col0 + window_px]
            if v.mean() < 0.98:
                continue
            fractions = {code: float(((z == code) & v).sum() / v.sum()) for code in range(9)}
            crop = z == 4
            # Crop-edge density favors fragmented agricultural mosaics without using rice evidence.
            crop_edge = crop ^ ndimage.binary_erosion(crop, structure=np.ones((3, 3)))
            edge_density = float((crop_edge & v).sum() / v.sum())
            diversity = sum(fractions[c] >= 0.01 for c in range(8))
            context_ok = fractions[0] >= 0.01 and fractions[6] >= 0.01 and (fractions[1] + fractions[2] + fractions[5]) >= 0.02
            if not (0.30 <= fractions[4] <= 0.82 and context_ok and diversity >= 4):
                continue
            score = (
                2.2 * fractions[4]
                + 6.0 * edge_density
                + 1.5 * min(fractions[0], 0.10)
                + 1.5 * min(fractions[6], 0.10)
                + 0.8 * min(fractions[1] + fractions[2] + fractions[5], 0.15)
                + 0.03 * diversity
            )
            west, north = transform * (col0, row0)
            east, south = transform * (col0 + window_px, row0 + window_px)
            rows.append({
                "row": row0, "col": col0, "west": west, "south": south, "east": east, "north": north,
                "valid_fraction": float(v.mean()), "crop_fraction": fractions[4], "water_fraction": fractions[0],
                "built_fraction": fractions[6], "tree_grass_shrub_fraction": fractions[1] + fractions[2] + fractions[5],
                "crop_edge_density": edge_density, "class_diversity_ge_1pct": diversity, "selection_score": score,
            })
    ranking = pd.DataFrame(rows).sort_values(["selection_score", "row", "col"], ascending=[False, True, True]).reset_index(drop=True)
    if ranking.empty:
        raise RuntimeError("No candidate met the predeclared land-cover mixture rules")
    selected = ranking.iloc[0].to_dict()
    geom = box(selected["west"], selected["south"], selected["east"], selected["north"])
    tx = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon_min, lat_min = tx.transform(selected["west"], selected["south"])
    lon_max, lat_max = tx.transform(selected["east"], selected["north"])
    props = {
        "aoi_id": "chongming_field_test_5km_v2_label_free",
        "selection_method": "dynamic_world_landcover_mixture_v2",
        "selection_data": "2022 Dynamic World temporal mode only; retained probability raster metadata supplied study extent/grid",
        "rice_labels_used": False, "rice_prediction_values_used": False,
        "width_m": 5000, "height_m": 5000, "crs_epsg": crs.to_epsg(),
        "wgs84_lon_min": lon_min, "wgs84_lat_min": lat_min,
        "wgs84_lon_max": lon_max, "wgs84_lat_max": lat_max,
        **{k: (int(v) if k in {"row", "col", "class_diversity_ge_1pct"} else float(v)) for k, v in selected.items()},
    }
    collection = {"type": "FeatureCollection", "name": "field_delineation_test_aoi", "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::32651"}}, "features": [{"type": "Feature", "properties": props, "geometry": mapping(geom)}]}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(collection, ensure_ascii=False, indent=2), encoding="utf-8")
    METADATA.write_text(json.dumps({"selected": props, "top_10_candidates": ranking.head(10).to_dict(orient="records"), "selection_raster": str(DW.relative_to(ROOT)), "rice_labels_used": False, "rice_prediction_values_used": False}, indent=2), encoding="utf-8")
    print(json.dumps(props, indent=2))


if __name__ == "__main__":
    main()

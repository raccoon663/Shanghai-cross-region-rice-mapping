from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from scipy import ndimage
from shapely.geometry import box, mapping, shape


ROOT = Path(__file__).resolve().parents[3]
DW = ROOT / "work/field_delineation/input_data/chongming_dynamic_world_mode_2022_20m.tif"
AOI1 = ROOT / "data/field_delineation_test_aoi.geojson"
OUT = ROOT / "outputs/field_delineation/ftw_test_aoi_2/aoi_2.geojson"
REPORT = ROOT / "reports/phase3b_aoi2_freeze.md"
META = ROOT / "work/field_delineation/ftw/aoi2_selection_label_free.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    window_px, stride_px = 250, 25  # fixed 5 km window, 500 m stride
    aoi1_doc = json.loads(AOI1.read_text(encoding="utf-8"))
    aoi1 = shape(aoi1_doc["features"][0]["geometry"])
    with rasterio.open(DW) as src:
        labels = src.read(1)
        transform, crs, bounds = src.transform, src.crs, src.bounds

    rows: list[dict] = []
    for row0 in range(0, labels.shape[0] - window_px + 1, stride_px):
        for col0 in range(0, labels.shape[1] - window_px + 1, stride_px):
            z = labels[row0:row0 + window_px, col0:col0 + window_px]
            fractions = {code: float((z == code).mean()) for code in range(9)}
            crop = z == 4
            crop_edge = crop ^ ndimage.binary_erosion(crop, structure=np.ones((3, 3)))
            edge_density = float(crop_edge.mean())
            diversity = sum(fractions[c] >= 0.01 for c in range(8))
            context_ok = fractions[0] >= 0.01 and fractions[6] >= 0.01 and (fractions[1] + fractions[2] + fractions[5]) >= 0.02
            if not (0.30 <= fractions[4] <= 0.82 and context_ok and diversity >= 4):
                continue
            west, north = transform * (col0, row0)
            east, south = transform * (col0 + window_px, row0 + window_px)
            candidate = box(west, south, east, north)
            distance = float(candidate.distance(aoi1))
            if candidate.intersects(aoi1) or distance < 2000:
                continue
            score = (
                2.2 * fractions[4] + 6.0 * edge_density
                + 1.5 * min(fractions[0], 0.10) + 1.5 * min(fractions[6], 0.10)
                + 0.8 * min(fractions[1] + fractions[2] + fractions[5], 0.15)
                + 0.03 * diversity
            )
            rows.append({
                "row": row0, "col": col0, "west": west, "south": south, "east": east, "north": north,
                "crop_fraction": fractions[4], "water_fraction": fractions[0], "built_fraction": fractions[6],
                "tree_grass_shrub_fraction": fractions[1] + fractions[2] + fractions[5],
                "crop_edge_density": edge_density, "class_diversity_ge_1pct": diversity,
                "distance_from_aoi1_m": distance, "selection_score": score,
            })
    ranking = pd.DataFrame(rows).sort_values(["selection_score", "row", "col"], ascending=[False, True, True]).reset_index(drop=True)
    if ranking.empty:
        raise RuntimeError("No non-overlapping AOI 2 candidate met the frozen label-free mixture rules")
    selected = ranking.iloc[0].to_dict()
    geom = box(selected["west"], selected["south"], selected["east"], selected["north"])
    tx = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon_min, lat_min = tx.transform(selected["west"], selected["south"])
    lon_max, lat_max = tx.transform(selected["east"], selected["north"])
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    props = {
        "aoi_id": "chongming_field_test_5km_phase3b_aoi2_label_free",
        "selection_method": "frozen_dynamic_world_landcover_mixture_v2_with_nonoverlap",
        "selection_data": "2022 Dynamic World temporal mode only; no rice probability values or rice labels inspected",
        "selection_timestamp": timestamp,
        "rice_labels_used": False, "rice_prediction_values_used": False,
        "width_m": 5000, "height_m": 5000, "crs_epsg": int(crs.to_epsg()),
        "wgs84_lon_min": lon_min, "wgs84_lat_min": lat_min,
        "wgs84_lon_max": lon_max, "wgs84_lat_max": lat_max,
        **{k: (int(v) if k in {"row", "col", "class_diversity_ge_1pct"} else float(v)) for k, v in selected.items()},
    }
    doc = {
        "type": "FeatureCollection", "name": "phase3b_aoi_2",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::32651"}},
        "features": [{"type": "Feature", "properties": props, "geometry": mapping(geom)}],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    aoi_hash = sha256(OUT)
    META.write_text(json.dumps({
        "selected": props, "top_10_eligible_candidates": ranking.head(10).to_dict(orient="records"),
        "selection_raster": str(DW.relative_to(ROOT)), "selection_raster_sha256": sha256(DW),
        "selection_raster_bounds": list(bounds), "aoi1_sha256": sha256(AOI1), "aoi2_sha256": aoi_hash,
        "rice_labels_used": False, "rice_prediction_values_used": False,
    }, indent=2), encoding="utf-8")
    REPORT.write_text(f"""# Phase 3B AOI 2 freeze

- Freeze timestamp: `{timestamp}`
- AOI file: `outputs/field_delineation/ftw_test_aoi_2/aoi_2.geojson`
- AOI SHA-256: `{aoi_hash}`
- CRS: `EPSG:32651`
- Bounds: `({selected['west']:.1f}, {selected['south']:.1f}, {selected['east']:.1f}, {selected['north']:.1f})`
- Size: `5,000 m x 5,000 m` (`25 km2`)
- Minimum edge-to-edge distance from AOI 1: `{selected['distance_from_aoi1_m']:.1f} m`
- Overlap with AOI 1: `0 m2`
- WGS84 bounds: `({lon_min:.7f}, {lat_min:.7f}, {lon_max:.7f}, {lat_max:.7f})`

## Label-free selection rationale

AOI 2 is the highest-ranked eligible 5 km window under the same predeclared Dynamic World mixture logic used for AOI 1, after excluding every window that overlaps AOI 1 or lies within 2 km of it. The selected window contains a dense but fragmented crop mosaic together with water/canals, built land, and tree/grass/shrub cover. Its Dynamic World fractions are crops `{selected['crop_fraction']:.4f}`, water `{selected['water_fraction']:.4f}`, built `{selected['built_fraction']:.4f}`, and tree/grass/shrub `{selected['tree_grass_shrub_fraction']:.4f}`. Crop-edge density is `{selected['crop_edge_density']:.4f}`.

The selection used only the 2022 Dynamic World temporal-mode land-cover raster and the retained study raster's extent/grid metadata. It did not read rice probability values, official Shanghai rice products, or manually labelled rice/non-rice parcels.

> AOI 2 was frozen before inspecting rice classification outcomes.
""", encoding="utf-8")
    print(json.dumps({"aoi2": props, "aoi2_sha256": aoi_hash}, indent=2))


if __name__ == "__main__":
    main()

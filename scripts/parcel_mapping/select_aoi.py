from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.features import bounds as geometry_bounds
from scipy import ndimage
from shapely.geometry import box, mapping


ROOT = Path(__file__).resolve().parents[2]
PROBABILITY = ROOT / "outputs/experiments/rf_baseline_m2_v1/shanghai_source_only_probability_s1_s2_fusion.tif"
REFERENCE = ROOT / "inputs/official_reference/shanghai_2022_rice_aligned_20m.tif"
OUTPUT = ROOT / "data/field_delineation_test_aoi.geojson"
METADATA = ROOT / "work/field_delineation/aoi_selection.json"


def main() -> None:
    window_px = 250  # 5 km on the retained 20 m grid
    stride_px = 25   # evaluate candidates every 500 m

    with rasterio.open(PROBABILITY) as prob_ds, rasterio.open(REFERENCE) as ref_ds:
        if (
            prob_ds.crs != ref_ds.crs
            or prob_ds.transform != ref_ds.transform
            or prob_ds.shape != ref_ds.shape
        ):
            raise ValueError("Probability and weak-reference rasters are not aligned")
        probability = prob_ds.read(1).astype("float32")
        reference = ref_ds.read(1) > 0
        valid = np.isfinite(probability) & (probability != prob_ds.nodata)
        transform = prob_ds.transform
        crs = prob_ds.crs

    rows: list[dict] = []
    for row0 in range(0, probability.shape[0] - window_px + 1, stride_px):
        for col0 in range(0, probability.shape[1] - window_px + 1, stride_px):
            rs = slice(row0, row0 + window_px)
            cs = slice(col0, col0 + window_px)
            local_valid = valid[rs, cs]
            valid_fraction = float(local_valid.mean())
            if valid_fraction < 0.98:
                continue
            rice = reference[rs, cs] & local_valid
            rice_fraction = float(rice.sum() / local_valid.sum())
            if not 0.12 <= rice_fraction <= 0.70:
                continue
            eroded = ndimage.binary_erosion(rice)
            boundary_density = float((rice ^ eroded).sum() / local_valid.sum())
            components, n_components = ndimage.label(rice, structure=np.ones((3, 3)))
            component_sizes = np.bincount(components.ravel())[1:]
            substantial_components = int((component_sizes >= 9).sum())
            p = probability[rs, cs][local_valid]
            probability_std = float(np.std(p))
            probability_mid_fraction = float(((p >= 0.2) & (p <= 0.8)).mean())
            mix_score = 4.0 * rice_fraction * (1.0 - rice_fraction)
            score = (
                2.0 * mix_score
                + 12.0 * boundary_density
                + 0.8 * probability_std
                + 0.4 * probability_mid_fraction
                + 0.002 * min(substantial_components, 250)
            )
            west, north = transform * (col0, row0)
            east, south = transform * (col0 + window_px, row0 + window_px)
            rows.append(
                {
                    "row": row0,
                    "col": col0,
                    "west": west,
                    "south": south,
                    "east": east,
                    "north": north,
                    "valid_fraction": valid_fraction,
                    "rice_fraction_weak_reference": rice_fraction,
                    "boundary_density": boundary_density,
                    "substantial_components": substantial_components,
                    "probability_std": probability_std,
                    "probability_mid_fraction": probability_mid_fraction,
                    "selection_score": score,
                }
            )

    ranking = pd.DataFrame(rows).sort_values("selection_score", ascending=False).reset_index(drop=True)
    if ranking.empty:
        raise RuntimeError("No AOI candidate satisfied the declared constraints")
    selected = ranking.iloc[0].to_dict()
    geom = box(selected["west"], selected["south"], selected["east"], selected["north"])
    to_wgs84 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon_min, lat_min = to_wgs84.transform(selected["west"], selected["south"])
    lon_max, lat_max = to_wgs84.transform(selected["east"], selected["north"])
    feature = {
        "type": "Feature",
        "properties": {
            "aoi_id": "chongming_field_test_5km_v1",
            "selection_method": "automatic_20m_window_score_v1",
            "selection_uses_weak_reference_only_as_sampling_aid": True,
            "width_m": 5000,
            "height_m": 5000,
            "crs_epsg": crs.to_epsg(),
            "wgs84_lon_min": lon_min,
            "wgs84_lat_min": lat_min,
            "wgs84_lon_max": lon_max,
            "wgs84_lat_max": lat_max,
            **{key: (int(value) if key in {"row", "col", "substantial_components"} else float(value))
               for key, value in selected.items()},
        },
        "geometry": mapping(geom),
    }
    collection = {
        "type": "FeatureCollection",
        "name": "field_delineation_test_aoi",
        "crs": {"type": "name", "properties": {"name": f"urn:ogc:def:crs:EPSG::{crs.to_epsg()}"}},
        "features": [feature],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(collection, ensure_ascii=False, indent=2), encoding="utf-8")
    METADATA.write_text(
        json.dumps(
            {
                "selected": feature["properties"],
                "top_10_candidates": ranking.head(10).to_dict(orient="records"),
                "probability_raster": str(PROBABILITY.relative_to(ROOT)),
                "weak_reference_raster": str(REFERENCE.relative_to(ROOT)),
                "note": "Weak reference is used only to locate a mixed agricultural feasibility AOI, not to tune or evaluate field boundaries.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(feature["properties"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

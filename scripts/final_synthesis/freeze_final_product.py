from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import rasterio


ROOT = Path(__file__).resolve().parents[2]
P5 = ROOT / "outputs" / "phase5_parcel_safeguards" / "products"
P4 = ROOT / "outputs" / "phase4_chongming_staged" / "products"
OUT = ROOT / "outputs" / "final_chongming_parcel_product"
MANIFEST = ROOT / "outputs" / "manifests" / "final_chongming_parcel_product_manifest.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def copy_exact(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    if sha256(src) != sha256(dst):
        raise RuntimeError(f"Copy verification failed: {src}")


def vector_metadata(path: Path) -> dict:
    frame = gpd.read_file(path)
    return {
        "feature_count": int(len(frame)),
        "crs": str(frame.crs),
        "bounds": [float(x) for x in frame.total_bounds],
        "columns": list(frame.columns),
    }


def raster_metadata(path: Path) -> dict:
    with rasterio.open(path) as src:
        return {
            "crs": str(src.crs),
            "bounds": [src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top],
            "shape": [src.height, src.width],
            "count": src.count,
            "dtype": list(src.dtypes),
            "nodata": src.nodata,
            "transform": list(src.transform)[:6],
        }


def write_projection(source: Path, destination: Path, columns: list[str]) -> None:
    frame = gpd.read_file(source)
    keep = [c for c in columns if c in frame.columns] + ["geometry"]
    frame[keep].to_file(destination, driver="GPKG", layer=destination.stem)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    sources = {
        "product_A_raw_ftw_parcels": P5 / "A_raw_ftw_parcels.gpkg",
        "product_B_deployment_safe_parcels": P5 / "B_deployment_safe_parcels.gpkg",
        "product_C_QA_risk_excluded_parcels": P5 / "C_QA_risk_only.gpkg",
        "final_M0_raw_probability": P4 / "M0_raw_probability.tif",
        "final_safe_M1_probability": P5 / "M1_deployment_safe_probability.tif",
        "final_safe_M1b_probability": P5 / "M1b_deployment_safe_probability.tif",
    }
    destinations = {
        key: OUT / src.name for key, src in sources.items()
    }
    for key, src in sources.items():
        if not src.exists():
            raise FileNotFoundError(src)
        copy_exact(src, destinations[key])

    m2 = P5 / "M2_deployment_safe_parcel_predictions.gpkg"
    probability = OUT / "final_parcel_probability.gpkg"
    classes = OUT / "final_parcel_class.gpkg"
    qa = OUT / "final_parcel_QA.gpkg"
    common = ["field_id", "source_tile_id", "area_ha", "repaired_area_m2", "mapping_status"]
    write_projection(m2, probability, common + [
        "n_valid_pixels_phase5", "rice_prob_mean_phase5", "rice_prob_median_phase5",
        "rice_prob_std_phase5", "rice_positive_fraction_phase5",
    ])
    write_projection(m2, classes, common + [
        "class_by_mean_phase5", "class_by_median_phase5",
        "class_by_positive_fraction_phase5", "summary_disagreement_phase5",
        "parcel_class_phase5",
    ])
    write_projection(m2, qa, common + [
        "parcel_valid_for_mapping", "parcel_exclusion_reason", "parcel_geometry_risk",
        "parcel_landcover_risk", "safeguard_QA_flag", "phase5_qa_flag_count",
        "geometry_repaired_flag", "deployment_edge_flag",
        "cross_tile_reconciliation_flag", "tile_disagreement_fraction",
    ])
    destinations.update({
        "final_parcel_probability": probability,
        "final_parcel_class": classes,
        "final_parcel_QA": qa,
    })

    artifacts = {}
    for role, path in destinations.items():
        suffix = path.suffix.lower()
        metadata = vector_metadata(path) if suffix == ".gpkg" else raster_metadata(path)
        artifacts[role] = {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "metadata": metadata,
        }

    manifest = {
        "manifest_version": "1.0",
        "product_version": "final_chongming_parcel_prototype_v1_20260814",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Chongming prototype only; no all-Shanghai extension",
        "scientific_result_lock": (
            "Artifacts listed below are frozen scientific results. Figures, tables, and reports "
            "must reference them without rewriting them."
        ),
        "provenance": {
            "phase5_safeguard_contract_sha256": "39c1b1f8b54ad6b85cebcea4b1626ceba7b2dbc5125c03b43adc0a16c17a77ba",
            "phase4_raw_parcel_sha256": "b41c4fbd825057efe4dca4743f4aca79b524446d7e7b11fc0f3cc9be96dce194",
            "classifier": "unchanged Jiangxi-trained source-only RF; no Shanghai retraining",
            "field_model": "frozen FTW PRUE workflow; no Phase-5 or final-phase tuning",
            "safeguards": "label-independent Phase-5 geometry/land-cover/deployment safeguards",
        },
        "claim_limits": [
            "Deployment and structural QA metrics are not independent Shanghai rice accuracy.",
            "No Shanghai rice reference labels were used to tune FTW or safeguards.",
            "Independent parcel validation remains pending.",
        ],
        "artifacts": artifacts,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(MANIFEST)
    print(json.dumps({k: v["sha256"] for k, v in artifacts.items()}, indent=2))


if __name__ == "__main__":
    main()

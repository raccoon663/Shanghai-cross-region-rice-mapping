"""Submit restartable Earth Engine exports for frozen Presto point inputs.

This stage reconstructs native S1/S2 values at the exact frozen sample
locations and temporal windows. It does not construct embeddings, fit models,
or evaluate target labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/eofm_input_reconstruction.yaml"
DEFAULT_STATUS = ROOT / "outputs/eofm/ee/presto_point_exports.json"
EXPORT_PROPERTIES = ["sample_id", "region", "class_id", "label_source", "spatial_block", "split"]
TERMINAL_TASK_STATES = {"COMPLETED", "FAILED", "CANCELLED", "CANCEL_REQUESTED"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_contract(config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    grid_path = ROOT / config["temporal_grid_manifest"]
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    if grid["timestep_count"] != len(grid["windows"]) or grid["timestep_count"] != 23:
        raise ValueError("The frozen temporal grid must contain exactly 23 windows")
    tags = [window["tag"] for window in grid["windows"]]
    if tags != [f"t{index:02d}" for index in range(1, 24)]:
        raise ValueError("Frozen temporal tags are missing or out of order")
    return config, grid


def validate_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"sample_id": str})
    required = set(EXPORT_PROPERTIES + ["longitude", "latitude"])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Sample manifest is missing columns: {missing}")
    if len(frame) != 13429:
        raise ValueError(f"Expected 13,429 frozen rows, found {len(frame):,}")
    if frame.duplicated(["region", "sample_id"]).any():
        raise ValueError("Duplicate frozen region/sample_id keys")
    if frame.groupby(["region", "spatial_block"]).split.nunique().max() != 1:
        raise ValueError("Spatial blocks cross frozen splits")
    if frame.region.value_counts().to_dict() != {"shanghai": 12000, "jiangxi": 1429}:
        raise ValueError("Unexpected frozen region counts")
    return frame


def band_schema(config: dict[str, Any], grid: dict[str, Any]) -> list[str]:
    ee_config = config["earth_engine"]
    result: list[str] = []
    for window in grid["windows"]:
        tag = window["tag"]
        result.extend(f"s2_{band.lower()}_{tag}" for band in ee_config["sentinel_2_bands"])
        result.extend([f"s2_ndvi_{tag}", f"s2_valid_{tag}", f"s2_count_{tag}"])
        result.extend([f"s1_vv_{tag}", f"s1_vh_{tag}", f"s1_rvi_{tag}", f"s1_valid_{tag}", f"s1_count_{tag}"])
    return result


def chunk_plan(rows: int, chunk_size: int) -> list[dict[str, int | str]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    plan = []
    for chunk_index, start in enumerate(range(0, rows, chunk_size)):
        stop = min(start + chunk_size, rows)
        plan.append({
            "chunk_index": chunk_index,
            "start": start,
            "stop": stop,
            "rows": stop - start,
            "description": f"eofm_presto_points_2022_c{chunk_index:03d}_{start:05d}_{stop:05d}",
        })
    return plan


def _feature_collection(ee: Any, frame: pd.DataFrame) -> Any:
    features = []
    for row in frame.itertuples(index=False):
        properties = {name: getattr(row, name) for name in EXPORT_PROPERTIES}
        properties["class_id"] = int(properties["class_id"])
        features.append(
            ee.Feature(ee.Geometry.Point([float(row.longitude), float(row.latitude)]), properties)
        )
    return ee.FeatureCollection(features)


def _stack(ee: Any, config: dict[str, Any], grid: dict[str, Any], region: Any) -> Any:
    ee_config = config["earth_engine"]
    missing = float(ee_config["missing_value"])
    s2_bands = list(ee_config["sentinel_2_bands"])
    stack = ee.Image([])

    def mask_s2(image: Any) -> Any:
        scl = image.select("SCL")
        clear = ee.Image(1)
        for code in ee_config["sentinel_2_scl_excluded"]:
            clear = clear.And(scl.neq(int(code)))
        return image.updateMask(clear)

    for window in grid["windows"]:
        tag = window["tag"]
        s2_collection = (
            ee.ImageCollection(ee_config["sentinel_2_collection"])
            .filterBounds(region)
            .filterDate(window["start_inclusive"], window["end_exclusive"])
            .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", ee_config["sentinel_2_scene_cloud_max_percent"]))
        )
        clear_s2 = s2_collection.map(mask_s2)
        s2_native = clear_s2.select(s2_bands).median()
        s2_names = [f"s2_{band.lower()}_{tag}" for band in s2_bands]
        s2_valid = s2_native.select("B2").mask().rename(f"s2_valid_{tag}").unmask(0).toUint8()
        s2_count = clear_s2.select("B2").count().rename(f"s2_count_{tag}").unmask(0).toUint16()
        legacy_ndvi = (
            clear_s2.map(lambda image: image.normalizedDifference(["B8", "B4"]))
            .median()
            .rename(f"s2_ndvi_{tag}")
            .unmask(missing)
        )
        stack = stack.addBands(s2_native.rename(s2_names).unmask(missing))
        stack = stack.addBands(legacy_ndvi).addBands(s2_valid).addBands(s2_count)

        s1_collection = (
            ee.ImageCollection(ee_config["sentinel_1_collection"])
            .filterBounds(region)
            .filterDate(window["start_inclusive"], window["end_exclusive"])
            .filter(ee.Filter.eq("instrumentMode", ee_config["sentinel_1_instrument_mode"]))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .select(["VV", "VH"])
        )
        s1_native = s1_collection.median()
        vv = s1_native.select("VV")
        vh = s1_native.select("VH")
        vv_linear = ee.Image(10).pow(vv.divide(10))
        vh_linear = ee.Image(10).pow(vh.divide(10))
        rvi = vh_linear.multiply(4).divide(vv_linear.add(vh_linear)).rename(f"s1_rvi_{tag}")
        s1_valid = vv.mask().rename(f"s1_valid_{tag}").unmask(0).toUint8()
        s1_count = s1_collection.select("VV").count().rename(f"s1_count_{tag}").unmask(0).toUint16()
        stack = stack.addBands(vv.rename(f"s1_vv_{tag}").unmask(missing))
        stack = stack.addBands(vh.rename(f"s1_vh_{tag}").unmask(missing))
        stack = stack.addBands(rvi.unmask(missing))
        stack = stack.addBands(s1_valid).addBands(s1_count)
    return stack


def _load_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"manifest_version": "1.0", "tasks": []}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_status(path: Path, status: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(status, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--project", help="Authorized Google Cloud project registered for Earth Engine")
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--start-chunk", type=int, default=0)
    parser.add_argument("--max-chunks", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--refresh-status", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config, grid = load_contract(config_path)
    manifest_path = ROOT / config["sample_manifest"]
    frame = validate_manifest(manifest_path)
    chunk_size = args.chunk_size or int(config["earth_engine"]["chunk_size"])
    plan = chunk_plan(len(frame), chunk_size)
    schema = band_schema(config, grid)
    summary = {
        "sample_manifest": str(manifest_path),
        "sample_manifest_sha256": sha256(manifest_path),
        "rows": len(frame),
        "chunks": len(plan),
        "chunk_size": chunk_size,
        "feature_bands": len(schema),
        "first_bands": schema[:8],
        "last_bands": schema[-8:],
        "drive_folder": config["earth_engine"]["drive_folder"],
    }
    if args.dry_run:
        print(json.dumps({**summary, "plan": plan}, indent=2))
        return
    project = args.project or config["earth_engine"].get("project")
    if not project:
        raise SystemExit("--project is required for Earth Engine task submission")

    import ee

    ee.Initialize(project=project)
    status = _load_status(args.status)
    existing = {int(task["chunk_index"]): task for task in status.get("tasks", [])}
    if args.refresh_status:
        for task_record in existing.values():
            task = ee.batch.Task(task_record["task_id"])
            task_record.update(task.status())
        status.update(summary)
        _write_status(args.status, status)

    selected = [item for item in plan if int(item["chunk_index"]) >= args.start_chunk]
    if args.max_chunks is not None:
        selected = selected[: args.max_chunks]
    selectors = EXPORT_PROPERTIES + schema
    for item in selected:
        chunk_index = int(item["chunk_index"])
        previous = existing.get(chunk_index)
        if previous and previous.get("state") not in TERMINAL_TASK_STATES:
            print(f"skip active chunk {chunk_index}: {previous.get('state')}")
            continue
        if previous and previous.get("state") == "COMPLETED":
            print(f"skip completed chunk {chunk_index}")
            continue
        part = frame.iloc[int(item["start"]):int(item["stop"])]
        collection = _feature_collection(ee, part)
        image = _stack(ee, config, grid, collection.geometry().bounds())
        sampled = image.sampleRegions(
            collection=collection,
            properties=EXPORT_PROPERTIES,
            projection=ee.Projection(
                config["earth_engine"]["sample_crs"],
                config["earth_engine"]["sample_crs_transform"],
            ),
            geometries=False,
            tileScale=4,
        )
        task = ee.batch.Export.table.toDrive(
            collection=sampled,
            description=str(item["description"]),
            folder=config["earth_engine"]["drive_folder"],
            fileNamePrefix=str(item["description"]),
            fileFormat="CSV",
            selectors=selectors,
        )
        task.start()
        record = {**item, "task_id": task.id, "state": task.status().get("state", "SUBMITTED")}
        existing[chunk_index] = record
        status.update(summary)
        status["project"] = project
        status["tasks"] = [existing[index] for index in sorted(existing)]
        _write_status(args.status, status)
        print(json.dumps(record))


if __name__ == "__main__":
    main()

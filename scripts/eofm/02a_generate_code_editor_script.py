"""Generate a private Earth Engine Code Editor fallback submission script.

The generated JavaScript embeds frozen point coordinates, so its default
output is under the gitignored runtime directory and must not be committed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/eofm_input_reconstruction.yaml"
DEFAULT_OUTPUT = ROOT / "outputs/eofm/ee/presto_code_editor_private.js"


def compact(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-chunk", type=int, default=0)
    parser.add_argument("--end-chunk", type=int, default=27,
                        help="Exclusive chunk bound; defaults to all remaining chunks")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    ee_config = config["earth_engine"]
    grid = json.loads((ROOT / config["temporal_grid_manifest"]).read_text(encoding="utf-8"))
    frame = pd.read_csv(ROOT / config["sample_manifest"])
    if len(frame) != 13429 or frame.region.tolist() != ["jiangxi"] * 1429 + ["shanghai"] * 12000:
        raise ValueError("Frozen sample order or region membership changed")
    if not 0 <= args.start_chunk < args.end_chunk <= 27:
        raise ValueError("chunk bounds must satisfy 0 <= start < end <= 27")

    selected_start = args.start_chunk * 500
    selected_stop = min(args.end_chunk * 500, len(frame))
    selected = frame.iloc[selected_start:selected_stop]
    xy = selected[["longitude", "latitude"]].astype(float).values.tolist()
    windows = [
        [item["tag"], item["start_inclusive"], item["end_exclusive"]]
        for item in grid["windows"]
    ]
    s2_bands = list(ee_config["sentinel_2_bands"])
    feature_bands: list[str] = []
    for tag, _, _ in windows:
        feature_bands.extend(f"s2_{band.lower()}_{tag}" for band in s2_bands)
        feature_bands.extend([f"s2_ndvi_{tag}", f"s2_valid_{tag}", f"s2_count_{tag}"])
        feature_bands.extend([f"s1_vv_{tag}", f"s1_vh_{tag}", f"s1_rvi_{tag}", f"s1_valid_{tag}", f"s1_count_{tag}"])

    lines = [
        f"var xy={compact(xy)};",
        f"var selectedStart={selected_start};",
        "var points=ee.FeatureCollection(xy.map(function(p,i){var row=i+selectedStart;return ee.Feature(ee.Geometry.Point(p),{manifest_row:row,region:row<1429?'jiangxi':'shanghai'});}));",
        f"var windows={compact(windows)};",
        f"var s2Bands={compact(s2_bands)};",
        f"var excluded={compact(ee_config['sentinel_2_scl_excluded'])};",
        f"var missing={float(ee_config['missing_value'])};",
        f"var projection=ee.Projection({compact(ee_config['sample_crs'])},{compact(ee_config['sample_crs_transform'])});",
        "function maskS2(image){var scl=image.select('SCL');var clear=ee.Image(1);excluded.forEach(function(code){clear=clear.and(scl.neq(code));});return image.updateMask(clear);}",
        "var stack=ee.Image([]);",
        "windows.forEach(function(w){var tag=w[0],start=w[1],end=w[2];"
        f"var s2=ee.ImageCollection('{ee_config['sentinel_2_collection']}').filterBounds(points).filterDate(start,end).filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE',{ee_config['sentinel_2_scene_cloud_max_percent']}));"
        "var clearS2=s2.map(maskS2);var s2Native=clearS2.select(s2Bands).median();"
        "var s2Names=s2Bands.map(function(b){return 's2_'+b.toLowerCase()+'_'+tag;});"
        "var s2Valid=s2Native.select('B2').mask().rename('s2_valid_'+tag).unmask(0).toUint8();"
        "var s2Count=clearS2.select('B2').count().rename('s2_count_'+tag).unmask(0).toUint16();"
        "var ndvi=clearS2.map(function(image){return image.normalizedDifference(['B8','B4']);}).median().rename('s2_ndvi_'+tag).unmask(missing);"
        "stack=stack.addBands(s2Native.rename(s2Names).unmask(missing)).addBands(ndvi).addBands(s2Valid).addBands(s2Count);"
        f"var s1=ee.ImageCollection('{ee_config['sentinel_1_collection']}').filterBounds(points).filterDate(start,end).filter(ee.Filter.eq('instrumentMode','{ee_config['sentinel_1_instrument_mode']}')).filter(ee.Filter.listContains('transmitterReceiverPolarisation','VV')).filter(ee.Filter.listContains('transmitterReceiverPolarisation','VH')).select(['VV','VH']);"
        "var s1Native=s1.median(),vv=s1Native.select('VV'),vh=s1Native.select('VH');"
        "var vvLinear=ee.Image(10).pow(vv.divide(10)),vhLinear=ee.Image(10).pow(vh.divide(10));"
        "var rvi=vhLinear.multiply(4).divide(vvLinear.add(vhLinear)).rename('s1_rvi_'+tag);"
        "var s1Valid=vv.mask().rename('s1_valid_'+tag).unmask(0).toUint8();"
        "var s1Count=s1.select('VV').count().rename('s1_count_'+tag).unmask(0).toUint16();"
        "stack=stack.addBands(vv.rename('s1_vv_'+tag).unmask(missing)).addBands(vh.rename('s1_vh_'+tag).unmask(missing)).addBands(rvi.unmask(missing)).addBands(s1Valid).addBands(s1Count);});",
        f"var selectors=['manifest_row','region'].concat({compact(feature_bands)});",
        f"for(var c={args.start_chunk};c<{args.end_chunk};c++){{var start=c*500,stop=Math.min(start+500,13429);var desc='eofm_presto_points_2022_c'+('000'+c).slice(-3)+'_'+('00000'+start).slice(-5)+'_'+('00000'+stop).slice(-5);var part=ee.FeatureCollection(points.toList(stop-start,start-selectedStart));var sampled=stack.sampleRegions({{collection:part,properties:['manifest_row','region'],projection:projection,geometries:false,tileScale:4}});Export.table.toDrive({{collection:sampled,description:desc,folder:'{ee_config['drive_folder']}',fileNamePrefix:desc,fileFormat:'CSV',selectors:selectors}});}}",
        "print('selected_rows',points.size());print('feature_bands',stack.bandNames().size());",
    ]
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"output": str(output), "rows": len(frame), "selected_rows": len(selected), "start_chunk": args.start_chunk, "end_chunk": args.end_chunk, "tasks": args.end_chunk - args.start_chunk, "feature_bands": len(feature_bands)}))


if __name__ == "__main__":
    main()

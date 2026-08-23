"""Generate private GEE Code Editor exports for frozen 3x3 Galileo patches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/eofm_input_reconstruction.yaml"
DEFAULT_OUTPUT = ROOT / "outputs/eofm/ee/galileo_patch_code_editor_private.js"


def compact(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True)


def parse_chunks(value: str) -> list[int]:
    if value.strip().lower() == "all":
        return list(range(27))
    chunks = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not chunks or chunks[0] < 0 or chunks[-1] > 26:
        raise ValueError("chunks must be a comma-separated subset of 0..26 or 'all'")
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--chunks", default="0,3",
        help="Center-row chunks; 0 and 3 cover the source/target smoke populations",
    )
    args = parser.parse_args()

    config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    ee_config = config["earth_engine"]
    galileo = config["galileo_input"]
    if galileo["patch_pixels"] != 3 or galileo["sample_resolution_m"] != 10:
        raise ValueError("This frozen generator requires the primary 3x3 at 10 m design")
    grid = json.loads((ROOT / config["temporal_grid_manifest"]).read_text(encoding="utf-8"))
    frame = pd.read_csv(ROOT / config["sample_manifest"])
    if len(frame) != 13429 or frame.region.tolist() != ["jiangxi"] * 1429 + ["shanghai"] * 12000:
        raise ValueError("Frozen sample order or region membership changed")
    chunks = parse_chunks(args.chunks)
    selected = frame.iloc[
        [row for chunk in chunks for row in range(chunk * 500, min((chunk + 1) * 500, len(frame)))]
    ]
    centers = [
        [int(row.Index), float(row.longitude), float(row.latitude), int(row.Index // 500)]
        for row in selected.itertuples()
    ]
    windows = [
        [item["tag"], item["start_inclusive"], item["end_exclusive"]]
        for item in grid["windows"]
    ]
    s2_bands = list(ee_config["sentinel_2_bands"])
    feature_bands: list[str] = []
    for tag, _, _ in windows:
        feature_bands.extend(f"s2_{band.lower()}_{tag}" for band in s2_bands)
        feature_bands.extend([f"s2_ndvi_{tag}", f"s2_valid_{tag}", f"s2_count_{tag}"])
        feature_bands.extend(
            [f"s1_vv_{tag}", f"s1_vh_{tag}", f"s1_rvi_{tag}",
             f"s1_valid_{tag}", f"s1_count_{tag}"]
        )
    offsets = [
        {"patch_row": row, "patch_col": col, "dx": col - 1, "dy": 1 - row}
        for row in range(3) for col in range(3)
    ]
    transform = [10, 0, ee_config["sample_crs_transform"][2],
                 0, -10, ee_config["sample_crs_transform"][5]]
    lines = [
        f"var centerRows={compact(centers)};",
        "var centers=ee.FeatureCollection(centerRows.map(function(p){return ee.Feature(ee.Geometry.Point([p[1],p[2]]),{manifest_row:p[0],chunk_id:p[3],region:p[0]<1429?'jiangxi':'shanghai'});}));",
        f"var offsets={compact(offsets)};",
        f"var projection=ee.Projection({compact(ee_config['sample_crs'])},{compact(transform)});",
        "var points=ee.FeatureCollection(centers.map(function(f){var xy=ee.List(f.geometry().transform(projection,1).coordinates());return ee.FeatureCollection(offsets.map(function(o){var x=ee.Number(xy.get(0)).add(ee.Number(o.dx).multiply(10));var y=ee.Number(xy.get(1)).add(ee.Number(o.dy).multiply(10));return ee.Feature(ee.Geometry.Point([x,y],projection),{manifest_row:f.get('manifest_row'),chunk_id:f.get('chunk_id'),region:f.get('region'),patch_row:o.patch_row,patch_col:o.patch_col});}));}).flatten());",
        f"var windows={compact(windows)};",
        f"var s2Bands={compact(s2_bands)};",
        f"var excluded={compact(ee_config['sentinel_2_scl_excluded'])};",
        f"var missing={float(ee_config['missing_value'])};",
        "function maskS2(image){var scl=image.select('SCL');var clear=ee.Image(1);excluded.forEach(function(code){clear=clear.and(scl.neq(code));});return image.updateMask(clear);}",
        "var stack=ee.Image([]);",
        "windows.forEach(function(w){var tag=w[0],start=w[1],end=w[2];"
        f"var s2=ee.ImageCollection('{ee_config['sentinel_2_collection']}').filterBounds(centers).filterDate(start,end).filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE',{ee_config['sentinel_2_scene_cloud_max_percent']}));"
        "var clearS2=s2.map(maskS2);var s2Native=clearS2.select(s2Bands).median();"
        "var s2Names=s2Bands.map(function(b){return 's2_'+b.toLowerCase()+'_'+tag;});"
        "var s2Valid=s2Native.select('B2').mask().rename('s2_valid_'+tag).unmask(0).toUint8();"
        "var s2Count=clearS2.select('B2').count().rename('s2_count_'+tag).unmask(0).toUint16();"
        "var ndvi=clearS2.map(function(image){return image.normalizedDifference(['B8','B4']);}).median().rename('s2_ndvi_'+tag).unmask(missing);"
        "stack=stack.addBands(s2Native.rename(s2Names).unmask(missing)).addBands(ndvi).addBands(s2Valid).addBands(s2Count);"
        f"var s1=ee.ImageCollection('{ee_config['sentinel_1_collection']}').filterBounds(centers).filterDate(start,end).filter(ee.Filter.eq('instrumentMode','{ee_config['sentinel_1_instrument_mode']}')).filter(ee.Filter.listContains('transmitterReceiverPolarisation','VV')).filter(ee.Filter.listContains('transmitterReceiverPolarisation','VH')).select(['VV','VH']);"
        "var s1Native=s1.median(),vv=s1Native.select('VV'),vh=s1Native.select('VH');"
        "var vvLinear=ee.Image(10).pow(vv.divide(10)),vhLinear=ee.Image(10).pow(vh.divide(10));"
        "var rvi=vhLinear.multiply(4).divide(vvLinear.add(vhLinear)).rename('s1_rvi_'+tag);"
        "var s1Valid=vv.mask().rename('s1_valid_'+tag).unmask(0).toUint8();"
        "var s1Count=s1.select('VV').count().rename('s1_count_'+tag).unmask(0).toUint16();"
        "stack=stack.addBands(vv.rename('s1_vv_'+tag).unmask(missing)).addBands(vh.rename('s1_vh_'+tag).unmask(missing)).addBands(rvi.unmask(missing)).addBands(s1Valid).addBands(s1Count);});",
        f"var selectors=['manifest_row','region','patch_row','patch_col'].concat({compact(feature_bands)});",
        f"var chunks={compact(chunks)};",
        "chunks.forEach(function(c){var start=c*500,stop=Math.min(start+500,13429);var desc='eofm_galileo_patch3_2022_c'+('000'+c).slice(-3)+'_'+('00000'+start).slice(-5)+'_'+('00000'+stop).slice(-5);var part=points.filter(ee.Filter.eq('chunk_id',c));var sampled=stack.sampleRegions({collection:part,properties:['manifest_row','region','patch_row','patch_col'],projection:projection,geometries:false,tileScale:4});Export.table.toDrive({collection:sampled,description:desc,folder:'rice_eofm_galileo_patch3_2022',fileNamePrefix:desc,fileFormat:'CSV',selectors:selectors});});",
        "print('center_rows',centers.size());print('patch_rows',points.size());print('feature_bands',stack.bandNames().size());",
    ]
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({
        "output": str(output), "chunks": chunks, "center_rows": len(selected),
        "patch_rows": len(selected) * 9, "patch_pixels": [3, 3],
        "resolution_m": 10, "feature_bands": len(feature_bands),
    }))


if __name__ == "__main__":
    main()

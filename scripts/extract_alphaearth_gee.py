"""Download 2022 AlphaEarth point embeddings in restartable batches.

Requires an authenticated Earth Engine account. Run `earthengine authenticate`
once, then pass the Cloud project registered for Earth Engine with --project.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import ee
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
BANDS = [f"A{i:02d}" for i in range(64)]
KEYS = ["sample_id", "region", "year", "class_id", "label_source", "spatial_block", "split"]


def feature_collection(frame: pd.DataFrame) -> ee.FeatureCollection:
    features = []
    for row in frame.itertuples(index=False):
        properties = {key: getattr(row, key) for key in KEYS}
        features.append(ee.Feature(ee.Geometry.Point([row.longitude, row.latitude]), properties))
    return ee.FeatureCollection(features)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", help="Optional Google Cloud project registered for Earth Engine")
    parser.add_argument("--manifest", default=ROOT / "data_metadata/alphaearth_sample_manifest.csv", type=Path)
    parser.add_argument("--output-dir", default=ROOT / "outputs/alphaearth/raw_batches", type=Path)
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()

    if args.project:
        ee.Initialize(project=args.project)
    else:
        ee.Initialize()
    image = (ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
             .filterDate("2022-01-01", "2023-01-01").mosaic().select(BANDS))
    frame = pd.read_csv(args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selectors = KEYS + BANDS

    for start in range(0, len(frame), args.batch_size):
        stop = min(start + args.batch_size, len(frame))
        output = args.output_dir / f"alphaearth_2022_{start:05d}_{stop:05d}.csv"
        if output.exists() and output.stat().st_size > 100:
            continue
        batch = frame.iloc[start:stop]
        sampled = image.sampleRegions(collection=feature_collection(batch), scale=10, geometries=False, tileScale=4)
        url = sampled.getDownloadURL(filetype="CSV", selectors=selectors, filename=output.stem)
        response = requests.get(url, timeout=180)
        response.raise_for_status()
        output.write_bytes(response.content)
        print(f"Downloaded {start}:{stop} -> {output.name}")
        time.sleep(0.5)


if __name__ == "__main__":
    main()

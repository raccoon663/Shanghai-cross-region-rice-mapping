from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject


def align(reference: Path, template: Path, output: Path) -> Path:
    with rasterio.open(template) as target, rasterio.open(reference) as source:
        destination = np.zeros((target.height, target.width), dtype="uint8")
        reproject(
            source=rasterio.band(source, 1),
            destination=destination,
            src_transform=source.transform,
            src_crs=source.crs,
            dst_transform=target.transform,
            dst_crs=target.crs,
            src_nodata=source.nodata,
            dst_nodata=0,
            resampling=Resampling.nearest,
        )
        destination = (destination > 0).astype("uint8")
        profile = target.profile.copy()
        profile.update(count=1, dtype="uint8", nodata=None, compress="deflate")
    output.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output, "w", **profile) as destination_file:
        destination_file.write(destination, 1)
        destination_file.set_band_description(1, "official_product_weak_label")
        destination_file.update_tags(
            label_source="official_product",
            evaluation_role="weak_label_only_not_independent_ground_truth",
            source_file=reference.name,
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(align(args.reference, args.template, args.output).resolve())


if __name__ == "__main__":
    main()


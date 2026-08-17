# Data availability and recovery

This repository is designed to be useful without committing multi-gigabyte
GeoTIFF exports. All sensor and embedding inputs are from public Google Earth
Engine collections. The Shanghai map used for evaluation is an **official
product used as a weak reference**, not independent field ground truth.

## What is retained

- the 13,429-row frozen Jiangxi/Shanghai sample and split manifest;
- data registry, schemas, hashes, and reconstruction instructions;
- lightweight final tables, manifests, label-free AOIs, and a blank validation sample;
- publication figures and compact headline metrics;
- SHA-256 values and schemas for the deleted historical Sentinel exports;
- a deterministic public-data reconstruction entry point.

Complete extracted embeddings, feature tables, model binaries, rasters,
GeoPackages, per-tile outputs, and logs are runtime artifacts. They are not
tracked in the public repository.

Run:

```bash
python scripts/check_data.py
```

This prints every expected asset, its resolved location and recovery route.
Use `--strict` in a fully provisioned environment or CI job that is expected to
contain the large data.

## Where to put large data

Either place files under `data/raw/`, or keep them outside the repository and
set `RICE_FUSION_DATA_ROOT` to a directory with the same relative layout:

```text
data/raw/
  jiangxi_official_2022_binary_features_gee_s1_s2.csv
  jiangxi_official_2022_ancillary_features.csv
  shanghai_light_2022_fusion_92_20m.tif
inputs/official_reference/
  shanghai_2022_rice_aligned_20m.tif
```

PowerShell example:

```powershell
$env:RICE_FUSION_DATA_ROOT = 'D:\geo-data\rice-transfer'
python scripts/check_data.py
```

Linux/macOS example:

```bash
export RICE_FUSION_DATA_ROOT=/data/rice-transfer
python scripts/check_data.py
```

## Public Earth Engine sources

- Sentinel-1 GRD: `COPERNICUS/S1_GRD`
- Sentinel-2 surface reflectance: `COPERNICUS/S2_SR_HARMONIZED`
- AlphaEarth annual embeddings: `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`

The 23 documented temporal anchors run from 2022-03-02 to 2022-11-12 and
are stored explicitly in `gee/01_rebuild_public_s1_s2_exports.js`.

## Rebuilding the deleted Sentinel exports

1. Upload `data_metadata/alphaearth_sample_manifest.csv` to Earth Engine.
2. Open `gee/01_rebuild_public_s1_s2_exports.js` in the Code Editor.
3. Replace the manifest asset placeholder.
4. Run the Jiangxi table and Shanghai raster exports.
5. Place the downloaded files under `data/raw/` (or your external data root).
6. Run `python scripts/check_data.py --strict`.
7. Run `python scripts/validate_rebuilt_sentinel.py` before any experiment.

The original files were deleted before the historical GEE script was
recovered. The supplied script reconstructs the documented public collections,
dates, 92-band schema, sample IDs, target bounds, CRS and resolution. It is a
**new data version unless the historical SHA-256 is reproduced**. Never merge a
reconstructed export into published result tables without rerunning the frozen
baseline and recording the new hash.

Historical hashes and expected dimensions are recorded in
`data_metadata/data_registry.yaml`.

## Rebuilding AlphaEarth point embeddings

```bash
pip install -r requirements-alphaearth.txt
earthengine authenticate
python scripts/extract_alphaearth_gee.py --project YOUR_GCP_PROJECT
python scripts/validate_alphaearth_export.py
```

The Code Editor fallback is `gee/03_export_alphaearth_samples.js`.

## Shanghai labels

The official Shanghai product can reproduce the weak-reference experiments,
but it cannot establish field accuracy. The raster itself is not redistributed:
its redistribution rights were not established for this repository. A provider
file list is retained as `data_metadata/official_reference_ftp_manifest.txt`,
and `scripts/download_official_rice_ftp.py` documents the acquisition route.

The repository includes a frozen, blank-label 400-parcel validation sample at
`results/selected_outputs/independent_validation_sample_400.csv`. Independent
labels are optional for running the code but required before making
ground-truth accuracy or true-error claims.

No manual labels are fabricated or inferred from the official product.

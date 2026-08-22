# EO foundation-model extension audit

## Status

This audit was completed before any Presto or Galileo adapter, embedding, or
comparison run was started. The frozen legacy experiments have not been
modified.

The repository contains enough metadata to define a controlled four-way
benchmark, but the current workspace does not contain the raw structured
Sentinel inputs required by Presto or Galileo. It also does not contain several
ignored runtime artifacts from the legacy Temporal-92D and AlphaEarth runs.
Model extraction must remain blocked until the data contract in this document
is satisfied.

## Authoritative sample population and splits

`data_metadata/alphaearth_sample_manifest.csv` is the authoritative shared
sample and split manifest. It has 13,429 rows, no duplicate `(region,
sample_id)` keys, and SHA-256
`04c4e7454ead6cd415f5d761593d9e88b8041dd6611f630064fca798ccf6421b`.

### Jiangxi source

The source population contains 1,429 binary samples: 622 non-rice and 807
rice. The labels are recorded as `official_product`. The spatial split was
created with seed `20260803`, 0.05-degree blocks, a 20% test fraction, and a
20% overall validation fraction. Spatial blocks do not cross splits.

| Split | Non-rice | Rice | Total | Blocks |
|---|---:|---:|---:|---:|
| `source_train` | 370 | 480 | 850 | 49 |
| `source_val` | 115 | 150 | 265 | 17 |
| `source_test` | 137 | 177 | 314 | 17 |

The source validation split may be used for source-only engineering decisions.
The source test split remains held out for final source reporting.

### Shanghai target

The target population contains 12,000 samples derived from the official
Shanghai product used as a weak reference. It was initially sampled as 6,000
non-rice and 6,000 rice pixels after Temporal-92D availability filtering. The
target uses 100-pixel spatial blocks on the 20 m grid and a group-disjoint 20%
validation split with seed `20260803`.

| Split | Non-rice | Rice | Total | Blocks |
|---|---:|---:|---:|---:|
| `target_pool` | 4,872 | 4,377 | 9,249 | 132 |
| `target_val` | 1,128 | 1,623 | 2,751 | 33 |

The grouped split is intentionally not class-balanced within each partition.
`target_pool` supplies weak labels for few-shot adaptation. `target_val` is the
fixed spatially disjoint weak-reference evaluation set. It is not independent
field truth and is separate from the blank 400-parcel independent-validation
sample.

## Frozen Temporal-92D representation

Temporal-92D consists of 23 Sentinel-2 NDVI values followed by 23 Sentinel-1
VV, 23 VH, and 23 derived RVI values. `src/data/common.py` enforces this exact
92-column order.

The documented temporal anchors are:

```text
2022-03-02, 2022-03-12, 2022-04-06, 2022-04-11, 2022-04-21,
2022-05-06, 2022-05-16, 2022-06-25, 2022-07-10, 2022-08-04,
2022-08-09, 2022-08-19, 2022-08-24, 2022-09-13, 2022-09-18,
2022-09-28, 2022-10-03, 2022-10-13, 2022-10-18, 2022-10-23,
2022-11-02, 2022-11-07, 2022-11-12
```

For each anchor, `gee/01_rebuild_public_s1_s2_exports.js` uses a window from
eight days before through nine days after the anchor. Sentinel-2 comes from
`COPERNICUS/S2_SR_HARMONIZED`; scenes with
`CLOUDY_PIXEL_PERCENTAGE > 80` are excluded and SCL classes 3, 8, 9, 10, and
11 are masked before median NDVI from B8/B4 is calculated. Sentinel-1 comes
from dual-polarization `COPERNICUS/S1_GRD` IW observations; median VV and VH
are retained in dB and RVI is calculated after conversion to linear units.

The historical source CSV and target 92-band raster are not present. Their
recorded original hashes are:

- source CSV:
  `042425e9a1d754191c6626379d43b5ee812656ea5b5b937ca1c8c9c1dc0b00dc`;
- target raster:
  `16efb56adf0853bdd49fa0ed1e623ae8650ad32c92292d5c7521a3456ff7cd19`.

The recovery script reconstructs the public collections, schema, sample IDs,
target grid, and dates, but the repository explicitly treats its output as a
new data version unless both historical hashes are reproduced.

## Frozen AlphaEarth representation

AlphaEarth is the 64-dimensional 2022 annual embedding `A00` through `A63`
from `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`, extracted at the same manifest
locations. `scripts/extract_alphaearth_gee.py` and
`scripts/validate_alphaearth_export.py` define its reconstruction and
validation route.

The combined embedding table
`outputs/alphaearth/alphaearth_samples_2022.csv` is not present in the current
workspace. Consequently its historical content hash, complete-row membership,
and exact common-sample intersection cannot currently be verified.

## Existing downstream and few-shot protocol

The legacy classifier family is a random forest with `max_features: sqrt`,
minimum leaf size 2, and a fixed 0.50 decision threshold. Source baselines use
500 trees, source-train median imputation, class-balanced subsampling, and seed
`20260803`. No longitude, latitude, sample ID, domain, or spatial-block value is
passed to the classifier.

The preserved few-shot budgets are total label counts
`[20, 50, 100, 200, 500]`. The original runs use seeds `[42, 43, 44]` and the
methods `source_only`, `target_only`, and `joint`. Each nonzero draw is balanced
between classes and prioritizes one sample per spatial block before reusing a
block. Temporal and AlphaEarth use equivalent sampling code, and Phase III
constructs one draw before applying both representations. Phase III expands to
seeds 42 through 71 and separately audits distributed and clustered label
placement.

Exact selected sample-ID lists were not committed as immutable draw manifests.
The EOFM benchmark must materialize and hash shared draw membership before any
new representation is evaluated.

## Zero-shot and OOD protocol

Zero-shot models are trained only on `source_train`; source validation may
support source-only decisions. Models, preprocessing, pooling, and the 0.50
threshold must be frozen before target weak-reference labels or metrics are
loaded.

The current OOD implementation operates primarily in AlphaEarth space. It
computes standardized 1-NN distance, cosine 1-NN distance, mean 10-NN cosine
distance, shrinkage Mahalanobis distance, and a Jiangxi-versus-Shanghai domain
probability. Phase III divides target-validation spatial blocks into tuning and
held sets, selects selective-prediction thresholds on tuning blocks, and reports
held-block coverage and disagreement. EOFM representations must reuse the same
target rows, blocks, target predictions, and methodological family.

## Local data availability

Present:

- `data_metadata/alphaearth_sample_manifest.csv`;
- `data_metadata/data_registry.yaml`;
- `inputs/official_reference/shanghai_2022_rice_aligned_20m.tif`;
- committed aggregate headline, deployment, and documentation artifacts.

Missing:

- `data/raw/jiangxi_official_2022_binary_features_gee_s1_s2.csv`;
- `data/raw/jiangxi_official_2022_ancillary_features.csv`;
- `data/raw/shanghai_light_2022_fusion_92_20m.tif`;
- `outputs/alphaearth/alphaearth_samples_2022.csv` and its raw batches;
- legacy source split, RF model, per-run, zero-shot, few-shot, and OOD runtime
  artifacts under `outputs/experiments/`, `results/alphaearth_minimal/`,
  `results/phase2/`, and `results/phase3/`;
- raw multispectral Sentinel-2 time series for either region;
- native Sentinel-1/Sentinel-2 sample patches for either region;
- Presto and Galileo packages, pinned upstream commits, checkpoints, and
  checkpoint hashes.

`RICE_FUSION_DATA_ROOT` and `WALL_TO_WALL_DATA_ROOT` are unset in the audited
workspace, so no external data root resolves these missing artifacts.

## Presto compatibility

The existing flattened 92D representation is not a valid native Presto input.
The repository can recover VV and VH at the documented anchors, but the current
GEE recovery script exports only derived Sentinel-2 NDVI rather than the
multispectral reflectance required for a meaningful Presto representation.

Before a Presto smoke test, reconstruct or restore, for every frozen source and
target sample:

- a deterministic `[timesteps, bands]` tensor;
- S1 VV and VH in the units expected by official Presto preprocessing;
- available S2 reflectance explicitly mapped to the official schema, at least
  B2, B3, B4, B5, B6, B7, B8, B8A, B11, and B12 for the standard supported
  optical groups;
- observation masks, temporal order, and month handling;
- Dynamic World values or the official ignored/missing value;
- enough extraction metadata to reproduce compositing, scaling, masking, and
  row order.

ERA5 and SRTM may be supplied if reconstructed consistently, or masked through
the official API. Missing values must never be fabricated. Whether coordinates
enter the frozen encoder must be declared; raw coordinates remain forbidden in
the downstream classifier.

The 23 irregular anchors fit Presto's 1-to-24 length limit, but they are not a
regular monthly sequence. The chosen temporal-position policy must therefore be
validated against the official API and frozen before target evaluation.

## Galileo compatibility

The existing point-level 92D representation is also not a native Galileo
spatiotemporal input. The preferred benchmark requires co-registered patches
around every frozen sample, not just sampled point values.

Before a Galileo smoke test, reconstruct or restore:

- S1 VV/VH and S2 B2, B3, B4, B5, B6, B7, B8, B8A, B11, and B12 tensors;
- a common 10 m grid and explicit `[height, width, timesteps, bands]` order;
- deterministic 3-by-3 or 5-by-5 candidate patches with edge and nodata masks;
- explicit month indices and the same frozen temporal policy used by the
  benchmark;
- official Galileo normalization and modality masks;
- a source-only, predeclared choice of one primary patch size and pooling rule.

The physical footprints would be 30 m by 30 m for a 3-by-3 patch or 50 m by
50 m for a 5-by-5 patch at 10 m. Patch size must not be selected from Shanghai
metrics.

## Common-sample implications

The current manifest defines the intended population, but an EOFM common-sample
freeze cannot be finalized until Temporal-92D, Presto, Galileo, and AlphaEarth
availability is known per row. After extraction, the benchmark must:

1. intersect valid rows across all four representations without outcome-driven
   rebalancing;
2. report original and retained counts by region, split, class, and spatial
   block;
3. audit geographic and class selection effects;
4. freeze ordered-row, membership, schema, and shared-draw hashes;
5. keep private per-sample identifiers out of committed aggregate artifacts.

## Blocking decision

The data audit is complete, but Presto and Galileo extraction is blocked by
missing native S1/S2 tensors and patches. The AlphaEarth table and detailed
legacy runtime outputs must also be restored or reproducibly regenerated before
a paired four-way common-sample benchmark can be frozen. No model comparison,
target evaluation, parcel regeneration, or README result update is authorized
until those inputs exist and pass validation.

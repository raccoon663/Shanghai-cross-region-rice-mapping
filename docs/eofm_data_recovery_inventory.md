# EO foundation-model data recovery inventory

## Scope and outcome

This inventory continues the completed repository audit on branch
`feature/eo-foundation-model-benchmark`. It searches the public checkout,
ignored runtime paths, sibling Codex workspaces, and earlier rice-project
directories. No legacy result or deployment artifact was modified or copied
into Git.

The useful historical workspace is:

```text
C:\Users\ASUS\Documents\Codex\2026-07-26\sentinel-fusion-transfer-rice
```

The search also inspected:

```text
C:\Users\ASUS\Documents\Codex\2026-07-26\sentinel12-rice-fusion-research
C:\Users\ASUS\Documents\Codex\2026-07-22\undergraduate-student-with-a-dual-degree-2\projects\sentinel_rice_fusion
```

The first contains the legacy runtime state. The second contains a small 2024
Chongming demonstration, not the frozen 2022 sample population. The third
contains documentation only.

## Recovered authoritative geometry

The exact frozen sample locations did not need to be regenerated:

- all 1,429 Jiangxi source geometries occur in the historical
  `source_splits.csv`;
- all 12,000 Shanghai geometries occur in historical target split/sample
  files;
- source identifiers, labels, blocks, splits, and coordinates match the
  committed manifest, allowing for CSV type parsing of numeric source IDs and
  at most `3.6e-15` degrees of coordinate roundoff;
- target identifiers, labels, blocks, and splits match exactly, and reprojection
  of historical EPSG:32651 coordinates differs from the committed longitude and
  latitude by at most `1.5e-14` degrees.

The frozen geometry/sample manifest remains
`data_metadata/alphaearth_sample_manifest.csv`, with 13,429 rows and SHA-256
`04c4e7454ead6cd415f5d761593d9e88b8041dd6611f630064fca798ccf6421b`.

## Useful historical artifacts

Paths below are local recovery locations and must not be committed as portable
repository paths. “Dates” means explicit observation-date columns; the feature
names encode timestep numbers, while their dates are supplied by the frozen
temporal-grid manifest.

| Artifact | Type / size | Rows or shape | Schema and metadata | Frozen-sample match |
|---|---|---:|---|---|
| `C:\Users\ASUS\Documents\Codex\2026-07-26\sentinel-fusion-transfer-rice\outputs\experiments\rf_baseline_m2_v1\source_splits.csv` | CSV, 130,703 B | 1,429 × 8 | Sample IDs, lon/lat, labels, region, blocks, splits; no dates | All Jiangxi rows match; SHA-256 `75bfc6d5...d03b4` |
| `C:\Users\ASUS\Documents\Codex\2026-07-26\sentinel-fusion-transfer-rice\outputs\experiments\weak_label_budget_m4_v1\target_weak_splits.csv` | CSV, 1,129,935 B | 12,000 × 10 | Sample IDs, EPSG:32651 x/y, labels, region, blocks, splits; no dates | All Shanghai rows match; SHA-256 `251d3b6b...6c0f` |
| `C:\Users\ASUS\Documents\Codex\2026-07-26\sentinel-fusion-transfer-rice\outputs\experiments\weak_label_budget_m4_v1\target_weak_samples.csv` | CSV, 12,807,702 B | 12,000 × 102 | IDs, x/y, labels/splits plus all 92 Temporal features; no explicit dates | Exact target membership; 15,894 `-9999` feature cells; SHA-256 `6e20ce81...e558` |
| `C:\Users\ASUS\Documents\Codex\2026-07-26\sentinel-fusion-transfer-rice\outputs\alphaearth\alphaearth_samples_2022.csv` | CSV, 17,397,043 B | 13,429 × 71 | IDs, labels/splits and `A00`–`A63`; coordinates and dates absent | Row-for-row manifest match; all embeddings finite; SHA-256 `e84a4c43...8ef0` |
| `C:\Users\ASUS\Documents\Codex\2026-07-26\sentinel-fusion-transfer-rice\outputs\alphaearth\raw_batches\alphaearth_2022_*.csv` | 27 CSV chunks, about 0.59–0.69 MB each | 13,429 total | IDs and 64 annual embedding bands; local extraction batches | Reassembles recovered AlphaEarth table |
| `C:\Users\ASUS\Documents\Codex\2026-07-26\sentinel-fusion-transfer-rice\outputs\experiments\rf_baseline_m2_v1\rf_s1_s2_fusion.joblib` | Joblib, 4,408,759 B | Frozen model bundle | RF, 92 ordered columns, source medians, training split, seed; no raw training rows | Frozen Temporal model; SHA-256 `93e296a2...bd24` |
| `C:\Users\ASUS\Documents\Codex\2026-07-26\sentinel-fusion-transfer-rice\outputs\experiments\rf_baseline_m2_v1\rf_metrics.csv` | CSV, 1,385 B | 6 × 17 | Aggregate source validation/test metrics; no coordinates or IDs | Frozen source results; SHA-256 `1da489e6...42e4` |
| `C:\Users\ASUS\Documents\Codex\2026-07-26\sentinel-fusion-transfer-rice\outputs\experiments\weak_label_budget_m4_v1\rf_budget_runs.csv` | CSV, 9,921 B | 51 runs | Few-shot method, budget, seed, aggregate metrics; no coordinates/IDs | Frozen Temporal few-shot runs; SHA-256 `4b20ca96...53e4` |
| `C:\Users\ASUS\Documents\Codex\2026-07-26\sentinel-fusion-transfer-rice\results\phase2\target_ood_predictions.csv` | CSV, 4,093,428 B | 2,751 × 84 | Target IDs, labels, 64D embeddings, predictions and OOD scores; no coordinates/dates | Exact target-validation membership; SHA-256 `15dd80b3...7516` |
| `C:\Users\ASUS\Documents\Codex\2026-07-26\sentinel-fusion-transfer-rice\results\phase2\ood_analysis.csv` | CSV, 847 B | 5 metrics | Aggregate AlphaEarth OOD results | Frozen OOD table; SHA-256 `b59be293...87c` |
| `C:\Users\ASUS\Documents\Codex\2026-07-26\sentinel-fusion-transfer-rice\results\phase3\label_efficiency_robust.csv` | CSV, 3,784 B | 20 × 12 | Distributed/clustered representation summaries over 30 seeds | Frozen robustness results; SHA-256 `e62e4e1f...566` |
| `C:\Users\ASUS\Documents\Codex\2026-07-26\sentinel-fusion-transfer-rice\data\raw\jiangxi_official_2022_ancillary_features.csv` | CSV, 1,274,849 B | 3,000 × 34 | IDs, labels, summary indices, RGB and geometry JSON; no 92D/raw series | Contains a broader source candidate set, not a substitute for the frozen 1,429-row temporal table; SHA-256 `89ebb117...e06` |

Additional historical RF probability/class rasters, pseudo-label outputs,
calibration outputs, Phase II/III tables, figures, and parcel products are
present. They are useful for provenance but do not contain the missing native
multispectral point-time-series values.

## Historical artifacts not found

The following original inputs were not found anywhere in the searched local
rice-project directories, including under alternative names by matching their
recorded SHA-256 values:

- `jiangxi_official_2022_binary_features_gee_s1_s2.csv`, expected SHA-256
  `042425e9a1d754191c6626379d43b5ee812656ea5b5b937ca1c8c9c1dc0b00dc`;
- `shanghai_light_2022_fusion_92_20m.tif`, expected SHA-256
  `16efb56adf0853bdd49fa0ed1e623ae8650ad32c92292d5c7521a3456ff7cd19`;
- any 2022 multispectral S2 point time series for the frozen 13,429 samples;
- any 2022 native S1/S2 spatial patches centered on those samples;
- any Presto or Galileo embedding generated for this rice benchmark.

The recovered 12,000-row Shanghai sample table preserves the target-side
Temporal-92D values, but no equivalent source feature table was found. A saved
RF model is not reversible into its training observations.

## Temporal provenance recovered

The authoritative cadence is fully recovered from
`gee/01_rebuild_public_s1_s2_exports.js`, whose SHA-256 is
`1f54ab3bcd992041c5676dc77781d7c8b6cd57825ac2d091373d27e1ea48fbec`.
The exact window table and processing contract are frozen in
`results/manifests/eofm_temporal_grid_freeze.json`.

Important details are:

- 23 irregular anchors in 2022;
- each Earth Engine `filterDate` starts eight days before the anchor and ends
  nine days after it (end exclusive), representing 17 calendar days;
- S1 GRD IW, required VV and VH, no orbit-direction filter, median reducer,
  VV/VH retained in dB, RVI calculated from linearized median VV/VH;
- S2 SR Harmonized, scene cloud percentage at most 80, SCL masking of classes
  3/8/9/10/11, per-image NDVI followed by the median reducer;
- 20 m sampling; target CRS/grid EPSG:32651 with the recorded fixed transform.

## Reconstruction decision

Local recovery supplies authoritative geometries, the target legacy values,
AlphaEarth, models, and result provenance, but not native Presto inputs. The
next scientifically valid step is a deterministic Earth Engine point export
over the frozen manifest. `scripts/eofm/02_submit_presto_point_exports.py`
implements chunked S1/S2 extraction, native S2 bands, explicit observation
counts/masks, and simultaneous legacy NDVI/RVI columns for target-side
reproduction checks. Sampling is explicitly locked to the historical target
EPSG:32651 20 m transform rather than relying on an image collection's implicit
first-band projection.

Earth Engine credentials exist locally. A later review of the historical Codex
task and the still-authorized Code Editor session recovered the selected Google
Cloud project as `eng-artifact-503507-k7`. The earlier conclusion that no
authorized project could be found was therefore incorrect. Python API access
still requires a working TLS connection to `earthengine.googleapis.com`; the
Code Editor session itself remains authenticated. The script records task IDs
and is restartable when that API connection is available.

From the repository root, the submission and resume procedure is:

```powershell
.\.venv\Scripts\python.exe scripts\eofm\02_submit_presto_point_exports.py --project eng-artifact-503507-k7
.\.venv\Scripts\python.exe scripts\eofm\02_submit_presto_point_exports.py --project eng-artifact-503507-k7 --refresh-status
```

The first command skips active and completed chunks recorded in the local,
gitignored `outputs/eofm/ee/presto_point_exports.json`. The second refreshes
their Earth Engine states before applying the same skip rules. Use
`--start-chunk` and `--max-chunks` for a bounded restart. The manifest records
task IDs, descriptions, expected row ranges, output prefixes, and the Drive
folder; it should be retained locally until all 27 chunks have completed.

## Earth Engine submission result (2026-08-23)

All 27 deterministic chunks were submitted through the authenticated Earth
Engine Code Editor under project `eng-artifact-503507-k7`. Task IDs and states
are frozen in `results/manifests/eofm_ee_submission_freeze.json`.

- c000 completed and wrote the first 500 rows to Google Drive;
- c001-c026 failed after computation because the destination Drive had
  insufficient free space;
- the inspected c026 error was: `Not enough space in Google Drive (need 1.4MB
  for this export). (Error code: 3)`;
- no task remains active, so the point-time-series reconstruction is not yet
  complete and Presto tensor construction must not start.

Free at least 40 MB plus a safety margin in the signed-in Google Drive, then
resubmit c001-c026. Chunk c000 must be retained and skipped. This storage-quota
failure does not change the frozen sample, temporal, band, projection, or
missingness contracts.

# Independent parcel validation protocol

## Current status

The Chongming prototype is ready for independent parcel-level validation. No independent Shanghai parcel labels have been entered yet. Existing Shanghai scores are comparisons with an official rice product used as a weak reference, while geometry, coherence, and mapped-area statistics describe the deployment workflow rather than crop-classification accuracy.

The validation sample was defined on 2026-08-14, before any independent labels were added. The sampling manifest is [`results/summary/independent_validation_sampling_manifest.json`](../results/summary/independent_validation_sampling_manifest.json), and the blank 400-parcel table is [`results/selected_outputs/independent_validation_sample_400.csv`](../results/selected_outputs/independent_validation_sample_400.csv).

## Sample design

The sample contains 400 unique parcels in EPSG:32651, covers all 30 deployment tiles, and uses random seed `20260814`. Seven mutually exclusive strata intentionally include more difficult mapping contexts:

| Stratum | n | Purpose |
|---|---:|---|
| Clear predicted Rice | 140 | Main positive map population |
| Clear predicted Non-rice | 60 | Confident negative parcels |
| Retained QA-risk | 80 | Test whether warnings identify difficult cases |
| Excluded built/greenhouse-like | 50 | Evaluate settlement and protected-cultivation cases |
| Excluded water/coastal | 30 | Evaluate shoreline, water, and tidal-flat failures |
| Large or extreme objects | 20 | Test under-segmentation and mixed parcels |
| Cross-tile or deployment-edge | 20 | Test reconciliation and edge behavior |

Allocation is proportional across source tiles where possible, with at least one sample from represented tiles when the stratum size permits. Because rare QA cases are deliberately oversampled, population estimates require inverse-inclusion-probability or post-stratification weights. Unweighted results should be treated as sample diagnostics rather than population estimates.

## Reference evidence and labeling

Reference evidence should be prioritized as follows:

1. contemporaneous field survey or official 2022 parcel/crop records;
2. multi-date high-resolution imagery from the 2022 rice season;
3. independent multi-temporal Sentinel-2 interpretation when better evidence is unavailable.

Model predictions, M0/M1/M2 values, FTW attributes, and Dynamic World attributes should be hidden from labelers until interpretation is complete.

Primary reference labels are `Rice`, `Non-rice cropland`, `Non-agricultural`, `Mixed / boundary invalid`, and `Unresolvable`. Each record should also store confidence, evidence source, imagery dates, field-visit date where applicable, and notes. `Mixed / boundary invalid` describes a geometry problem and should not automatically be counted as a rice-classification error.

Two interpreters independently label all QA-risk, large/extreme, coastal/water, built/greenhouse, and cross-tile cases. At least 20% of clear-parcel samples are also double-labeled. A third reviewer resolves disagreements without seeing model predictions. Inter-interpreter agreement and Cohen's kappa are reported before adjudication.

## Model comparison

All methods are evaluated on the same validation geometries without changing thresholds after labeling begins:

1. **M0:** aggregate the raw 20 m source-only probability;
2. **M1:** use the FTW-gated raster with the inherited rice threshold;
3. **M2:** use parcel aggregation before the final QA rules;
4. **QA-aware M2:** use the final parcel predictions together with exclusion/QA status.

M1b remains a secondary ablation. Excluded parcels are treated as abstentions when reporting coverage and are evaluated separately for how often the exclusion was appropriate; they are not silently scored as ordinary Rice/Non-rice predictions.

## Metrics

Report 95% confidence intervals and stratum-weighted estimates when making population-level claims:

- Rice precision, recall, and F1;
- parcel overall and balanced accuracy;
- full Rice / Non-rice cropland / Non-agricultural confusion matrix;
- binary Rice versus non-Rice confusion matrix;
- optional area-weighted metrics where parcel geometry is reliable;
- abstention/exclusion rate and selective accuracy;
- precision of the QA exclusions and plausible-agriculture false-exclusion rate;
- geometry-validity and mixed/boundary-invalid rates;
- metrics by QA stratum, parcel-size band, coastal context, and tile;
- paired bootstrap differences among M0, M1, M2, and QA-aware M2.

Bootstrap by parcel with stratification and include a tile-clustered sensitivity interval. Keep unresolvable records in the dataset and report their weighted fraction even if they are excluded from the primary confusion matrix.

## Preventing leakage after labeling starts

The sampling frame, stratum allocation, parcel IDs, seed, and original blank sample are kept unchanged once labeling starts. Model parameters, FTW settings, parcel-QA thresholds, the rice threshold, and validation geometries should likewise remain unchanged until the first independent evaluation is complete.

Reference labels should be stored in a new versioned file rather than overwriting the blank sample. Labelers should not use the official Shanghai rice product or model outputs as reference evidence. Pixels or parcels removed by gating should continue to be described in terms of coverage or selectivity until independent labels show whether they were correct exclusions.

Independent Shanghai parcel accuracy should only be reported after the adjudicated reference set is complete. Until then, the Chongming output should be described as a prototype awaiting independent validation rather than a validated all-Shanghai rice map.

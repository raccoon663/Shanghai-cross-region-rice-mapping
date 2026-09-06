# Representation inputs and embedding validation

The benchmark uses complete frozen monthly Presto and Galileo representations.
This document records input contracts, missing-data handling, and validation;
the downstream findings are in the [case study](eofm_benchmark_case_study.md).

## Authoritative state

The consolidated machine-readable entry point is
`results/manifests/eofm_full_representation_readiness.json`.

| Representation | Validated chunks | Input rows | Centers | Full embedding | Repeat check |
| --- | ---: | ---: | ---: | --- | --- |
| Galileo nano, 3×3 at 10 m, encoder patch_size=1, 23 windows | 27/27 | 120,861 patch-pixel rows | 13,429 | 13,429 × 128 float32 | Bitwise identical |
| presto_primary, 12 natural calendar months, January start | 27/27 | 13,429 point rows | 13,429 | 13,429 × 128 float32 | Bitwise identical |

There are no missing chunks or embeddings. Both outputs have the exact same
frozen manifest-row order: 1,429 Jiangxi source samples and 12,000 Shanghai
target samples. All embedding values are finite. Labels and splits remain join metadata and are excluded from downstream
features. This input freeze preceded downstream evaluation.

The frozen sample-manifest SHA-256 remains
`04c4e7454ead6cd415f5d761593d9e88b8041dd6611f630064fca798ccf6421b`.

Logical hashes cover ordered int32 manifest rows followed by float32 embeddings:

- Galileo: `70f9071c3da30221d2f2ac2a81297a7779af02803d21403af5da2dfcafdf839b`
- presto_primary: `5242245319350c184898071f33664e28c2b8e1ca212c8dd43317f2b039e42a30`

## Detailed manifests

All paths below are under `results/manifests/`:

- `eofm_galileo_recovery_validation.json`: 27 input chunks, schemas, identities, hashes.
- `eofm_galileo_full_patch_validation.json`: full patch order, bands, months, masks.
- `eofm_galileo_full_embedding_freeze.json`: checkpoint, normalization, deterministic full extraction.
- `eofm_presto_primary_recovery_validation.json`: 27 monthly chunks and hashes.
- `eofm_presto_primary_input_validation.json`: canonical monthly tensor and masks.
- `eofm_presto_primary_smoke_validation.json`: 100-sample constructor/encoder check.
- `eofm_presto_primary_embedding_freeze.json`: deterministic full monthly extraction.

Raw CSVs, patch tensors, point embeddings, checkpoints and detailed cloud-file
state remain in ignored runtime storage. Only aggregate manifests and code are
published. Existing Temporal-92D, AlphaEarth, and the 23-window Presto cadence
sensitivity freeze remain unchanged.

## Missing-data handling

The full Galileo input contains four S2 exported-mask versus numeric-sentinel
disagreements (the original smoke already contained one). The existing frozen
preparation rule combines the explicit sensor mask with numeric availability,
so these observations are masked rather than fabricated. S1 disagreements: zero.
The monthly Presto tensor has 49 missing S1 and 2,411 missing S2 sample-months;
the official constructor receives the existing explicit missing-data masks.

## Encoder runtimes and extraction

The upstream commits, checkpoint identities, normalization and output hashes
are recorded in the full embedding freeze manifests under `results/manifests/`.
Both extractions were repeated with bitwise-identical results on CUDA. Runtime
dependencies included einops 0.7.0 for Galileo and 0.6.0 for Presto in separate
environments. This repeat check applies to the recorded execution environment.

Galileo preparation accepts repeated `--chunk-dir` arguments in
`scripts/eofm/09_prepare_galileo_smoke_patches.py`; the encoder in
`scripts/eofm/10_smoke_test_galileo.py` accepts `--extra-site-packages` for an
isolated dependency directory. Monthly Presto preparation and extraction use
`scripts/eofm/08_prepare_presto_monthly_inputs.py` and
`scripts/eofm/05_extract_presto_embeddings.py`. Run each script with `--help`
for its input, checkpoint and output arguments.

The [temporal-semantics analysis](presto_temporal_semantics_audit.md) explains
why the primary Presto input uses 12 calendar months. The earlier 23-step
cadence is retained as a sensitivity representation and is not one of the
four primary benchmark rows. See the [reproduction guide](eofm_benchmark_reproduction.md)
for the required downstream runtime paths and exact-input constraints.

# EOFM export recovery checkpoint — 2026-09-02

This is an input-recovery checkpoint, not a representation or benchmark freeze.
The frozen scientific configuration and sample population are unchanged.

## Locally verified inventory

- Galileo: c000–c017, **18/27 chunks**, 9,000 centers and 81,000 patch-pixel rows.
- Missing local Galileo chunks: c018–c026 (4,429 centers / 39,861 patch rows).
- c017 was downloaded in this session and passed all checks. Its Downloads
  copy and retained repository runtime copy have identical SHA-256
  `4f07327851311df23778d91b3fac55cbf1620e8d50a2e24423cfaad625761edd`.
- The c017 temporary Drive copy was moved to recoverable Trash. It was **not**
  permanently purged in this session. No local backup or original TIFF was deleted.
- The full aggregate ledger is
  `results/manifests/eofm_galileo_recovery_validation.json`. It contains file
  hashes and aggregate validation results, not coordinates or feature values.
- No monthly Presto CSV was found in the configured local download directory.
  The GEE task list shows completed monthly c000/c001 exports; these have not
  been counted as locally recovered or validated. Monthly input readiness is
  therefore not established.

## Current blocker and exact resume point

c018 was resubmitted with the frozen generator and GEE reported `completed`.
Drive exposes the expected `eofm_galileo_patch3_2022_c018_09000_09500.csv`.
The first download attempt failed with a visible Drive HTTP error; a retry
timed out, and a further retry after refreshing Drive also timed out. No local
c018 file was found after these attempts. The HTTP status/cause is not known.

Keep the c018 cloud copy. Resume by downloading this **already completed**
export, not by blindly submitting another task. Validate and hash the local
file before any cleanup or c019 submission. c019's private script is prepared
locally but has not been submitted in this session.

Drive showed approximately 14.93 GB of 15 GB used. Storage is tight, but the
observed c018 failure was a download error, not an export quota failure.

## Reproducible per-chunk validation

`scripts/eofm/11_validate_recovery_chunks.py` reuses the existing frozen schema
and hash helpers. It validates exact chunk filenames, every sample/patch key,
every row's region, row count, schema/band ordering, zero duplicate keys, and
finite/readable numeric values. Canonical row-major ordering is checked after
sorting, as in the existing tensor-preparation code. It rejects changed hashes,
missing previously validated files, and changed sample/grid contracts.

```text
python scripts/eofm/11_validate_recovery_chunks.py --representation galileo --chunk-dir outputs/eofm/galileo_smoke_downloaded --chunk-dir outputs/eofm/galileo_full_downloaded --report results/manifests/eofm_galileo_recovery_validation.json
```

The ledger does not claim cloud cleanup, mask-semantic validation, or embedding
validation. Those remain separate from download-integrity checks.

## Verification and restrictions

- EOFM reconstruction and recovery tests: **16 passed**.
- `python -m pytest -q`: **22 passed, 13 failed**. All failures were caused by
  missing `geopandas` in the current Python environment when loading the
  existing wall-to-wall evaluator; its evaluation did not run.
- `python -m compileall src scripts`: passed.
- No four-way/source/target/few-shot/OOD benchmark, paired statistics, parcel
  deployment update, or README metric change was performed.
- Full Galileo and monthly `presto_primary` inputs/embeddings are **not ready**.
  The existing 23-window Presto freeze is preserved unchanged.

## Git handoff

The validation code, tests, aggregate ledger, and this checkpoint are committed
on `feature/eo-foundation-model-benchmark`. Push was rejected because the remote
contains unseen commits. Fetch then stalled; a bounded retry failed to connect
to GitHub on port 443. No force-push, merge, or remote-history rewrite was made.
Fetch and inspect the remote changes before reconciling and pushing this local
checkpoint. Raw exports and private runtime scripts remain ignored.

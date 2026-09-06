# EOFM full input and representation freeze — 2026-09-05

Both new full representations are frozen and validated. This completes the input
recovery and embedding stage; the four-representation downstream benchmark has
not been run.

## Authoritative state

The consolidated machine-readable entry point is
`results/manifests/eofm_full_representation_readiness.json`.

| Representation | Validated chunks | Input rows | Centers | Full embedding | Repeat check |
| --- | ---: | ---: | ---: | --- | --- |
| Galileo nano, 3×3 at 10 m, encoder patch_size=1, 23 windows | 27/27 | 120,861 patch-pixel rows | 13,429 | 13,429 × 128 float32 | Bitwise identical |
| presto_primary, 12 natural calendar months, January start | 27/27 | 13,429 point rows | 13,429 | 13,429 × 128 float32 | Bitwise identical |

There are no missing chunks or embeddings. Both outputs have the exact same
frozen manifest-row order: 1,429 Jiangxi source samples and 12,000 Shanghai
target samples. All embedding values are finite. Labels/splits remain join
metadata; no downstream training, target evaluation, or model selection ran.

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

## Recovery and execution

This continuation recovered Galileo c019–c026 and all monthly Presto chunks.
Eight new Galileo exports and 25 new monthly Presto exports were submitted;
the two earlier completed monthly exports were downloaded without resubmission.
No export failure or duplicate submission was recorded in this continuation.
Two Galileo status-query interruptions were recovered from the existing cloud
files. The serial runner now retries both native and SDK-wrapped DNS failures
on read-only queries, without retrying export submission blindly.

Cloud cleanup was limited to validated temporary CSVs with retained local
backups. The current recovery ledger records five Galileo and twenty Presto
CSV purges for quota; four Galileo and seven Presto CSVs remain in recoverable
Trash. The Galileo counts cover c018–c026 only and do not claim cleanup history
for earlier chunks. Retained backup hashes were rechecked at final audit.

Use `scripts/eofm/12_recover_exports_serial.py` for restartable recovery.
`--proxy-port 0` selects direct access; an explicit local proxy port can be
supplied when required. Resume at the first missing chunk from the ledger.
`--backup-dir` must point to retained local storage distinct from the working
copy. Frozen project, sample and Galileo grid checks remain enforced.

Full Galileo preparation accepts repeated `--chunk-dir` arguments, so smoke
and later chunks can remain in their original directories. The full encoder
accepts `--extra-site-packages` for a separate dependency directory. The pinned
runtime dependency versions used here were einops 0.7.0 for Galileo and 0.6.0
for Presto; model checkpoints and scientific settings were not changed.

## Validation and Git

- Full repository suite: **42 passed**.
- `python -m compileall src scripts`: passed.
- Both full extractions repeated with bitwise equality on CUDA.
- Final audit checked exact sample/region order, output hashes, finite values,
  full input inventories, and retained local backup hashes.

Work remains on `feature/eo-foundation-model-benchmark`. The old local and
remote histories diverged, but their prior checkpoint trees differed only in
`AGENTS.md`. Publication applies this completed change on top of the existing
remote feature history, preserving both histories without a force-push or a
merge into main. See the final Git commit for the published snapshot.

The next scientific stage is the downstream representation benchmark. It is
not part of this completed input/embedding freeze.

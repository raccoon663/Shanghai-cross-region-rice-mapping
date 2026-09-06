# Presto temporal-semantics audit

## Verdict

`CURRENT_23_STEP_INPUT_APPROXIMATE`

This audit applies to `nasaharvest/presto` commit
`11e207a668a34336ced1d8e492a1bd5849b96c4a` and is based on the pinned source
and tests, not only on README wording.

## Source-code behavior

In `presto/presto.py:197-212`, `month_to_tensor` expands an integer with
`arange(month, month + seq_len) % 12` for every sample. A one-dimensional
tensor supplies one starting month per sample and is expanded identically. A
two-dimensional tensor passes through unchanged after the `< 12` assertion.

The encoder calls this function at line 341, looks up a frozen 12-entry
sinusoidal month embedding at line 342, and concatenates it with channel and
independent timestep positional embeddings at lines 341-371 and 388. Thus
scalar `month=2` and 23 timesteps means 23 consecutive zero-based months
`2,3,...,11,0,1,...,0`, spanning almost two years—not 23 windows in 2022.

## Tests and supported contract

The pinned tests pass scalar `month=1`. The explicit embedding test at
`tests/test_presto.py:201-230` constructs expected month encodings with
`month_to_tensor(1, 1, num_timesteps)`, confirming consecutive-month behavior.
No pinned test exercises a `[batch, timestep]` month tensor, repeated months,
or irregular sub-monthly observations.

A two-dimensional per-timestep month tensor therefore works incidentally in
this commit, but is not an official tested interface contract. Repeated month
indices are mechanically representable but not tested as multiple observations
within a month. There is no sub-monthly interval or day-of-year encoding; the
separate positional embedding preserves order, not true elapsed time.

## Project interpretation

The frozen input contains 23 irregular, overlapping, approximately 17-day
windows from March through November 2022. Passing `month=2` preserves values,
masks, order, and deterministic inference, but misstates their timing as
consecutive months. It is retained without overwrite as
`presto_legacy_cadence_sensitivity`.

The primary official-semantics representation is `presto_primary`: 12 natural
calendar-month composites for 2022 with scalar zero-based `month=0`. The
upstream model and checkpoint remain unmodified.

The verdict is **approximate**, rather than invalid, because the tensor and
inference contracts are valid; the mismatch is the calendar meaning assigned
to valid timesteps.

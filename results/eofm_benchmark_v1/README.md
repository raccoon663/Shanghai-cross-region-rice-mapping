# Four-representation benchmark evidence

Experiment `eofm_four_way_v1`. These files contain aggregate metrics and hashes;
they contain no per-sample features, predictions, coordinates, or model weights.

- `source_zero_shot_runs.csv`, `source_zero_shot_summary.csv`: source and target metrics.
- `fewshot_runs.csv`, `fewshot_summary.csv`: 2,400 fits and 80 setting summaries.
- `paired_f1_intervals.csv`: 126 descriptive paired spatial-block comparisons.
- `ood_summary.csv`, `ood_deciles.csv`: five error-ranking diagnostics per representation.
- `domain_audit.csv`: true and permuted domain-label diagnostics.
- `risk_coverage.csv`, `risk_coverage_summary.csv`: tuning/held-block selective prediction.
- `shared_draw_summary.csv`: 300 draw hashes and aggregate coverage counts.
- `protocol_freeze.json`, `source_prediction_freeze.json`: input, config and source artifact hashes.
- `execution_environment.json`: recorded runtime versions and code/config hashes.
- `artifact_manifest.json`: SHA-256 values for the curated tables, provenance, and figures.

The experiment namespace is separate from historical results. See the
[case study](../../docs/eofm_benchmark_case_study.md) and
[reproduction guide](../../docs/eofm_benchmark_reproduction.md).

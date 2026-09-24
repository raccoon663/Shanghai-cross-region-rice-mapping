# Transfer and reliability results

Shanghai metrics use an official-product weak reference. Read [the research report](../../RESULTS.md)
for interpretation and [the protocol](../../docs/geoai_research_protocol.md) for definitions.

| Evidence | File |
|---|---|
| Representation comparison | [Table 1](table1_representation.md) · [CSV](table1_representation.csv) |
| Adaptation comparison | [Table 2](table2_adaptation.md) · [CSV](table2_adaptation.csv) |
| OOD / failure prediction | [Table 3](table3_ood_failure.md) · [CSV](table3_ood_failure.csv) |
| Per-seed classification | [RQ1](rq1_runs.csv) · [RQ2](rq2_runs.csv) |
| Complete risk–coverage curves | [Compressed CSV](rq3_risk_coverage.csv.gz) |
| Acceptance operating points | [CSV](rq3_operating_points.csv) |
| Pool-derived thresholds | [CSV](rq3_pool_thresholds.csv) |
| Score quantiles | [CSV](rq3_quantiles.csv) |

Six figures are saved as PNG and SVG in `assets/figures/geoai_rqs_v1/`:
framework, representation transfer, few-shot curves, domain versus failure
discrimination, quantile error, and risk–coverage curves.

`protocol_freeze.json` records the original execution inputs and environment.
`artifact_manifest.json` identifies this published evidence and reproduction code.
The sample audit, shared-draw hashes, weight diagnostics and unit-weight control
are retained as supporting evidence. Private prediction files are not distributed.

```bash
python scripts/validate_geoai_results.py
python scripts/make_paper_figures.py
```

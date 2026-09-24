# README evidence guide

The README summarizes three experiment versions. Values are rounded only for
display; the saved tables remain authoritative and unchanged.

| Version | Scope | Evidence |
|---|---|---|
| Historical transfer and OOD | Original Sentinel/AlphaEarth experiments; maximum retained distance diagnostic | [Headline values](../results/summary/headline_metrics.csv), [historical experiments](experiments.md), [methodology](methodology.md) |
| Matched four-representation benchmark | Rebuilt Temporal, AlphaEarth, monthly Presto and Galileo; three source seeds and thirty shared few-shot draws | [Frozen protocol](../results/eofm_benchmark_v1/protocol_freeze.json), [source runs](../results/eofm_benchmark_v1/source_zero_shot_runs.csv), [few-shot runs](../results/eofm_benchmark_v1/fewshot_runs.csv) |
| Adaptation and reliability | Five-seed adaptation; additional 250-label fits; eleven risk signals on frozen seed-42 source predictions | [Research protocol](geoai_research_protocol.md), [configuration](../configs/geoai_rqs.yaml), [research results](../RESULTS.md) |

## Numerical sources

| README content | Saved source and interpretation |
|---|---|
| Historical source F1 0.947/0.962 and target F1 0.813/0.836 | `source` and `target_weak_reference` rows of [headline_metrics.csv](../results/summary/headline_metrics.csv); retain the historical precision and model version. |
| Historical distance error AUROC 0.800 | `ood,distance_error_auroc` in the same table; maximum retained diagnostic, not specifically the new cosine 10-NN condition. |
| Sample counts and representation dimensions | [Integrity audit](../results/geoai_rqs_v1/integrity_audit.json): 13,429 rows, split counts and feature shapes. Jiangxi totals 1,429; Shanghai totals 12,000. |
| Current source/zero-shot F1 and 50/500-label F1 | [Table 1](../results/geoai_rqs_v1/table1_representation.csv), backed by [RQ1 runs](../results/geoai_rqs_v1/rq1_runs.csv). Source/zero-shot use three seeds; label-efficiency columns use target-only, distributed, thirty-draw means. |
| 2,400 existing few-shot fits and 80 additional fits | [Benchmark runs](../results/eofm_benchmark_v1/fewshot_runs.csv); [RQ1 runs](../results/geoai_rqs_v1/rq1_runs.csv) with `provenance=new_250_label_fit`. The additional fits use four representations, two regimes, two methods and five seeds. |
| Six zero-shot paired intervals include zero | `regime=zero_shot` in [paired_f1_intervals.csv](../results/eofm_benchmark_v1/paired_f1_intervals.csv). |
| Adaptation F1 and gains | [Table 2](../results/geoai_rqs_v1/table2_adaptation.csv): five-seed, 500-tree rows. Galileo 0.821962 → 0.829897; Temporal gain +0.000507. These are descriptive changes in means, not significance claims. |
| Weight collapse and Galileo effective sample size 43.24 | [Importance diagnostics](../results/geoai_rqs_v1/importance_diagnostics.csv) and [constant-weight control](../results/geoai_rqs_v1/constant_weight_diagnostic.json). Floor-clipped near-uniform weights and concentrated weights are different failure modes. |
| Domain and error AUROC | AlphaEarth rows of [rq3_scores.csv](../results/geoai_rqs_v1/rq3_scores.csv): domain probability 0.9999988423433749 domain AUROC and 0.46300576385322145 error AUROC; cosine 10-NN 0.7976629212133226 error AUROC. |
| Selective disagreement 22.79% → 5.38% | AlphaEarth `cosine_knn` in [operating points](../results/geoai_rqs_v1/rq3_operating_points.csv): risk 0.22791712104689205 at full coverage; 0.05377906976744186 at requested 0.5 coverage, actual 0.500181752090149. No guaranteed risk bound. |
| Parcel count, large objects, overlaps and water-dominant area | [Final deployment effects](../results/tables/final_deployment_effects.csv); structural QA, not measured classification-error removal. |
| Historical 20 m raster and unchanged threshold 0.50 | [Methodology](methodology.md#representations-and-source-classifier), [frozen product manifest](../results/summary/final_chongming_parcel_product_manifest.json) and parcel configurations. The label-free comparison AOI is documented in the methodology. |
| 400-parcel validation sample | [Sampling manifest](../results/summary/independent_validation_sampling_manifest.json); labels remain uncollected, not a completed accuracy evaluation. |
| Wall-to-wall F1, coverage, precision, recall and reference rice area | [Metric table](../results/tables/wall_to_wall_weak_reference_metrics.csv). Mode A and B have different evaluation populations. M0 reference rice area is 7,134.28 ha. |
| Grid, pixel count and reference coding | [Alignment audit](../results/summary/wall_to_wall_alignment_audit.json) and the metric table: EPSG:32651, 20 m, 1,500,751 common valid pixels; reference is fully coded with no nodata. Low rice prevalence is not missing reference coverage. |
| Improved paired-block counts | [Block summary](../results/tables/wall_to_wall_block_summary.csv): M1 22/35, M1b 21/32 evaluable blocks; M2/M2 QA median change zero. |

The historical domain-probability error AUROC (0.415 in the historical narrative)
and the current value (0.463006) belong to different domain-classifier protocols.
Likewise, historical rejection using tuning blocks differs from the current
offline ranking curves and pool-quantile threshold transfer. They should not be
combined into a single leaderboard.

The parcel deployment retains its frozen historical classifier. Neither the
representation comparison nor sample-level selective prediction silently
changes the deployed model or adds a wall-to-wall OOD surface.

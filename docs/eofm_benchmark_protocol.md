# Four-representation benchmark protocol

The comparison uses `configs/eofm_benchmark.yaml`; generated runtime artifacts
go to ignored `outputs/eofm_benchmark_v1/`. Curated aggregate evidence is in
`results/eofm_benchmark_v1/`. The experiment does not replace the earlier
frozen headline tables or parcel products.

## Population and representations

All 13,429 frozen samples are retained, with 850 source-training, 265
source-validation, 314 source-test, 9,249 target-pool and 2,751 target-validation
rows. Blocks remain disjoint between splits within each domain. The four inputs
are reconstructed Temporal-92D, historical AlphaEarth-64D, monthly Presto-128D,
and Galileo-128D. Their input hashes and common sample membership are frozen
before model fitting. There is no performance-based sample exclusion.

The Temporal table is the new reconstructed data version, not the missing
historical Jiangxi feature table. Its results must not be presented as a rerun
of the legacy Temporal headline. This compares representation systems with
different upstream inputs: AlphaEarth annual inputs, monthly Presto with
encoder-only coordinates, Galileo spatial context, and point temporal features.
It does not isolate neural architecture effects.

## Classifiers and evaluation order

Source RFs use 500 trees, minimum leaf size 2, square-root feature sampling,
balanced-subsample class weighting, and seeds 42–44. Only the 850 source-training
rows enter fitting. There is no hyperparameter search. All source-model files
and predictions are saved and hashed before any Shanghai performance metric is
computed. The common decision threshold is 0.50. Source validation and test
remain separate report rows; source validation is not added to training.

Missing temporal features are imputed from training-row medians; a feature that
is entirely missing in those training rows uses zero. No evaluation rows enter
imputation. Downstream feature matrices contain representation columns only,
never coordinates, labels, IDs, domain indicators or block identifiers.

## Shared few-shot experiments

Both existing distributed and clustered samplers are reused from
`run_phase3_robustness.py`. For each regime, budget 20/50/100/200/500 and seed
42–71, balanced target-pool draws are generated once and saved before outcomes.
Every representation receives exactly the same draw. Budgets are independently
sampled under the inherited sampler, not claimed to be nested.

Each representation runs target-only and joint adaptation with 300 RF trees
and otherwise the same fixed RF settings. In joint fitting, source and target
have equal total sample weight and class weighting is disabled, following the
existing joint comparator. Each model imputes from its own permitted training
rows. This gives 2,400 matched adaptation fits, with 30 seeds per setting.

## OOD and selective prediction

The seed-42 source model defines each representation's errors. Fixed scores are
1-NN standardized Euclidean distance, 1-NN raw cosine, mean 10-NN raw cosine,
Ledoit-Wolf Mahalanobis distance and feature-only domain RF probability.
Scaling, imputation and covariance fit on source-training rows only. The domain
diagnostic uses class-matched source-training/target-pool samples, with true and
permuted domain-label controls; it never tunes the rice classifier.

Error AUROC uses spatial-block bootstrap intervals; error AUPRC is reported
alongside each classifier's error prevalence. Across representations, errors
are different outcomes, so OOD AUROC is not a head-to-head rice-accuracy metric.

Confidence and 10-NN cosine selective prediction use the same 30 random
half-block tuning/held-out partitions. Thresholds are quantiles of tuning-block
scores at prespecified coverages 0.9–0.5, without consulting labels. Held-block
coverage and weak-reference disagreement are reported. No score or threshold
is selected based on the observed target results.

## Paired uncertainty and interpretation

Each pair of representations is compared with the same 1,000 spatial-block
bootstrap draws. Intervals estimate the difference in mean F1 over the fixed
seed set; seed standard deviations are reported separately. Intervals are
descriptive and are not multiplicity-adjusted significance claims.

All Shanghai results are agreement with a product-derived weak reference, not
independent field accuracy. Negative results and unstable ordering are retained.
Figures and summaries cover the complete experiment suite.

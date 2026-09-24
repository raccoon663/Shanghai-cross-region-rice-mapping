# GeoAI RQ extension: methods and interpretation

The implementation is `src/experiments/run_geoai_rqs.py`; all critical settings
are in `configs/geoai_rqs.yaml`. The legacy config is inherited for RF leaf size,
feature subsampling, class weighting, representations, splits and old draws.
This extension is exploratory: earlier target results were already available.
It preserves `target_val` as the evaluation split, without calling it a newly
collected or untouched test set.

## RQ1: representation systems

Reuse 12 saved 500-tree source RFs and 2,400 saved 300-tree few-shot RF predictions.
Recalculate their metrics (including new AUPRC) and assert that saved F1 agrees.
Every representation uses exactly the same evaluation rows and adaptation draw.
The additional 250-label condition uses five seeds (42–46), two sampling regimes
and two fitting strategies, giving 80 new fits. Old 20/50/100/200/500 conditions
retain 30 seeds each. Budgets are independently sampled, not nested.

Joint fitting preserves the original equal total source/target weight and
disables class weighting. Target-only fitting uses only the chosen pool labels.
All training-specific imputation remains inside the fit. Source validation is
never folded into source training. RQ1 means use three frozen source seeds;
RQ2 uses five source seeds to match its adaptation runs. These means should
not be conflated. All seed-level predictions and metrics are retained.

## RQ2: controlled simple adaptation

Five seeds, four representations, 500 trees, fixed threshold 0.5 for all methods
below. Hyperparameters are not tuned against target evaluation. Only target-pool
features may enter unsupervised fitting; this is inductive adaptation with a
separate unlabeled pool, not transductive use of test features.

- **Source-only:** identical frozen source RFs for seeds 42–44, two additional
  source-trained RFs for seeds 45–46.
- **CORAL variant:** fit medians and StandardScaler on source training, estimate
  source and target-pool covariances using Ledoit–Wolf shrinkage, and transform
  source row vectors with `Cs^(-1/2) Ct^(1/2)`, followed by target-pool mean
  translation. Fit the RF on the transformed source; apply it to target features
  standardized with the original source scaler. Eigenvalues have a numerical
  floor. This is a regularized, mean-aligning variant of
  [CORAL (Sun, Feng & Saenko, 2016)](https://ojs.aaai.org/index.php/AAAI/article/view/10306),
  not a reproduction of every original benchmark setting.
- **Importance weighting:** five group-disjoint folds of source and pool samples
  train logistic domain classifiers. Each held-out source point receives domain
  odds `P(target|x)/P(source|x)` times that fold's source/target count ratio.
  Clip ratios to [0.1, 10], normalize their mean to one and fit a source RF with
  these weights (retaining baseline balanced-subsample class weighting). The
  covariance-shift premise is conditional stability plus support overlap; neither
  is established merely by fitting a domain classifier. See
  [Sugiyama et al. (2007)](https://jmlr.org/papers/volume8/sugiyama07a/sugiyama07a.pdf)
  for the covariate-shift framework. This implementation does not claim to be
  their importance-weighted cross-validation algorithm. Report clipping fraction
  and effective source sample size; high domain separability can cause weight
  collapse. Source imputation precedes domain cross-fitting and uses permitted
  source-training data only; fold scaling is fit on each domain-training fold.
  In the recorded scikit-learn 1.9 runtime, explicit weights enter bootstrap
  sampling via weighted `choice`, while absent weights use `randint`. Even unit
  weights therefore change the random stream for the same seed. The script
  `scripts/check_geoai_weight_degeneracy.py` checks this on AlphaEarth seed 42:
  uniform-weight probabilities exactly reproduce the importance run, and a fit
  with no weights exactly reproduces the original source RF. Its tiny observed
  mean difference is not evidence of useful density-ratio correction.
- **Pseudo-labeling:** one source-RF prediction pass on the target pool; accept
  confidence ≥0.95, at most 850 highest-confidence samples. Ties resolve with
  seeded random keys. Give accepted pseudo samples the same total weight as the
  source (while retaining balanced-subsample weighting); fit a new RF. No target
  rice labels enter selection or pseudo-label assignment. If nothing is accepted,
  preserve source predictions. This is not the old three-round pseudo experiment.
- **Full-pool supervised reference:** fit an RF on all 9,249 target-pool weak
  labels with pool-only imputation. It is a reference, not a guaranteed upper bound.

Legacy joint/target-only few-shot methods use 300 trees. Their rows are explicitly
identified separately in Table 2; do not attribute a source-vs-few-shot difference
solely to adaptation while ignoring this inherited capacity difference. The
zero-label adaptation comparison itself has matched 500-tree capacity. Gain and
reference-gap columns are descriptive differences of seed means, not paired
inferential confidence intervals.

## RQ3: separate domain and error outcomes

Risk scores use the frozen seed-42 source model in every representation:

| Score | Definition and fitting rows |
|---|---|
| Euclidean 1-NN | Nearest source-training point in source-standardized features |
| Cosine 1-NN / 10-NN | Nearest / mean ten nearest raw imputed embedding distances |
| Mahalanobis | Square root of Ledoit–Wolf distance in source-standardized features |
| Centroid | Minimum Euclidean distance to source class centroids |
| Entropy | Binary entropy of RF probability |
| Confidence | One minus maximum RF class probability |
| Margin | One minus absolute binary probability difference |
| Vote uncertainty | One minus maximum fraction of per-tree hard votes |
| Tree variance | Variance of per-tree rice probabilities |
| Domain probability | Domain RF trained on source-training and target-pool features only |

Source scaling, covariance, centroids and neighbor reference sets never use
evaluation rows. The domain RF uses domain labels with domain class balancing,
not rice labels. This differs from the earlier class-matched diagnostic and is
kept in the new namespace. Scores, row IDs and source probabilities are saved
and hashed before target labels are read for scoring. Direction is fixed as
large = risky; no flipping of below-chance scores is allowed.

Domain AUROC/AUPRC use the same score on 314 source-test and 2,751 target-test
points. Their sampled domain/class mixture is not prevalence-matched; report
domain-target prevalence (and do not claim architecture-invariant separation).
Error AUROC/AUPRC rank target disagreement, using a different outcome. Report
error prevalence as the no-skill AUPRC reference. Bootstrap target spatial
blocks 500 times; skip single-outcome draws and report the valid count.
These intervals condition on one source model, not model-training uncertainty.

Sample Spearman relates score to the binary error indicator; block Spearman
relates average score to empirical error rate in each block. They are different
estimands. Do not replace one with the higher number. Quantile bins preserve
ties, so fewer than ten bins can occur. Retain all negative findings.

## Selective prediction

The offline curve retains k=1..n target predictions in increasing score order.
Within exact tied groups use expected confusion counts under uniform random
acceptance. This makes expected error independent of row order. F1 computed
from these counts is explicitly named `f1_expected_counts`; because F1 is
nonlinear it is not the expectation of F1 over tie permutations. AURC equals
the arithmetic mean of the n retained-prefix risks, a right-step empirical
integral over coverage. There is no invented risk at coverage zero.

At 50/70/80/90/100%, use ceil(coverage × n) predictions and report actual coverage,
risk, accuracy and F1. Labels only evaluate accepted sets; they do not construct
scores or choose an operating point. A separate inductive experiment thresholds
held-out scores at target-pool score quantiles and reports actual achieved
coverage and risk. Tied threshold scores are all accepted, so coverage can exceed
the nominal value. Scores on the pool are in-sample for the domain classifier;
this can affect threshold transfer and is not hidden by relabeling nominal
coverage as actual coverage. Neither protocol guarantees a target risk bound.

Figures 5–6 display fixed diverse score families, not a method selected on the
target outcome; complete CSV curves retain all 11 scores. Binary confidence,
margin and entropy induce equivalent rankings mathematically and are redundant
comparators, not independent corroborations. Differences from floating-point
ties, if any, are numerical rather than scientific distinctions.

## Integrity and reproducibility boundaries

The audit validates original artifact hashes, canonical arrays, exact source
prediction rows, all 300 original draws, duplicate coordinates/features and
block disjointness. A hash of every old few-shot probability artifact is newly
recorded before reuse; equality to its saved F1 is checked. Unlike the old source
freeze, old few-shot NPZs did not carry individual historical hashes or row IDs:
row order is inherited from the original runner and validated by length and
metrics, not retrospectively proven from a preexisting per-file freeze.

Per-sample data and coordinates stay under ignored outputs. New aggregate
results have their own namespace. No original result, model, manifest or
parcel output is overwritten. Code and configuration changes invalidate cached predictions. Use the reproduction
configuration for a fresh run in a separate output directory. Figure
reproduction requires only aggregate CSVs. Full experiments require the exact
local canonical data, legacy model bundles, probabilities and original config.

New results do not establish independent Shanghai accuracy, conformal coverage,
transfer to another year, or parcel-level generalization. Spatial disjointness
does not rule out adjacent-block correlation or overlap of foundation-model
context windows. No independent geographic pretraining-overlap audit is available.


## Reproduction

The recorded execution configuration is `configs/geoai_rqs.yaml`. To repeat the
experiment without overwriting the published results, use the equivalent settings
in `configs/geoai_rqs_reproduce.yaml`. Only output directories differ.

```bash
pip install -e ".[dev,earthengine]"
python -m src.experiments.run_geoai_rqs --config configs/geoai_rqs_reproduce.yaml --stage audit
python scripts/run_rq1.py --config configs/geoai_rqs_reproduce.yaml
python scripts/run_rq2.py --config configs/geoai_rqs_reproduce.yaml
python scripts/run_rq3.py --config configs/geoai_rqs_reproduce.yaml
python scripts/make_paper_figures.py --config configs/geoai_rqs_reproduce.yaml
```

Full execution requires the canonical arrays, shared draws, source RF bundles and
few-shot predictions described in the [four-way reproduction guide](eofm_benchmark_reproduction.md).
Preserved input hashes are checked before fitting. The historical execution manifest
records the runtime configuration hashes; the published base configuration differs
only in release metadata. Fresh runs record their own input and environment hashes.

The public evidence can be checked without private runtime files:

```bash
python scripts/validate_geoai_results.py
python scripts/make_paper_figures.py
```

The complete risk–coverage CSV is gzip-compressed without changing any rows or
values. Pandas reads it directly. Tables report every score and method, including
negative results; the report highlights representative comparisons.

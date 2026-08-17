# Experiments and results

## Overview

| Experiment | Evaluation | Main result | Limitation |
|---|---|---|---|
| Source representation | Jiangxi source test | S2 F1 0.930; S1 F1 0.921; S1/S2 fusion F1 0.947; AlphaEarth F1 0.962 | Source-domain performance only |
| Zero-shot transfer | Shanghai official-product weak reference | Temporal fusion ≈0.813; AlphaEarth ≈0.836 | Not independent Shanghai accuracy; ordering reverses on a stricter subset |
| Low-label adaptation | Target-label budgets | At 500 labels, temporal target-only ≈0.854 and AlphaEarth ≈0.895 | Labels are weak-reference samples |
| OOD ranking | Held Shanghai spatial blocks | Distance error AUROC up to 0.800; nearest/farthest 10-NN deciles ≈1.1%/50.2% disagreement | Predicts weak-reference disagreement, not verified field error |
| Field model comparison | Same label-free 5 × 5 km AOI | FTW 231 objects vs DAv2 116; mean area 5.647 vs 13.990 ha | Workflow comparison, not architecture-only isolation |
| Parcel mapping | Two AOIs and 30-tile prototype | Strong mean/median consistency and large fragmentation reduction | Spatial coherence is not rice accuracy |
| Parcel QA | Label-independent rules | 8,227 raw objects → 7,330 parcels retained for mapping | Removed objects/pixels are not automatically classification errors |

Machine-readable selected results are under [`results/`](../results/README.md).

## Representation transfer

On the Jiangxi source test, S1/S2 temporal fusion improves on either sensor alone, while AlphaEarth gives the strongest source-domain result. Transfer to Shanghai is harder. On the full weak-reference validation set, AlphaEarth reaches approximately 0.836 F1 compared with 0.813 for temporal fusion.

That ordering is not universal. On a stricter interior/large-patch subset derived from the same reference product, temporal fusion reaches 0.749 and AlphaEarth 0.735. I therefore treat the AlphaEarth gain as reference-sensitive rather than evidence that one representation is always better.

With 500 distributed Shanghai weak labels, target-only temporal and AlphaEarth models reach approximately 0.854 and 0.895 F1. Across the larger repeated-seed analysis, the corresponding means are approximately 0.856 and 0.898. Spatially clustered supervision is less stable than distributing labels across blocks.

Several adaptation attempts did not improve the result. Three rounds of high-confidence pseudo-labeling reduce temporal agreement from 0.813 to 0.795, suggesting error reinforcement under domain shift. Directly concatenating the 92 temporal features with the 64 AlphaEarth dimensions also overfits at small label budgets. PCA compression reduces the penalty but still does not beat AlphaEarth alone.

## OOD and calibration

A feature-only Jiangxi-versus-Shanghai domain classifier reaches AUROC near 1, while permutation controls return to approximately 0.5. This confirms that the two regions are easily separable in representation space.

However, domain-classifier probability is not a useful score for ranking which Shanghai samples are most likely to fail: its error AUROC is 0.415. Distance from the Jiangxi training distribution performs much better, with error AUROC up to 0.800 and spatial-block Spearman up to 0.843.

The nearest and farthest 10-NN OOD deciles show approximately 1.1% and 50.2% weak-reference disagreement. When a risk threshold selected on tuning blocks is transferred to held spatial blocks, OOD-only rejection retains 49.4% coverage with 5.7% disagreement.

These numbers are useful for target-domain triage, but they are still evaluated against an official-product weak reference rather than independent field truth.

## DAv2 versus FTW

DAv2 completed CUDA inference on the test AOI, but the resulting field objects were too coarse for the parcel-mapping workflow. The 116 cleaned objects had a mean area of 13.99 ha, 11 objects exceeded 50 ha, and the largest object reached 201.96 ha. Several objects also included substantial non-crop context.

FTW produced 231 cleaned objects on the same 5 × 5 km AOI. Median area fell from 2.305 to 1.650 ha, mean area from 13.990 to 5.647 ha, non-crop-dominant objects from 27.59% to 9.52%, and Dynamic World water capture from 41.94% to 0.48%.

FTW is not perfect: some 50–133 ha blocks remain merged and some village/building structures are still detected. The comparison also uses different model inputs—FTW uses two-season RGB+NIR while DAv2 uses RGB—so the difference cannot be attributed to architecture alone.

## Independent AOI check

A second 5 × 5 km AOI, 12.98 km from the first and with a different land-cover mixture, was used to check whether the parcel behavior carried over spatially. It produced 577 accepted parcels, a mean/median probability correlation of 0.99487, 98.61% agreement among the three parcel summaries, and an 87.29% reduction in neighbor disagreement.

Among parcels additionally removed by the Dynamic World M1b gate, 72.38% were built-majority. These results supported moving to a larger Chongming prototype, but not a claim of full-Shanghai deployment accuracy.

## Chongming prototype and parcel QA

Running the workflow over 30 tiles produced 8,227 reconciled raw parcels. At that scale, several failure cases became obvious: very large coastal/water objects, greenhouse and built mosaics, deployment-edge artifacts, six residual overlap pairs, and clusters of tile warnings.

The parcel QA rules reduce these problems without using Shanghai rice labels to tune the classifier. The resulting product retains 7,330 parcels for mapping and keeps excluded or ambiguous objects separately so that the filtering process remains inspectable.

| Metric | Raw objects | After QA |
|---|---:|---:|
| Parcel count | 8,227 | 7,330 |
| Parcel area | 16,020.72 ha | 8,287.04 ha |
| Water-dominant area | 5,477.04 ha | 56.65 ha |
| Built-dominant area | 654.99 ha | 305.94 ha |
| Tree-dominant area | 745.52 ha | 444.23 ha |
| Objects >20 ha | 69 | 40 |
| Objects >50 ha | 28 | 0 |
| Residual overlap pairs | 6 | 0 |
| Warning tiles | 26/30 | 13/30 |

The large area reduction is mostly caused by enormous coastal and water objects in the raw field output. It should not be interpreted as measured false-positive removal.

## Pixel-to-parcel ablation

M0 predicts 31,079.76 ha above the inherited rice threshold. Before the final QA step, M1 and M1b retain 9,919.00 and 8,399.88 ha; after QA, they retain 7,728.64 and 6,786.84 ha. These values describe how field geometry and land-cover gates change mapped extent.

For parcel aggregation, the raw M2 product has mean/median correlation 0.991681, three-summary agreement 98.50%, and a 92.62% reduction in neighboring prediction disagreement. After QA, these values are 0.991621, 98.65%, and 90.74% respectively.

The final parcel output contains 4,324 Rice, 545 Non-rice, and 2,461 Uncertain/QA-risk parcels. These are model predictions awaiting independent validation.

## Wall-to-wall weak-reference consistency evaluation

The sampled weak-reference scores above use balanced 6,000 + 6,000 product samples. I also ran a wall-to-wall consistency evaluation that scores the full deployment rasters against the official Shanghai product as a weak reference. This is a deployment-consistency check, not an independent accuracy estimate.

All deployment rasters (M0 raw probability, M1 FTW-gated, M1b FTW + Dynamic World-gated, M2 parcel class, M2 QA final parcel) and the reference raster are co-registered at EPSG:32651, 20 m, 1071 × 1437. Each product is clipped to the common M0 valid footprint (1,500,751 pixels), which makes the predicted rice area invariant between the two evaluation modes and removes a previously reported M2 area discrepancy.

Two evaluation modes are reported:

- **Mode A — full-grid non-rice interpretation:** pixels a product excludes or abstains on are scored as predicted non-rice. This is the whole-map reading.
- **Mode B — conditional retained-coverage agreement:** agreement is computed only on the pixels each product retains. It is a conditional statistic on a subset and must not be read as a same-population improvement over M0.

Predicted rice area after clipping to the M0 footprint (ha):

| Product | Predicted rice area (ha) | Spatial retention |
|---|---:|---:|
| M0 raw | 31,079.76 | 100% |
| M1 FTW | 9,919.00 | 28.1% |
| M1b FTW + DW | 8,399.88 | 15.7% |
| M2 parcel | 4,646.80 | 8.7% |
| M2 QA | 4,646.80 | 8.7% |

These M0-footprint-clipped raster areas differ from the parcel-QA-retained area figures in the Pixel-to-parcel ablation section above, because they are measured on the binary raster within the M0 coverage and before the parcel-QA trim.

Weak-reference F1:

| Product | Mode A F1 (P / R) | Mode B F1 (P / R) |
|---|---|---|
| M0 raw | 0.329 (0.202 / 0.882) | 0.329 (0.202 / 0.882) |
| M1 FTW | 0.510 (0.438 / 0.610) | 0.602 (0.438 / 0.960) |
| M1b FTW + DW | 0.528 (0.488 / 0.575) | 0.648 (0.488 / 0.964) |
| M2 parcel | 0.359 (0.455 / 0.296) | 0.621 (0.455 / 0.978) |
| M2 QA | 0.359 (0.455 / 0.296) | 0.621 (0.455 / 0.978) |

The low weak-reference F1 is driven by **precision**, not recall. The official reference is sparse (7,134 ha within the M0 region) while the deployment products are wall-to-wall, so most predicted rice pixels fall outside the reference rice footprint and count as false positives. This is expected for a consistency check against a partial reference and does not by itself imply poor field-level accuracy.

The FTW gates (M1, M1b) raise both conditional-retained recall and agreement, but in Mode A they remove reference rice they abstain on; full-grid recall therefore falls (M0 0.882 → M1 0.610 → M1b 0.575) and spatial retention shrinks to 28.1% and 15.7%. M2 and M2 QA produce identical binary masks, so QA does not change the weak-reference score; QA only trims the Uncertain/QA-risk tail that the binary evaluation already treats as non-rice.

A 36-block paired analysis shows M1 improves block-level F1 versus M0 in 22/35 blocks (median ΔF1 ≈ +0.060) and M1b in 21/32 blocks (median ΔF1 ≈ +0.067), while M2 and M2 QA are neutral (median 0.0; 14 improved vs 13 worsened). The common grid contains 36 blocks, but paired ΔF1 is computed only where F1 is defined for both M0 and the compared product (n_evaluable: 35 M1, 32 M1b, 31 M2/M2_QA). The negative result is preserved: gating and parcel aggregation do not uniformly beat M0 in full-grid weak-reference F1.

Machine-readable outputs: [`results/tables/wall_to_wall_weak_reference_metrics.csv`](../results/tables/wall_to_wall_weak_reference_metrics.csv), [`results/tables/wall_to_wall_block_summary.csv`](../results/tables/wall_to_wall_block_summary.csv), and the narrative in [`results/summary/wall_to_wall_weak_reference_report.md`](../results/summary/wall_to_wall_weak_reference_report.md).

## What did not work

- DAv2 ran successfully but produced objects that were too coarse for this parcel workflow.
- A nearly perfect domain classifier did not identify which target samples were risky.
- Pseudo-labeling reinforced target-domain errors instead of improving transfer.
- Raw high-dimensional Sentinel + AlphaEarth concatenation overfit when only a few target labels were available.
- Gating and parcel QA strongly change mapped area, but removed area cannot be counted as labeled error.
- FTW provides useful agricultural structure, but it is not cadastral field truth.

The project chronology is in [`experiment_history.md`](experiment_history.md), methods are described in [`methodology.md`](methodology.md), and the independent evaluation plan is in [`validation.md`](validation.md).

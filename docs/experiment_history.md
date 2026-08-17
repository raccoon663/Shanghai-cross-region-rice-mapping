# Experiment history

This page records how the project developed from the original Jiangxi Sentinel experiment into the current cross-region transfer and parcel-mapping study. It also keeps older results separate from the later controlled experiments so that incompatible sample designs are not mixed.

## Sample designs

| Study | Samples | Use |
|---|---|---|
| Early ENVI experiment | 1,118 four-class Jiangxi ROI rows with 23 NDVI dates | Original Sentinel baseline and project provenance |
| Controlled binary transfer | 1,429 Jiangxi rice/non-rice samples; 850 train, 265 validation, 314 test | Sentinel and AlphaEarth source comparison |
| Shanghai weak-reference study | 12,000 balanced product-derived samples; 9,249 pool, 2,751 validation | Zero-shot transfer, adaptation, and OOD analysis |
| Chongming parcel experiment | 20 m source-only Sentinel probability raster | Field delineation, parcel aggregation, and QA |

The early four-class experiment and the later binary transfer study use different sample definitions. Their counts and metrics should not be compared directly.

## 1. Initial Sentinel transfer experiment

The project began by asking whether Sentinel-1/2 temporal fusion could improve geographic transfer of a rice classifier from Jinxian, Jiangxi, to Shanghai.

The original dataset contained 1,118 ENVI ROI samples and 23 dates from 2022-03-02 to 2022-11-12. I compared:

- Sentinel-2 NDVI only;
- Sentinel-1 VV/VH/RVI only;
- Sentinel-1/2 temporal fusion.

The earlier ENVI workflow produced very high within-source performance, but that result did not answer the harder question of geographic transfer. This motivated rebuilding the experiment with explicit binary labels and spatially separated splits.

## 2. Controlled spatial split

The transfer study was rebuilt around 1,429 valid Jiangxi rice/non-rice samples with group-disjoint spatial blocks. This made the source evaluation more conservative and gave a consistent sample membership for comparing Sentinel temporal features with AlphaEarth.

Source-test F1 scores were:

- Sentinel-2 only: **0.930**;
- Sentinel-1 only: **0.921**;
- Sentinel-1/2 fusion: **0.947**;
- AlphaEarth: **0.962**.

Three-seed neural baselines did not improve on the Random Forest baseline; the best S2 MLP reached about 0.939 F1.

## 3. Shanghai zero-shot transfer and adaptation

Applying the Jiangxi-trained models directly to Shanghai produced a clear performance drop. On the official-product weak reference, temporal fusion reached about **0.813 F1** and AlphaEarth about **0.836**.

Adding small amounts of Shanghai weak-reference data improved performance, but the sampling strategy mattered. At 500 labels, target-only temporal training reached about **0.854**, while AlphaEarth reached roughly **0.895–0.898**.

Not every adaptation method helped. Three rounds of high-confidence pseudo-labeling reduced temporal agreement from **0.813 to 0.795**, suggesting that source-model errors were being reinforced rather than corrected.

## 4. Feature-fusion experiments

I also tested whether the Sentinel time series and AlphaEarth embeddings were complementary enough to combine directly.

A raw 156-dimensional concatenation overfit under small target-label budgets. PCA compression reduced much of the penalty but still did not outperform AlphaEarth alone. This is why the project does not claim that a larger fused feature vector is automatically better.

At 500 labels, the two representations also made partly different errors, which motivated keeping the comparison rather than collapsing everything into one final feature set.

## 5. OOD analysis

The next question was whether the model could identify target samples that were far from the Jiangxi training distribution.

A Jiangxi-vs-Shanghai domain classifier achieved AUROC near **1.0**, showing that the regions are easy to distinguish in representation space. However, its probability was poor at ranking which Shanghai samples were likely to be misclassified (error AUROC about **0.415**).

Nearest-source distance worked much better:

- best error AUROC up to **0.800**;
- block-level Spearman up to **0.843**;
- nearest vs. farthest 10-NN distance deciles had roughly **1.1% vs. 50.2%** weak-reference disagreement.

This distinction became an important result of the project: **detecting that two regions are different is not the same as identifying which target samples are risky.**

## 6. Reference sensitivity

The Shanghai labels used in these experiments come from an official product rather than independent field observations. To check whether the main conclusions depended strongly on uncertain boundaries or tiny patches, I repeated key comparisons on a stricter interior/large-patch subset.

On that subset, the zero-shot ordering between AlphaEarth and temporal features reversed slightly. This is one reason the repository reports Shanghai numbers as weak-reference agreement rather than as independent target-domain accuracy.

## 7. From pixels to parcels

After the transfer experiments, I extended the project from sample-level evaluation to a spatial mapping prototype in Chongming.

Delineate Anything v2 provided a usable pretrained baseline, but on the shared 5 × 5 km AOI it frequently merged multiple visible fields and leaked into non-crop surfaces. FTW PRUE produced more narrow-field structure and was therefore used for the parcel experiment.

The 20 m Jiangxi-trained Sentinel probability raster was kept unchanged. FTW field polygons were used to aggregate the same probabilities into parcels, followed by label-independent QA for obvious water, built-up areas, extreme geometry, tile disagreement, deployment edges, and overlaps.

The full Chongming run exposed several practical problems that were not visible in the small AOI tests, especially very large coastal/water polygons and cross-tile overlap cases. These were handled as geometry/deployment issues rather than being counted as rice-classification errors.

## 8. Current endpoint

The repository now contains three connected pieces:

1. **cross-region transfer** — Jiangxi source models evaluated in Shanghai;
2. **representation and OOD analysis** — Sentinel vs. AlphaEarth, adaptation, calibration, and risk ranking;
3. **parcel deployment extension** — FTW field structure, probability aggregation, and QA in Chongming.

Independent Shanghai parcel labels are still missing. The current parcel product should therefore be treated as a mapping prototype prepared for validation, not as a final accuracy-validated crop map.

For the current methods and retained quantitative results, see:

- [`methodology.md`](methodology.md)
- [`experiments.md`](experiments.md)
- [`validation.md`](validation.md)
- [`../results/`](../results/README.md)

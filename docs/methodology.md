# Methodology

## Study scope

This repository studies geographic transfer of a 2022 rice classifier from Jiangxi to central/eastern Chongming, Shanghai. There are two connected questions: how well a source-trained representation transfers to Shanghai, and how pixel-level probabilities can be converted into a usable parcel map.

The source rice classifier is kept unchanged during parcel integration. FTW field boundaries and parcel QA rules are defined without using Shanghai rice labels to tune the classifier.

## Data and spatial splits

The controlled binary source study contains 1,429 valid Jiangxi samples: 622 non-rice and 807 rice. Group-disjoint spatial blocks define 850 training, 265 validation, and 314 source-test samples.

The Shanghai weak-reference study contains 12,000 balanced samples derived from an official product. These are divided into 9,249 target-pool samples and 2,751 spatially disjoint target-validation samples. Their shared manifest is [`data_metadata/alphaearth_sample_manifest.csv`](../data_metadata/alphaearth_sample_manifest.csv).

These samples are separate from the earlier 1,118-row, four-class ENVI ROI experiment. Metrics from the legacy and controlled studies are therefore reported separately.

## Representations and source classifier

The Sentinel temporal representations use 23 documented dates in 2022:

- Sentinel-2: 23 NDVI observations;
- Sentinel-1: 23 observations each of VV, VH, and RVI;
- temporal fusion: all 92 Sentinel features;
- AlphaEarth comparison: 64-dimensional 2022 annual embeddings for the same sample membership.

The parcel-mapping extension starts from the source-only 92-feature random-forest probability raster at 20 m in EPSG:32651. The rice decision threshold remains 0.50 during field gating and parcel aggregation.

Configurations are stored under [`configs/`](../configs/), and public collection reconstruction is documented in [`DATA_AVAILABILITY.md`](../DATA_AVAILABILITY.md).

## Transfer and OOD analysis

The representation study compares source performance, Shanghai weak-reference agreement, low-label adaptation, calibration, domain separability, and distance from the Jiangxi training distribution.

OOD scores are computed from representation features rather than coordinates or sample identifiers. Thresholds used for selective prediction are chosen on tuning blocks and evaluated on separate spatial blocks.

The domain-classifier probability and nearest-source distance answer different questions. A model can easily distinguish Jiangxi from Shanghai while still providing little information about which individual Shanghai samples are likely to fail. The AlphaEarth/OOD analysis is therefore kept as a sample-level diagnostic; this repository does not claim a wall-to-wall AlphaEarth OOD map for Shanghai.

## Field delineation

A label-free 5 × 5 km Chongming AOI was used to compare two pretrained field-delineation workflows.

DAv2 used a georeferenced 2022 Sentinel-2 RGB composite. The baseline used the AGPL-3.0 `Lavreniuk/Delineate-Anything` code at commit `278a3d91e78535174e2a80865a67c77e855d3a91` and the Hugging Face `MykolaL/DelineateAnything` revision `cff6ad11e3a0a7ccdcf1261fefb148309b5d6a8a`. The one-class Ultralytics checkpoint `DelineateAnythingv2.pt` has SHA-256 `46700b8a279b07922953a11adaeb5e658d9a2384b6334c8e0a3090886218915a`. CUDA inference was run through the recorded local geospatial tiler; the repository's GDAL entry point was not run end to end. Neither code nor checkpoint is redistributed here.

FTW PRUE used two seasonal Sentinel-2 RGB+NIR windows in `B04,B03,B02,B08` order, stacked as eight 10 m channels. The selected model is the official `FTW_PRUE_EFNET_B5_CCBY` release, a three-class U-Net with an EfficientNet-B5 encoder that predicts background, field interior, and boundary. The v3.1 CC-BY-4.0 checkpoint `prue_efnetb5_ccby_checkpoint.ckpt` has SHA-256 `f87c041c1a98e1c7f5366012d2a07d4372eb392f36c4bccee4779e73189bcf97`. Model weights are not redistributed by this repository.

FTW semantic predictions are polygonized into candidate field instances. Overlapping tiles are reconciled before the vectors are aligned with the 20 m rice-probability grid. The 10 m field product and 20 m probability raster share EPSG:32651 and an exact 2:1 grid relationship.

## Parcel-mapping ablation

Four spatial products are compared:

- **M0:** unchanged wall-to-wall source-only rice probability;
- **M1:** M0 restricted to FTW field geometry;
- **M1b:** M1 intersected with an independent Dynamic World crop/flooded-vegetation gate;
- **M2:** parcel summaries of the same probabilities using mean, median, and positive-pixel fraction.

M1 and M1b change where predictions are mapped, not the underlying probability values. Pixels removed by a gate are therefore not counted as classification errors without independent reference labels. M2 reduces fragmentation, but increased spatial smoothness alone is not evidence of higher thematic accuracy.

## Parcel QA rules

Large-scale mapping exposed several geometry and land-cover failures that were uncommon in the small AOIs, including coastal/water polygons, built-up mosaics, very large objects, tile-edge disagreement, and residual overlaps.

The final QA rules use Dynamic World class fractions, parcel area and shape, FTW tile disagreement, cross-tile reconciliation metadata, and deployment-edge context. Rice probabilities and Shanghai rice labels are not used to define these rules.

Hard exclusions are:

- Dynamic World water fraction at least 0.80;
- Dynamic World built fraction at least 0.80;
- area above 20 ha with combined water and bare fraction at least 0.70;
- original parcel area above 50 ha;
- a repaired overlap fragment below 2,500 m².

Moderate water/built fractions, tree dominance, 20–50 ha area, deployment-edge contact, tile disagreement of at least 0.25, and repaired overlap participation are retained as QA flags rather than automatically removed.

The parcel outputs are organized as:

- **Product A:** raw reconciled FTW parcels;
- **Product B:** parcels retained for mapping, including clearly marked soft-risk cases;
- **Product C:** hard-excluded and soft-flagged objects with recorded reasons.

The repository stores schemas, selected tables, figures, and checksums. Full GeoTIFF/GPKG runtime products remain local under the ignored `outputs/` convention; their checksums are listed in [`results/summary/final_chongming_parcel_product_manifest.json`](../results/summary/final_chongming_parcel_product_manifest.json).

## Reproducibility and claim limits

Scientific imports and runtime paths continue to use `outputs/`; excluding generated products from Git does not change code behavior. External checkpoints, large reconstructed imagery, the official Shanghai reference raster, virtual environments, and complete generated output trees are not included in the repository.

The current evidence supports conclusions about source performance, weak-reference transfer, OOD risk ranking, field geometry, deployment coverage, parcel coherence, and QA behavior. It does not yet support independent Shanghai parcel precision, recall, F1, overall accuracy, or area accuracy.

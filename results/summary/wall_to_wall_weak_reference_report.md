# Wall-to-Wall Deployment Evaluation — Weak-Reference Consistency

**Repository:** Shanghai-cross-region-rice-mapping (Chongming, Shanghai prototype)
**Experiment family:** `wall_to_wall_deployment` (deployment products M0 → M1 → M1b → M2 → M2_QA)
**Reference:** Shanghai official-product rice raster, used here as a **weak reference only**.
**Status:** frozen, reproducible, no threshold tuning, no retraining.

> **This is NOT an accuracy study.** The Shanghai official product is used as a
> *weak reference*. Agreement with it is reported as **wall-to-wall weak-reference
> consistency**, not as independent Shanghai rice accuracy. No independent
> parcel/field ground truth is available. All deployment products use the
> inherited Jiangxi-trained classifier and the frozen `0.50` threshold; nothing
> is retrained or tuned against the official product, and the official product
> is never used to build the FTW parcels.

---

## 1. What is evaluated

The same frozen artifacts the deployment pipeline produced are evaluated against
the official-product weak reference:

| Label | Product | Source artifact |
|---|---|---|
| M0 | **Raw transfer** | `outputs/final_chongming_parcel_product/M0_raw_probability.tif` (Jiangxi-trained source-only RF, inherited 0.50 threshold) |
| M1 | **FTW-gated** | `outputs/phase4_chongming_staged/products/M1_ftw_gated_probability.tif` (M0 restricted to accepted FTW field mask) |
| M1b | **FTW + independent land-cover gate** | `outputs/phase4_chongming_staged/products/M1b_ftw_dynamicworld_gated_probability.tif` (M1 ∩ Dynamic World crop/flooded-veg) |
| M2 | **Parcel aggregation** | `outputs/phase4_chongming_staged/products/parcel_class_20m.tif` (0=Non-rice, 1=Rice, 2=Uncertain/QA-risk abstained) |
| M2_QA | **Final deployment-safe parcel product** | rasterized `outputs/final_chongming_parcel_product/final_parcel_class.gpkg` (Rice/Non-rice committed; Uncertain/QA-risk abstained) |

A **separate** evaluation family, `sampled_representation`, reports the earlier
balanced-sample weak-reference agreement (temporal fusion F1 = 0.813; AlphaEarth
F1 = 0.836). These are *sampled* representation results on balanced 6000-rice +
6000-non-rice weak-reference pixels and are **never** conflated with the
wall-to-wall deployment products. In particular, **AlphaEarth 0.836 is NOT M1b**.

---

## 2. Two evaluation modes (and why both are reported)

To keep the positive prediction mask — and therefore predicted rice area —
**identical across modes**, every product is clipped to a common evaluation grid
(the M0 valid footprint intersected with the reference extent). This also fixes
a pre-audit discrepancy where 535 M2 rice pixels (≈21.4 ha) fell outside the M0
footprint; after clipping, predicted rice area is invariant across modes by
construction (asserted in code and in `tests/test_wall_to_wall_metrics.py`).

- **Mode A — Full-grid weak-reference agreement** (non-rice interpretation of
  excluded pixels). The deployed product is read as a complete rice map: pixels
  the product does **not** retain are treated as operational **NON-RICE**
  ("rice only where the product maps rice"). This answers the real deployment
  question: *what does the whole TIFF look like once spatial priors are applied
  and unmapped area is read as non-rice?*

- **Mode B — Conditional retained-coverage agreement.** Classification metrics
  are computed **only** over the pixels the product actually retains (committed
  Rice/Non-rice) within the common grid; coverage is reported separately. These
  metrics are **conditioned on the retained region**, so they measure
  *quality-at-coverage*, **not** a same-population gain over products that
  retain more area. Mode B numbers must never be read as "M1b is more accurate
  than M0."

---

## 3. Alignment audit

| Check | Result |
|---|---|
| CRS | all `EPSG:32651` |
| Pixel size | 20 m |
| Shape | all `1071 × 1437` |
| Transform | identical across M0/M1/M1b/M2/M2_QA/reference |
| Verdict | **PASS** — all products perfectly co-registered; no resampling applied |

(See `results/summary/wall_to_wall_alignment_audit.json`.)

Evaluation grid: M0 valid footprint, 1,500,751 valid pixels; reference rice
inside the grid = 178,357 px = 7,134.28 ha.

---

## 4. Results

### 4.1 Mode A — Full-grid (operational whole-map interpretation)

| Product | Precision | Recall | F1 | IoU | Coverage | Pred. rice (ha) | Ref. rice (ha) | Area Δ (ha) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M0 (raw transfer) | 0.202 | 0.882 | **0.329** | 0.197 | 100.0% | 31,079.76 | 7,134.28 | +23,945.48 |
| M1 (FTW) | 0.438 | 0.610 | **0.510** | 0.342 | 28.1% | 9,919.00 | 7,134.28 | +2,784.72 |
| M1b (FTW+DW) | 0.488 | 0.575 | **0.528** | 0.359 | 15.7% | 8,399.88 | 7,134.28 | +1,265.60 |
| M2 (parcel) | 0.455 | 0.296 | **0.359** | 0.219 | 8.7% | 4,646.80 | 7,134.28 | −2,487.48 |
| M2_QA (final) | 0.455 | 0.296 | **0.359** | 0.219 | 8.7% | 4,646.80 | 7,134.28 | −2,487.48 |

### 4.2 Mode B — Conditional retained-coverage (quality-at-coverage)

| Product | Precision | Recall | F1 | IoU | Coverage | Pred. rice (ha) | Ref. rice retained (ha) | Area Δ (ha) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M0 | 0.202 | 0.882 | 0.329 | 0.197 | 100.0% | 31,079.76 | 7,134.28 | +23,945.48 |
| M1 | 0.438 | 0.960 | **0.602** | 0.431 | 28.1% | 9,919.00 | 4,527.76 | +5,391.24 |
| M1b | 0.488 | 0.964 | **0.648** | 0.480 | 15.7% | 8,399.88 | 4,254.76 | +4,145.12 |
| M2 | 0.455 | 0.978 | **0.621** | 0.450 | 8.7% | 4,646.80 | 2,161.88 | +2,484.92 |
| M2_QA | 0.455 | 0.978 | **0.621** | 0.450 | 8.7% | 4,646.80 | 2,161.88 | +2,484.92 |

### 4.3 Sampled representation (separate family — NOT wall-to-wall)

| Representation | Weak-reference F1 | Population |
|---|---:|---|
| Temporal fusion | 0.813 | balanced 6000 rice + 6000 non-rice weak-reference pixels |
| AlphaEarth | 0.836 | balanced 6000 rice + 6000 non-rice weak-reference pixels |

---

## 5. Interpretation

### 5.1 Full-grid (Mode A): gating raises precision, but at the cost of recalled reference rice

- **M0** over-predicts rice massively relative to the weak reference: 619,672
  false positives vs 157,322 true positives → precision **0.20**. The low F1
  (0.33) is driven by this **class imbalance** (the reference is predominantly
  non-rice across the wall-to-wall grid), not by a precision/recall trade-off
  alone.
- **M1** improves precision to **0.44** and full-grid F1 to **0.51**; recall
  drops from 0.88 → 0.61 because FTW gating removes reference-rice pixels that
  fall outside the accepted field mask (in full-grid mode those become
  operational non-rice).
- **M1b** adds the independent Dynamic World gate: precision **0.49**, F1
  **0.53**, recall 0.57.
- **M2 / M2_QA** show the strongest precision (**0.45**) but the **lowest
  recall (0.30)** and only **8.7% coverage** — parcel aggregation concentrates
  on coherent field objects and leaves most of the grid (including much
  reference rice) unmapped. Full-grid F1 is **0.36**.

### 5.2 Conditional-retained (Mode B): high quality-at-coverage, but on a small, selected subset

- M1 reaches **0.60**, M1b **0.65**, M2/M2_QA **0.62** retained-coverage F1,
  with recall **0.96–0.98**. These numbers describe *how good the product is
  where it chooses to map*, **not** how much of the region it covers. Coverage
  is 28.1% (M1), 15.7% (M1b), 8.7% (M2). They must **not** be cited as
  "M1b accuracy > M0 accuracy."

### 5.3 The negative result is preserved

- M2/M2_QA full-grid recall (0.30) means **most reference rice is unmapped** by
  the parcel product. This is a genuine limitation, not a failure of the metric.
- Paired block analysis over the **same 36 blocks** (full-grid F1 vs M0):
  - M1 improves **22/35** blocks (median ΔF1 +0.060);
  - M1b improves **21/32** blocks (median ΔF1 +0.067);
  - M2/M2_QA are **neutral** (median ΔF1 = 0.0; 14 improved / 13 worsened /
    4 unchanged).
- Net: field gating (M1/M1b) broadly *improves* weak-reference consistency
  block-by-block; parcel aggregation (M2/M2_QA) does not, despite high
  quality-at-coverage on the retained subset.

---

## 6. Reproducibility

- Fully config-driven: `configs/wall_to_wall_weak_reference.yaml`
  (repo-relative input paths, overridable via `WALL_TO_WALL_DATA_ROOT`).
- No machine-specific absolute paths; all outputs written relative to the
  repository root.
- Deterministic; no randomness. Fails loudly on missing inputs, raster
  misalignment, or invalid class coding.
- Entry point:
  `python scripts/parcel_mapping/evaluate_wall_to_wall_weak_reference.py --config configs/wall_to_wall_weak_reference.yaml`
- Block summary: `python scripts/parcel_mapping/analyze_wall_to_wall_blocks.py`
- Tests: `pytest -q` (logic tests always run; raster tests skip without frozen
  inputs).

---

## 7. Artifacts produced

| File | Purpose |
|---|---|
| `results/tables/wall_to_wall_weak_reference_metrics.csv` | Per-product, per-mode metrics (both families) |
| `results/tables/wall_to_wall_block_metrics.csv` | Per-block full-grid F1 over 36 paired blocks |
| `results/tables/wall_to_wall_block_summary.csv` | Paired block improvement/worsening counts |
| `results/summary/wall_to_wall_alignment_audit.json` | Co-registration audit |
| `results/summary/wall_to_wall_experiment_summary.json` | Compact summary |
| `results/summary/wall_to_wall_weak_reference_report.md` | This report |
| `assets/figures/wall_to_wall_performance_comparison.png` | Fig 1: full-grid P/R/F1/IoU |
| `assets/figures/wall_to_wall_area_coverage_tradeoff.png` | Fig 2: predicted vs reference area + coverage |
| `assets/figures/wall_to_wall_block_delta.png` | Fig 3: block-level ΔF1 (M2_QA − M0) |

All large rasters/GPKG inputs remain git-ignored (`outputs/`, `inputs/`); only
the compact tables, figures, config, scripts, and this report are tracked.

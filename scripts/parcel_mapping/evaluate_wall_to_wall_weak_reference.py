#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wall-to-wall weak-reference consistency evaluation
=================================================

Evaluates the Shanghai (Chongming prototype) rice-mapping products against the
Shanghai *official-product* raster used in this project as a WEAK REFERENCE.

This is NOT independent ground truth. The official product is a weak reference;
agreement with it is "wall-to-wall weak-reference consistency", not "accuracy".

Products evaluated (the same frozen artifacts the deployment pipeline produced):

  M0      RAW TRANSFER
          outputs/final_chongming_parcel_product/M0_raw_probability.tif
          Jiangxi-trained source-only RF probability at inherited threshold 0.50.

  M1      FTW GATED
          outputs/phase4_chongming_staged/products/M1_ftw_gated_probability.tif
          M0 restricted to the accepted FTW agricultural-field mask.

  M1b     FTW + INDEPENDENT LAND-COVER GATE
          outputs/phase4_chongming_staged/products/M1b_ftw_dynamicworld_gated_probability.tif
          M1 intersected with an independent Dynamic World crop/flooded-vegetation gate.

  M2      PARCEL AGGREGATION (Phase-4 raw FTW parcels)
          outputs/phase4_chongming_staged/products/parcel_class_20m.tif
          0=Non-rice, 1=Rice, 2=Uncertain/QA-risk (abstained), 255=nodata.

  M2_QA   FINAL DEPLOYMENT-SAFE PARCEL PRODUCT
          Rasterized outputs/final_chongming_parcel_product/final_parcel_class.gpkg
          Rice / Non-rice committed pixels mapped, Uncertain/QA-risk abstained.

We DO NOT retrain, DO NOT tune the threshold, DO NOT optimize FTW/safeguard
rules against the official labels, and DO NOT use the official product to build
the FTW parcels. The frozen classifier and deployment rules are unchanged.

Two evaluation modes are reported for every deployment product. To keep the
positive prediction mask - and therefore predicted rice area - identical across
modes, EVERY product is clipped to the same common evaluation grid (the M0
valid footprint intersected with the reference extent).

  MODE A - FULL-GRID weak-reference agreement
      under a non-rice interpretation of excluded pixels.
      The deployed product is read as a complete rice map. Pixels the product
      does not retain are treated as operational NON-RICE ("rice only where the
      product maps rice"). This is documented, not silent. It answers the real
      deployment question: "what does the whole TIFF look like once spatial
      priors are applied and unmapped area is read as non-rice?"

  MODE B - CONDITIONAL RETAINED-COVERAGE AGREEMENT
      Classification metrics are computed ONLY over the pixels the product
      actually retains (committed Rice/Non-rice) within the common grid.
      Coverage is reported separately. These metrics are CONDITIONED on the
      retained region, so they measure quality-at-coverage, not a same-population
      improvement over products that retain more area.

Outputs (all written relative to the repository root, never to a scratch dir):
  results/tables/wall_to_wall_weak_reference_metrics.csv
  results/tables/wall_to_wall_block_metrics.csv
  results/summary/wall_to_wall_alignment_audit.json
  results/summary/wall_to_wall_experiment_summary.json
  assets/figures/wall_to_wall_performance_comparison.png
  assets/figures/wall_to_wall_area_coverage_tradeoff.png
  assets/figures/wall_to_wall_block_delta.png

Deterministic. No randomness. Fails loudly on missing inputs, raster
misalignment, or invalid class coding.

Usage:
  python scripts/parcel_mapping/evaluate_wall_to_wall_weak_reference.py \
      --config configs/wall_to_wall_weak_reference.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
import pandas as pd
import yaml

# ----------------------------------------------------------------------------
# Repository-relative path resolution (no machine-specific absolute paths).
# ----------------------------------------------------------------------------
SCRIPT = Path(__file__).resolve()
PARCEL_DIR = SCRIPT.parent                     # scripts/parcel_mapping
REPO = SCRIPT.parents[2]                        # repository root

PIXEL_AREA_HA = 20.0 * 20.0 / 10_000.0          # 0.04 ha per 20 m pixel

RICE_CLASS = 1
NONRICE_CLASS = 0
UNCERTAIN_CLASS = 2      # parcel_class_20m "Uncertain / QA-risk" -> abstained
NODATA_BYTE = 255


# ----------------------------------------------------------------------------
# Metric helpers (pure, no I/O)
# ----------------------------------------------------------------------------
def confusion(pred, ref, valid):
    """pred, ref, valid: 1-D bool/int arrays over the SAME pixel set.

    Returns dict of TP, FP, FN, TN counts.
    """
    p = np.asarray(pred)[np.asarray(valid)].astype(bool)
    r = np.asarray(ref)[np.asarray(valid)].astype(bool)
    tp = int(np.count_nonzero(p & r))
    fp = int(np.count_nonzero(p & ~r))
    fn = int(np.count_nonzero(~p & r))
    tn = int(np.count_nonzero(~p & ~r))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def metrics_from_conf(c: dict):
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    n = tp + fp + fn + tn
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else float("nan")
    acc = (tp + tn) / n if n else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    bal = (rec + spec) / 2.0
    agreement = (tp + tn) / n if n else float("nan")
    return {
        "precision": prec, "recall": rec, "f1": f1, "iou": iou,
        "accuracy": acc, "balanced_accuracy": bal, "agreement_fraction": agreement,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


# ----------------------------------------------------------------------------
# Input loading + alignment audit
# ----------------------------------------------------------------------------
def load_reference(path: Path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Official weak-reference raster not found: {path}\n"
            f"Place it under inputs/official_reference/ or set the "
            f"WALL_TO_WALL_DATA_ROOT environment variable. See DATA_AVAILABILITY.md."
        )
    with rasterio.open(path) as s:
        a = s.read(1)
        prof = {
            "crs": str(s.crs), "transform": list(s.transform),
            "shape": list(a.shape), "dtype": str(s.dtypes[0]),
            "nodata": s.nodata, "bounds": list(s.bounds),
            "width": s.width, "height": s.height,
        }
        uniq = np.unique(a)
        if not set(uniq.tolist()).issubset({0, 1}):
            raise ValueError(f"Reference has unexpected class codes: {uniq.tolist()}")
        ref_bin = (a == 1).astype(np.uint8)
    return ref_bin, prof


def load_probability(path: Path, threshold: float):
    if not path.is_file():
        raise FileNotFoundError(
            f"Probability raster not found: {path}\n"
            f"Restore the frozen pipeline output or set WALL_TO_WALL_DATA_ROOT. "
            f"See DATA_AVAILABILITY.md."
        )
    with rasterio.open(path) as s:
        a = s.read(1).astype(np.float32)
        prof = {
            "crs": str(s.crs), "transform": list(s.transform),
            "shape": list(a.shape), "dtype": str(s.dtypes[0]),
            "nodata": s.nodata, "bounds": list(s.bounds),
            "width": s.width, "height": s.height,
        }
        valid = ~np.isclose(a, s.nodata) if s.nodata is not None else np.ones(a.shape, bool)
        pred_bin = (a >= threshold) & valid
    return pred_bin, valid, prof


def load_m2_parcel(path: Path):
    """parcel_class_20m: 0=Non-rice, 1=Rice, 2=Uncertain, 255=nodata."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Parcel class raster not found: {path}\n"
            f"Restore the frozen pipeline output or set WALL_TO_WALL_DATA_ROOT. "
            f"See DATA_AVAILABILITY.md."
        )
    with rasterio.open(path) as s:
        a = s.read(1).astype(np.uint8)
        prof = {
            "crs": str(s.crs), "transform": list(s.transform),
            "shape": list(a.shape), "dtype": str(s.dtypes[0]),
            "nodata": s.nodata, "bounds": list(s.bounds),
            "width": s.width, "height": s.height,
        }
        uniq = np.unique(a)
        if not set(uniq.tolist()).issubset({0, 1, 2, 255}):
            raise ValueError(f"M2 parcel_class has unexpected codes: {uniq.tolist()}")
        valid = a != NODATA_BYTE
        committed = valid & (a != UNCERTAIN_CLASS)     # 0 / 1 only
        pred_bin = np.zeros(a.shape, bool)
        pred_bin[committed & (a == RICE_CLASS)] = True
    return pred_bin, valid, committed, prof


def rasterize_m2qa(path: Path, template_profile):
    """Rasterize final_parcel_class.gpkg: Rice->1, Non-rice->0,
    Uncertain/QA-risk -> abstained (nodata)."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Final parcel class GeoPackage not found: {path}\n"
            f"Restore the frozen pipeline output or set WALL_TO_WALL_DATA_ROOT. "
            f"See DATA_AVAILABILITY.md."
        )
    gdf = gpd.read_file(path)
    if "parcel_class_phase5" not in gdf.columns:
        raise KeyError("parcel_class_phase5 missing in final parcel class gpkg")
    geom_rice, geom_nonrice = [], []
    for _, row in gdf.iterrows():
        cls = row["parcel_class_phase5"]
        if cls == "Rice":
            geom_rice.append(row.geometry)
        elif cls == "Non-rice":
            geom_nonrice.append(row.geometry)
    out = np.full(template_profile["shape"], NODATA_BYTE, dtype=np.uint8)
    if geom_nonrice:
        rz = rasterize(
            [(g, NONRICE_CLASS) for g in geom_nonrice],
            out_shape=template_profile["shape"],
            transform=rasterio.Affine(*template_profile["transform"]),
            fill=NODATA_BYTE, dtype=np.uint8)
        out[rz != NODATA_BYTE] = NONRICE_CLASS
    if geom_rice:
        rz = rasterize(
            [(g, RICE_CLASS) for g in geom_rice],
            out_shape=template_profile["shape"],
            transform=rasterio.Affine(*template_profile["transform"]),
            fill=NODATA_BYTE, dtype=np.uint8)
        out[rz == RICE_CLASS] = RICE_CLASS
    valid = out != NODATA_BYTE
    committed = valid                       # no third class rasterized
    pred_bin = np.zeros(out.shape, bool)
    pred_bin[out == RICE_CLASS] = True
    prof = {"crs": str(gdf.crs), "source": str(path.name),
            "shape": list(out.shape), "transform": list(template_profile["transform"]),
            "nodata": NODATA_BYTE,
            "n_rice_parcels": len(geom_rice), "n_nonrice_parcels": len(geom_nonrice),
            "n_total_parcels": len(gdf)}
    return pred_bin, valid, committed, prof


def verify_alignment(profiles: dict, ref_shape, ref_transform):
    """Fail loudly if any product is not perfectly co-registered with ref."""
    issues = []
    for name, p in profiles.items():
        if name == "__ref__":
            continue
        if tuple(p["shape"]) != tuple(ref_shape):
            issues.append(f"{name}: shape {p['shape']} != ref {ref_shape}")
        if list(p["transform"]) != list(ref_transform):
            issues.append(f"{name}: transform mismatch vs ref")
        if p["crs"] != profiles["__ref__"]["crs"]:
            issues.append(f"{name}: CRS {p['crs']} != ref CRS")
    if issues:
        raise RuntimeError("RASTER MISALIGNMENT:\n" + "\n".join(issues))
    return True


# ----------------------------------------------------------------------------
# Evaluation object
# ----------------------------------------------------------------------------
@dataclass
class ProductEval:
    name: str
    pred_common: np.ndarray     # 2-D committed-rice prediction over the common grid
    retained_common: np.ndarray # 2-D True where product makes a committed prediction
    # within the common grid (already clipped to common_region)
    is_prob: bool


def build_eval(name, pred_bin, valid, committed, common_region):
    """Clip everything to the common grid so predicted rice area is invariant.

    pred_common = rice-class committed pixels within the common grid.
    retained_common = committed (Rice/Non-rice) pixels within the common grid.
    """
    pred_common = np.zeros(common_region.shape, bool)
    retained_common = np.zeros(common_region.shape, bool)
    cc = common_region
    pred_common[cc & valid & committed & pred_bin] = True
    retained_common[cc & valid & committed] = True
    return ProductEval(name=name, pred_common=pred_common,
                       retained_common=retained_common, is_prob=False)


# ----------------------------------------------------------------------------
# Block metrics (paired, full-grid, identical 36 blocks)
# ----------------------------------------------------------------------------
def compute_block_metrics(evals, ref_bin, common_region, nb, min_px):
    nrows, ncols = common_region.shape
    br = nrows // nb
    bc = ncols // nb
    rows = []
    for bi in range(nb):
        for bj in range(nb):
            r0, r1 = bi * br, (bi + 1) * br if bi < nb - 1 else nrows
            c0, c1 = bj * bc, (bj + 1) * bc if bj < nb - 1 else ncols
            sub_region = common_region[r0:r1, c0:c1]
            n_block = int(sub_region.sum())
            if n_block < min_px:
                continue
            sub_ref = ref_bin[r0:r1, c0:c1][sub_region]
            ref_rice_block = int(sub_ref.sum())
            rec = {"block_row": bi, "block_col": bj, "block_id": f"R{bi}C{bj}",
                    "n_pixels": n_block, "reference_rice_pixels": ref_rice_block,
                    "reference_rice_area_ha": ref_rice_block * PIXEL_AREA_HA,
                    "zero_reference_rice": ref_rice_block == 0}
            for name in ["M0", "M1", "M1b", "M2", "M2_QA"]:
                prod = evals[name]
                # full-grid interpretation: within the block, abstained -> non-rice
                sub_pred = prod.pred_common[r0:r1, c0:c1][sub_region]
                c = confusion(sub_pred, sub_ref, np.ones(n_block, bool))
                m = metrics_from_conf(c)
                cov = int(prod.retained_common[r0:r1, c0:c1][sub_region].sum()) / n_block
                rec[f"{name}_coverage"] = cov
                rec[f"{name}_f1"] = m["f1"]
                rec[f"{name}_precision"] = m["precision"]
                rec[f"{name}_recall"] = m["recall"]
                rec[f"{name}_iou"] = m["iou"]
                rec[f"{name}_pred_rice_area_ha"] = int(sub_pred.sum()) * PIXEL_AREA_HA
                rec[f"{name}_delta_f1_vs_M0"] = float("nan")
            m0_f1 = rec["M0_f1"]
            for name in ["M1", "M1b", "M2", "M2_QA"]:
                v = rec[f"{name}_f1"]
                rec[f"{name}_delta_f1_vs_M0"] = (
                    (v - m0_f1) if (not np.isnan(v) and not np.isnan(m0_f1)) else float("nan"))
            rows.append(rec)
    return rows


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def make_figures(df, bdf, order, ref_shape, out_fig):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prod_label = {"M0": "M0_raw", "M1": "M1_FTW", "M1b": "M1b_FTW_DW",
                  "M2": "M2_parcel", "M2_QA": "M2_QA"}
    _rec = {(r.representation_or_product, r.evaluation_mode): r for r in df.itertuples()}
    # Full-grid values are the headline (Mode A).
    def g(name, field):
        return getattr(_rec[(prod_label[name], "A_full_grid_nonrice_interpretation")], field)

    colors = {"M0": "#315A7D", "M1": "#4472A6", "M1b": "#4D8B5E",
              "M2": "#D19A45", "M2_QA": "#694F8E"}

    # ---- Figure 1: full-grid performance (Precision/Recall/F1/IoU) ----
    fig, axs = plt.subplots(1, 4, figsize=(15, 5))
    metrics4 = [("precision", "Precision"), ("recall", "Recall"),
                ("f1", "F1"), ("iou", "IoU / Jaccard")]
    for ax, (m, title) in zip(axs, metrics4):
        vals = [g(n, m) for n in order]
        ax.bar(order, vals, color=[colors[n] for n in order])
        for i, v in enumerate(vals):
            ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.tick_params(labelsize=8)
    fig.suptitle("Figure 1. Full-grid wall-to-wall weak-reference performance\n"
                 "(non-rice interpretation of excluded pixels; same frozen classifier)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_fig / "wall_to_wall_performance_comparison.png", dpi=200)
    plt.close(fig)

    # ---- Figure 2: predicted rice area + coverage trade-off ----
    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(order))
    pred_area = [g(n, "predicted_rice_area_ha") for n in order]
    ref_area = g("M0", "reference_rice_area_ha")
    ax1.bar(x - 0.2, pred_area, width=0.4, color="#C0504D", label="Predicted rice area (ha)")
    ax1.axhline(ref_area, color="#2E75B6", ls="--", lw=2,
                label=f"Official weak-ref rice area = {ref_area:,.0f} ha")
    ax1.set_ylabel("Rice area (ha)")
    ax1.set_xticks(x); ax1.set_xticklabels(order)
    ax1.tick_params(labelsize=9)
    ax2 = ax1.twinx()
    cov = [g(n, "coverage_fraction") for n in order]
    ax2.plot(x, cov, "o-", color="#000000", lw=2, label="Retained coverage fraction")
    ax2.set_ylabel("Retained coverage fraction", color="#000000")
    ax2.set_ylim(0, 1.05)
    for i, c in enumerate(cov):
        ax2.text(i, c + 0.02, f"{c:.1%}", ha="center", fontsize=8)
    for i, pa in enumerate(pred_area):
        ax1.text(i - 0.2, pa + max(pred_area) * 0.02, f"{pa:,.0f}", ha="center", fontsize=8)
    ax1.set_title("Figure 2. Predicted rice area vs official weak-reference area, and retained coverage",
                  fontsize=11, fontweight="bold")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_fig / "wall_to_wall_area_coverage_tradeoff.png", dpi=200)
    plt.close(fig)

    # ---- Figure 3: spatial block delta F1 map (paired, full-grid, same 36 blocks) ----
    nb = 6
    grid = np.full((nb, nb), np.nan)
    for _, r in bdf.iterrows():
        grid[int(r["block_row"]), int(r["block_col"])] = r["M2_QA_delta_f1_vs_M0"]
    fig, ax = plt.subplots(figsize=(8.5, 7))
    cmap = plt.cm.RdBu_r
    vmax = np.nanmax(np.abs(grid))
    vmax = max(vmax, 0.05)
    im = ax.imshow(grid, cmap=cmap, vmin=-vmax, vmax=vmax, origin="upper")
    ax.set_title("Figure 3. Block-level \u0394F1 (M2_QA \u2212 M0)\n"
                 "full-grid weak-reference F1 change over the same 36 blocks",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Block column"); ax.set_ylabel("Block row")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("\u0394F1")
    fig.tight_layout()
    fig.savefig(out_fig / "wall_to_wall_block_delta.png", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def resolve_input(cfg, key, repo_root):
    root = Path(os.environ.get(cfg.get("data_root_env"), repo_root))
    return (root / cfg["inputs"][key]).resolve()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(REPO / "configs" / "wall_to_wall_weak_reference.yaml"))
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    threshold = float(cfg["parameters"]["inherited_threshold"])
    nb = int(cfg["parameters"]["block_grid"])
    min_px = int(cfg["parameters"]["min_block_pixels"])

    out_tables = REPO / "results" / "tables"
    out_summary = REPO / "results" / "summary"
    out_fig = REPO / "assets" / "figures"
    for d in (out_tables, out_summary, out_fig):
        d.mkdir(parents=True, exist_ok=True)

    # ---- load reference ----
    ref_bin, ref_prof = load_reference(resolve_input(cfg, "reference", REPO))
    ref_shape = ref_bin.shape
    ref_transform = ref_prof["transform"]

    # ---- load products ----
    profiles = {"__ref__": ref_prof}
    p0, v0, pr0 = load_probability(resolve_input(cfg, "M0", REPO), threshold)
    profiles["M0"] = pr0
    p1, v1, pr1 = load_probability(resolve_input(cfg, "M1", REPO), threshold)
    profiles["M1"] = pr1
    p1b, v1b, pr1b = load_probability(resolve_input(cfg, "M1b", REPO), threshold)
    profiles["M1b"] = pr1b
    p2, v2, c2, pr2 = load_m2_parcel(resolve_input(cfg, "M2", REPO))
    profiles["M2"] = pr2
    p2qa, v2qa, c2qa, pr2qa = rasterize_m2qa(
        resolve_input(cfg, "M2_QA", REPO),
        {"shape": ref_shape, "transform": ref_transform})
    profiles["M2_QA"] = pr2qa

    # ---- alignment audit (fail loudly on misalignment) ----
    verify_alignment(profiles, ref_shape, ref_transform)
    audit = {
        "crs": ref_prof["crs"],
        "pixel_size_m": float(cfg["parameters"]["pixel_size_m"]),
        "shape_rows_cols": list(ref_shape),
        "transform": ref_transform,
        "reference_class_coding": "0=non-rice, 1=rice; no nodata (fully coded)",
        "reference_unique_values": [0, 1],
        "products": {k: {kk: vv for kk, vv in v.items() if kk != "source"}
                     for k, v in profiles.items() if k != "__ref__"},
        "alignment_result": "PASS - all products perfectly co-registered (identical CRS/transform/shape)",
        "threshold_used": threshold,
        "pixel_area_ha": PIXEL_AREA_HA,
        "common_region_basis": cfg["common_region"]["basis"],
    }
    (out_summary / "wall_to_wall_alignment_audit.json").write_text(json.dumps(audit, indent=2))

    # ---- common evaluation grid = M0 valid footprint ----
    common_region = v0.copy()
    n_region = int(common_region.sum())
    ref_rice_total = int(ref_bin[common_region].sum())

    # ---- build ProductEval objects (all clipped to common_region) ----
    evals = {
        "M0": build_eval("M0", p0, v0, v0, common_region),
        "M1": build_eval("M1", p1, v1, v1, common_region),
        "M1b": build_eval("M1b", p1b, v1b, v1b, common_region),
        "M2": build_eval("M2", p2, v2, c2, common_region),
        "M2_QA": build_eval("M2_QA", p2qa, v2qa, c2qa, common_region),
    }

    # ---- per-product metrics (Mode A full-grid & Mode B conditional retained) ----
    rows = []
    for name in ["M0", "M1", "M1b", "M2", "M2_QA"]:
        prod = evals[name]
        # Mode A: full grid, abstained -> non-rice (valid = whole common_region)
        ma = metrics_from_conf(confusion(prod.pred_common, ref_bin, common_region))
        # Mode B: conditional retained coverage (only retained_common pixels)
        mb = metrics_from_conf(confusion(prod.pred_common, ref_bin, prod.retained_common))

        # predicted rice area must be invariant across modes by construction
        pred_rice_A = int(prod.pred_common[common_region].sum())
        pred_rice_B = int(prod.pred_common[prod.retained_common].sum())
        assert pred_rice_A == pred_rice_B, (
            f"Predicted rice area not invariant for {name}: "
            f"{pred_rice_A} (full-grid) vs {pred_rice_B} (retained)")
        pred_rice_ha = pred_rice_A * PIXEL_AREA_HA

        n_retained = int(prod.retained_common.sum())
        coverage = n_retained / n_region if n_region else float("nan")
        ref_rice_B = int(ref_bin[prod.retained_common].sum())

        # invariant-area test also guards against mode-dependent area reporting
        for m, label in ((ma, "A_full_grid_nonrice_interpretation"),
                         (mb, "B_conditional_retained_coverage")):
            rows.append({
                "evaluation_family": "wall_to_wall_deployment",
                "representation_or_product": {
                    "M0": "M0_raw", "M1": "M1_FTW", "M1b": "M1b_FTW_DW",
                    "M2": "M2_parcel", "M2_QA": "M2_QA"}[name],
                "evaluation_mode": label,
                "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
                "iou": m["iou"], "accuracy": m["accuracy"],
                "balanced_accuracy": m["balanced_accuracy"],
                "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "tn": m["tn"],
                "agreement_fraction": m["agreement_fraction"],
                "predicted_rice_area_ha": pred_rice_ha,
                "reference_rice_area_ha": (ref_rice_total if label.startswith("A")
                                           else ref_rice_B) * PIXEL_AREA_HA,
                "area_difference_ha": ((pred_rice_A - (ref_rice_total if label.startswith("A")
                                                       else ref_rice_B)) * PIXEL_AREA_HA),
                "coverage_fraction": coverage,
                "valid_evaluation_pixels": n_region if label.startswith("A") else n_retained,
                "retained_pixels": n_retained,
                "excluded_area_ha": (n_region - n_retained) * PIXEL_AREA_HA,
            })

    cols = ["evaluation_family", "representation_or_product", "evaluation_mode",
            "precision", "recall", "f1", "iou", "accuracy", "balanced_accuracy",
            "tp", "fp", "fn", "tn", "agreement_fraction",
            "predicted_rice_area_ha", "reference_rice_area_ha", "area_difference_ha",
            "coverage_fraction", "valid_evaluation_pixels", "retained_pixels", "excluded_area_ha"]

    # ---- append the SAMPLED REPRESENTATION family (clearly separated) ----
    sampled = cfg.get("sampled_representation", {})
    for key in ("temporal", "alphaearth"):
        if key not in sampled:
            continue
        rows.append({
            "evaluation_family": "sampled_representation",
            "representation_or_product": key,
            "evaluation_mode": "balanced_6000_6000_weak_reference",
            "precision": float("nan"), "recall": float("nan"), "f1": float(sampled[key]["f1"]),
            "iou": float("nan"), "accuracy": float("nan"), "balanced_accuracy": float("nan"),
            "tp": float("nan"), "fp": float("nan"), "fn": float("nan"), "tn": float("nan"),
            "agreement_fraction": float("nan"),
            "predicted_rice_area_ha": float("nan"),
            "reference_rice_area_ha": float("nan"),
            "area_difference_ha": float("nan"),
            "coverage_fraction": float("nan"),
            "valid_evaluation_pixels": float("nan"),
            "retained_pixels": float("nan"),
            "excluded_area_ha": float("nan"),
        })

    df = pd.DataFrame(rows)[cols]
    df.to_csv(out_tables / "wall_to_wall_weak_reference_metrics.csv", index=False)
    df = pd.read_csv(out_tables / "wall_to_wall_weak_reference_metrics.csv")
    print("Wrote", out_tables / "wall_to_wall_weak_reference_metrics.csv")

    # ---- spatial block metrics (paired, full-grid, identical blocks) ----
    block_rows = compute_block_metrics(evals, ref_bin, common_region, nb, min_px)
    bdf = pd.DataFrame(block_rows)
    bdf.to_csv(out_tables / "wall_to_wall_block_metrics.csv", index=False)
    print("Wrote", out_tables / "wall_to_wall_block_metrics.csv")

    # ---- figures ----
    order = ["M0", "M1", "M1b", "M2", "M2_QA"]
    make_figures(df, bdf, order, ref_shape, out_fig)

    # ---- summary ----
    prod_label = {"M0": "M0_raw", "M1": "M1_FTW", "M1b": "M1b_FTW_DW",
                  "M2": "M2_parcel", "M2_QA": "M2_QA"}
    _rec = {(r.representation_or_product, r.evaluation_mode): r for r in df.itertuples()}
    def _g(name, mode, field):
        return getattr(_rec[(prod_label[name], mode)], field)
    summ_fields = ["f1", "precision", "recall", "coverage_fraction",
                   "predicted_rice_area_ha", "reference_rice_area_ha"]
    summary = {
        "region_pixels": n_region,
        "reference_rice_pixels_region": ref_rice_total,
        "reference_rice_area_ha_region": ref_rice_total * PIXEL_AREA_HA,
        "common_region_basis": cfg["common_region"]["basis"],
        "predicted_rice_area_invariant_across_modes": True,
        "metrics_modeB_conditional_retained": {
            n: {k: _g(n, "B_conditional_retained_coverage", k) for k in summ_fields}
            for n in order},
        "metrics_modeA_full_grid": {
            n: {k: _g(n, "A_full_grid_nonrice_interpretation", k) for k in summ_fields}
            for n in order},
        "sampled_representation": cfg.get("sampled_representation", {}),
    }
    (out_summary / "wall_to_wall_experiment_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))
    print("Done.")


if __name__ == "__main__":
    sys.exit(main())

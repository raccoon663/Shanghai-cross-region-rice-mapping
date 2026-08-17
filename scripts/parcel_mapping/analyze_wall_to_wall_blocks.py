#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paired spatial-robustness analysis for the wall-to-wall weak-reference experiment.

Reads results/tables/wall_to_wall_block_metrics.csv (produced by
evaluate_wall_to_wall_weak_reference.py) and writes a paired summary:

  results/tables/wall_to_wall_block_summary.csv

The block metrics are computed on the SAME 36 spatial blocks for every product
(full-grid interpretation, abstained -> non-rice), so comparisons are paired:
every product is evaluated over the identical block pixel set. This avoids the
earlier mistake of comparing medians computed over different block sets.

Reported per product (M1, M1b, M2, M2_QA):
  - median paired Delta F1 vs M0
  - number / percentage of blocks whose F1 improved (Delta > 0)
  - number / percentage of blocks whose F1 worsened (Delta < 0)
  - number of zero-reference-rice blocks (reported separately; F1 undefined there)

Usage:
  python scripts/parcel_mapping/analyze_wall_to_wall_blocks.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
BLOCK_CSV = REPO / "results" / "tables" / "wall_to_wall_block_metrics.csv"
SUMMARY_CSV = REPO / "results" / "tables" / "wall_to_wall_block_summary.csv"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--blocks", default=str(BLOCK_CSV))
    ap.add_argument("--out", default=str(SUMMARY_CSV))
    args = ap.parse_args(argv)

    bdf = pd.read_csv(args.blocks)
    products = ["M1", "M1b", "M2", "M2_QA"]

    n_blocks = len(bdf)
    n_zero_ref = int(bdf["zero_reference_rice"].sum())
    # blocks with a valid M0 F1 and a valid product F1 (paired, comparable)
    paired_rows = []
    for _, r in bdf.iterrows():
        m0 = r["M0_f1"]
        if np.isnan(m0):
            continue
        row = {"block_id": r["block_id"], "M0_f1": m0,
               "reference_rice_pixels": int(r["reference_rice_pixels"]),
               "zero_reference_rice": bool(r["zero_reference_rice"])}
        for p in products:
            row[f"{p}_f1"] = r[f"{p}_f1"]
            row[f"{p}_delta_f1_vs_M0"] = r[f"{p}_delta_f1_vs_M0"]
        paired_rows.append(row)
    pdata = pd.DataFrame(paired_rows)

    summary_rows = []
    for p in products:
        deltas = pdata[f"{p}_delta_f1_vs_M0"].dropna().to_numpy(dtype=float)
        n_comp = len(deltas)
        n_imp = int((deltas > 0).sum())
        n_wor = int((deltas < 0).sum())
        n_same = int((deltas == 0).sum())
        summary_rows.append({
            "product": p,
            "n_paired_blocks": n_comp,
            "median_paired_delta_f1_vs_M0": float(np.median(deltas)) if n_comp else float("nan"),
            "mean_paired_delta_f1_vs_M0": float(np.mean(deltas)) if n_comp else float("nan"),
            "n_blocks_improved": n_imp,
            "pct_blocks_improved": (n_imp / n_comp) if n_comp else float("nan"),
            "n_blocks_worsened": n_wor,
            "pct_blocks_worsened": (n_wor / n_comp) if n_comp else float("nan"),
            "n_blocks_unchanged": n_same,
        })
    sdf = pd.DataFrame(summary_rows)
    sdf.to_csv(args.out, index=False)
    print("Wrote", args.out)

    meta = {
        "n_total_blocks": n_blocks,
        "n_zero_reference_rice_blocks": n_zero_ref,
        "n_paired_comparable_blocks": len(pdata),
        "paired_design": "identical 36-block full-grid evaluation for all products",
        "products": {r["product"]: {
            "median_paired_delta_f1_vs_M0": r["median_paired_delta_f1_vs_M0"],
            "pct_improved": r["pct_blocks_improved"],
            "pct_worsened": r["pct_blocks_worsened"],
        } for _, r in sdf.iterrows()},
    }
    (Path(args.out).with_suffix("")).parent.joinpath(
        "wall_to_wall_block_summary.json").write_text(json.dumps(meta, indent=2, default=str))
    print("Wrote block summary JSON")
    return sdf


if __name__ == "__main__":
    main()

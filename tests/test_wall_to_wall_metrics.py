"""Tests for the wall-to-wall weak-reference consistency evaluation.

Pure-logic tests run without any raster data. Raster/data-dependent tests skip
gracefully when the frozen pipeline inputs are absent (a Git clone without the
git-ignored outputs/ and inputs/ directories).
"""
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
PARCEL_DIR = REPO / "scripts" / "parcel_mapping"


def _E():
    """Lazily import the evaluator module (needs rasterio/geopandas)."""
    import importlib.util
    import sys
    name = "evaluate_wall_to_wall_weak_reference"
    spec = importlib.util.spec_from_file_location(
        name, PARCEL_DIR / "evaluate_wall_to_wall_weak_reference.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Confusion + metric logic
# ---------------------------------------------------------------------------
def test_confusion_counts():
    E = _E()
    pred = np.array([1, 1, 0, 0, 1, 0])
    ref = np.array([1, 0, 0, 1, 1, 0])
    valid = np.ones(6, bool)
    c = E.confusion(pred, ref, valid)
    assert c == {"tp": 2, "fp": 1, "fn": 1, "tn": 2}


def test_metrics_perfect():
    E = _E()
    c = {"tp": 10, "fp": 0, "fn": 0, "tn": 10}
    m = E.metrics_from_conf(c)
    assert m["precision"] == 1.0 and m["recall"] == 1.0 and m["f1"] == 1.0
    assert m["iou"] == 1.0 and m["accuracy"] == 1.0


def test_metrics_no_positive_prediction():
    # all predicted non-rice, but reference has rice -> precision undefined, F1 = 0
    E = _E()
    c = {"tp": 0, "fp": 0, "fn": 5, "tn": 5}
    m = E.metrics_from_conf(c)
    assert np.isnan(m["precision"])
    assert abs(m["recall"]) < 1e-9
    assert m["f1"] == 0.0


def test_metrics_no_positive_reference():
    # reference all non-rice -> recall undefined, F1 = 0 (denominator > 0)
    E = _E()
    c = {"tp": 0, "fp": 3, "fn": 0, "tn": 7}
    m = E.metrics_from_conf(c)
    assert abs(m["precision"] - 0.0) < 1e-9
    assert np.isnan(m["recall"])
    assert m["f1"] == 0.0


def test_f1_iou_relationship():
    E = _E()
    c = {"tp": 3, "fp": 1, "fn": 1, "tn": 5}
    m = E.metrics_from_conf(c)
    # F1 = 2*TP/(2TP+FP+FN); IoU = TP/(TP+FP+FN)
    assert abs(m["f1"] - 2 * 3 / (2 * 3 + 1 + 1)) < 1e-12
    assert abs(m["iou"] - 3 / (3 + 1 + 1)) < 1e-12
    assert m["f1"] > m["iou"]


# ---------------------------------------------------------------------------
# Nodata / excluded-pixel semantics
# ---------------------------------------------------------------------------
def test_excluded_pixels_ignored():
    E = _E()
    pred = np.array([1, 1, 0, 1])
    ref = np.array([1, 0, 0, 1])
    valid = np.array([True, True, True, False])   # last pixel excluded
    c = E.confusion(pred, ref, valid)
    # only first 3 pixels count: tp=1 (p1,r1), fp=1 (p1,r0), fn=0, tn=1 (p0,r0)
    assert c == {"tp": 1, "fp": 1, "fn": 0, "tn": 1}


# ---------------------------------------------------------------------------
# Retained-coverage semantics + "not a same-population improvement"
# ---------------------------------------------------------------------------
def test_retained_coverage_is_conditional():
    E = _E()
    # 10 px region. Reference rice on positions 0-5 (six rice pixels).
    region = np.ones(10, bool)
    ref = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
    # M0 over-predicts rice (positions 0-8) but catches all reference rice.
    pred_m0 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0])
    m0_full = E.metrics_from_conf(E.confusion(pred_m0, ref, region))

    # M2 retains ONLY the two easy pixels (0,1), abstains on the rest.
    # Within the retained set it is perfectly classified.
    retained_m2 = np.array([True, True, False, False, False, False, False, False, False, False])
    pred_m2 = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    m2_retained = E.metrics_from_conf(E.confusion(pred_m2, ref, retained_m2))

    # Retained F1 of M2 looks perfect and beats M0 full F1 ...
    assert m2_retained["f1"] > m0_full["f1"]
    # ... but coverage is only 20% and the full-grid interpretation (abstained ->
    # non-rice) is actually WORSE than M0, because M2 misses reference rice it
    # abstained on.
    m2_full = E.metrics_from_conf(E.confusion(pred_m2, ref, region))
    assert m2_full["f1"] < m0_full["f1"]
    # Therefore M2 retained F1 must NOT be read as a same-population gain over M0.
    assert m2_retained["f1"] > m0_full["f1"] > m2_full["f1"]
    # And coverage is small, so the retained number is conditioned on a subset.
    assert retained_m2.mean() == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Full-grid non-rice interpretation of excluded pixels
# ---------------------------------------------------------------------------
def test_full_grid_excluded_is_nonrice():
    E = _E()
    # 4 px region. Product retains only px0 (rice, correct). px1,2,3 abstained.
    region = np.ones(4, bool)
    pred_common = np.array([1, 0, 0, 0])   # abstained -> non-rice everywhere
    ref = np.array([1, 0, 0, 1])           # px3 is reference rice but unmapped
    m = E.metrics_from_conf(E.confusion(pred_common, ref, region))
    # px0: TP, px1: TN, px2: TN, px3: FN (reference rice, predicted non-rice)
    assert m["tp"] == 1 and m["tn"] == 2 and m["fn"] == 1 and m["fp"] == 0
    assert m["recall"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Raster alignment failure
# ---------------------------------------------------------------------------
def test_alignment_raises_on_mismatch():
    E = _E()
    profiles = {
        "__ref__": {"crs": "EPSG:32651", "shape": [10, 10], "transform": [20, 0, 0, 0, -20, 0]},
        "M0": {"crs": "EPSG:32651", "shape": [10, 10], "transform": [20, 0, 0, 0, -20, 0]},
        "M1": {"crs": "EPSG:32651", "shape": [11, 10], "transform": [20, 0, 0, 0, -20, 0]},
    }
    with pytest.raises(RuntimeError):
        E.verify_alignment(profiles, [10, 10], [20, 0, 0, 0, -20, 0])


def test_alignment_passes_when_identical():
    E = _E()
    profiles = {
        "__ref__": {"crs": "EPSG:32651", "shape": [10, 10], "transform": [20, 0, 0, 0, -20, 0]},
        "M0": {"crs": "EPSG:32651", "shape": [10, 10], "transform": [20, 0, 0, 0, -20, 0]},
    }
    assert E.verify_alignment(profiles, [10, 10], [20, 0, 0, 0, -20, 0]) is True


# ---------------------------------------------------------------------------
# Predicted-rice-area invariance across evaluation modes (clipping to common grid)
# ---------------------------------------------------------------------------
def test_predicted_rice_area_invariant_across_modes():
    E = _E()
    # 10x10 grid. common_region excludes the last row.
    common = np.ones((10, 10), bool)
    common[9, :] = False
    # A parcel product whose raw raster has rice OUTSIDE the common grid (row 9).
    pred_bin = np.zeros((10, 10), bool)
    pred_bin[0, 0] = True          # inside common -> should count
    pred_bin[9, 0] = True          # outside common -> should be clipped out
    valid = np.ones((10, 10), bool)
    committed = np.ones((10, 10), bool)
    prod = E.build_eval("M2", pred_bin, valid, committed, common)

    # After clipping to the common grid, the outside-common rice is gone.
    assert int(prod.pred_common.sum()) == 1
    # Mode A (full common grid) and Mode B (retained within common) see the same
    # positive mask, so predicted rice counts are identical.
    cA = E.confusion(prod.pred_common, np.zeros((10, 10), bool), common)
    cB = E.confusion(prod.pred_common, np.zeros((10, 10), bool), prod.retained_common)
    assert (cA["tp"] + cA["fp"]) == (cB["tp"] + cB["fp"]) == 1


# ---------------------------------------------------------------------------
# Real-data guarded tests (skip without frozen inputs)
# ---------------------------------------------------------------------------
def _inputs_present():
    ref = REPO / "inputs" / "official_reference" / "shanghai_2022_rice_aligned_20m.tif"
    m0 = REPO / "outputs" / "final_chongming_parcel_product" / "M0_raw_probability.tif"
    return ref.is_file() and m0.is_file()


@pytest.mark.skipif(not _inputs_present(), reason="frozen pipeline inputs not present")
def test_real_predicted_rice_area_invariant():
    E = _E()
    # Run the full evaluator into a temp location by monkeypatching outputs.
    import yaml
    cfg = yaml.safe_load(
        (REPO / "configs" / "wall_to_wall_weak_reference.yaml").read_text(encoding="utf-8"))
    # load + audit + build evals, then assert invariance per product.
    ref_bin, ref_prof = E.load_reference(REPO / cfg["inputs"]["reference"])
    threshold = float(cfg["parameters"]["inherited_threshold"])
    p0, v0, _ = E.load_probability(REPO / cfg["inputs"]["M0"], threshold)
    p1, v1, _ = E.load_probability(REPO / cfg["inputs"]["M1"], threshold)
    p1b, v1b, _ = E.load_probability(REPO / cfg["inputs"]["M1b"], threshold)
    p2, v2, c2, _ = E.load_m2_parcel(REPO / cfg["inputs"]["M2"])
    p2qa, v2qa, c2qa, _ = E.rasterize_m2qa(
        REPO / cfg["inputs"]["M2_QA"],
        {"shape": ref_bin.shape, "transform": list(ref_prof["transform"])})
    common = v0.copy()
    evals = {
        "M0": E.build_eval("M0", p0, v0, v0, common),
        "M1": E.build_eval("M1", p1, v1, v1, common),
        "M1b": E.build_eval("M1b", p1b, v1b, v1b, common),
        "M2": E.build_eval("M2", p2, v2, c2, common),
        "M2_QA": E.build_eval("M2_QA", p2qa, v2qa, c2qa, common),
    }
    for name, prod in evals.items():
        a = int(prod.pred_common[common].sum())
        b = int(prod.pred_common[prod.retained_common].sum())
        assert a == b, f"predicted rice area not invariant for {name}: {a} vs {b}"


@pytest.mark.skipif(not _inputs_present(), reason="frozen pipeline inputs not present")
def test_real_metrics_csv_families_and_no_alphaearth_mislabel():
    """Run the evaluator and verify the published CSV structure."""
    import subprocess, sys
    out = subprocess.run(
        [sys.executable, str(PARCEL_DIR / "evaluate_wall_to_wall_weak_reference.py")],
        cwd=str(REPO), capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    csv_path = REPO / "results" / "tables" / "wall_to_wall_weak_reference_metrics.csv"
    df = __import__("pandas").read_csv(csv_path)
    # Both families present
    assert set(df["evaluation_family"].unique()) >= {"wall_to_wall_deployment", "sampled_representation"}
    # AlphaEarth must be its own representation row, never under an M1b product label
    sampled = df[df["evaluation_family"] == "sampled_representation"]
    assert "alphaearth" in set(sampled["representation_or_product"])
    w2w = df[df["evaluation_family"] == "wall_to_wall_deployment"]
    assert "M1b_FTW_DW" in set(w2w["representation_or_product"])
    # Predicted rice area must be identical across modes for every deployment product
    for prod in ["M0_raw", "M1_FTW", "M1b_FTW_DW", "M2_parcel", "M2_QA"]:
        sub = w2w[w2w["representation_or_product"] == prod]
        areas = sub["predicted_rice_area_ha"].dropna().unique()
        assert len(areas) == 1, f"area not invariant for {prod}: {areas}"

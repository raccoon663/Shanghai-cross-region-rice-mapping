from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/phase3"
AEFILE = ROOT / "outputs/alphaearth/alphaearth_samples_2022.csv"
TARGET = ROOT / "outputs/experiments/weak_label_budget_m4_v1/target_weak_samples.csv"
OODFILE = ROOT / "results/phase2/target_ood_predictions.csv"
REF = ROOT / "inputs/official_reference/shanghai_2022_rice_aligned_20m.tif"
TEMP_MODEL = ROOT / "outputs/experiments/rf_baseline_m2_v1/rf_s1_s2_fusion.joblib"
AE = [f"A{i:02d}" for i in range(64)]
BUDGETS = [20, 50, 100, 200, 500]
SEEDS = list(range(42, 72))


def ece(y, p, bins=10):
    y, p = np.asarray(y), np.asarray(p)
    total = 0.0
    for lo, hi in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.any(): total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def rf(seed):
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=2, max_features="sqrt",
                                  class_weight="balanced_subsample", random_state=seed, n_jobs=-1)


def distributed_sample(pool, budget, seed):
    """Balanced total budget, spread across blocks before reusing a block."""
    rng = np.random.default_rng(seed); parts = []
    for label in [0, 1]:
        candidates = pool[pool.class_id == label]
        blocks = candidates.spatial_block.unique().copy(); rng.shuffle(blocks)
        picked = []; need = budget // 2
        cycles = 0
        while len(picked) < need:
            for block in blocks:
                available = candidates[(candidates.spatial_block == block) & (~candidates.index.isin(picked))]
                if len(available): picked.append(rng.choice(available.index))
                if len(picked) == need: break
            cycles += 1
            if cycles > need + 2: raise RuntimeError("Insufficient samples")
        parts.append(pool.loc[picked])
    return pd.concat(parts).sample(frac=1, random_state=seed)


def clustered_sample(pool, budget, seed):
    """Balanced sample from the minimum randomly ordered blocks needed per class."""
    rng = np.random.default_rng(seed); parts = []
    for label in [0, 1]:
        candidates = pool[pool.class_id == label]
        blocks = candidates.spatial_block.unique().copy(); rng.shuffle(blocks)
        chosen = []; need = budget // 2
        for block in blocks:
            idx = candidates[candidates.spatial_block == block].index.to_numpy(); rng.shuffle(idx)
            chosen.extend(idx[:need-len(chosen)])
            if len(chosen) >= need: break
        if len(chosen) < need: raise RuntimeError("Insufficient clustered samples")
        parts.append(pool.loc[chosen])
    return pd.concat(parts).sample(frac=1, random_state=seed)


def metrics(y, p):
    pred = p >= .5
    return {"f1": f1_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0), "error_rate": np.mean(pred != y),
            "ece": ece(y, p), "brier": brier_score_loss(y, p)}


def reference_flags(frame):
    with rasterio.open(REF) as ds: arr = ds.read(1).astype(bool)
    boundary = ndimage.binary_dilation(arr) ^ ndimage.binary_erosion(arr)
    distance = ndimage.distance_transform_edt(~boundary)
    labels, _ = ndimage.label(arr, structure=np.ones((3, 3)))
    sizes = np.bincount(labels.ravel())
    rows, cols = frame.row.astype(int).to_numpy(), frame.col.astype(int).to_numpy()
    frame["boundary_distance_px"] = distance[rows, cols]
    frame["rice_patch_size_px"] = np.where(frame.class_id.eq(1), sizes[labels[rows, cols]], np.nan)
    frame["reference_subset"] = np.where(
        (frame.boundary_distance_px >= 2) & ((frame.class_id == 0) | (frame.rice_patch_size_px >= 25)),
        "interior_large_patch", "all_only")
    return frame


def ood_robustness(ood):
    scores = ["nn_standardized", "nn_cosine", "knn10_cosine", "mahalanobis", "domain_probability"]
    rows = []
    rng = np.random.default_rng(2026); blocks = ood.spatial_block.unique()
    for score in scores:
        block = ood.groupby("spatial_block").agg(score=(score, "mean"), error=("error", "mean"))
        q = pd.qcut(ood[score], 10, labels=False, duplicates="drop")
        dec = ood.assign(decile=q).groupby("decile").error.mean()
        boot = []
        for _ in range(500):
            chosen = rng.choice(blocks, len(blocks), replace=True)
            part = pd.concat([ood[ood.spatial_block == b] for b in chosen], ignore_index=True)
            if part.error.nunique() == 2: boot.append(roc_auc_score(part.error, part[score]))
        rows.append({"row_type":"method", "method_1":score, "method_2":"", "error_auroc":roc_auc_score(ood.error, ood[score]),
                     "error_auroc_ci_low":np.quantile(boot,.025), "error_auroc_ci_high":np.quantile(boot,.975),
                     "error_auprc":average_precision_score(ood.error, ood[score]),
                     "block_spearman":spearmanr(block.score, block.error).statistic,
                     "decile_spearman":spearmanr(dec.index, dec.values).statistic,
                     "nearest_decile_error":dec.iloc[0], "farthest_decile_error":dec.iloc[-1], "rank_spearman":np.nan})
    for i, a in enumerate(scores):
        for b in scores[i+1:]:
            rows.append({"row_type":"pairwise_rank", "method_1":a, "method_2":b,
                         "rank_spearman":spearmanr(ood[a], ood[b]).statistic})
    pd.DataFrame(rows).to_csv(OUT / "ood_metric_robustness.csv", index=False)


def frozen_risk(ood):
    rows=[]; coverages=[.9,.8,.7,.6,.5]; unique=np.array(sorted(ood.spatial_block.unique()))
    for seed in SEEDS:
        rng=np.random.default_rng(seed); blocks=unique.copy(); rng.shuffle(blocks)
        tune_blocks=set(blocks[:len(blocks)//2]); tune=ood[ood.spatial_block.isin(tune_blocks)].copy(); test=ood[~ood.spatial_block.isin(tune_blocks)].copy()
        for part in [tune,test]: part["confidence"] = 1-np.abs(2*part.probability-1)
        for score in ["confidence", "knn10_cosine", "combined"]:
            if score == "combined":
                mu_c,sd_c=tune.confidence.mean(),tune.confidence.std(); mu_o,sd_o=tune.knn10_cosine.mean(),tune.knn10_cosine.std()
                tune[score]=(tune.confidence-mu_c)/sd_c+(tune.knn10_cosine-mu_o)/sd_o
                test[score]=(test.confidence-mu_c)/sd_c+(test.knn10_cosine-mu_o)/sd_o
            for cov in coverages:
                threshold=tune[score].quantile(cov); kept=test[test[score] <= threshold]
                row={"seed":seed,"risk_score":score,"target_coverage":cov,"threshold":threshold,
                     "actual_coverage":len(kept)/len(test),"rejected_fraction":1-len(kept)/len(test),"n":len(kept)}
                row.update(metrics(kept.class_id, kept.probability)); rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "frozen_risk_coverage.csv", index=False)


def adaptation(frame, ood):
    temporal=[c for c in frame if c.startswith("s1_") or c.startswith("s2_")]
    pool=frame[frame.split=="target_pool"].copy(); val=frame[frame.split=="target_val"].copy()
    val=val.merge(ood[["sample_id","knn10_cosine"]],on="sample_id",validate="one_to_one")
    val_ood=val[["sample_id","knn10_cosine"]]
    q20,q80=val_ood.knn10_cosine.quantile([.2,.8]); bands=np.where(val_ood.knn10_cosine<=q20,"nearest20",np.where(val_ood.knn10_cosine>=q80,"farthest20","middle60"))
    robust=[]; block_rows=[]; complement=[]; sensitivity=[]
    for regime,sampler in [("distributed",distributed_sample),("clustered",clustered_sample)]:
      for seed in SEEDS:
        preds={}
        for budget in BUDGETS:
          train=sampler(pool,budget,seed); med=train[temporal].replace(-9999,np.nan).median()
          matrices={"alphaearth":(train[AE],val[AE]),"temporal":(train[temporal].replace(-9999,np.nan).fillna(med),val[temporal].replace(-9999,np.nan).fillna(med))}
          for rep,(xtr,xv) in matrices.items():
            model=rf(seed); model.fit(xtr,train.class_id); p=model.predict_proba(xv)[:,1]; preds[rep]=p
            row={"regime":regime,"seed":seed,"budget_total":budget,"representation":rep,"n_train":len(train),"n_train_blocks":train.spatial_block.nunique()}; row.update(metrics(val.class_id,p)); robust.append(row)
            for band in ["nearest20","middle60","farthest20"]:
                m=bands==band; sensitivity.append({"analysis":"ood_band_adaptation","subset":band,"regime":regime,"seed":seed,"budget_total":budget,"representation":rep,"n":m.sum(),**metrics(val.class_id.to_numpy()[m],p[m])})
            hq=val.reference_subset.eq("interior_large_patch").to_numpy(); sensitivity.append({"analysis":"reference_quality_adaptation","subset":"interior_large_patch","regime":regime,"seed":seed,"budget_total":budget,"representation":rep,"n":hq.sum(),**metrics(val.class_id.to_numpy()[hq],p[hq])})
            if budget in [50,500] and regime=="distributed":
              temp=val[["spatial_block","class_id","knn10_cosine"]].copy(); temp["p"]=p
              for b,g in temp.groupby("spatial_block"):
                block_rows.append({"seed":seed,"budget_total":budget,"representation":rep,"spatial_block":b,"n":len(g),"mean_ood":g.knn10_cosine.mean(),**metrics(g.class_id,g.p)})
          a=preds["alphaearth"]>=.5; t=preds["temporal"]>=.5; y=val.class_id.to_numpy()
          groups=np.select([(a==y)&(t==y),(a==y)&(t!=y),(a!=y)&(t==y)], ["both_correct","alphaearth_only_correct","temporal_only_correct"],default="both_wrong")
          for group in np.unique(groups): complement.append({"regime":regime,"seed":seed,"budget_total":budget,"error_group":group,"n":sum(groups==group),"fraction":np.mean(groups==group)})
    runs=pd.DataFrame(robust); runs.to_csv(OUT/"spatial_adaptation.csv",index=False)
    summary=(runs.groupby(["regime","representation","budget_total"]).agg(n_seeds=("seed","nunique"),f1_mean=("f1","mean"),f1_sd=("f1","std"),precision_mean=("precision","mean"),recall_mean=("recall","mean"),ece_mean=("ece","mean"),ece_sd=("ece","std")).reset_index())
    summary["f1_ci_low"]=summary.f1_mean-1.96*summary.f1_sd/np.sqrt(summary.n_seeds); summary["f1_ci_high"]=summary.f1_mean+1.96*summary.f1_sd/np.sqrt(summary.n_seeds)
    summary.to_csv(OUT/"label_efficiency_robust.csv",index=False)
    pd.DataFrame(block_rows).to_csv(OUT/"spatial_block_stability.csv",index=False)
    pd.DataFrame(complement).to_csv(OUT/"error_complementarity_robust.csv",index=False)
    return pd.DataFrame(sensitivity)


def zero_shot_sensitivity(frame, ood, sensitivity):
    val=frame[frame.split=="target_val"].copy(); ae=pd.read_csv(AEFILE); src=ae[(ae.region=="jiangxi")&(ae.split=="source_train")]
    model=rf(42); model.fit(src[AE],src.class_id); pa=model.predict_proba(val[AE])[:,1]
    saved=joblib.load(TEMP_MODEL); cols=saved["feature_columns"]; xv=val[cols].replace(-9999,np.nan).fillna(saved["feature_medians"]); pt=saved["model"].predict_proba(xv)[:,1]
    merged=val[["sample_id","class_id","reference_subset"]].merge(ood[["sample_id","knn10_cosine"]],validate="one_to_one")
    rows=[]
    for subset,mask in [("all",np.ones(len(val),bool)),("interior_large_patch",val.reference_subset.eq("interior_large_patch").to_numpy())]:
      for rep,p in [("alphaearth",pa),("temporal",pt)]: rows.append({"analysis":"zero_shot","subset":subset,"regime":"source_only","seed":42,"budget_total":0,"representation":rep,"n":mask.sum(),**metrics(val.class_id.to_numpy()[mask],p[mask])})
      m=mask; err=(pa>=.5)!=val.class_id.to_numpy(); rows.append({"analysis":"ood_error","subset":subset,"regime":"source_only","seed":42,"budget_total":0,"representation":"alphaearth","n":m.sum(),"ood_error_auroc":roc_auc_score(err[m],merged.knn10_cosine.to_numpy()[m]),"ood_error_auprc":average_precision_score(err[m],merged.knn10_cosine.to_numpy()[m])})
    pd.concat([sensitivity,pd.DataFrame(rows)],ignore_index=True).to_csv(OUT/"reference_sensitivity.csv",index=False)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    explicit=pd.read_csv(TARGET); ae=pd.read_csv(AEFILE); ae=ae[ae.region=="shanghai"][["sample_id"]+AE]
    frame=reference_flags(explicit.merge(ae,on="sample_id",validate="one_to_one"))
    ood=pd.read_csv(OODFILE)
    ood_robustness(ood); frozen_risk(ood)
    sensitivity=adaptation(frame,ood); zero_shot_sensitivity(frame,ood,sensitivity)
    print("Phase III robustness tables complete:", OUT)


if __name__ == "__main__": main()

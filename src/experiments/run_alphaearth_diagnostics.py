from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "outputs/alphaearth/alphaearth_samples_2022.csv"
OUT = ROOT / "results/alphaearth_minimal"
AE = [f"A{i:02d}" for i in range(64)]


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1); value = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if mask.any(): value += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return value


def main() -> None:
    frame = pd.read_csv(DATA).dropna(subset=AE)
    train = frame[(frame.region == "jiangxi") & (frame.split == "source_train")]
    source_test = frame[(frame.region == "jiangxi") & (frame.split == "source_test")]
    target_pool = frame[(frame.region == "shanghai") & (frame.split == "target_pool")]
    target_val = frame[(frame.region == "shanghai") & (frame.split == "target_val")].copy()

    rice = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, max_features="sqrt",
                                  class_weight="balanced_subsample", random_state=42, n_jobs=-1)
    rice.fit(train[AE], train.class_id)
    rows = []
    for name, part in [("source_test", source_test), ("target_zero_shot", target_val)]:
        p = rice.predict_proba(part[AE])[:, 1]; y = part.class_id.to_numpy()
        rows.append({"diagnostic": "calibration", "split": name, "brier": brier_score_loss(y, p),
                     "ece_10bin": ece(y, p), "f1": f1_score(y, p >= .5)})
        if name == "target_zero_shot": target_val["probability"] = p; target_val["error"] = (p >= .5) != y

    domain = pd.concat([train.assign(domain=0), target_pool.assign(domain=1)], ignore_index=True)
    domain["domain_block"] = domain.region + "_" + domain.spatial_block.astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=.25, random_state=42)
    tr, te = next(splitter.split(domain, domain.domain, domain.domain_block))
    domain_model = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, max_features="sqrt",
                                          class_weight="balanced", random_state=42, n_jobs=-1)
    domain_model.fit(domain.iloc[tr][AE], domain.iloc[tr].domain)
    domain_p = domain_model.predict_proba(domain.iloc[te][AE])[:, 1]
    rows.append({"diagnostic": "domain_separability", "split": "held_spatial_blocks",
                 "auroc": roc_auc_score(domain.iloc[te].domain, domain_p)})

    nn = NearestNeighbors(n_neighbors=1, metric="cosine", n_jobs=-1).fit(train[AE])
    target_val["source_cosine_distance"] = nn.kneighbors(target_val[AE], return_distance=True)[0][:, 0]
    rho, pvalue = spearmanr(target_val.source_cosine_distance, target_val.error.astype(int))
    rows.append({"diagnostic": "distance_error_association", "split": "target_zero_shot",
                 "spearman_rho": rho, "p_value": pvalue})
    target_val["distance_quintile"] = pd.qcut(target_val.source_cosine_distance, 5, labels=False, duplicates="drop") + 1
    bins = (target_val.groupby("distance_quintile")
            .agg(n=("sample_id", "size"), mean_distance=("source_cosine_distance", "mean"),
                 error_rate=("error", "mean"), mean_probability=("probability", "mean"))
            .reset_index())
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "domain_uncertainty_diagnostics.csv", index=False)
    bins.to_csv(OUT / "distance_error_quintiles.csv", index=False)
    target_val[["sample_id", "class_id", "probability", "error", "source_cosine_distance", "distance_quintile"]].to_csv(
        OUT / "target_zero_shot_diagnostics.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(bins.to_string(index=False))


if __name__ == "__main__":
    main()

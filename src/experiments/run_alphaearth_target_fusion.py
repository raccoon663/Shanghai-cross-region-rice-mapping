from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

from .run_alphaearth_benchmark import sample_total_budget


ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "outputs/experiments/weak_label_budget_m4_v1/target_weak_samples.csv"
EMBEDDINGS = ROOT / "outputs/alphaearth/alphaearth_samples_2022.csv"
OUT = ROOT / "results/alphaearth_minimal"
AE = [f"A{i:02d}" for i in range(64)]


def main() -> None:
    explicit = pd.read_csv(TARGET)
    embeddings = pd.read_csv(EMBEDDINGS)
    embeddings = embeddings[embeddings.region == "shanghai"][["sample_id"] + AE]
    frame = explicit.merge(embeddings, on="sample_id", validate="one_to_one")
    temporal = [c for c in explicit.columns if c.startswith("s1_") or c.startswith("s2_")]
    features = temporal + AE
    frame[temporal] = frame[temporal].replace(-9999, np.nan)
    pool, val = frame[frame.split == "target_pool"].copy(), frame[frame.split == "target_val"].copy()
    rows = []
    for seed in [42, 43, 44]:
        for budget in [20, 50, 100, 200, 500]:
            adaptation = sample_total_budget(pool, budget, seed).copy()
            medians = adaptation[features].median()
            xtrain = adaptation[features].fillna(medians)
            xval = val[features].fillna(medians)
            model = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, max_features="sqrt",
                                           class_weight="balanced_subsample", random_state=seed, n_jobs=-1)
            model.fit(xtrain, adaptation.class_id)
            pred = model.predict(xval)
            rows.append({"representation": "alphaearth_plus_temporal", "strategy": "target_only",
                         "budget_total": budget, "seed": seed,
                         "precision": precision_score(val.class_id, pred, zero_division=0),
                         "recall": recall_score(val.class_id, pred, zero_division=0),
                         "f1": f1_score(val.class_id, pred, zero_division=0)})
    runs = pd.DataFrame(rows)
    summary = (runs.groupby(["representation", "strategy", "budget_total"])
               .agg(f1_mean=("f1", "mean"), f1_std=("f1", "std"),
                    precision_mean=("precision", "mean"), recall_mean=("recall", "mean"))
               .reset_index())
    summary["f1_ci95"] = 1.96 * summary.f1_std / np.sqrt(3)
    OUT.mkdir(parents=True, exist_ok=True)
    runs.to_csv(OUT / "target_fusion_runs.csv", index=False)
    summary.to_csv(OUT / "target_fusion_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

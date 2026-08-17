from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .run_alphaearth_benchmark import sample_total_budget


ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "outputs/experiments/weak_label_budget_m4_v1/target_weak_samples.csv"
EMBEDDINGS = ROOT / "outputs/alphaearth/alphaearth_samples_2022.csv"
OOD = ROOT / "results/phase2/target_ood_predictions.csv"
OUT = ROOT / "results/phase2"
AE = [f"A{i:02d}" for i in range(64)]
BUDGETS = [20, 50, 100, 200, 500]
SEEDS = [42, 43, 44]


def rf(seed, **kwargs):
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=kwargs.pop("min_samples_leaf", 2),
                                  max_features="sqrt", class_weight="balanced_subsample",
                                  random_state=seed, n_jobs=-1, **kwargs)


def score_rows(name, budget, seed, y_train, pred_train, y_val, pred_val):
    return {"representation": name, "budget_total": budget, "seed": seed,
            "train_f1": f1_score(y_train, pred_train), "val_f1": f1_score(y_val, pred_val),
            "val_precision": precision_score(y_val, pred_val, zero_division=0),
            "val_recall": recall_score(y_val, pred_val, zero_division=0),
            "generalization_gap": f1_score(y_train, pred_train) - f1_score(y_val, pred_val)}


def main() -> None:
    explicit = pd.read_csv(TARGET)
    embeddings = pd.read_csv(EMBEDDINGS)
    embeddings = embeddings[embeddings.region == "shanghai"][["sample_id"] + AE]
    frame = explicit.merge(embeddings, on="sample_id", validate="one_to_one")
    temporal = [c for c in explicit.columns if c.startswith("s1_") or c.startswith("s2_")]
    frame[temporal] = frame[temporal].replace(-9999, np.nan)
    pool, val = frame[frame.split == "target_pool"].copy(), frame[frame.split == "target_val"].copy()
    rows = []
    predictions_500 = {}
    for seed in SEEDS:
        for budget in BUDGETS:
            train = sample_total_budget(pool, budget, seed).copy()
            temporal_medians = train[temporal].median()
            xt_train = train[temporal].fillna(temporal_medians).to_numpy()
            xt_val = val[temporal].fillna(temporal_medians).to_numpy()
            xa_train, xa_val = train[AE].to_numpy(), val[AE].to_numpy()
            y_train, y_val = train.class_id.to_numpy(), val.class_id.to_numpy()
            concat_train, concat_val = np.c_[xa_train, xt_train], np.c_[xa_val, xt_val]

            models = []
            models.append(("alphaearth_rf", rf(seed), xa_train, xa_val))
            models.append(("temporal_rf", rf(seed), xt_train, xt_val))
            models.append(("raw_concat_rf", rf(seed), concat_train, concat_val))
            models.append(("regularized_concat_rf", rf(seed, min_samples_leaf=max(2, budget // 50), max_depth=10), concat_train, concat_val))
            models.append(("standardized_concat_logistic_l2", make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=3000, class_weight="balanced", random_state=seed)), concat_train, concat_val))

            components = max(2, min(16, budget - 2, len(temporal), len(AE)))
            st = StandardScaler().fit(xt_train); pca_t = PCA(n_components=components, random_state=seed).fit(st.transform(xt_train))
            pcat_train, pcat_val = pca_t.transform(st.transform(xt_train)), pca_t.transform(st.transform(xt_val))
            models.append(("pca_temporal_plus_ae_rf", rf(seed), np.c_[xa_train, pcat_train], np.c_[xa_val, pcat_val]))
            sa = StandardScaler().fit(xa_train); pca_a = PCA(n_components=components, random_state=seed).fit(sa.transform(xa_train))
            pcaa_train, pcaa_val = pca_a.transform(sa.transform(xa_train)), pca_a.transform(sa.transform(xa_val))
            models.append(("pca_ae_plus_temporal_rf", rf(seed), np.c_[pcaa_train, xt_train], np.c_[pcaa_val, xt_val]))

            for name, model, x_train, x_val in models:
                model.fit(x_train, y_train); pred_train = model.predict(x_train); pred_val = model.predict(x_val)
                rows.append(score_rows(name, budget, seed, y_train, pred_train, y_val, pred_val))
                if seed == 42 and budget == 500 and name in {"alphaearth_rf", "temporal_rf", "raw_concat_rf"}:
                    predictions_500[name] = pred_val

    runs = pd.DataFrame(rows)
    summary = (runs.groupby(["representation", "budget_total"])
               .agg(train_f1_mean=("train_f1", "mean"), val_f1_mean=("val_f1", "mean"), val_f1_std=("val_f1", "std"),
                    precision_mean=("val_precision", "mean"), recall_mean=("val_recall", "mean"),
                    generalization_gap_mean=("generalization_gap", "mean"))
               .reset_index())
    summary["val_f1_ci95"] = 1.96 * summary.val_f1_std / np.sqrt(len(SEEDS))
    OUT.mkdir(parents=True, exist_ok=True)
    runs.to_csv(OUT / "fusion_diagnostics.csv", index=False)
    summary.to_csv(OUT / "fusion_diagnostics_summary.csv", index=False)

    ood = pd.read_csv(OOD, usecols=["sample_id", "knn10_cosine"])
    overlap = val[["sample_id", "x", "y", "class_id", "spatial_block"]].merge(ood, on="sample_id", validate="one_to_one")
    ae_correct = predictions_500["alphaearth_rf"] == val.class_id.to_numpy()
    temporal_correct = predictions_500["temporal_rf"] == val.class_id.to_numpy()
    overlap["error_group"] = np.select(
        [ae_correct & temporal_correct, ae_correct & ~temporal_correct, ~ae_correct & temporal_correct],
        ["both_correct", "alphaearth_only_correct", "temporal_only_correct"], default="both_wrong")
    overlap["concat_correct"] = predictions_500["raw_concat_rf"] == val.class_id.to_numpy()
    overlap.to_csv(OUT / "error_overlap_samples.csv", index=False)
    overlap_summary = (overlap.groupby("error_group")
                       .agg(n=("sample_id", "size"), mean_ood=("knn10_cosine", "mean"), concat_accuracy=("concat_correct", "mean"))
                       .reset_index())
    overlap_summary["fraction"] = overlap_summary.n / len(overlap)
    overlap_summary.to_csv(OUT / "error_overlap_summary.csv", index=False)

    chosen = ["alphaearth_rf", "temporal_rf", "raw_concat_rf", "pca_temporal_plus_ae_rf", "standardized_concat_logistic_l2"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    for name in chosen:
        part = summary[summary.representation == name]
        axes[0].plot(part.budget_total, part.val_f1_mean, marker="o", label=name)
        axes[1].plot(part.budget_total, part.generalization_gap_mean, marker="o", label=name)
    axes[0].set(xlabel="Target weak labels (total)", ylabel="Target-val F1", title="Controlled fusion variants")
    axes[1].set(xlabel="Target weak labels (total)", ylabel="Train F1 - target-val F1", title="Adaptation overfitting gap")
    for ax in axes: ax.grid(alpha=.25)
    axes[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(OUT / "fig_fusion_diagnostics.png", dpi=200)

    colors = {"both_correct": "#2a9d8f", "alphaearth_only_correct": "#457b9d", "temporal_only_correct": "#f4a261", "both_wrong": "#d62828"}
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for group, part in overlap.groupby("error_group"):
        ax.scatter(part.x, part.y, s=8, alpha=.65, label=f"{group} (n={len(part)})", color=colors[group])
    ax.set_aspect("equal"); ax.set(xlabel="Easting (m)", ylabel="Northing (m)", title="Shanghai error complementarity at 500 weak labels")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(OUT / "fig_error_overlap_map.png", dpi=200)

    print(summary.to_string(index=False))
    print(overlap_summary.to_string(index=False))


if __name__ == "__main__":
    main()

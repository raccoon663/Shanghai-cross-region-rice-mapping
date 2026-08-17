from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .run_alphaearth_benchmark import sample_total_budget


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "outputs/alphaearth/alphaearth_samples_2022.csv"
OUT = ROOT / "results/phase2"
AE = [f"A{i:02d}" for i in range(64)]
SEEDS = [42, 43, 44]


def ece(y, p, bins=10):
    y, p = np.asarray(y), np.asarray(p); edges = np.linspace(0, 1, bins + 1); value = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if mask.any(): value += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(value)


def cluster_ci(frame, value="error", reps=1000, seed=42):
    rng = np.random.default_rng(seed); blocks = frame.spatial_block.unique(); estimates = []
    for _ in range(reps):
        chosen = rng.choice(blocks, len(blocks), replace=True)
        parts = [frame[frame.spatial_block == block] for block in chosen]
        estimates.append(pd.concat(parts, ignore_index=True)[value].mean())
    return np.quantile(estimates, [.025, .975])


def matched_by_class(frame, counts, seed):
    return pd.concat([frame[frame.class_id == label].sample(n=n, random_state=seed + label)
                      for label, n in counts.items()], ignore_index=True)


def main() -> None:
    frame = pd.read_csv(DATA).dropna(subset=AE)
    source_train = frame[(frame.region == "jiangxi") & (frame.split == "source_train")].copy()
    source_test = frame[(frame.region == "jiangxi") & (frame.split == "source_test")].copy()
    target_pool = frame[(frame.region == "shanghai") & (frame.split == "target_pool")].copy()
    target_val = frame[(frame.region == "shanghai") & (frame.split == "target_val")].copy()
    OUT.mkdir(parents=True, exist_ok=True)

    # Domain audit: same year/schema, complete vectors, class-matched train/test, representations only.
    audit_rows = []
    train_counts = source_train.class_id.value_counts().to_dict()
    test_counts = source_test.class_id.value_counts().to_dict()
    for seed in SEEDS:
        target_train = matched_by_class(target_pool, train_counts, seed)
        target_test = matched_by_class(target_val, test_counts, seed + 100)
        domain_train = pd.concat([source_train.assign(domain=0), target_train.assign(domain=1)], ignore_index=True)
        domain_test = pd.concat([source_test.assign(domain=0), target_test.assign(domain=1)], ignore_index=True)
        for model_name, model in [
            ("rf", RandomForestClassifier(n_estimators=300, min_samples_leaf=2, max_features="sqrt", class_weight="balanced", random_state=seed, n_jobs=-1)),
            ("logistic", make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=3000, class_weight="balanced", random_state=seed))),
        ]:
            model.fit(domain_train[AE], domain_train.domain)
            probability = model.predict_proba(domain_test[AE])[:, 1]
            audit_rows.append({"model": model_name, "seed": seed, "test": "true_domain", "auroc": roc_auc_score(domain_test.domain, probability)})
            permuted = domain_train.domain.sample(frac=1, random_state=seed + 900).to_numpy()
            model.fit(domain_train[AE], permuted)
            probability = model.predict_proba(domain_test[AE])[:, 1]
            audit_rows.append({"model": model_name, "seed": seed, "test": "permuted_train_labels", "auroc": roc_auc_score(domain_test.domain, probability)})
    pd.DataFrame(audit_rows).to_csv(OUT / "domain_classifier_audit.csv", index=False)

    rice = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, max_features="sqrt",
                                  class_weight="balanced_subsample", random_state=42, n_jobs=-1)
    rice.fit(source_train[AE], source_train.class_id)
    target_val["probability"] = rice.predict_proba(target_val[AE])[:, 1]
    target_val["prediction"] = (target_val.probability >= .5).astype(int)
    target_val["error"] = target_val.prediction.ne(target_val.class_id).astype(int)

    scaler = StandardScaler().fit(source_train[AE])
    xs = scaler.transform(source_train[AE]); xt = scaler.transform(target_val[AE])
    target_val["nn_standardized"] = NearestNeighbors(n_neighbors=1).fit(xs).kneighbors(xt)[0][:, 0]
    cosine_nn = NearestNeighbors(n_neighbors=10, metric="cosine").fit(source_train[AE])
    cosine_dist = cosine_nn.kneighbors(target_val[AE])[0]
    target_val["nn_cosine"] = cosine_dist[:, 0]
    target_val["knn10_cosine"] = cosine_dist.mean(axis=1)
    covariance = LedoitWolf().fit(xs)
    target_val["mahalanobis"] = np.sqrt(covariance.mahalanobis(xt))
    target_domain = matched_by_class(target_pool, train_counts, 42)
    domain_train = pd.concat([source_train.assign(domain=0), target_domain.assign(domain=1)], ignore_index=True)
    domain_rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1)
    domain_rf.fit(domain_train[AE], domain_train.domain)
    target_val["domain_probability"] = domain_rf.predict_proba(target_val[AE])[:, 1]

    score_columns = ["nn_standardized", "nn_cosine", "knn10_cosine", "mahalanobis", "domain_probability"]
    ood_rows, decile_rows = [], []
    for score in score_columns:
        rho, pvalue = spearmanr(target_val[score], target_val.error)
        block = target_val.groupby("spatial_block").agg(score=(score, "mean"), error=("error", "mean"))
        block_rho, block_p = spearmanr(block.score, block.error)
        ood_rows.append({"representation": "alphaearth", "score": score, "error_auroc": roc_auc_score(target_val.error, target_val[score]),
                         "error_auprc": average_precision_score(target_val.error, target_val[score]), "spearman_rho": rho,
                         "spearman_p": pvalue, "block_spearman_rho": block_rho, "block_spearman_p": block_p})
        target_val["decile"] = pd.qcut(target_val[score], 10, labels=False, duplicates="drop") + 1
        for decile, group in target_val.groupby("decile"):
            low, high = cluster_ci(group)
            decile_rows.append({"representation": "alphaearth", "score": score, "decile": int(decile), "n": len(group),
                                "score_mean": group[score].mean(), "error_rate": group.error.mean(), "ci95_low": low, "ci95_high": high})
    ood = pd.DataFrame(ood_rows); deciles = pd.DataFrame(decile_rows)
    ood.to_csv(OUT / "ood_analysis.csv", index=False); deciles.to_csv(OUT / "ood_error_deciles.csv", index=False)

    # Risk-coverage: lower distrust is retained first.
    target_val["confidence_distrust"] = 1 - np.abs(2 * target_val.probability - 1)
    target_val["ood_distrust"] = target_val.knn10_cosine.rank(pct=True)
    target_val["combined_distrust"] = (target_val.confidence_distrust.rank(pct=True) + target_val.ood_distrust) / 2
    risk_rows = []
    for ranking in ["confidence_distrust", "ood_distrust", "combined_distrust"]:
        ordered = target_val.sort_values(ranking)
        for coverage in [1, .95, .9, .8, .7, .6, .5]:
            kept = ordered.head(round(len(ordered) * coverage)); y, pred = kept.class_id, kept.prediction
            risk_rows.append({"ranking": ranking, "coverage": coverage, "rejected_fraction": 1-coverage, "n": len(kept),
                              "f1": f1_score(y, pred), "precision": precision_score(y, pred), "recall": recall_score(y, pred),
                              "error_rate": pred.ne(y).mean()})
    risk = pd.DataFrame(risk_rows); risk.to_csv(OUT / "risk_coverage.csv", index=False)

    # Calibration for source, zero-shot target and 500-label target adaptation.
    source_test["probability"] = rice.predict_proba(source_test[AE])[:, 1]
    adaptation = sample_total_budget(target_pool, 500, 42)
    adapted = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, max_features="sqrt", class_weight="balanced_subsample", random_state=42, n_jobs=-1)
    adapted.fit(adaptation[AE], adaptation.class_id)
    target_val["adapted_probability"] = adapted.predict_proba(target_val[AE])[:, 1]
    calibration_rows = []
    for model_name, split_name, part, pcol in [
        ("source_only", "source_test", source_test.assign(distance_band="all"), "probability"),
        ("source_only", "target_zero_shot", target_val, "probability"),
        ("target_only_500", "target_val", target_val, "adapted_probability"),
    ]:
        if split_name == "source_test": bands = [("all", part)]
        else:
            q20, q80 = target_val.knn10_cosine.quantile([.2, .8])
            bands = [("nearest_20", part[part.knn10_cosine <= q20]), ("middle_60", part[(part.knn10_cosine > q20) & (part.knn10_cosine < q80)]), ("farthest_20", part[part.knn10_cosine >= q80]), ("all", part)]
        for band, group in bands:
            y, p = group.class_id.to_numpy(), group[pcol].to_numpy(); pred = p >= .5; confidence = np.maximum(p, 1-p)
            calibration_rows.append({"representation": "alphaearth", "model": model_name, "split": split_name, "distance_band": band,
                                     "n": len(group), "f1": f1_score(y, pred), "brier": brier_score_loss(y, p), "ece_10bin": ece(y, p),
                                     "accuracy": (pred == y).mean(), "mean_confidence": confidence.mean(),
                                     "confidence_minus_accuracy": confidence.mean() - (pred == y).mean(),
                                     "mean_confidence_when_wrong": confidence[pred != y].mean() if (pred != y).any() else np.nan})
    calibration = pd.DataFrame(calibration_rows); calibration.to_csv(OUT / "calibration_metrics.csv", index=False)
    target_val.to_csv(OUT / "target_ood_predictions.csv", index=False)

    # Figures.
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for score in score_columns:
        part = deciles[deciles.score == score]
        ax.plot(part.decile, part.error_rate, marker="o", label=score)
    ax.set(xlabel="OOD score decile (low to high)", ylabel="Classification error rate", title="AlphaEarth zero-shot error rises with source-domain distance")
    ax.grid(alpha=.25); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT / "fig_ood_error_deciles.png", dpi=200)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for ranking, group in risk.groupby("ranking"):
        ax.plot(group.coverage, group.error_rate, marker="o", label=ranking)
    ax.set(xlabel="Coverage retained", ylabel="Error rate", title="OOD-aware selective risk under geographic transfer")
    ax.invert_xaxis(); ax.grid(alpha=.25); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT / "fig_risk_coverage.png", dpi=200)

    plot = calibration[(calibration.split != "source_test") & (calibration.distance_band != "all")]
    fig, ax = plt.subplots(figsize=(8, 5))
    for model_name, group in plot.groupby("model"):
        order = pd.Categorical(group.distance_band, ["nearest_20", "middle_60", "farthest_20"], ordered=True)
        group = group.assign(order=order).sort_values("order")
        ax.plot(group.distance_band, group.confidence_minus_accuracy, marker="o", label=model_name)
    ax.axhline(0, color="black", lw=1); ax.set(ylabel="Mean confidence - accuracy", title="Overconfidence increases with AlphaEarth OOD distance")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(OUT / "fig_calibration_by_domain_distance.png", dpi=200)

    print(pd.DataFrame(audit_rows).groupby(["model", "test"]).auroc.agg(["mean", "std"]).to_string())
    print(ood.to_string(index=False))
    print(calibration.to_string(index=False))


if __name__ == "__main__":
    main()

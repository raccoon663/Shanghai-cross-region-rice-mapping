from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
P1 = ROOT / "results/alphaearth_minimal"
P2 = ROOT / "results/phase2"


def reliability(y, p, bins=8):
    edges = np.linspace(0, 1, bins + 1); rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if mask.any(): rows.append((p[mask].mean(), y[mask].mean(), mask.sum()))
    return pd.DataFrame(rows, columns=["mean_probability", "observed_frequency", "n"])


def first_threshold(frame, representation, threshold):
    part = frame[(frame.representation == representation) & (frame.val_f1_mean >= threshold)].sort_values("budget_total")
    return int(part.iloc[0].budget_total) if len(part) else np.nan


def main() -> None:
    P2.mkdir(parents=True, exist_ok=True)
    fusion_runs = pd.read_csv(P2 / "fusion_diagnostics.csv")
    fusion = pd.read_csv(P2 / "fusion_diagnostics_summary.csv")
    ood = pd.read_csv(P2 / "ood_analysis.csv")
    calibration = pd.read_csv(P2 / "calibration_metrics.csv")
    audit = pd.read_csv(P2 / "domain_classifier_audit.csv")
    overlap = pd.read_csv(P2 / "error_overlap_summary.csv")
    risk = pd.read_csv(P2 / "risk_coverage.csv")
    predictions = pd.read_csv(P2 / "target_ood_predictions.csv")
    master_p1 = pd.read_csv(P1 / "master_representation_comparison.csv")

    selected = fusion_runs[fusion_runs.representation.isin(["temporal_rf", "alphaearth_rf", "raw_concat_rf"])].copy()
    selected["representation"] = selected.representation.map({"temporal_rf": "temporal", "alphaearth_rf": "alphaearth", "raw_concat_rf": "raw_concat"})
    selected.to_csv(P2 / "label_efficiency_repeated.csv", index=False)

    thresholds = []
    for representation in ["temporal_rf", "alphaearth_rf", "raw_concat_rf"]:
        for threshold in [.85, .87, .89]:
            thresholds.append({"representation": representation, "f1_threshold": threshold,
                               "minimum_observed_labels": first_threshold(fusion, representation, threshold)})
    pd.DataFrame(thresholds).to_csv(P2 / "labels_to_threshold.csv", index=False)

    best_ood = ood.sort_values("error_auroc", ascending=False).iloc[0]
    full = risk[risk.coverage == 1].iloc[0]
    combined50 = risk[(risk.ranking == "combined_distrust") & (risk.coverage == .5)].iloc[0]
    far = calibration[(calibration.model == "source_only") & (calibration.distance_band == "farthest_20")].iloc[0]
    near = calibration[(calibration.model == "source_only") & (calibration.distance_band == "nearest_20")].iloc[0]
    summary_rows = [
        {"finding": "phase1_representation", "metric": "alphaearth_zero_shot_f1", "value": master_p1.iloc[1].shanghai_zero_shot_f1},
        {"finding": "domain_audit", "metric": "rf_true_domain_auroc_mean", "value": audit[(audit.model == "rf") & (audit.test == "true_domain")].auroc.mean()},
        {"finding": "domain_audit", "metric": "rf_permuted_auroc_mean", "value": audit[(audit.model == "rf") & (audit.test == "permuted_train_labels")].auroc.mean()},
        {"finding": "ood_error", "metric": "best_error_auroc", "value": best_ood.error_auroc},
        {"finding": "ood_error", "metric": "best_block_spearman_rho", "value": best_ood.block_spearman_rho},
        {"finding": "calibration", "metric": "nearest20_error_rate", "value": 1-near.accuracy},
        {"finding": "calibration", "metric": "farthest20_error_rate", "value": 1-far.accuracy},
        {"finding": "calibration", "metric": "farthest20_confidence_minus_accuracy", "value": far.confidence_minus_accuracy},
        {"finding": "selective_prediction", "metric": "full_coverage_error_rate", "value": full.error_rate},
        {"finding": "selective_prediction", "metric": "combined_50coverage_error_rate", "value": combined50.error_rate},
        {"finding": "complementarity", "metric": "alphaearth_only_correct_fraction", "value": overlap[overlap.error_group == "alphaearth_only_correct"].fraction.iloc[0]},
        {"finding": "complementarity", "metric": "temporal_only_correct_fraction", "value": overlap[overlap.error_group == "temporal_only_correct"].fraction.iloc[0]},
    ]
    pd.DataFrame(summary_rows).to_csv(P2 / "phase2_master_table.csv", index=False)

    # Detailed label efficiency.
    fig, ax = plt.subplots(figsize=(8, 5.2))
    colors = {"temporal_rf": "#e76f51", "alphaearth_rf": "#2a9d8f", "raw_concat_rf": "#457b9d"}
    for representation in colors:
        part = fusion[fusion.representation == representation]
        ax.plot(part.budget_total, part.val_f1_mean, marker="o", label=representation, color=colors[representation])
        ax.fill_between(part.budget_total, part.val_f1_mean-part.val_f1_ci95, part.val_f1_mean+part.val_f1_ci95, alpha=.15, color=colors[representation])
    for threshold in [.85, .87, .89]: ax.axhline(threshold, color="gray", ls="--", lw=.8)
    ax.set(xlabel="Shanghai weak labels (total)", ylabel="F1 agreement with reference product", title="Label efficiency under frozen target evaluation")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(P2 / "fig_label_efficiency_detailed.png", dpi=200)

    # OOD error curve: best three legitimate source-distance measures.
    deciles = pd.read_csv(P2 / "ood_error_deciles.csv")
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for score in ["nn_cosine", "knn10_cosine", "nn_standardized"]:
        part = deciles[deciles.score == score]
        ax.plot(part.decile * 10, part.error_rate, marker="o", label=score)
        if score == "knn10_cosine": ax.fill_between(part.decile * 10, part.ci95_low, part.ci95_high, alpha=.15)
    ax.set(xlabel="OOD percentile bin", ylabel="Error rate", title="Transfer error versus source-domain distance")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(P2 / "fig_ood_error_curve.png", dpi=200)

    # Reliability curves by distance and model state.
    q20, q80 = predictions.knn10_cosine.quantile([.2, .8])
    bands = {"nearest_20": predictions.knn10_cosine <= q20,
             "middle_60": (predictions.knn10_cosine > q20) & (predictions.knn10_cosine < q80),
             "farthest_20": predictions.knn10_cosine >= q80}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
    for ax, pcol, title in [(axes[0], "probability", "Zero-shot source model"), (axes[1], "adapted_probability", "Target-only, 500 labels")]:
        for band, mask in bands.items():
            rel = reliability(predictions.loc[mask, "class_id"].to_numpy(), predictions.loc[mask, pcol].to_numpy())
            ax.plot(rel.mean_probability, rel.observed_frequency, marker="o", label=band)
        ax.plot([0, 1], [0, 1], "k--", lw=1); ax.set(xlabel="Mean predicted probability", ylabel="Observed rice frequency", title=title)
        ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.suptitle("Calibration stratified by AlphaEarth OOD distance"); fig.tight_layout(); fig.savefig(P2 / "fig_calibration_by_domain_distance.png", dpi=200)

    # Confidence distributions.
    predictions["correct"] = predictions.prediction.eq(predictions.class_id)
    confidence = np.maximum(predictions.probability, 1-predictions.probability)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.hist(confidence[predictions.correct], bins=20, alpha=.55, density=True, label="correct")
    ax.hist(confidence[~predictions.correct], bins=20, alpha=.55, density=True, label="incorrect")
    ax.set(xlabel="Model confidence", ylabel="Density", title="Zero-shot confidence for correct and incorrect Shanghai predictions")
    ax.legend(); fig.tight_layout(); fig.savefig(P2 / "fig_confidence_correct_incorrect.png", dpi=200)
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(pd.DataFrame(thresholds).to_string(index=False))


if __name__ == "__main__":
    main()

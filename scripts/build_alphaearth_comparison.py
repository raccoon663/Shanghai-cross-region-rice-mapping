from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
R0_SOURCE = ROOT / "outputs/experiments/rf_baseline_m2_v1/rf_metrics.csv"
R0_BUDGET = ROOT / "outputs/experiments/weak_label_budget_m4_v1/rf_budget_summary.csv"
R1 = ROOT / "results/alphaearth_minimal/alphaearth_summary.csv"
OUT = ROOT / "results/alphaearth_minimal"


def one(frame, **filters):
    mask = pd.Series(True, index=frame.index)
    for key, value in filters.items():
        mask &= frame[key].eq(value)
    return frame.loc[mask].iloc[0]


def main() -> None:
    r0s, r0b, r1 = pd.read_csv(R0_SOURCE), pd.read_csv(R0_BUDGET), pd.read_csv(R1)
    r0_source = one(r0s, experiment="s1_s2_fusion", split="source_test")
    r0_zero = one(r0b, method="source_only", budget_total=0)
    r0_adapt = one(r0b, method="target_only", budget_total=500)
    r1_source = one(r1, strategy="source_only", budget_total=0, split="source_test")
    r1_zero = one(r1, strategy="source_only", budget_total=0, split="target_zero_shot")
    r1_adapt = one(r1, strategy="target_only", budget_total=500, split="target_val")
    master = pd.DataFrame([
        {"representation": "R0 Sentinel-1/2 temporal fusion", "source_test_f1": r0_source.f1,
         "shanghai_zero_shot_f1": r0_zero.f1_mean, "transfer_drop": r0_source.f1-r0_zero.f1_mean,
         "adapted_500_total_f1": r0_adapt.f1_mean, "zero_shot_precision": r0_zero.precision_mean,
         "zero_shot_recall": r0_zero.recall_mean},
        {"representation": "R1 AlphaEarth 64D", "source_test_f1": r1_source.f1_mean,
         "shanghai_zero_shot_f1": r1_zero.f1_mean, "transfer_drop": r1_source.f1_mean-r1_zero.f1_mean,
         "adapted_500_total_f1": r1_adapt.f1_mean, "zero_shot_precision": r1_zero.precision_mean,
         "zero_shot_recall": r1_zero.recall_mean},
    ])
    OUT.mkdir(parents=True, exist_ok=True)
    master.to_csv(OUT / "master_representation_comparison.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    x = [0, 20, 50, 100, 200, 500]
    for strategy, style in [("joint", "-o"), ("target_only", "--s")]:
        old = [one(r0b, method="source_only", budget_total=0).f1_mean] + [one(r0b, method=strategy, budget_total=b).f1_mean for b in x[1:]]
        new = [one(r1, strategy="source_only", budget_total=0, split="target_zero_shot").f1_mean] + [one(r1, strategy=strategy, budget_total=b, split="target_val").f1_mean for b in x[1:]]
        axes[0].plot(x, old, style, label=f"R0 {strategy}")
        axes[1].plot(x, new, style, label=f"R1 {strategy}")
    for ax, title in zip(axes, ["R0 Sentinel-1/2 temporal fusion", "R1 AlphaEarth 64D"]):
        ax.set_title(title); ax.set_xlabel("Shanghai weak labels (total)"); ax.set_ylabel("Target-val F1")
        ax.set_ylim(0.64, 0.92); ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.suptitle("Label-efficient geographic adaptation under the same weak-label protocol")
    fig.tight_layout()
    fig.savefig(OUT / "fig_label_efficiency.png", dpi=200, bbox_inches="tight")
    print(master.to_string(index=False))


if __name__ == "__main__":
    main()

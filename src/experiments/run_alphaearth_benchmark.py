from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


ROOT = Path(__file__).resolve().parents[2]
AE = [f"A{i:02d}" for i in range(64)]


def fit_rf(x, y, seed: int, sample_weight=None, joint: bool = False, n_estimators: int = 500) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=n_estimators, min_samples_leaf=2, max_features="sqrt",
        class_weight=None if joint else "balanced_subsample", random_state=seed, n_jobs=-1,
    )
    model.fit(x, y, sample_weight=sample_weight)
    return model


def metrics(y: pd.Series, prediction: np.ndarray) -> dict:
    cm = confusion_matrix(y, prediction, labels=[0, 1])
    return {
        "n": len(y), "precision": precision_score(y, prediction, zero_division=0),
        "recall": recall_score(y, prediction, zero_division=0),
        "f1": f1_score(y, prediction, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y, prediction),
        "tn": cm[0, 0], "fp": cm[0, 1], "fn": cm[1, 0], "tp": cm[1, 1],
    }


def sample_total_budget(pool: pd.DataFrame, budget: int, seed: int) -> pd.DataFrame:
    """Match the preserved R0 sampler: balanced total budget, blocks first."""
    rng = np.random.default_rng(seed)
    chosen = []
    for label in (0, 1):
        need = budget // 2
        part = pool[pool.class_id == label].sample(frac=1, random_state=seed + label)
        diverse = part.drop_duplicates("spatial_block").index.to_numpy()
        if len(diverse) >= need:
            picked = rng.choice(diverse, need, replace=False)
        else:
            rest = np.setdiff1d(part.index.to_numpy(), diverse)
            picked = np.r_[diverse, rng.choice(rest, need - len(diverse), replace=False)]
        chosen.extend(picked)
    return pool.loc[np.asarray(chosen)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=Path, default=ROOT / "outputs/alphaearth/alphaearth_samples_2022.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results/alphaearth_minimal")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--budgets-total", type=int, nargs="+", default=[20, 50, 100, 200, 500])
    args = parser.parse_args()

    frame = pd.read_csv(args.embeddings)
    missing = sorted(set(AE) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing AlphaEarth columns: {missing}")
    frame = frame.dropna(subset=AE).copy()
    source_train = frame[(frame.region == "jiangxi") & (frame.split == "source_train")]
    source_val = frame[(frame.region == "jiangxi") & (frame.split == "source_val")]
    source_test = frame[(frame.region == "jiangxi") & (frame.split == "source_test")]
    target_pool = frame[(frame.region == "shanghai") & (frame.split == "target_pool")]
    target_val = frame[(frame.region == "shanghai") & (frame.split == "target_val")]
    if min(map(len, [source_train, source_val, source_test, target_pool, target_val])) == 0:
        raise ValueError("One or more frozen splits are empty")

    rows = []
    for seed in args.seeds:
        source_model = fit_rf(source_train[AE], source_train.class_id, seed)
        for split_name, split in [("source_val", source_val), ("source_test", source_test), ("target_zero_shot", target_val)]:
            pred = source_model.predict(split[AE])
            rows.append({"representation": "alphaearth", "strategy": "source_only", "budget_total": 0,
                         "seed": seed, "split": split_name, **metrics(split.class_id, pred)})
        for budget in args.budgets_total:
            adaptation = sample_total_budget(target_pool, budget, seed)
            target_model = fit_rf(adaptation[AE], adaptation.class_id, seed, n_estimators=300)
            joint = pd.concat([source_train, adaptation], ignore_index=True)
            weights = np.r_[np.ones(len(source_train)), np.full(len(adaptation), len(source_train) / len(adaptation))]
            joint_model = fit_rf(joint[AE], joint.class_id, seed, sample_weight=weights, joint=True, n_estimators=300)
            for strategy, model in [("target_only", target_model), ("joint", joint_model)]:
                pred = model.predict(target_val[AE])
                rows.append({"representation": "alphaearth", "strategy": strategy, "budget_total": budget,
                             "seed": seed, "split": "target_val", **metrics(target_val.class_id, pred)})

    runs = pd.DataFrame(rows)
    runs["budget_total"] = runs.get("budget_total", pd.Series(0, index=runs.index)).fillna(0).astype(int)
    summary = (runs.groupby(["representation", "strategy", "budget_total", "split"])
               .agg(f1_mean=("f1", "mean"), f1_std=("f1", "std"),
                    precision_mean=("precision", "mean"), recall_mean=("recall", "mean"), n_runs=("seed", "nunique"))
               .reset_index())
    summary["f1_ci95"] = 1.96 * summary.f1_std / np.sqrt(summary.n_runs)
    args.output.mkdir(parents=True, exist_ok=True)
    runs.to_csv(args.output / "alphaearth_runs.csv", index=False)
    summary.to_csv(args.output / "alphaearth_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

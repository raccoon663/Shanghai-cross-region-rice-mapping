from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import rasterio
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data.common import feature_groups, load_config, resolve_data_path, resolve_project_path
from src.data.inspect_dataset import inspect
from src.data.splits import build_splits


def expected_calibration_error(y: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = len(y)
    value = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (probability >= lower) & (probability < upper if upper < 1 else probability <= upper)
        if mask.any():
            value += mask.sum() / total * abs(y[mask].mean() - probability[mask].mean())
    return float(value)


def scores(y: np.ndarray, probability: np.ndarray) -> dict:
    prediction = (probability >= 0.5).astype("uint8")
    cm = confusion_matrix(y, prediction, labels=[0, 1])
    return {
        "n": int(len(y)),
        "precision": precision_score(y, prediction, zero_division=0),
        "recall": recall_score(y, prediction, zero_division=0),
        "f1": f1_score(y, prediction, zero_division=0),
        "iou": jaccard_score(y, prediction, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y, prediction),
        "auroc": roc_auc_score(y, probability),
        "auprc": average_precision_score(y, probability),
        "brier": brier_score_loss(y, probability),
        "ece_10bin": expected_calibration_error(y, probability),
        "prediction_coverage": 1.0,
        "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
    }


def predict_target(
    model: RandomForestClassifier,
    features: list[str],
    medians: np.ndarray,
    target_path: Path,
    output_dir: Path,
    experiment: str,
) -> None:
    with rasterio.open(target_path) as source:
        descriptions = list(source.descriptions)
        if len(descriptions) != len(set(descriptions)):
            raise ValueError("Target raster band descriptions are missing or duplicated")
        lookup = {name: index + 1 for index, name in enumerate(descriptions)}
        missing = sorted(set(features) - set(lookup))
        if missing:
            raise ValueError(f"Target raster is missing model features: {missing}")
        indexes = [lookup[name] for name in features]
        probability_profile = source.profile.copy()
        probability_profile.update(count=1, dtype="float32", nodata=-9999.0, compress="deflate")
        class_profile = source.profile.copy()
        class_profile.update(count=1, dtype="uint8", nodata=255, compress="deflate")
        probability_path = output_dir / f"shanghai_source_only_probability_{experiment}.tif"
        class_path = output_dir / f"shanghai_source_only_binary_{experiment}.tif"
        with rasterio.open(probability_path, "w", **probability_profile) as probability_dst, rasterio.open(
            class_path, "w", **class_profile
        ) as class_dst:
            for _, window in source.block_windows(1):
                cube = source.read(indexes, window=window, out_dtype="float32")
                shape = cube.shape[1:]
                x = cube.reshape(len(features), -1).T
                observed = np.isfinite(x) & (x != source.nodata)
                valid = observed.sum(axis=1) >= max(6, int(len(features) * 0.25))
                x = np.where(observed, x, medians)
                probability = np.full(len(x), -9999.0, dtype="float32")
                prediction = np.full(len(x), 255, dtype="uint8")
                if valid.any():
                    probability[valid] = model.predict_proba(x[valid])[:, 1].astype("float32")
                    prediction[valid] = (probability[valid] >= 0.5).astype("uint8")
                probability_dst.write(probability.reshape(shape), 1, window=window)
                class_dst.write(prediction.reshape(shape), 1, window=window)


def run(config_file: str | Path) -> Path:
    config, config_path = load_config(config_file)
    output_dir = resolve_project_path(config["project"]["output_dir"], config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    inspect(config_file)
    split_frame = build_splits(config_file)
    source_path = resolve_data_path(config["data"]["source_features"], config_path)
    target_path = resolve_data_path(config["data"]["target_features"], config_path)
    source = pd.read_csv(source_path)
    source = source.merge(split_frame[["sample_id", "spatial_block", "split"]], on="sample_id", validate="one_to_one")
    groups = feature_groups(list(source.columns))
    seed = int(config["project"]["seed"])
    rf_config = config["random_forest"]
    rows, block_rows = [], []
    for experiment, features in groups.items():
        train = source.split == "source_train"
        medians = source.loc[train, features].median().to_numpy(dtype="float32")
        x = source[features].to_numpy(dtype="float32")
        x = np.where(np.isfinite(x), x, medians)
        y = source.class_id.to_numpy(dtype="uint8")
        model = RandomForestClassifier(
            n_estimators=int(rf_config["n_estimators"]),
            min_samples_leaf=int(rf_config["min_samples_leaf"]),
            max_features=rf_config["max_features"],
            class_weight=rf_config["class_weight"],
            random_state=seed,
            n_jobs=int(rf_config["n_jobs"]),
        )
        model.fit(x[train], y[train])
        for split_name in ("source_val", "source_test"):
            mask = source.split == split_name
            probability = model.predict_proba(x[mask])[:, 1]
            rows.append({"experiment": experiment, "split": split_name, **scores(y[mask], probability)})
            indices = np.where(mask)[0]
            for block, block_indices in source.iloc[indices].groupby("spatial_block").groups.items():
                idx = np.asarray(list(block_indices), dtype=int)
                if len(np.unique(y[idx])) < 2:
                    continue
                block_rows.append({
                    "experiment": experiment,
                    "split": split_name,
                    "spatial_block": block,
                    **scores(y[idx], model.predict_proba(x[idx])[:, 1]),
                })
        joblib.dump(
            {
                "model": model,
                "feature_columns": features,
                "feature_medians": dict(zip(features, medians.astype(float))),
                "training_split": "source_train",
                "seed": seed,
            },
            output_dir / f"rf_{experiment}.joblib",
        )
        predict_target(
            model,
            features,
            medians,
            target_path,
            output_dir,
            experiment,
        )
    pd.DataFrame(rows).to_csv(output_dir / "rf_metrics.csv", index=False)
    pd.DataFrame(block_rows).to_csv(output_dir / "rf_block_metrics.csv", index=False)
    environment = {
        "python_packages": {"numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "joblib": joblib.__version__},
        "seed": seed,
        "git_commit": None,
        "note": "No target metrics: independent target_test is unavailable.",
    }
    (output_dir / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    shutil.copy2(config_path, output_dir / "config.yaml")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    output = run(args.config)
    print(pd.read_csv(output / "rf_metrics.csv").to_string(index=False))
    print(f"Outputs: {output}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import yaml
from sklearn.model_selection import GroupShuffleSplit

from src.data.common import feature_groups, load_config, resolve_data_path, resolve_project_path


def build(config_file: str | Path) -> Path:
    config_path = Path(config_file).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_config, data_config_path = load_config(config["data_config"])
    target_path = resolve_data_path(data_config["data"]["target_features"], data_config_path)
    reference_path = resolve_data_path(data_config["data"]["target_reference"], data_config_path)
    output_dir = resolve_project_path(config["output_dir"], config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260803)
    with rasterio.open(reference_path) as reference, rasterio.open(target_path) as target:
        labels = reference.read(1)
        names = list(target.descriptions)
        feature_groups(names)
        positions = []
        request = int(config["target_sample_per_class"] * 1.5)
        for label in (0, 1):
            rows, cols = np.where(labels == label)
            chosen = rng.choice(len(rows), size=min(request, len(rows)), replace=False)
            positions.extend(zip(rows[chosen], cols[chosen], [label] * len(chosen)))
        frame = pd.DataFrame(positions, columns=["row", "col", "class_id"])
        coords = [target.xy(int(row), int(col)) for row, col in zip(frame.row, frame.col)]
        values = np.asarray(list(target.sample(coords)), dtype="float32")
        observed = np.isfinite(values) & (values > -9990)
        valid = observed.sum(1) >= max(6, int(values.shape[1] * 0.25))
        frame, values = frame.loc[valid].reset_index(drop=True), values[valid]
        selected = []
        for label in (0, 1):
            idx = np.where(frame.class_id.to_numpy() == label)[0]
            requested = int(config["target_sample_per_class"])
            if len(idx) < requested:
                raise ValueError(f"Only {len(idx)} valid weak-label samples available for class {label}; requested {requested}")
            selected.extend(rng.choice(idx, size=requested, replace=False))
        selected = np.asarray(selected)
        frame, values = frame.iloc[selected].reset_index(drop=True), values[selected]
        xs, ys = target.xy(frame.row.to_numpy(), frame.col.to_numpy())
        frame["x"], frame["y"] = xs, ys
    block_size = int(config["target_block_pixels"])
    frame["spatial_block"] = (frame.row // block_size).astype(str) + "_" + (frame.col // block_size).astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=float(config["target_val_fraction"]), random_state=20260803)
    pool_idx, val_idx = next(splitter.split(frame, frame.class_id, frame.spatial_block))
    frame["split"] = "target_pool"; frame.loc[val_idx, "split"] = "target_val"
    frame["region"] = "shanghai"; frame["label_source"] = "official_product_weak_label"
    frame.insert(0, "sample_id", [f"SH_W_{index:05d}" for index in range(len(frame))])
    feature_frame = pd.DataFrame(values, columns=names)
    result = pd.concat([frame.reset_index(drop=True), feature_frame], axis=1)
    if result.groupby("spatial_block").split.nunique().max() != 1:
        raise AssertionError("Target spatial block leakage")
    path = output_dir / "target_weak_samples.csv"
    result.to_csv(path, index=False)
    result[["sample_id","region","label_source","class_id","row","col","x","y","spatial_block","split"]].to_csv(output_dir / "target_weak_splits.csv", index=False)
    return path


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",default="configs/weak_label_budget.yaml"); args=parser.parse_args()
    path=build(args.config); frame=pd.read_csv(path); print(frame.groupby(["split","class_id"]).size().unstack(fill_value=0)); print(path.resolve())


if __name__ == "__main__": main()

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .common import load_config, parse_gee_points, resolve_data_path, resolve_project_path


def spatial_blocks(frame: pd.DataFrame, degrees: float) -> pd.Series:
    bx = np.floor(frame.longitude / degrees).astype(int)
    by = np.floor(frame.latitude / degrees).astype(int)
    return bx.astype(str) + "_" + by.astype(str)


def _best_group_split(
    indices: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    local_labels = labels[indices]
    local_groups = groups[indices]
    global_rate = local_labels.mean()
    best = None
    for offset in range(100):
        splitter = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed + offset)
        left_local, right_local = next(
            splitter.split(indices, local_labels, local_groups)
        )
        left, right = indices[left_local], indices[right_local]
        if len(np.unique(labels[left])) < 2 or len(np.unique(labels[right])) < 2:
            continue
        score = abs(labels[left].mean() - global_rate) + abs(labels[right].mean() - global_rate)
        if best is None or score < best[0]:
            best = (score, left, right)
    if best is None:
        raise RuntimeError("Could not create a group-disjoint split containing both classes")
    return best[1], best[2]


def build_splits(config_file: str | Path) -> pd.DataFrame:
    config, config_path = load_config(config_file)
    source_path = resolve_data_path(config["data"]["source_features"], config_path)
    frame = parse_gee_points(pd.read_csv(source_path))
    frame["spatial_block"] = spatial_blocks(frame, float(config["split"]["block_degrees"]))
    labels = frame.class_id.to_numpy(dtype="uint8")
    groups = frame.spatial_block.to_numpy()
    all_idx = np.arange(len(frame))
    train_val, test = _best_group_split(
        all_idx, labels, groups, float(config["split"]["test_fraction"]), int(config["project"]["seed"])
    )
    relative_val = float(config["split"]["val_fraction"]) / (1 - float(config["split"]["test_fraction"]))
    train, val = _best_group_split(
        train_val,
        labels,
        groups,
        relative_val,
        int(config["project"]["seed"]) + 1000,
    )
    split = np.full(len(frame), "", dtype=object)
    split[train], split[val], split[test] = "source_train", "source_val", "source_test"
    result = frame[["sample_id", "class_id", "longitude", "latitude", "spatial_block"]].copy()
    result.insert(1, "region", "jiangxi")
    result.insert(3, "label_source", "official_product")
    result["split"] = split
    if result.groupby("spatial_block").split.nunique().max() != 1:
        raise AssertionError("Spatial block leakage detected")
    output_dir = resolve_project_path(config["project"]["output_dir"], config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "source_splits.csv", index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    result = build_splits(args.config)
    print(result.groupby(["split", "class_id"]).size().unstack(fill_value=0))
    print("blocks:", result.groupby("split").spatial_block.nunique().to_dict())


if __name__ == "__main__":
    main()

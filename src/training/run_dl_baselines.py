from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.common import feature_groups, load_config, resolve_data_path, resolve_project_path
from src.models import MLPClassifier, TemporalCNNClassifier


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def standardize_train_only(x: np.ndarray, train: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(x[train], axis=0).astype("float32")
    std = np.nanstd(x[train], axis=0).astype("float32")
    std[std < 1e-6] = 1.0
    filled = np.where(np.isfinite(x), x, mean)
    return ((filled - mean) / std).astype("float32"), mean, std


def tensor_inputs(x: np.ndarray, feature_set: str, model_name: str):
    flat = torch.from_numpy(x)
    if model_name == "mlp": return (flat,)
    use_s1 = feature_set in ("s1_only", "s1_s2_fusion")
    use_s2 = feature_set in ("s2_only", "s1_s2_fusion")
    offset = 23 if feature_set == "s1_s2_fusion" else 0
    s2 = flat[:, :23].reshape(-1, 1, 23) if use_s2 else torch.empty(len(x), 0, 23)
    s1_flat = flat[:, offset:] if use_s1 else torch.empty(len(x), 0)
    s1 = s1_flat.reshape(-1, 3, 23) if use_s1 else torch.empty(len(x), 0, 23)
    return s1, s2


def forward(model, batch, model_name: str):
    return model(batch[0]) if model_name == "mlp" else model(batch[0], batch[1])


@torch.no_grad()
def predict(model, inputs, indices, labels, model_name, device):
    model.eval(); probabilities = []
    dataset = TensorDataset(*(item[indices] for item in inputs), torch.from_numpy(labels[indices]))
    for batch in DataLoader(dataset, batch_size=256, shuffle=False):
        tensors = [item.to(device) for item in batch[:-1]]
        probabilities.append(torch.softmax(forward(model, tensors, model_name), 1)[:, 1].cpu().numpy())
    return np.concatenate(probabilities)


def run_one(source, features, split, model_name, feature_set, config, seed, output_dir, device):
    seed_everything(seed)
    x_raw = source[features].to_numpy("float32")
    train = np.where(split == "source_train")[0]; val = np.where(split == "source_val")[0]; test = np.where(split == "source_test")[0]
    x, mean, std = standardize_train_only(x_raw, train)
    inputs = tensor_inputs(x, feature_set, model_name)
    y = source.class_id.to_numpy("int64")
    dataset = TensorDataset(*(item[train] for item in inputs), torch.from_numpy(y[train]))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, generator=generator)
    if model_name == "mlp":
        model = MLPClassifier(len(features), config["hidden_size"], config["dropout"])
    else:
        model = TemporalCNNClassifier(feature_set != "s2_only", feature_set != "s1_only", config["hidden_size"], config["dropout"])
    model.to(device)
    counts = np.bincount(y[train], minlength=2); weights = len(train) / (2 * counts)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)
    best_f1, best_state, stale, logs = -1.0, None, 0, []
    started = time.time()
    for epoch in range(1, config["max_epochs"] + 1):
        model.train(); losses = []
        for batch in loader:
            tensors = [item.to(device) for item in batch[:-1]]; target = batch[-1].to(device)
            optimizer.zero_grad(set_to_none=True); loss = criterion(forward(model, tensors, model_name), target)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"]); optimizer.step(); losses.append(loss.item())
        val_probability = predict(model, inputs, val, y, model_name, device)
        val_f1 = f1_score(y[val], val_probability >= 0.5); scheduler.step(val_f1)
        logs.append({"epoch": epoch, "train_loss": np.mean(losses), "val_f1": val_f1, "lr": optimizer.param_groups[0]["lr"]})
        if val_f1 > best_f1 + 1e-6:
            best_f1, best_state, stale = val_f1, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else: stale += 1
        if stale >= config["patience"]: break
    model.load_state_dict(best_state)
    probability = predict(model, inputs, test, y, model_name, device); prediction = probability >= 0.5
    run_name = f"{model_name}_{feature_set}_seed{seed}"
    pd.DataFrame(logs).to_csv(output_dir / f"{run_name}_log.csv", index=False)
    torch.save({"state_dict": best_state, "features": features, "mean": mean, "std": std, "seed": seed}, output_dir / f"{run_name}.pt")
    return {
        "model": model_name, "feature_set": feature_set, "seed": seed, "parameters": sum(p.numel() for p in model.parameters()),
        "best_epoch": int(pd.DataFrame(logs).val_f1.idxmax() + 1), "epochs_run": len(logs), "seconds": time.time() - started,
        "precision": precision_score(y[test], prediction), "recall": recall_score(y[test], prediction), "f1": f1_score(y[test], prediction),
        "auroc": roc_auc_score(y[test], probability), "auprc": average_precision_score(y[test], probability),
    }


def run(config_file: str | Path):
    config_path = Path(config_file).resolve(); config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_config, data_path = load_config(config["data_config"])
    output_dir = resolve_project_path(config["output_dir"], config_path); output_dir.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(resolve_data_path(data_config["data"]["source_features"], data_path))
    splits_path = resolve_project_path(data_config["project"]["output_dir"], data_path) / "source_splits.csv"
    splits = pd.read_csv(splits_path); source = source.merge(splits[["sample_id", "split"]], on="sample_id", validate="one_to_one")
    groups = feature_groups(list(source.columns)); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for model_name in config["models"]:
        for feature_set in config["features"]:
            for seed in config["seeds"]:
                rows.append(run_one(source, groups[feature_set], source.split.to_numpy(), model_name, feature_set, config["training"], seed, output_dir, device))
    results = pd.DataFrame(rows); results.to_csv(output_dir / "runs.csv", index=False)
    summary = results.groupby(["model", "feature_set"]).agg(f1_mean=("f1","mean"),f1_std=("f1","std"),auprc_mean=("auprc","mean"),auprc_std=("auprc","std"),parameters=("parameters","first"),seconds_mean=("seconds","mean")).reset_index()
    summary.to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "environment.json").write_text(json.dumps({"torch":torch.__version__,"cuda":torch.cuda.is_available(),"device":str(device)},indent=2),encoding="utf-8")
    return summary


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",default="configs/dl_source_only.yaml"); args=parser.parse_args()
    print(run(args.config).to_string(index=False))


if __name__ == "__main__": main()

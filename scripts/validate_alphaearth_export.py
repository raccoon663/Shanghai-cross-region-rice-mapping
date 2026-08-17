from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BANDS = [f"A{i:02d}" for i in range(64)]
RAW = ROOT / "outputs/alphaearth/raw_batches"
OUTPUT = ROOT / "outputs/alphaearth/alphaearth_samples_2022.csv"


def main() -> None:
    files = sorted(RAW.glob("alphaearth_2022_*.csv"))
    if not files:
        raise FileNotFoundError(f"No exported batches in {RAW}")
    frame = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    missing = sorted(set(BANDS) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing embedding bands: {missing}")
    if frame.duplicated(["region", "sample_id"]).any():
        raise ValueError("Duplicate exported sample keys")
    values = frame[BANDS].to_numpy(float)
    valid = np.isfinite(values).all(axis=1)
    norms = np.linalg.norm(values[valid], axis=1)
    summary = {
        "rows": len(frame), "valid_rows": int(valid.sum()),
        "missing_rows": int((~valid).sum()), "norm_mean": float(norms.mean()),
        "norm_min": float(norms.min()), "norm_max": float(norms.max()),
    }
    if valid.mean() < 0.98:
        raise ValueError(f"Excessive missing embeddings: {summary}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT, index=False)
    print(summary)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()

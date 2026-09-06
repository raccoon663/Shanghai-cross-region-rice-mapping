# Reproducing the four-representation benchmark

Run commands from the repository root. The frozen configuration is
[`configs/eofm_benchmark.yaml`](../configs/eofm_benchmark.yaml). The public
aggregate evidence can be inspected without Earth Engine, GPU access, or private
sample outputs. Full retraining requires the exact frozen runtime inputs below.

## Inspect the evidence without retraining

```bash
python -m venv .venv
# Activate .venv using the command for your shell.
pip install -e ".[dev]"
python -c "import pandas as pd; x=pd.read_csv('results/eofm_benchmark_v1/fewshot_summary.csv'); print(x[x.budget.eq(500)].to_string(index=False))"
```

The case-study figures are in `assets/figures/eofm_benchmark_v1/`. To verify the
curated evidence, run this Python snippet from the repository root:

```python
import hashlib
import json
from pathlib import Path

manifest = json.loads(Path('results/eofm_benchmark_v1/artifact_manifest.json').read_text())
for name, expected in manifest['files'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
print('All curated artifact hashes match.')
```

Hashes verify artifact identity; they are not a substitute for rerunning model
training. Aggregate tables cannot reconstruct individual predictions or redo
the spatial-block bootstrap without the private per-sample outputs.

## Required runtime inputs

| Input | Location or setting | Identity check |
|---|---|---|
| Shared sample manifest | `data_metadata/alphaearth_sample_manifest.csv` | Config SHA-256 and fixed split counts |
| Rebuilt Temporal | 27 `eofm_presto_points_2022_c*.csv` files in `outputs/eofm/ee/downloaded_20260823/` | `results/manifests/eofm_presto_input_validation.json` |
| AlphaEarth | `outputs/alphaearth/alphaearth_samples_2022.csv` under `RICE_FUSION_LEGACY_ROOT` | Config SHA-256 and sample metadata equality |
| Monthly Presto | `outputs/eofm/presto_primary_embeddings_2022_monthly.npz` | `results/manifests/eofm_presto_primary_embedding_freeze.json` |
| Galileo | `outputs/eofm/galileo_full_embeddings_2022.npz` | `results/manifests/eofm_galileo_full_embedding_freeze.json` |

Set the AlphaEarth runtime root in your shell. For example, in PowerShell:

```powershell
$env:RICE_FUSION_LEGACY_ROOT = 'D:/rice-runtime'
```

The example expects the AlphaEarth CSV beneath
`D:/rice-runtime/outputs/alphaearth/`. Use your own retained runtime directory.
The [representation input specification](eofm_representation_inputs.md), export and
encoder scripts in `scripts/eofm/`, and embedding freeze manifests document
upstream extraction. Presto and Galileo use separate pinned encoder runtimes;
the downstream RF benchmark consumes their extracted arrays and does not need
a GPU. Downloading checkpoints and exporting public imagery do not recreate
unavailable reference labels or guarantee byte-identical historical inputs.

The runner checks the exact frozen hashes and fails on substitutes. If any
required input is missing, the published tables remain inspectable but full
numerical reproduction is unavailable until that input is restored. A rebuilt
dataset requires a new documented experiment namespace and input freeze; do not
replace hashes merely to make this experiment accept different data.

## Execute the frozen stages

With the matching inputs present:

```bash
python -m src.experiments.run_eofm_benchmark --stage prepare
python -m src.experiments.run_eofm_benchmark --stage source
python -m src.experiments.run_eofm_benchmark --stage fewshot
python -m src.experiments.run_eofm_benchmark --stage ood
python -m src.experiments.run_eofm_benchmark --stage paired
python scripts/eofm/14_report_benchmark.py
```

`source` freezes all source predictions before evaluating Shanghai. Keep this
ordering. `fewshot` uses four concurrent fits with one RF worker each and reuses
completed per-fit artifacts when resumed. Private features, draws, models and
predictions stay under ignored `outputs/eofm_benchmark_v1/private/`; generated
tables and reports go to its parent. These commands do not publish to GitHub.

The expected inventory is 12 source fits, 300 shared draws, 2,400 adaptation
fits, 80 adaptation summary rows, 20 OOD summary rows, and 126 paired intervals.
The reported environment versions and source/config hashes are retained in
[`execution_environment.json`](../results/eofm_benchmark_v1/execution_environment.json).
The package dependency ranges are broader than that recorded environment;
installing the latest compatible dependencies is not a bitwise-reproduction claim.

## Validation and interpretation

```bash
python -m pytest -q
python -m compileall src scripts
```

Read the [protocol](eofm_benchmark_protocol.md) for imputation, weighting,
sampling and uncertainty details. The reconstructed Temporal results belong
to this extension only. Shanghai metrics remain weak-reference agreement;
replicating the computation does not create independent field validation.

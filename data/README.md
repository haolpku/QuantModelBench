# Data access

The public GitHub repository contains code only. The dataset is maintained at
[ArolaLuo/QuantModelBench](https://huggingface.co/datasets/ArolaLuo/QuantModelBench)
and currently requires authorization. Possession of code does not provide data
access or redistribution permission. Check the relevant JKP and
WRDS/CRSP/Compustat terms for your intended use.

## Download selected components

Activate the repository virtual environment, then install the optional client:

```bash
python -m pip install -r requirements-data.txt
hf auth login
python scripts/download_data.py --component task-b-semantic --dry-run
python scripts/download_data.py --component task-b-semantic
```

Alternatively, supply `HF_TOKEN` through your environment. Never put it in a
command committed to this repository. The downloader defaults to the exact HF
revision recorded in [provenance](../docs/PROVENANCE.md), not moving `main`.

Available components:

| Component | Contents |
|---|---|
| `task-b` | 294 numeric cases and their evaluator files |
| `task-b-semantic` | Numeric base plus semantic overlay |
| `task-c` | 207 numeric cases and their evaluator files |
| `task-c-semantic` | Numeric base plus semantic overlay |
| `evaluator` | Task C stock panel, daily returns and size cutoffs |
| `jkp` | Processed factor returns, cards and metadata |
| `construction` | Download manifests and shared selection plan |

Task C rolling evaluation also needs `evaluator`. Dataset construction and
baseline scripts may need `jkp` and `construction`. Components may be several GB;
request only what the experiment needs. Raw WRDS exports and experiment runs are
not downloaded by this helper.

```bash
python scripts/download_data.py --component task-c-semantic evaluator
```

Download into the repository root to preserve `data/processed/...` paths, or use
`--local-dir` and pass explicit paths to runners. Some imported dataset metadata
may retain producer-side absolute paths; run the audits below and inspect any
reported path issues before scheduling model calls.

## Validate and use

```bash
python task_b/code/scripts/audit_task_b_dataset.py \
  --dataset data/processed/task_b_usa_frozen_raw_rolling_294
python task_c/code/scripts/audit_task_c_dataset.py \
  --dataset data/processed/task_c_frozen_raw_rolling_207
```

Split manifests have an index record as well as case records. Select records with
`record_type == "case"`; do not count the header as an extra case. Semantic
overlays share evaluator files with a numeric base, so preserve their relative
directory relationship and use the shared overlay resolver.

**Public-to-model is not public-to-redistribute.** Historical payloads may still
derive from licensed data. Downloaded case folders also include hidden truth,
feature/ID maps and rolling schedules. Pass only permitted historical payloads
and artifacts into development tools; never give an agent this entire directory.

All downloaded files under `data/` are ignored by Git, except this README.

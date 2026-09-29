# QuantModelBench

**Can LLMs Build Financial Models That Generalize?**

QuantModelBench evaluates whether LLMs can turn existing financial information
into executable modeling procedures that generalize across time. An LLM develops
a Python `fit_and_predict` program using permitted historical data and bounded
validation feedback. Selected source code is frozen before 12 monthly evaluation
origins; the same procedure is refitted locally with no further LLM calls.

This repository contains the research code, prompts, schemas, dataset builders,
baselines, and evaluation utilities. Restricted datasets and experiment artifacts
are maintained separately on [Hugging Face](https://huggingface.co/datasets/ArolaLuo/QuantModelBench).
The original project name, FactorModelBench, may remain in historical protocol
identifiers. Those identifiers and the `factoropt` import paths are retained for
compatibility.

## Tasks

| | Task B: factor timing | Task C: stock selection |
|---|---|---|
| Input | Historical returns of existing factors | Historical stock characteristics and labels |
| Output | A program scoring next-month factor returns | A program scoring next-month stock returns |
| Evaluator | Fixed factor portfolio rules | Fixed JKP-compatible stock portfolio and daily settlement |
| Cases | 294 (84 dev / 210 test) | 207 (84 dev / 123 test) |
| Hidden origins per case | 12 | 12 |

The 501 cases precede duplication across information conditions:
**Numeric** uses anonymized numerical inputs; **Semantic** adds neutral factor
descriptions to the same numerical case. Execution reliability, predictive
quality, and economic value are reported separately. Cases sharing temporal blocks
or feature subsets are dependent; they are not 501 independent market regimes.

## Quick start: no credentials or market data required

Use Python 3.11 and run commands from this repository's root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-smoke.txt
python task_b/code/scripts/smoke_task_b.py
python task_c/code/scripts/smoke_task_c.py
```

These checks use synthetic fixtures and mock model responses. They exercise
candidate validation, source selection, frozen rolling execution, semantic
metadata filtering, and Task C portfolio accounting. They do not reproduce the
paper's empirical results or call a paid model endpoint.

For actual model comparisons, install **all** allowed model families:

```bash
python -m pip install -r requirements.txt
```

The lightweight smoke environment is not a benchmark runtime: missing an allowed
library could otherwise turn valid generated code into an execution failure.
Dependency ranges are not a historical lockfile. Record exact installed versions,
hardware, seeds, and numerical-library thread settings for each experiment.

## Data and experiments

The HF repository requires authorized access. No dataset token is stored here.
Follow [data access](data/README.md) to download selected components, then use the
[Task B guide](task_b/README.md) or [Task C guide](task_c/README.md).
The [evaluation guide](task_bc_eval/README.md) explains the run registry and the
development-frozen scoring configuration.

The submission entrypoint is:

```python
def fit_and_predict(train_df, valid_df, predict_df, metadata):
    # Fit using permitted historical labels; predict_df has no target column.
    # Return exactly one row per prediction id, with columns: id, score.
    ...
```

Full response contracts are defined by the task-specific JSON schemas. Tool calls,
candidate revisions, candidate selection, and hidden rolling evaluation are
separate stages; make their budgets explicit rather than relying on CLI defaults.

## Repository map

```text
shared/           Shared tool loop, bounded Python tools, semantic overlay support
task_b/           Factor-timing builders, prompts, schemas, baselines and runners
task_c/           Stock-model builders, prompts, schemas, runners and backtester
task_bc_eval/     Outcome/process metrics, aggregation, rubric and judge preparation
scripts/          Dataset download and public-release checks
docs/             Protocol, execution boundary, and source provenance
data/README.md    Access instructions; actual data remain local and ignored
```

Task B and Task C retain separate packages named `factoropt`. Use their script
entrypoints in separate processes; do not put both code directories on a single
global `PYTHONPATH`. Task B reuses the Task C execution boundary internally, so
keep both task directories when installing this source checkout.

## Scope and status

- This is a curated source release, not a new experimental run or final leaderboard.
- The existing runner provides Python policy checks and subprocess execution;
  it is **not** a hardened container/VM sandbox. See [execution boundaries](docs/EXECUTION.md).
- Harbor/Terminal-Bench adaptation is not implemented in this release.
- Dataset access and redistribution are governed by the underlying providers;
  public source availability does not grant rights to WRDS/CRSP/Compustat data.
- No open-source license grant has been added in this import. Code licensing
  remains to be specified by the repository owner.

See [release provenance](docs/PROVENANCE.md) for the pinned HF revision and checks,
and [protocol](docs/PROTOCOL.md) for the interpretation of the two tasks.

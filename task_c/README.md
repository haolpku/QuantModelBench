# Task C: multifactor stock prediction

Generate a program mapping historical stock characteristics to next-month stock
scores. Portfolio formation, daily settlement, turnover and transaction costs
are computed by the fixed evaluator, not by the LLM's program.
The canonical inventory has 207 cases (84 dev, 123 test), each with 12 origins.

## Offline check

```bash
python task_c/code/scripts/smoke_task_c.py
```

This uses synthetic data and a mock provider. No WRDS data or API key is needed.

## Run a case

Follow [data access](../data/README.md) and install the full requirements first.
Task C reads `OPENAI_API_KEY`, `TASK_C_MODEL`, and `TASK_C_BASE_URL` for an
OpenAI-compatible provider. Set all three explicitly for a real experiment.

```bash
CASE_DIR=data/processed/task_c_frozen_raw_rolling_207/cases/BLOCK/CASE_ID
python task_c/code/scripts/run_task_c_codegen.py \
  --payload "$CASE_DIR/payload.json" \
  --output outputs/task_c/example/revision_audit.json \
  --max-revisions 2 --selection-policy llm --no-stop-on-final
python task_c/code/scripts/run_task_c_rolling.py \
  --frozen-response outputs/task_c/example/revision_audit.json \
  --case-dir "$CASE_DIR" \
  --panel data/processed/task_c_evaluator_panel/panel.parquet \
  --daily data/processed/task_c_evaluator_panel/daily_ret.parquet \
  --nyse-cutoffs data/processed/task_c_evaluator_panel/nyse_cutoffs.csv \
  --output-dir outputs/task_c/example/rolling
```

Replace `BLOCK/CASE_ID` with an actual case from a split manifest. Codegen and
rolling are separate stages. For semantic codegen, use the semantic payload and add `--artifact-root` pointing
to the matching numeric case directory, which holds the shared historical
Parquet files. The parallel codegen launcher resolves this automatically via
`shared.dataset_overlay`. Use the semantic case directory for rolling: it carries
its own cards, manifest, evaluator maps and truth schedule. See runner `--help`
for metadata options.

## Components

- `task_c.py` and `task_c_data.py`: input contracts and temporal data handling.
- `task_c_code_runner.py`: source policy checks and subprocess execution.
- `task_c_iterative.py`: bounded revisions, candidate selection and rolling refits.
- `task_c_backtest.py`: fixed portfolio construction and daily accounting.
- `agentic/`: task adapter over the shared tool loop.

Builders, audit commands, baseline sweeps, and parallel runners live in
[code/scripts/](code/scripts/). Prompts and response schemas are in
[prompt/](prompt/) and [code/schemas/](code/schemas/). Use script `--help` to inspect
required data and budget arguments. Batch shell scripts use `python3` from the
active environment; `QUANTMODEL_PYTHON` overrides that interpreter.

`rolling_ridge` is the primary same-information baseline. The signed JKP equal
weight portfolio is a domain-informed reference. Keep prediction metrics and
portfolio outcomes separate; a valid program need not beat either baseline.

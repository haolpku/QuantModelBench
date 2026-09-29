# Task B: factor timing

Predict the relative next-month returns of existing factors by generating a
`fit_and_predict` program. The evaluator forms portfolios using fixed rules.
The canonical USA inventory has 294 cases (84 dev, 210 test), each with 12 origins.
Numeric and Semantic are paired views of the same numerical case.

## Offline check

From the repository root, in the activated environment:

```bash
python task_b/code/scripts/smoke_task_b.py
```

## Run a case

First follow [data access](../data/README.md) and install the full requirements.
Set `CASE_DIR` to a numeric or semantic case directory. The API key is read from
`LLM_API_KEY`; set `LLM_MODEL` and `LLM_BASE_URL` for your provider.

```bash
CASE_DIR=data/processed/task_b_usa_frozen_raw_rolling_294/cases/BLOCK/CASE_ID
python task_b/code/scripts/run_task_b_codegen.py \
  --payload "$CASE_DIR/payload.json" \
  --output outputs/task_b/example/revision_audit.json \
  --max-revisions 2 --selection-policy llm --no-stop-on-final
python task_b/code/scripts/run_task_b_rolling.py \
  --frozen-response outputs/task_b/example/revision_audit.json \
  --rolling-truth "$CASE_DIR/rolling_truth.jsonl" \
  --output-dir outputs/task_b/example/rolling_result
```

Replace `BLOCK/CASE_ID` with an actual manifest entry. For a semantic overlay,
resolve evaluator truth from the numeric base. `outputs/` is ignored by Git.
The rolling command cannot request new LLM-generated source. Its default window
configuration is read from the case protocol; specify overrides only for a named
sensitivity experiment.

## Entrypoints

| Script in `code/scripts/` | Purpose |
|---|---|
| `build_task_b_case.py` / `build_task_b_dataset.py` | Build historical cases and hidden rolling schedules |
| `build_task_b_semantic_overlay.py` | Attach the semantic treatment to existing numeric cases |
| `audit_task_b_dataset.py` | Check dataset structure and temporal design |
| `run_task_b_baseline.py` / `run_task_b_baselines_split.py` | Evaluate reference forecasting rules/models |
| `run_task_b_codegen_parallel.py` | Schedule a split with explicit model/tool budgets |
| `prepare_task_b_bundle.py` | Prepare a local execution bundle |

Use each script's `--help` for required inputs and options. Prompts are in
[prompt/](prompt/) and response schemas in [code/schemas/](code/schemas/).
Keep Task C's code directory: Task B reuses its generated-code execution boundary.

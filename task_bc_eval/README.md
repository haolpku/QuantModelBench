# Cross-task evaluation

This package consumes existing run artifacts; it does not generate market data
or run an LLM. Source is under `code/bceval/`, CLI entrypoints under `code/scripts/`.

The imported run registry in `code/bceval/runspec.py` records the historical run
layout. Those run folders are intentionally absent from this public repository.
To score your own experiments, add a `RunSpec` with the correct task, method,
track, split and artifact layout before building metrics.

```bash
python task_bc_eval/code/scripts/build_case_metrics.py --split dev
python task_bc_eval/code/scripts/compute_rubric.py --split dev
```

Run these only after arranging complete artifacts and required dataset manifests.
Task C OOS IC additionally requires the evaluator panel. `--skip-oos` is a diagnostic
option, not a substitute for the full metric calculation. Missing runs must not be
interpreted as completed evaluations or assigned artificial zero returns.

`config/dev_frozen_v1.json` is the imported development-frozen scoring
configuration. Preserve it for historical comparisons; do not fit new thresholds
on test results. The judge configuration includes deferred fields, and no completed
judge assessment or final leaderboard is claimed by this release.

For task definitions, weighting and score interpretation see
[the rubric specification](../task_bc_metric_rubric.md) and
[protocol notes](../docs/PROTOCOL.md). `build_judge_batch.py` prepares judge inputs;
it does not mean that judging has been run.

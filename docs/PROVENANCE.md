# Source release provenance

Imported on 2026-09-29 from the private dataset repository
`ArolaLuo/QuantModelBench`, pinned to HF revision
`275fa9840756e26f2aeebcbc831cf2385a728f15`.

[source_manifest.json](source_manifest.json) records original SHA-256 hashes and
whether each retained source file was changed during release preparation.
This import does not include the HF Git history.

## Included

- Task B and C source, builders, prompts, schemas, baselines and execution runners.
- Shared provider/tool loop and semantic overlay resolver.
- Cross-task metric/rubric code and the historical development-frozen configuration.
- Synthetic smoke fixtures already embedded in the source scripts.

## Excluded

- Raw/processed datasets, hidden targets, case artifacts and actual predictions.
- Experiment directories, environment files, logs, process IDs and generated reports.
- Machine-specific SSH synchronization, live-server comparison and operational
  rerun scripts; private server addresses are not needed to reproduce the code.
- Internal review notes and historical experiment narratives that are not public
  runtime documentation.

## Release changes

- Added a QuantModelBench README, current task guides, data-access instructions,
  protocol and execution-boundary documentation.
- Preserved program interfaces, prompts, model-selection logic and numerical
  evaluation logic; kept historical import/schema identifiers for compatibility.
- Removed machine-specific roots from the two Task C baseline shell launchers,
  added interpreter/data overrides, and replaced GNU-only `xargs` input flags.
- Declared SciPy explicitly because source modules import it directly.
- Added lightweight smoke requirements, a component-scoped HF downloader,
  source-release checks and GitHub Actions smoke CI.
- Tightened Git ignores to exclude data, runs, credentials and generated artifacts.

The original full requirements use version ranges; this release does not pretend
to recover an exact historical experiment environment. The local smoke environment
is recorded in [smoke-environment.txt](smoke-environment.txt) for diagnostic purposes,
not as a full generated-model runtime lockfile.

## Validation at import

- Task B synthetic smoke: 12 rolling origins, 72 prediction rows, zero LLM calls
  during rolling; candidate selection and semantic filtering checks passed.
- Task C synthetic smoke: two revisions, two rolling months, 60 prediction rows;
  public-artifact tool access, portfolio settlement and transaction-cost checks passed.
- All 28 non-smoke Python CLI entrypoints loaded successfully with `--help`.
- Both shell launchers passed `bash -n`; the parallel scheduler passed a synthetic
  dry run with paths containing spaces.
- Python syntax, JSON parsing, publication-path exclusions and credential-pattern
  checks passed for release files.

Full market-data experiments, paid model calls, GPU/model-family compatibility and
historical leaderboard reproduction were not run for this source release. The
automated checks do not certify arbitrary generated code as safe to execute.

No license grant was inferred from public repository visibility; code licensing
remains for the owner to specify. Underlying data-provider terms still apply.

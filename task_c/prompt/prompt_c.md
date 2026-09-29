# Task C: multi-factor stock signal model

The executable adapter tags this contract as
`task_c_code_model_v5_llm_candidate_selection`.

## 0. Role and task

You are a quantitative researcher writing a multi-factor stock signal model.
Your deliverable is **executable Python source**, not predictions, portfolio
weights, a backtest, or prose.

The input is a sanitized, masked monthly stock panel, not raw source data. Each
labelled historical row represents one stock at one relative formation month:
its numeric columns are masked factor-characteristic measurements or derived
transformations computed from information available by that month. Their
economic meanings are intentionally withheld on `numeric_only` and may be
described by neutral cards on `semantic`. `label` is the next month's realized
stock excess return. The current `t-0` cross-section has the same feature
columns but no current return.
This is a cross-sectional scoring and ranking problem, not a request to identify
securities or implement a portfolio. Stock ids and feature columns are masked; the
semantic track may additionally provide neutral definitions and formulas, but
no source provider, paper, calendar date, or unmasked security identifier is
supplied as an external prior.

For every monthly formation date the evaluator calls one function you write:

```python
def fit_and_predict(train_df, valid_df, predict_df, metadata) -> pandas.DataFrame
```

Here `valid_df` is the in-code name for the payload's `validation` frame; it is
not a second prediction set.

At each formation origin the evaluator rebuilds the three frames from one
chronological history. `train_df` is the older labelled stock-month block used
to fit the initial model. `valid_df` is the newer labelled block available for
time-ordered model selection and, after selection, an optional refit together
with `train_df`; in revision-time walk-forward validation it contains only
labels settled before the month being scored and is empty for the first scored
month. `predict_df` contains exactly the current formation month's (`t-0`)
stock features, without its current return. The evaluator keeps those returns
outside the function and expects one score per `predict_df["id"]`.

`score` is a cross-sectional signal: the evaluator ranks all stocks by `score`
in the formation month and builds high-minus-low factor portfolios from those
ranks, settling the returns itself. Only the ordering matters, not the scale.
Higher score means more attractive. You never see the test months, and the same
source is re-fitted from scratch at every origin.

You work in revisions. Each revision is one request: you may inspect the public
data with the tools in §3.4, then return the complete source together with the
JSON response object specified in §2. The runner executes each candidate on
historical validation data and sends back a bounded feedback envelope (§3.2);
you then return a complete replacement. After the last revision, a separate
selection turn asks you to choose which valid candidate to freeze (§3.3). That
frozen source is then executed on hidden test months.

**Final goal: source that keeps producing stable, economically meaningful
cross-sectional signals on unseen future months.** Validation metrics are
diagnostics that tell you whether a change helped; they are not the objective by
themselves and no single number decides which revision is best. On the
selection turn you choose the candidate; the host does not freeze the max Rank
IC by default.

## 1. Input protocol

### 1.1 Payload fields

| Field | Meaning |
| --- | --- |
| `case_id`, `task`, `horizon_months` | Identifiers; echo them nowhere in the response. |
| `track` | `numeric_only` or `semantic`; see the appended track block. |
| `feature_columns` | Names of the anonymous numeric factor features. Use them verbatim. |
| `factor_cards` | Semantic track only: one sanitized card per feature with the neutral name, definition, formula and formula-input descriptions. Empty on `numeric_only`. |
| `train` | Public labelled history as `rows`, with an optional public `artifact` descriptor. |
| `validation` | Public labelled history in the same row format as `train`, with an optional `artifact`. |
| `predict` | `{schema, label_included: false, rows_visible_to_llm: false}`; no row labels are public. |
| `metadata` | Public facts: id/time columns, `seed`, window sizes and refit policy. The evaluator adds the sanitized cards to `metadata["factor_cards"]` only inside the executable function; they are top-level `factor_cards` in this payload. |
| `artifact_root` | Case-relative evaluator hint for resolving a declared public artifact; it is not necessarily a path inside the `python_exec` workspace. |

When embedded, `train.rows` and `validation.rows` are arrays whose objects have
exactly the ordered keys `["id", "time_key", *feature_columns, "label"]`;
`predict.rows`, when present, uses the same ordered list without `label`. An
artifact descriptor's public `columns` list is that same historical list. The
in-prompt row copies may be
size-capped or empty. In the agentic runner, artifact-backed history is
materialized into `payload["train"]["rows"]` and
`payload["validation"]["rows"]` inside `python_exec` before the requested code
runs; use those public rows for inspection.

### 1.2 Panel geometry

`time_key` is a relative month offset from the formation month, encoded as
`t-<k>`. Historic formation months use successively larger offsets; the scored
formation month is `t-0` and lives in `predict_df`. At each origin the supplied
history is split chronologically into the older `train_df` block and newer
`valid_df` block (the exact lengths are in the payload metadata); only the
current `t-0` rows enter `predict_df`. Never assume a fixed row count or that a
later month's row is visible at an earlier origin.

### 1.3 Validation timeline and data quality

Validation is chronological: the evaluator passes only labels from formation
months strictly earlier than the one being scored. The first scored month may
therefore receive an empty `valid_df` intentionally—there is no earlier
validation label yet, while the evaluator still scores that month against its
held-out return outside the function. Handle an empty or short `valid_df`
explicitly. It is acceptable to use `valid_df` for model selection and then
refit on train plus validation.

Real factor panels contain missing and occasionally infinite numeric values.
Impute or otherwise sanitize train, validation and predict frames before
fitting, and never pass NaN/inf to an estimator. Use `metadata["seed"]` as your
only randomness source, and do not mutate the input frames.

### 1.4 Getting the rows

`python_exec` sees the whole public payload as `payload`. Before each
`python_exec` call the runner mounts any declared public train/validation
artifact and materializes it into the corresponding `rows` list. Read the
materialized public rows as follows:

```python
train = pd.DataFrame(payload["train"]["rows"])
valid = pd.DataFrame(payload["validation"]["rows"])
```

The descriptor's `path` and `format` are public case metadata used by the
evaluator-side mount; do not pass that case-relative path directly to
`pd.read_parquet` (it may not exist in the tool's current working directory),
and do not hard-code a filename or format. Use these materialized rows, not the
chat copy, for row-level inspection; they contain no test data. If a caller
provides embedded rows instead of an artifact, the same code works unchanged.

## 2. Output protocol

Return exactly **one** JSON object with exactly these ten top-level keys — no
Markdown fence, no commentary, no tool tags, no extra keys:

| Key | Type and constraint |
| --- | --- |
| `revision` | Integer; must equal the `CURRENT REVISION` value in this request. |
| `status` | `"candidate"` or `"final"`. |
| `language` | `"python"`. |
| `code` | Complete source, 20–120000 characters, defining `fit_and_predict`. |
| `entrypoint` | `"fit_and_predict"`. |
| `requirements` | Array of bare package names you import, no versions, no prose. |
| `input_contract` | Object with exactly `train_columns`, `validation_columns`, `predict_columns`. |
| `output_contract` | Object with exactly `columns` (`["id", "score"]`) and `one_row_per_id: true`. |
| `explanation` | ≤2000 characters of public reasoning about the modelling choice. |
| `revision_summary` | ≤1000 characters; what changed since the previous revision. |

Concrete instance for a payload with features `<f1>`, `<f2>` (this is an
*instance*, not a schema definition):

```json
{
  "revision": 0,
  "status": "candidate",
  "language": "python",
  "code": "def fit_and_predict(train_df, valid_df, predict_df, metadata):\n    ...",
  "entrypoint": "fit_and_predict",
  "requirements": ["numpy", "pandas", "scikit-learn"],
  "input_contract": {
    "train_columns": ["id", "time_key", "<f1>", "<f2>", "label"],
    "validation_columns": ["id", "time_key", "<f1>", "<f2>", "label"],
    "predict_columns": ["id", "time_key", "<f1>", "<f2>"]
  },
  "output_contract": {"columns": ["id", "score"], "one_row_per_id": true},
  "explanation": "short public explanation",
  "revision_summary": "short change summary"
}
```

Rules that the contract checker enforces:

- Copy every name in `feature_columns` into all three `*_columns` arrays; `label`
  belongs in `train_columns` and `validation_columns` only. The ellipses above
  are explanatory and must not appear in the returned arrays.
- `input_contract` and `output_contract` are plain metadata objects. Do **not**
  turn them into JSON-Schema objects and do not add `type`, `properties`,
  `required`, `description`, `metadata`, `train_df`, `valid_df`, or `predict_df`
  keys. Every `*_columns`/`columns` value is an array of plain strings.
- `requirements` lists only what you import, as bare names (`["numpy"]`, not
  `["numpy==2.1.0"]`, not a newline-separated string).

### 2.1 Code requirements

- Define `fit_and_predict(train_df, valid_df, predict_df, metadata)`; the
  evaluator imports this one name and calls it once per origin.
- Return a pandas DataFrame with exactly one row per `predict_df["id"]`, with a
  unique `id` column and finite numeric `score`. Do not reorder, drop, or
  duplicate ids, and keep scores deterministic for the given inputs and seed.
- Fit preprocessing, feature selection, hyperparameters and model parameters
  using only the supplied historical frames.
- Never read future returns, the hidden test panel, or calendar dates. The code
  must not contain calendar literals (a four-digit year such as `2016`, with or
  without a month suffix) and must not branch on absolute dates; express every
  window as a relative `time_key` offset.
- Do not implement a portfolio or a backtest — the evaluator ranks your scores
  cross-sectionally and settles portfolio returns itself.

### 2.2 Execution environment and safety

- The source runs as one sandboxed subprocess with fresh frames per origin. Each
  call has a 1800-second wall clock, an 8 GiB memory cap (enforced on Linux), and
  four numeric threads (`OMP`/`OPENBLAS`/`MKL`/`NUMEXPR`/`VECLIB`). Choose any
  model family that fits: regularized linear models, tree ensembles, gradient
  boosting, or a neural network trained from scratch. Prefer a compact, well
  regularized model over an exhaustive search, since the same call repeats at
  every origin.
- `numpy`, `pandas`, `scikit-learn`, `scipy`, `xgboost`, `lightgbm`, `catboost`,
  `statsmodels`, `optuna` and `torch`, plus the standard-library modules listed
  in the sandbox policy, are available. Anything else fails at import time.
- Forbidden regardless of package: filesystem access, network, subprocesses,
  environment variables, dynamic execution, and reading hidden truth.
- A response that violates the contract, fails to run, exceeds the timeout, or
  returns malformed scores is a failed revision, not a silent repair: the
  feedback envelope reports the errors and you return a corrected complete
  source in the next revision.

## 3. Revision protocol

Revision requests are **stateless**. Each one contains this task description,
the payload, and at most one feedback envelope; nothing from earlier turns is
retained except through that envelope. Always return the complete JSON object
and the complete source — never a diff, never a patch description, never a
partial file.

### 3.1 Request tail

The runner appends these blocks after this prompt, in order:

1. `SEMANTIC TRACK` or `NUMERIC-ONLY TRACK` — the information set for this case.
2. `REVISION PROTOCOL` — includes `CURRENT REVISION: <n>`, the integer to copy
   into `revision`, and the requirement to return a complete replacement source.
3. `PUBLIC VALIDATION FEEDBACK` — present from the second revision onward.
4. `MANDATORY RESPONSE INSTANCE FOR THIS TURN` — the exact feature names,
   contract arrays and current revision for this payload. Replace its
   placeholders and return that object. The static instance in §2 is only an
   illustrative example; the appended instance is authoritative.
5. After the last code revision, a `CANDIDATE SELECTION TURN` may replace (4)
   with a two-key object `{selected_revision, selection_reason}`.

The revision number belongs to this revision protocol. It is independent of
payload metadata and always takes precedence over a value remembered from an
earlier request.

### 3.2 Feedback envelope

The envelope contains only public information: `revision`,
`candidate_status`, `validation` aggregate metrics, `io_checks.execution_valid`,
`runtime`, at most 20 `errors`, `allowed_next_revision`, and
`previous_candidate` — the previous revision's public fields and source. Read it
to (a) understand the aggregate effect of the previous change, and (b) repair:
`previous_candidate` is repair context so you can keep the contract fields and
source that were already correct, fix every listed error, and change only what
the evidence supports. It is not a test signal and never contains hidden test
rows or labels.

### 3.3 Candidate selection turn

When revisions end and at least two candidates passed schema and execution
checks, the runner sends one extra request whose envelope has
`selection_request: true`, `allowed_revisions`, and a public table of each
valid candidate: revision index, `code_hash`, `code_chars`, `revision_summary`,
and the same aggregate validation metrics as §3.2. It does not include hidden
labels, raw source, or a required ranking formula.

Return only:

```json
{"selected_revision": 0, "selection_reason": "<why this candidate is frozen>"}
```

`selected_revision` must be one of `allowed_revisions`. Do not emit new source
and do not call tools on this turn. If the selection object is invalid, the
host freezes the last valid candidate and records a fallback.

### 3.4 Tools

`python_exec` runs sandboxed Python (up to 8 calls per revision, ≤12000
characters of output per call) in the workspace that holds `payload` and the
mounted public train/validation artifacts; `read_artifact` and `write_artifact`
move files. Use
tools to check shapes, missingness, cross-sectional coverage, label
distributions, and feature behaviour before committing to an architecture.
`MAX_CODE_CHARS` is 120000, so a full model definition fits. Spend the budget
efficiently; when tools are removed or finalization is requested, immediately
return the single JSON object and do not request another tool.

## 4. Style and hygiene

Keep `explanation` and `revision_summary` short, factual and auditable. Never
include row-level data, hidden targets, file paths, tool traces, credentials, or
private reasoning in the response. Comment the source where a choice is not
self-evident — sanitization, refit policy, regularization — and leave the
modelling decisions to the code rather than to prose.

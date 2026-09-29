# Task B: frozen-code factor-return model

The executable adapter tags this contract as
`task_b_factor_code_model_v6_llm_candidate_selection`.

## 0. Role and task

You are a quantitative researcher writing the factor-return forecasting model
for a rolling factor-allocation experiment. Your deliverable is **executable
Python source**, not a prediction file, portfolio weights, a backtest report, or
an explanation.

The input is a sanitized, masked monthly panel of factor observations, not raw
source data. Each labelled historical row represents one factor at one relative
monthly cutoff: its numeric columns are derived summaries of information
available for that factor through the cutoff (for example windowed return
statistics, risk/moment summaries, and data-availability indicators), and
`label` is the next month's realized factor excess return. The current `t-0` row
has the same kind of
features but its next return is withheld. This is a one-step-ahead factor-score
problem, not a request to identify the original factors or construct a
portfolio. Factor ids are masked; the semantic track may additionally provide
neutral definitions and formulas, but no source provider, paper, calendar date,
or an unmasked factor identifier is supplied as an external prior.

For every monthly origin the evaluator calls one function you write:

```python
def fit_and_predict(train_df, valid_df, predict_df, metadata) -> pandas.DataFrame
```

Here `valid_df` is the in-code name for the payload's `validation` frame; it is
not a second prediction set.

At each origin the evaluator rebuilds the three frames from one chronological
history. `train_df` is the older labelled factor-month block used to fit the
initial model. `valid_df` is the newer labelled block available for
time-ordered model selection and, after selection, an optional refit together
with `train_df`; in revision-time walk-forward validation it contains only
labels settled before the month being scored and is empty for the first scored
month. `predict_df` is exactly the current `t-0` feature row for each factor,
without its current/future label. The evaluator keeps those labels outside the
function and scores the one returned row per `predict_df["id"]` (higher means
more attractive).

You work in revisions. Each revision is one request: inspect the public data
with the tools described in §4 if that helps, then return the complete source
together with the JSON response object specified in §3. The runner scores each
candidate on historical validation data and sends back a bounded feedback
envelope (§3.2); you then return a complete replacement. After the last
revision, a separate selection turn asks you to choose which valid candidate
to freeze (§3.3). The evaluator then re-fits exactly that frozen source at
each of the next 12 monthly origins to score factors it has never seen.

**Final goal: source that keeps producing stable, economically meaningful
factor scores on unseen future months.** Validation metrics are diagnostics
that tell you whether a change helped; they are not the objective by themselves
and no single number decides which revision is best. On the selection turn you
choose the candidate; the host does not freeze the max Rank IC by default.

## 1. Input protocol

### 1.1 Payload fields

| Field | Meaning |
| --- | --- |
| `case_id`, `task`, `task_variant` | Identifiers; echo them nowhere in the response. |
| `track` | `numeric_only` or `semantic`; see the appended track block. |
| `feature_columns` | Names of the anonymous numeric factor features. Use them verbatim. |
| `factor_cards` | Semantic track only: one sanitized card per factor id with the neutral name, definition, formula and formula-input descriptions. Empty on `numeric_only`. |
| `train` | Public labelled history as `rows`, with an optional runner-added `artifact` descriptor. |
| `validation` | Public labelled history in the same row format as `train`, with an optional `artifact`. |
| `predict` | `{schema, label_included: false, rows_visible_to_llm: false}`. |
| `metadata` | Public runtime facts such as `seed`, window sizes and refit policy. The executable function receives sanitized `factor_cards` through runtime `metadata`; the provider payload keeps cards at the top level. |

When embedded, `train.rows` and `validation.rows` are arrays whose objects have
exactly the ordered keys `["id", "time_key", *feature_columns, "label"]`.
An artifact descriptor's public `columns` list is that same ordered list. Their
in-prompt copy is size-capped, so use the public rows materialized by
`python_exec` (§1.4) rather than relying on the chat copy. The agentic runner
may add an artifact descriptor `{path, format, rows, columns, bytes}` and mount
that public copy for the tool; the raw on-disk payload can instead contain
embedded rows.

### 1.2 Panel geometry

Historic rows are numbered backwards from the current formation month: a row at
cutoff `t-k` observes factor values as of that cutoff and carries the return
realised at `t-(k-1)`. So the newest historical row `t-1` already has a settled
label, while the `predict_df` row at `t-0` is labelled only after the next month
settles.

The three frames are rebuilt at every origin. In the default protocol the 12
oldest labelled cutoff months form `train_df` and the 12 most recent form
`valid_df`; other runs use the same chronological rule with different block
lengths. `predict_df` contains only the current `t-0` row; earlier months are
represented by the historical frames and may also be summarized inside a
rolling feature value, never as extra prediction rows. Never assume that a row
from a later month is available at an earlier origin.

### 1.3 Validation timeline and data quality

Validation is chronological and leave-future-out: for the validation month `j`
being scored, `valid_df` contains only labels that had already settled before
`j`. The first scored month therefore receives an empty `valid_df` intentionally:
there is no earlier validation label available yet; the evaluator still scores
that month against its held-out label outside the function. Handle an empty or
very short `valid_df` explicitly; never assume a non-empty validation frame.

Factor features may be missing, constant, or infinite. Sanitize every frame —
train, validation and predict — before fitting, and never pass NaN/inf to an
estimator. Use `metadata["seed"]` as your only randomness source so a run is
reproducible, and do not mutate the input frames.

### 1.4 Getting the rows

`python_exec` sees the whole public payload as `payload`. The runner mounts any
declared public history and makes the rows available in the payload before the
tool code runs. Prefer this format-independent access pattern:

```python
train = pd.DataFrame(payload["train"]["rows"])
valid = pd.DataFrame(payload["validation"]["rows"])
```

When the runner has staged an artifact, its descriptor path is a case-relative
mount hint and is readable only from that tool workspace; do not assume it is a
path in the raw case directory or hard-code a filename. These materialized
copies hold exactly the public train/validation rows, so no test data is
reachable through them.

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
*instance*, not a schema definition; the per-turn instance appended after this
prompt is authoritative):

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
  belongs in `train_columns` and `validation_columns` only.
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
  unique `id` column and a finite numeric `score`. Do not reorder, drop, or
  duplicate ids.
- Fit preprocessing, feature selection, hyperparameters and model parameters
  using only the supplied historical frames. Refitting on `train_df` plus
  `valid_df` before scoring `predict_df` is allowed; if there is no tunable
  choice, fit on the concatenated historical rows directly.
- Never read future labels, the hidden test panel, or calendar dates. The code
  must not contain calendar literals (a four-digit year such as `2016`, with or
  without a month suffix) and must not branch on absolute dates; express every
  window as a relative `time_key` offset.
- Do not implement a portfolio, a backtest, or any transaction-cost logic — the
  evaluator converts your scores into factors' Top-3 inverse-volatility routing
  and settles the return itself.

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
4. `MANDATORY RESPONSE INSTANCE FOR THIS TURN` — a payload-specific instance of
   the §2 object with the exact feature names. Replace its placeholders and
   return that object.
5. After the last code revision, a `CANDIDATE SELECTION TURN` may replace (4)
   with a two-key object `{selected_revision, selection_reason}`.

The revision number belongs to this revision protocol. It is independent of
payload metadata and always takes precedence over a value remembered from an
earlier request.

### 3.2 Feedback envelope

The envelope contains only public information: `revision`,
`candidate_status`, `validation` (`rank_ic_mean`, `rank_ic_std`, `mse`,
`directional_accuracy`, `n_months`, `n_rows`), `io_checks.execution_valid`,
`runtime.seconds`, at most 20 `errors`, `allowed_next_revision`, and
`previous_candidate` — the previous revision's public fields and source.

Read it to (a) understand the aggregate effect of the previous change, and
(b) repair: `previous_candidate` is repair context so you can keep the contract
fields and source that were already correct, fix every listed error, and change
only what the evidence supports. It is not a test signal and never contains
hidden rows or labels.

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
mounted Parquet files; `read_artifact` and `write_artifact` move files. Use
tools to check shapes, missingness, label distributions, and feature behaviour
before committing to an architecture. `MAX_CODE_CHARS` is 120000, so a full
model definition fits. If a turn's instructions say tools are unavailable,
finalize immediately with the JSON object.

## 4. Style and hygiene

Keep `explanation` and `revision_summary` short, factual and auditable. Never
include row-level labels, hidden targets, file paths, tool traces, credentials,
or private reasoning in the response. Comment the source where a choice is not
self-evident — sanitization, refit policy, regularization — and leave the
modelling decisions to the code rather than to prose.

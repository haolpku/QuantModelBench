# Evaluation protocol

## Unit of evaluation

A case specifies a task, temporal anchor, historical context length, and feature
subset. The model sees permitted history and a task contract and chooses its own
modeling procedure. Numeric and Semantic are paired information conditions.

1. Develop candidates using historical training/validation data and bounded tools.
2. Return the complete code response required by the task's JSON schema.
3. Select an execution-valid candidate and record its source hash.
4. Refit the same source at 12 monthly origins using only then-permitted history.
5. Evaluate predictions and portfolios outside the generated modeling procedure.

Frozen source does not imply frozen parameter values or a fixed feature subset:
a fixed data-dependent selection rule can choose different features after refitting.
Previously hidden labels may enter later fitting windows only when observable.

## Information conditions

Numeric exposes masked identifiers, relative times, and numerical history.
Semantic adds neutral names, definitions/formulas and input descriptions. Cards
exclude explicit direction, significance, theme labels and paper provenance.
Descriptions may still convey economic context. The intervention changes both
information and context length; it is not a perfectly length-matched treatment.

## Cases and splits

Task B has seven anchors, three history lengths (2/3/5 years), and 14 subsets
(eight with 38 factors, five with 76, and one with 153): 294 cases, 84 dev and
210 test. Task C has 207 cases, 84 dev and 123 test, with test blocks including
1960-12, 2008-12 and 2022-12. Use manifests for exact membership and window sizes;
do not infer a balanced 5 x 3 x 14 design for Task C.

Dev/test case membership is distinct from the hidden rolling months inside each
case. Development-case rolling performance is not a final test-set result. Task C
test anchors are not all chronologically later than development anchors.

## Metrics and baselines

Keep execution validity, ranking quality, and portfolio outcomes separate.
Primary baselines are `momentum_12` (B) and `rolling_ridge` (C).
`jkp_signed_equal_weight` is a domain-informed reference, not a same-information
baseline. Use common valid support for paired performance comparisons and retain
all scheduled cases in reliability denominators. Never substitute a zero return
for a failed execution. Shared blocks/subsets require dependence-aware uncertainty.

The imported rubric and `dev_frozen_v1.json` preserve the historical scoring
configuration. Deferred judge/champion fields are not completed experimental
evidence; do not derive new thresholds from test outcomes.

## Reproducibility

Record dataset revision and case IDs, payload/source hashes, prompts and model
endpoints, decoding settings, seeds, tool and revision budgets, execution limits,
dependency versions, BLAS threads, hardware and candidate-selection policy.
Single-case CLI defaults differ between tasks; a run configuration must explicitly
set the intended protocol (for example, `--max-revisions 2`).

Masking and frozen evaluation do not establish absence of pretraining contamination.
Task B's complete-data eligibility spans hidden intervals and may induce coverage
selection. These are limitations to measure, not properties fixed by packaging.

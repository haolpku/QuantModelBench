"""Registry of run roots evaluated by this package.

A `RunSpec` is the unit the rubric scores: one (task, kind, method, track)
combination backed by a directory or a results file on disk.  Adding a new
model / prompt / revision-strategy track means appending one `RunSpec` here;
nothing else in the package hard-codes a path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def repo_root() -> Path:
    """Repository root, resolved from this file's location."""
    return Path(__file__).resolve().parents[3]


# Layout tags tell `loaders` how to parse a run root.
LAYOUT_TASK_B_LLM = "task_b_llm"          # <root>/<block>/<case>/rolling_result/
LAYOUT_TASK_B_BASELINE = "task_b_baseline_jsonl"   # results.jsonl, one row per (case, method)
LAYOUT_TASK_C_DIRS = "task_c_dirs"        # <root>/<block>/<case>/ with summary.json etc.


@dataclass(frozen=True)
class RunSpec:
    """One scored track."""

    key: str
    task: str            # "task_b" | "task_c"
    kind: str            # "llm" | "baseline"
    method: str          # model tag or baseline name
    track: str | None    # "numeric_only" | "semantic" | None for baselines
    root: Path
    layout: str
    codegen_root: Path | None = None   # Task C keeps codegen artefacts separately
    baseline_method: str | None = None  # row filter for LAYOUT_TASK_B_BASELINE
    note: str = ""
    coverage_note: str = ""             # non-empty => partial coverage, excluded from macro
    tags: tuple[str, ...] = field(default_factory=tuple)
    # Which split the run's cases belong to.  Drives the family-key join
    # (`loaders.attach_family`) and nothing else: thresholds are never derived
    # here, so a test spec is scored with the dev-frozen config unchanged.
    split: str = "dev"

    @property
    def is_llm(self) -> bool:
        return self.kind == "llm"

    @property
    def full_coverage(self) -> bool:
        return not self.coverage_note


# --------------------------------------------------------------------------
# Dataset roots (case manifests + evaluator truth)
# --------------------------------------------------------------------------

TASK_B_DATASET = Path("data/processed/task_b_usa_frozen_raw_rolling_294")

# Task C now keeps dev and test in a single build, the same way Task B does.
# The 207-case build is the only one: it carries the 1960-12 test block plus
# every dev case, so both splits resolve here.
#
# The build was merged from the former 168-case dataset, which the 207 build
# strictly supersets -- payload.json, rolling_truth.jsonl and both evaluator
# maps are byte-identical for all 84 shared dev case IDs.  Only bookkeeping
# changed: 1960-12 sorts first, so blocks are now numbered block_00..block_04
# and the dev blocks moved from block_02/block_03 to block_01/block_02.  Run
# roots and frozen thresholds recorded against the old 168 paths still resolve
# by case ID, but any path recorded verbatim against `..._168` is stale.
TASK_C_DATASET = Path("data/processed/task_c_frozen_raw_rolling_207")
# Test split: 123 cases over anchors 1960-12 / 2008-12 / 2022-12.
TASK_C_TEST_DATASET = Path("data/processed/task_c_frozen_raw_rolling_207")
# Evaluator panel.  Cut to cover every anchor in the 207-case build -- the two
# dev anchors and the three test anchors -- with a year of slack outside the
# span the case manifests require.  Built by
# `task_c/code/scripts/build_task_c_evaluator_panel.py`, which verifies
# per-case coverage before writing.  Keep the evaluator on the canonical
# processed-data path; old server staging copies are disposable.
TASK_C_PANEL = Path("data/processed/task_c_evaluator_panel/panel.parquet")


def dataset_for(task: str, split: str = "dev") -> Path:
    """Dataset root that holds `split` for `task`.

    Both tasks keep development and test cases in their canonical build.
    """
    if task == "task_b":
        return TASK_B_DATASET
    if task != "task_c":
        raise ValueError(f"unknown task: {task}")
    return TASK_C_TEST_DATASET if split == "test" else TASK_C_DATASET

# --------------------------------------------------------------------------
# Full-dev run roots (84 cases per track)
# --------------------------------------------------------------------------

_B_LLM = Path("task_b/runs/dev/llm/deepseek_v4_pro_fulldev")
# Task B baseline results live under one root per split.  Dev keeps the
# historical path; test was run 2026-09-27 over the 210-case holdout.
_B_BASE = {
    "dev": Path("task_b/runs/dev/baselines/20260915_usa_dev_84_cases/results.jsonl"),
    "test": Path("task_b/runs/test/baselines/20260927_usa_test_210_cases/results.jsonl"),
}
_C_LLM = Path("task_c/runs/dev/llm/deepseek_v4_pro_fulldev")
_C_BASE_LINEAR = Path("task_c/runs/dev/baselines/20260911_dev_84")
_C_BASE_MLP = Path("task_c/runs/dev/baselines/20260910_dev_mlp")
# Task C test baselines.  Unlike Task B, which keeps both splits in one
# results.jsonl, Task C baselines are one directory per (method, block, case),
# so dev and test need separate roots.  Same six methods as dev, so the test
# registry mirrors the dev one exactly.
_C_BASE_TEST = Path("task_c/runs/test/baselines/20260927_usa_test_123_cases")
_C_BASE_TEST_METHODS = {
    "rolling_ridge": _C_BASE_TEST / "rolling_ridge",
    "rolling_elastic_net": _C_BASE_TEST / "rolling_elastic_net",
    "rolling_mlp": _C_BASE_TEST / "rolling_mlp",
    "rolling_mlp_deep": _C_BASE_TEST / "rolling_mlp_deep",
    "rolling_mlp_ensemble": _C_BASE_TEST / "rolling_mlp_ensemble",
    "jkp_signed_equal_weight": _C_BASE_TEST / "jkp_signed_equal_weight",
}

# --------------------------------------------------------------------------
# Local-model LLM runs (13-case reference plus canonical full-dev output)
# --------------------------------------------------------------------------

# Self-hosted Qwen3.8-27B, 84-case full-dev run (2026-09-26).  Same directory
# layout as the deepseek roots, so no new layout tag is needed.
#
# The 13-case v2 run (2026-09-25) is not registered and no longer exists
# locally: it was deleted 2026-09-28 under a "keep only final-protocol
# results" rule, leaving the server copies (`frozen_tracks_v1_qwen3_8_27b_v2_`)
# as the sole record.  Its Task B inputs came from a different dataset build
# and its Task C inputs were identical but served under a shorter client
# timeout, so scoring it beside the full-dev run would have mixed two
# configurations in one table.
_QWEN = "qwen3_8_27b"
# qwen's 5 unrecovered Task B cases are genuine model failures, not transport
# drops: numeric_only 83/84, semantic 80/84 = 163/168.  They are left absent
# rather than zero-filled, so `R_case` on a manifest denominator must count
# them as failures (see `QWEN_ABSENT_CASES` and the two-denominator rule in
# `process.reliability`).  This is full coverage in the sense that matters for
# threshold derivation, so it is NOT a partial-coverage run.
_QWEN_NOTE = ""
QWEN_ABSENT_CASES = {
    "task_b": {
        "numeric_only": ("case_230cf512b0232e04",),
        "semantic": (
            "case_0100922498302aa8",
            "case_375333223e289825",
            "case_5e5211e5e9dda0b3",
            "case_f192130a0e719acd",
        ),
    },
}
_QWEN_B = Path("task_b/runs/dev/llm/qwen3_8_27b_fulldev")
_QWEN_C = Path("task_c/runs/dev/llm/qwen3_8_27b_fulldev")
_QWEN_C_NOTE = ""

TASK_B_BASELINE_METHODS = (
    "momentum_12",
    "t_stat_12",
    "t_stat_36",
    "skip_month_momentum",
    "reversal_1",
    "equal_weight",
    "inverse_vol",
    "downside_vol",
    "logistic",
    "elastic_net",
    "tree",
    "mlp",
    "mlp_deep",
    "mlp_ensemble",
)

TASK_C_BASELINE_METHODS = {
    "rolling_ridge": _C_BASE_LINEAR / "rolling_ridge",
    "rolling_elastic_net": _C_BASE_LINEAR / "rolling_elastic_net",
    "rolling_mlp": _C_BASE_MLP / "rolling_mlp",
    "rolling_mlp_deep": _C_BASE_MLP / "rolling_mlp_deep",
    "rolling_mlp_ensemble": _C_BASE_MLP / "rolling_mlp_ensemble",
    # Rationale: the five rolling baselines above collapse out-of-sample
    # (OOS Rank IC ~0.00 vs a validation Rank IC of 0.04-0.07), which dragged
    # the Task C `oos_rank_ic` full-mark line down to 0.0087 and saturated the
    # metric for every LLM track.  The JKP signed equal-weight composite is
    # fit-free and no-lookahead (directions come from the official JKP factor
    # definitions, not from returns), so it has no validation-overfitting
    # failure mode and anchors the full-mark line at a meaningful level.
    # It is a privileged calibration/reference anchor: it uses the official
    # factor directions, which the Task C prompt never exposes.  It therefore
    # belongs in the frozen threshold/Active-IR reference set, but must stay
    # out of model-vs-baseline uplift pairing against an LLM track.
    "jkp_signed_equal_weight": _C_BASE_LINEAR / "jkp_signed_equal_weight",
}

# Baselines named by the rubric.  `frozen_set` is what dev thresholds are
# derived from; it must not change without re-freezing the config.
PRIMARY_BASELINE = {"task_b": "momentum_12", "task_c": "rolling_ridge"}
SECONDARY_BASELINE = {"task_b": "t_stat_12", "task_c": "rolling_mlp_ensemble"}
FROZEN_BASELINE_SET = {
    "task_b": TASK_B_BASELINE_METHODS,
    "task_c": tuple(TASK_C_BASELINE_METHODS),
}


def _key_prefix(task: str, split: str) -> str:
    """Run-key namespace for a split.

    Dev keys stay exactly as they were (`task_b/baseline/momentum_12`): every
    frozen threshold, the saved rubric output and the report all reference them,
    and renaming them would orphan that record.  Test keys get a suffix so both
    splits can sit in one frame without colliding.
    """
    return task if split == "dev" else f"{task}/{split}"


def baseline_key(task: str, method: str, split: str = "dev") -> str:
    """Run key of a frozen baseline within a split's namespace.

    Callers that look up `task_b/baseline/momentum_12` by hand break the moment
    the key gains a `/{split}` suffix, so the construction lives here and both
    `build_registry` and the scoring path go through it.
    """
    return f"{_key_prefix(task, split)}/baseline/{method}"


def build_registry(root: Path | None = None, split: str = "dev") -> list[RunSpec]:
    """Run roots for one split.

    Defaults to dev, so every existing caller keeps its exact behaviour.  A
    test registry differs in three ways and no others: the baseline results
    root points at the test run, keys carry a `/{split}` suffix, and each spec
    records `split` so `attach_family` joins the matching manifest.  Thresholds
    are NOT touched -- they stay dev-frozen, which is the whole point.
    """
    base = root or repo_root()
    specs: list[RunSpec] = []
    prefix = _key_prefix("task_b", split)
    if split not in _B_BASE:
        raise ValueError(f"no Task B baseline root for split {split!r}; known: {sorted(_B_BASE)}")

    # LLM tracks are registered for dev only.  Every LLM run root on disk is a
    # dev run, and each spec's `root` is a literal dev path -- registering them
    # into the test registry would silently re-read dev and double-count it.
    # When a test LLM run lands, add its spec here behind the same `split`
    # guard rather than reusing these roots.
    if split == "dev":
        for track in ("numeric_only", "semantic"):
            specs.append(
                RunSpec(
                    key=f"task_b/llm/deepseek_v4_pro/{track}",
                    task="task_b",
                    kind="llm",
                    method="deepseek_v4_pro",
                    track=track,
                    root=base / _B_LLM / track,
                    layout=LAYOUT_TASK_B_LLM,
                    tags=("full_dev",),
                )
            )
            specs.append(
                RunSpec(
                    key=f"task_c/llm/deepseek_v4_pro/{track}",
                    task="task_c",
                    kind="llm",
                    method="deepseek_v4_pro",
                    track=track,
                    root=base / _C_LLM / track / "rolling",
                    layout=LAYOUT_TASK_C_DIRS,
                    codegen_root=base / _C_LLM / track / "codegen",
                    tags=("full_dev",),
                )
            )

        for track in ("numeric_only", "semantic"):
            specs.append(
                RunSpec(
                    key=f"task_b/llm/{_QWEN}/{track}",
                    task="task_b",
                    kind="llm",
                    method=_QWEN,
                    track=track,
                    root=base / _QWEN_B / track,
                    layout=LAYOUT_TASK_B_LLM,
                    note="self-hosted vLLM, reasoning_effort=xhigh",
                    coverage_note=_QWEN_NOTE,
                    tags=("full_dev", "local_model", "has_absent_cases"),
                )
            )
            specs.append(
                RunSpec(
                    key=f"task_c/llm/{_QWEN}/{track}",
                    task="task_c",
                    kind="llm",
                    method=_QWEN,
                    track=track,
                    root=base / _QWEN_C / track / "rolling",
                    layout=LAYOUT_TASK_C_DIRS,
                    codegen_root=base / _QWEN_C / track / "codegen",
                    note="self-hosted vLLM, reasoning_effort=xhigh",
                    coverage_note=_QWEN_C_NOTE,
                    tags=("full_dev", "local_model"),
                )
            )

    for method in TASK_B_BASELINE_METHODS:
        specs.append(
            RunSpec(
                key=baseline_key("task_b", method, split),
                task="task_b",
                kind="baseline",
                method=method,
                track=None,
                root=base / _B_BASE[split],
                layout=LAYOUT_TASK_B_BASELINE,
                baseline_method=method,
                tags=("frozen_baseline",),
                split=split,
            )
        )

    c_roots = TASK_C_BASELINE_METHODS if split == "dev" else _C_BASE_TEST_METHODS
    for method, rel in c_roots.items():
        specs.append(
            RunSpec(
                key=baseline_key("task_c", method, split),
                task="task_c",
                kind="baseline",
                method=method,
                track=None,
                root=base / rel,
                layout=LAYOUT_TASK_C_DIRS,
                tags=("frozen_baseline",),
                split=split,
            )
        )

    return specs


def registry_by_key(root: Path | None = None, split: str = "dev") -> dict[str, RunSpec]:
    return {s.key: s for s in build_registry(root, split)}

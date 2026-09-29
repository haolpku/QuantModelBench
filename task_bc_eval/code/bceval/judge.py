"""LLM-as-Judge framework: prompt construction, output schema, agreement.

This module deliberately contains **no network client**.  It builds the batch
and parses/aggregates results; issuing the calls is a separate, explicit step
so that a judge run can never happen as a side effect of scoring.

Two invariants from rubric 4.3 are enforced here rather than left to the
prompt: the judge sees no outcome, baseline or ranking information, and a
failed real audit overrides any judge score.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

JUDGE_PROMPT_VERSION = "task_bc_judge_prompt.v1"
JUDGE_PASSES = 2

DIMENSIONS: tuple[tuple[str, str], ...] = (
    ("contract_adherence",
     "Does the code fully honour the declared input/output contract, entrypoint and evaluation protocol?"),
    ("point_in_time_leakage",
     "Are train/validation/predict boundaries explicit, with no future labels or future-derived statistics?"),
    ("statistical_soundness",
     "Are preprocessing, fitting and feature selection all confined to data visible at that point?"),
    ("numerical_robustness",
     "Are NaNs, empty frames, constant columns, extreme values and singular matrices handled?"),
    ("generalization",
     "Does it avoid hard-coded column counts, case-specific constants and hidden shape assumptions?"),
    ("code_clarity",
     "Is the structure clear, the complexity justified, and are there no dead or duplicated computations?"),
    ("reproducibility",
     "Are randomness, ordering, state and side effects controlled so a rerun reproduces the result?"),
)

SCORE_MIN, SCORE_MAX = 0, 4

OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["scores", "rationales"],
    "additionalProperties": False,
    "properties": {
        "scores": {
            "type": "object",
            "required": [name for name, _ in DIMENSIONS],
            "additionalProperties": False,
            "properties": {
                name: {"type": "integer", "minimum": SCORE_MIN, "maximum": SCORE_MAX}
                for name, _ in DIMENSIONS
            },
        },
        "rationales": {
            "type": "object",
            "required": [name for name, _ in DIMENSIONS],
            "additionalProperties": False,
            "properties": {name: {"type": "string", "maxLength": 600} for name, _ in DIMENSIONS},
        },
        "blocking_concerns": {"type": "array", "items": {"type": "string"}},
    },
}

SYSTEM_PROMPT = """You are reviewing the research quality of code written for a \
quantitative forecasting task. You score the code and the research process only.

You are NOT told, and must not attempt to infer, how the code performed: no \
backtest result, no Rank IC, no baseline comparison and no ranking is available \
to you. Do not reward code for "looking like" a winner.

Score each of the seven dimensions on an integer 0-4 scale, where 4 fully meets \
the stated standard and 0 fails it outright. Return JSON conforming exactly to \
the provided schema. Every score needs a one-or-two-sentence rationale citing \
something concrete in the code."""


@dataclass
class JudgeItem:
    """One (case, pass) unit of work."""

    item_id: str
    run_key: str
    task: str
    track: str | None
    case_id: str
    judge_pass: int
    prompt_version: str
    code_sha256: str
    system_prompt: str
    user_prompt: str

    def to_dict(self) -> dict:
        return asdict(self)


def _dimension_block() -> str:
    return "\n".join(f"{i}. {name}: {desc}" for i, (name, desc) in enumerate(DIMENSIONS, 1))


def build_user_prompt(task: str, contract: dict, code: str) -> str:
    """Judge input: contract + code only (rubric 4.3.1 forbids outcome leakage)."""
    task_note = {
        "task_b": "Predict next-month factor returns; the host selects a Top-3 factor portfolio from the scores.",
        "task_c": "Produce a cross-sectional stock score; the host builds a long-short portfolio from the scores.",
    }[task]
    return (
        f"## Task contract\n{task_note}\n\n"
        f"Entrypoint: {contract.get('entrypoint', 'fit_and_predict')}\n"
        f"Declared output columns: {contract.get('output_columns')}\n"
        f"Declared requirements: {contract.get('requirements')}\n"
        f"Number of feature columns supplied: {contract.get('n_feature_columns')}\n\n"
        f"## Scoring dimensions\n{_dimension_block()}\n\n"
        f"## Submitted code\n```python\n{code}\n```\n"
    )


def build_item(run_key: str, task: str, track: str | None, case_id: str,
               code: str, contract: dict, judge_pass: int) -> JudgeItem:
    code_hash = hashlib.sha256((code or "").encode()).hexdigest()
    return JudgeItem(
        item_id=f"{run_key}|{case_id}|pass{judge_pass}",
        run_key=run_key,
        task=task,
        track=track,
        case_id=case_id,
        judge_pass=judge_pass,
        prompt_version=JUDGE_PROMPT_VERSION,
        code_sha256=code_hash,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=build_user_prompt(task, contract, code or ""),
    )


def write_batch(items: list[JudgeItem], path) -> int:
    with open(path, "w") as fh:
        for item in items:
            fh.write(json.dumps(item.to_dict()) + "\n")
    return len(items)


# --------------------------------------------------------------------------
# Result aggregation (used once a judge run exists)
# --------------------------------------------------------------------------


def parse_results(records: list[dict]) -> pd.DataFrame:
    """Normalise raw judge output into one row per (case, pass) with `j` in [0,1]."""
    rows = []
    for rec in records:
        scores = rec.get("scores") or {}
        vals = [scores.get(name) for name, _ in DIMENSIONS]
        if any(v is None for v in vals):
            continue
        j_raw = float(np.mean(vals))
        rows.append(
            {
                "run_key": rec.get("run_key"),
                "case_id": rec.get("case_id"),
                "judge_pass": rec.get("judge_pass"),
                "j_raw": j_raw,
                "j": j_raw / SCORE_MAX,
                "blocking_concerns": len(rec.get("blocking_concerns") or []),
                **{f"dim_{name}": scores[name] for name, _ in DIMENSIONS},
            }
        )
    return pd.DataFrame(rows)


def agreement(scores: pd.DataFrame) -> dict:
    """Exact and within-one agreement between the two independent passes."""
    if scores is None or not len(scores):
        return {"available": False, "reason": "no_judge_results"}
    wide = scores.pivot_table(index=["run_key", "case_id"], columns="judge_pass",
                              values=[f"dim_{n}" for n, _ in DIMENSIONS])
    passes = sorted({p for _, p in wide.columns})
    if len(passes) < 2:
        return {"available": False, "reason": "single_pass_only"}
    a, b = passes[0], passes[1]
    exact, within1, total = 0, 0, 0
    for name, _ in DIMENSIONS:
        left = wide[(f"dim_{name}", a)]
        right = wide[(f"dim_{name}", b)]
        ok = left.notna() & right.notna()
        diff = (left[ok] - right[ok]).abs()
        exact += int((diff == 0).sum())
        within1 += int((diff <= 1).sum())
        total += int(ok.sum())
    per_case = scores.groupby(["run_key", "case_id"])["j"].agg(["mean", "std"])
    return {
        "available": True,
        "exact_agreement": exact / total if total else float("nan"),
        "within_one_agreement": within1 / total if total else float("nan"),
        "n_dimension_pairs": total,
        "mean_j": float(per_case["mean"].mean()),
        "mean_within_case_sd": float(per_case["std"].mean(skipna=True)),
    }


def apply_audit_override(scores: pd.DataFrame, failed_case_ids: dict[str, list[str]]) -> pd.DataFrame:
    """Rubric 4.3.3: a real audit failure is not cancelled by a high judge score.

    The case keeps its judge detail for inspection but is excluded from the
    Process judge component via `audit_override`.
    """
    if scores is None or not len(scores):
        return scores
    out = scores.copy()
    out["audit_override"] = [
        case_id in failed_case_ids.get(run_key, [])
        for run_key, case_id in zip(out["run_key"], out["case_id"])
    ]
    return out

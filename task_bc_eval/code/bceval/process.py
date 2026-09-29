"""Process axis: Reliability, Stability, LLM-as-Judge and Efficiency.

    P = 0.30 R + 0.15 S + 0.30 J + 0.25 E     (rubric 2.2)

Every sub-component may be absent.  Absent means *renormalise*, never
zero-fill: a track that simply did not persist token usage must not be scored
as if it had burned infinite tokens.  The dropped components travel with the
score so a reader can see which denominator produced it.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .loaders import EXPECTED_ORIGINS
from .metrics import clip01, nan_iqr, three_threshold_score, weighted_score

PROCESS_WEIGHTS = {"reliability": 0.30, "stability": 0.15, "judge": 0.30, "efficiency": 0.25}
RELIABILITY_WEIGHTS = {"case": 0.60, "origin": 0.40}
EFFICIENCY_WEIGHTS = {"cost": 0.45, "valid": 0.25, "retry": 0.15, "diversity": 0.15}

# The origin-level series whose dispersion defines Stability for each task.
STABILITY_METRIC = {"task_b": "rank_ic", "task_c": "rank_ic"}


# --------------------------------------------------------------------------
# Reliability
# --------------------------------------------------------------------------


def strict_completion(cases: pd.DataFrame, task: str) -> pd.Series:
    """Rubric 4.1 strict completion, evaluated per case.

    A case counts only if the frozen code executed on every origin, the
    evaluator terminated normally and the economic leg actually traded.  A
    case that produced no tradable month is a failure, not a zero return.
    """
    ok = cases.get("run_complete", pd.Series(True, index=cases.index)).fillna(False).astype(bool)
    origin_total = cases.get("origin_total", pd.Series(np.nan, index=cases.index))
    origin_valid = cases.get("origin_valid", pd.Series(np.nan, index=cases.index))
    ok &= (origin_valid.fillna(0) >= origin_total.fillna(np.inf))

    if task == "task_b":
        ok &= cases.get("rank_ic", pd.Series(np.nan, index=cases.index)).notna()
        ok &= cases.get("net_annual_return", pd.Series(np.nan, index=cases.index)).notna()
    else:
        ok &= cases.get("tradable_months", pd.Series(0, index=cases.index)).fillna(0) > 0
        ok &= cases.get("oos_rank_ic", pd.Series(np.nan, index=cases.index)).notna()
        ok &= cases.get("net_annual_return", pd.Series(np.nan, index=cases.index)).notna()
    return ok


def reliability(cases: pd.DataFrame, task: str,
                attempted_case_ids: Sequence[str] | None = None) -> dict:
    """Rubric 4.1 Reliability, on the *attempted* denominator.

    Two denominators, deliberately:

    * **Outcome** (everything downstream of this function) is computed on the
      case set the runs being compared actually share.  Scoring a model on
      cases its rival never ran would compare different questionnaires.
    * **Reliability** is computed on every case the track was *asked* for.
      It is the axis that measures failure, so narrowing its denominator to
      the survivors measures the wrong thing -- and does so asymmetrically,
      because the run that failed more cases has the smaller denominator.

    The bug this fixes is concrete: with the matching-only denominator, qwen's
    Task C track reported `R_case = 100.0` despite 7 no-candidate cases, because
    the 7 missing cases were absent from the frame entirely.  Task B happened to
    escape it only because its failures still leave a directory behind.

    `attempted_case_ids` is the manifest-level roster.  Cases in it but not in
    `cases` are injected as failed rows carrying `EXPECTED_ORIGINS` origins,
    none valid, so both denominators are fixed by the split rather than by how
    much of it survived: `R_case` is over the roster's case count (84 on dev)
    and `R_origin` over `EXPECTED_ORIGINS` times that count (1008 on dev).
    Neither touches any Outcome number.
    """
    scored = cases
    absent_ids: list[str] = []
    if attempted_case_ids is not None:
        present = set(cases["case_id"]) if len(cases) else set()
        absent_ids = [cid for cid in attempted_case_ids if cid not in present]
        if absent_ids:
            absent = pd.DataFrame({"case_id": absent_ids})
            absent["run_complete"] = False
            absent["origin_valid"] = 0
            # The denominator is the roster, not the survivors.  `nan` here
            # used to drop out of the `.fillna(0)` sum below, so an absent case
            # added 0 to both sides of `R_origin` and escaped the axis that
            # exists to measure failure -- qwen's Task C tracks scored a
            # perfect origin rate with three cases that never produced a
            # candidate.  `loaders` already settled the rule for the milder
            # version of this (a case directory with no audit is charged
            # `EXPECTED_ORIGINS` origins, none of them valid); a case with no
            # directory at all is charged the same way.
            absent["origin_total"] = float(EXPECTED_ORIGINS)
            for frame in (scored,):
                for col in cases.columns:
                    if col not in frame.columns:
                        frame[col] = np.nan
            scored = pd.concat([cases, absent], ignore_index=True)

    complete = strict_completion(scored, task)
    n_cases = int(len(scored))
    r_case = float(complete.sum()) / n_cases if n_cases else float("nan")

    origin_valid = float(scored.get("origin_valid", pd.Series(dtype=float)).fillna(0).sum())
    origin_total = float(scored.get("origin_total", pd.Series(dtype=float)).fillna(0).sum())
    r_origin = origin_valid / origin_total if origin_total else float("nan")

    score = weighted_score({"case": r_case, "origin": r_origin}, RELIABILITY_WEIGHTS)
    return {
        "score": score.value,
        "detail": score.to_dict(),
        "r_case": r_case,
        "r_origin": r_origin,
        "n_cases": n_cases,
        "n_cases_strict": int(complete.sum()),
        "n_cases_absent": len(absent_ids),
        "absent_case_ids": absent_ids,
        "denominator": "attempted_manifest" if attempted_case_ids is not None else "scored_only",
        "n_origin_valid": int(origin_valid),
        "n_origin_total": int(origin_total),
        "failed_case_ids": scored.loc[~complete, "case_id"].tolist(),
    }


def failure_taxonomy(cases: pd.DataFrame, task: str) -> dict[str, int]:
    """Why cases failed, so failures are visible rather than averaged away."""
    complete = strict_completion(cases, task)
    failed = cases.loc[~complete]
    tally: dict[str, int] = {}
    for _, row in failed.iterrows():
        if not bool(row.get("run_complete", False)):
            reason = str(row.get("failure_reason") or "run_incomplete")
        elif task == "task_c" and not (row.get("tradable_months") or 0):
            # More specific than "invalid_origin", which it always implies: the
            # frozen code ran but never produced a portfolio that could trade.
            reason = "no_tradable_month"
        elif (row.get("origin_valid") or 0) < (row.get("origin_total") or 0):
            reason = "invalid_origin"
        elif pd.isna(row.get("oos_rank_ic" if task == "task_c" else "rank_ic")):
            reason = "missing_prediction_metric"
        else:
            reason = "missing_economic_metric"
        tally[reason] = tally.get(reason, 0) + 1
    return tally


# --------------------------------------------------------------------------
# Stability
# --------------------------------------------------------------------------


def stability(origins: pd.DataFrame, cases: pd.DataFrame, a_s: float, task: str) -> dict:
    """`S = 1 - clip(IQR(origin metric) / a_S, 0, 1)`, averaged over cases.

    Also reports family-level dispersion, which answers a different question:
    whether performance hinges on one data subset or history length.
    """
    metric = STABILITY_METRIC[task]
    if origins is None or not len(origins) or metric not in origins.columns:
        return {"score": float("nan"), "a_s": a_s, "n_cases": 0, "median_iqr": float("nan")}

    per_case = origins.groupby("case_id")[metric].apply(nan_iqr)
    if not np.isfinite(a_s) or a_s <= 0:
        return {"score": float("nan"), "a_s": a_s, "n_cases": int(per_case.notna().sum()),
                "median_iqr": float(per_case.median(skipna=True))}

    scores = [1.0 - clip01(v / a_s) for v in per_case if np.isfinite(v)]
    family_sd = float("nan")
    if "family_id" in cases.columns:
        target = "oos_rank_ic" if task == "task_c" else "rank_ic"
        if target in cases.columns:
            fam = cases.groupby("family_id")[target].mean()
            family_sd = float(fam.std(ddof=1)) if fam.notna().sum() > 1 else float("nan")

    return {
        "score": float(np.mean(scores)) if scores else float("nan"),
        "a_s": float(a_s),
        "n_cases": len(scores),
        "median_iqr": float(per_case.median(skipna=True)),
        "family_dispersion_sd": family_sd,
        "metric": metric,
    }


# --------------------------------------------------------------------------
# Efficiency
# --------------------------------------------------------------------------


def efficiency(cases: pd.DataFrame, cost_scale: dict | None) -> dict:
    """`E = 0.45 cost + 0.25 valid + 0.15 retry + 0.15 diversity`.

    Cost uses a log scale so a 10x spend is not flattened by a linear one
    (rubric 4.4).  When `c_good` / `c_bad` are null -- Task B persisted no token
    usage this round -- the cost term is dropped and the remaining three are
    renormalised over 0.55, with `cost_available: false` attached.
    """
    e_cost, cost_detail = _cost_component(cases, cost_scale)

    valid = cases.get("valid_candidate_rate", pd.Series(dtype=float))
    e_valid = float(valid.mean(skipna=True)) if len(valid) else float("nan")

    # E_retry is the *first-pass* validation rate: the share of cases whose very
    # first revision already validated, before any repair round.
    #
    # It used to be `1 - revisions_used / max_revisions`, which was identically
    # zero for every LLM track: `revision_count` is 3.0 and `max_revisions` is 2
    # on all four tracks, so the ratio clipped to 1.0 everywhere and the term
    # contributed a constant 0 to E -- a 15% weight spent on nothing.  Every
    # track runs the full revision budget, so revision *count* carries no
    # signal; whether the first attempt was any good does.
    first_pass = cases.get("first_revision_valid", pd.Series(dtype=float)).astype(float)
    e_retry = float(first_pass.mean(skipna=True)) if len(first_pass.dropna()) else float("nan")

    distinct = cases.get("distinct_code_hashes", pd.Series(dtype=float)).astype(float)
    n_rev = cases.get("revision_count", pd.Series(dtype=float)).astype(float)
    div = (distinct / n_rev.replace(0, np.nan)).clip(0, 1)
    e_div = float(div.mean(skipna=True)) if len(div) else float("nan")

    score = weighted_score(
        {"cost": e_cost, "valid": e_valid, "retry": e_retry, "diversity": e_div},
        EFFICIENCY_WEIGHTS,
    )
    fallback = cases.get("selection_fallback", pd.Series(dtype=bool)).astype("boolean")
    return {
        "score": score.value,
        "detail": score.to_dict(),
        "cost_available": bool(np.isfinite(e_cost)),
        "cost_detail": cost_detail,
        "e_valid": e_valid,
        "e_retry": e_retry,
        "e_retry_definition": "share of cases whose first revision passed validation",
        "e_diversity": e_div,
        "selection_fallback_rate": float(fallback.fillna(False).astype(bool).mean()) if len(fallback) else float("nan"),
        "total_tokens": float(cases.get("total_tokens", pd.Series(dtype=float)).sum(skipna=True)),
        "tool_calls": float(cases.get("tool_call_count", pd.Series(dtype=float)).sum(skipna=True)),
        "wall_clock_seconds": float(
            cases.get("codegen_runtime_seconds", pd.Series(dtype=float)).sum(skipna=True)
        ),
    }


def _cost_component(cases: pd.DataFrame, cost_scale: dict | None) -> tuple[float, dict]:
    if not cost_scale:
        return float("nan"), {"reason": "cost_scale_not_frozen"}
    c_good = cost_scale.get("c_good")
    c_bad = cost_scale.get("c_bad")
    if c_good is None or c_bad is None:
        return float("nan"), {"reason": "cost_scale_null", "c_good": c_good, "c_bad": c_bad}

    tokens = cases.get("total_tokens", pd.Series(dtype=float)).astype(float)
    tokens = tokens[np.isfinite(tokens) & (tokens > 0)]
    if tokens.empty:
        return float("nan"), {"reason": "no_usage_recorded", "c_good": c_good, "c_bad": c_bad}

    span = np.log(c_bad) - np.log(c_good)
    if span <= 0:
        return float("nan"), {"reason": "degenerate_cost_scale"}
    scores = [(np.log(c_bad) - np.log(t)) / span for t in tokens]
    return float(np.mean([clip01(s) for s in scores])), {
        "c_good": float(c_good),
        "c_bad": float(c_bad),
        "n_cases_with_usage": int(len(tokens)),
        "n_cases_total": int(len(cases)),
        "median_tokens": float(tokens.median()),
    }


# --------------------------------------------------------------------------
# Judge (score is supplied externally; see judge.py)
# --------------------------------------------------------------------------


def judge_component(judge_scores: pd.DataFrame | None) -> dict:
    """Fold judge output into Process, honouring the audit-overrides-judge rule."""
    if judge_scores is None or not len(judge_scores):
        return {"score": float("nan"), "available": False,
                "reason": "judge_not_run_this_round", "n_cases": 0}
    per_case = judge_scores.groupby("case_id")["j"].mean()
    return {
        "score": float(per_case.mean()),
        "available": True,
        "n_cases": int(len(per_case)),
        "n_passes": int(judge_scores.groupby("case_id").size().median()),
    }


# --------------------------------------------------------------------------
# Process Score
# --------------------------------------------------------------------------


def process_score(reliability_score: float, stability_score: float,
                  judge_score: float, efficiency_score: float):
    return weighted_score(
        {
            "reliability": reliability_score,
            "stability": stability_score,
            "judge": judge_score,
            "efficiency": efficiency_score,
        },
        PROCESS_WEIGHTS,
    )

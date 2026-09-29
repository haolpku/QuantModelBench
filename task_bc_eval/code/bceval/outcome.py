"""Outcome axis: Absolute Quality, Baseline Uplift and the Outcome Score.

    O = 0.60 * Q_abs + 0.40 * Q_uplift        (rubric 2.2)

Scores are computed per case and then case-macro averaged, matching the
aggregation rule in rubric 1.2: aggregate within a case first, then across
cases.  Raw metrics are reported alongside every score so that a 0-100 number
never travels without the evidence behind it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import stats
from .metrics import three_threshold_score, weighted_score

OUTCOME_WEIGHTS = {"absolute_quality": 0.60, "baseline_uplift": 0.40}
UPLIFT_WEIGHTS = {"magnitude": 0.50, "improved_rate": 0.30, "active_ir": 0.20}


@dataclass(frozen=True)
class MetricSpec:
    """One Absolute-Quality metric.

    `column` is a case_metrics column already oriented so that higher is
    better.  Max drawdown is stored signed-negative, so -2% > -5% holds without
    a sign flip, and `delta = model - baseline` is exactly the rubric's
    `|MDD_baseline| - |MDD_model|`.
    """

    name: str
    column: str
    weight: float
    null_method: str
    label: str


# Rubric 3.1.2 -- Sharpe and MDD share the 10% risk slot, split evenly.
TASK_B_ABS: tuple[MetricSpec, ...] = (
    MetricSpec("rank_ic", "rank_ic", 0.35, "rank_perm", "rolling Rank IC"),
    MetricSpec("top3_hit_rate", "top3_hit_rate", 0.20, "top3_perm", "Top-3 hit rate"),
    MetricSpec("accuracy", "accuracy", 0.10, "binom", "Directional accuracy"),
    MetricSpec("net_annual_return", "net_annual_return", 0.25, "random_top3", "Net annual return"),
    MetricSpec("sharpe", "sharpe", 0.05, "random_top3", "Sharpe"),
    MetricSpec("max_drawdown", "max_drawdown", 0.05, "baseline_quantile", "Max drawdown"),
)

# Rubric 3.1.3
TASK_C_ABS: tuple[MetricSpec, ...] = (
    MetricSpec("oos_rank_ic", "oos_rank_ic", 0.40, "rank_perm", "rolling OOS Rank IC"),
    MetricSpec("net_annual_return", "net_annual_return", 0.25, "sign_flip", "Net annual return"),
    MetricSpec("sharpe", "sharpe", 0.20, "sign_flip", "Sharpe"),
    MetricSpec("max_drawdown", "max_drawdown", 0.10, "baseline_quantile", "Max drawdown"),
    MetricSpec("coverage", "coverage", 0.05, "baseline_quantile", "Coverage"),
)

# Rubric 3.2.2 lists Delta Rank IC / return / Sharpe / MDD (+ coverage for C).
TASK_B_UPLIFT_WEIGHTS = {
    "rank_ic": 0.40,
    "net_annual_return": 0.35,
    "sharpe": 0.15,
    "max_drawdown": 0.10,
}
TASK_C_UPLIFT_WEIGHTS = {
    "oos_rank_ic": 0.40,
    "net_annual_return": 0.30,
    "sharpe": 0.20,
    "max_drawdown": 0.05,
    "coverage": 0.05,
}


def abs_specs(task: str) -> tuple[MetricSpec, ...]:
    return TASK_B_ABS if task == "task_b" else TASK_C_ABS


def uplift_weights(task: str) -> dict[str, float]:
    return TASK_B_UPLIFT_WEIGHTS if task == "task_b" else TASK_C_UPLIFT_WEIGHTS


def spec_by_name(task: str, name: str) -> MetricSpec:
    for spec in abs_specs(task):
        if spec.name == name:
            return spec
    raise KeyError(name)


# --------------------------------------------------------------------------
# Absolute Quality
# --------------------------------------------------------------------------


def score_absolute_quality(cases: pd.DataFrame, thresholds: dict, task: str) -> pd.DataFrame:
    """Per-case Q_m and Q_abs.

    A case missing a metric (a runtime failure, an untradable stretch) gets NaN
    for that metric and has its weight renormalised away -- it is never scored
    as if the metric had been measured at zero.
    """
    out = cases.copy()
    specs = abs_specs(task)
    weights = {s.name: s.weight for s in specs}

    for spec in specs:
        th = thresholds.get(spec.name, {})
        tau_noise = th.get("tau_noise", np.nan)
        tau_target = th.get("tau_target", np.nan)
        tau_effect = th.get("tau_effect", np.nan)
        values = out[spec.column] if spec.column in out.columns else pd.Series(np.nan, index=out.index)
        out[f"q_{spec.name}"] = [three_threshold_score(v, tau_noise, tau_target) for v in values]
        out[f"above_effect_{spec.name}"] = [
            bool(np.isfinite(v) and np.isfinite(tau_effect) and v >= tau_effect) if np.isfinite(v)
            else None
            for v in values
        ]

    rows = []
    for _, row in out.iterrows():
        comp = {s.name: row.get(f"q_{s.name}") for s in specs}
        ws = weighted_score(comp, weights)
        rows.append({"q_abs": ws.value, "q_abs_weight_used": ws.weight_used,
                     "q_abs_dropped": ",".join(ws.dropped)})
    return pd.concat([out.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def above_effect_rates(scored: pd.DataFrame, task: str) -> dict[str, dict]:
    """`AboveEffectRate` per metric: share of cases clearing tau_effect."""
    result = {}
    for spec in abs_specs(task):
        col = f"above_effect_{spec.name}"
        if col not in scored.columns:
            continue
        series = scored[col].dropna()
        result[spec.name] = {
            "rate": float(series.mean()) if len(series) else float("nan"),
            "n_above": int(series.sum()) if len(series) else 0,
            "n_evaluated": int(len(series)),
        }
    return result


# --------------------------------------------------------------------------
# Baseline Uplift
# --------------------------------------------------------------------------


def score_uplift(
    model_cases: pd.DataFrame,
    baseline_cases: pd.DataFrame,
    model_monthly: pd.DataFrame | None,
    baseline_monthly: pd.DataFrame | None,
    frozen: dict,
    task: str,
) -> dict:
    """Q_uplift = 0.50 magnitude + 0.30 ImprovedRate + 0.20 ActiveIR.

    `(M - B) / |B|` is deliberately absent: it blows up whenever a baseline
    sits near zero (Hyndman & Koehler 2006), which several Task C baselines do.
    """
    scales = frozen.get("uplift_scales", {}).get(task, {})
    weights = uplift_weights(task)

    per_metric: dict[str, dict] = {}
    mag_components: dict[str, float | None] = {}
    rate_components: dict[str, float | None] = {}

    for name, _weight in weights.items():
        spec = spec_by_name(task, name)
        scale = scales.get(name, {}).get("a_m", np.nan)
        epsilon = scales.get(name, {}).get("epsilon_m", np.nan)
        paired = stats.paired_delta(model_cases, baseline_cases, spec.column)
        if paired.empty:
            per_metric[name] = {"n_pairs": 0}
            mag_components[name] = None
            rate_components[name] = None
            continue

        deltas = paired["delta"].to_numpy()
        magnitude = stats.uplift_magnitude(deltas, scale)
        rate = stats.improved_rate(deltas, epsilon)
        per_metric[name] = {
            "n_pairs": int(len(paired)),
            "mean_delta": float(np.mean(deltas)),
            "median_delta": float(np.median(deltas)),
            "win_cases": int((deltas > 0).sum()),
            "a_m": None if not np.isfinite(scale) else float(scale),
            "epsilon_m": None if not np.isfinite(epsilon) else float(epsilon),
            "u_magnitude_raw": None if not np.isfinite(magnitude) else float(magnitude),
            **rate,
        }
        # magnitude lives on [-1, 1]; map onto the [0, 1] score scale.
        mag_components[name] = (magnitude + 1.0) / 2.0 if np.isfinite(magnitude) else None
        rate_components[name] = rate["improved_rate"]

    magnitude_score = weighted_score(mag_components, weights)
    rate_score = weighted_score(rate_components, weights)

    ir = _active_ir_by_case(model_monthly, baseline_monthly)
    ir_target = frozen.get("active_ir", {}).get(task, {}).get("tau_target", np.nan)
    ir_values = [v for v in ir.values() if np.isfinite(v)]
    ir_point = float(np.mean(ir_values)) if ir_values else float("nan")
    ir_score = three_threshold_score(ir_point, 0.0, ir_target)

    uplift = weighted_score(
        {"magnitude": magnitude_score.value, "improved_rate": rate_score.value, "active_ir": ir_score},
        UPLIFT_WEIGHTS,
    )

    return {
        "q_uplift": uplift.to_dict(),
        "magnitude": magnitude_score.to_dict(),
        "improved_rate": rate_score.to_dict(),
        "active_ir": {
            "score": None if not np.isfinite(ir_score) else round(ir_score, 6),
            "mean_active_ir": None if not np.isfinite(ir_point) else round(ir_point, 6),
            "tau_target": None if not np.isfinite(ir_target) else round(float(ir_target), 6),
            "n_cases": len(ir_values),
        },
        "per_metric": per_metric,
        "active_ir_by_case": ir,
    }


def _active_ir_by_case(model_monthly: pd.DataFrame | None,
                       baseline_monthly: pd.DataFrame | None) -> dict[str, float]:
    if model_monthly is None or baseline_monthly is None:
        return {}
    if not len(model_monthly) or not len(baseline_monthly):
        return {}
    out: dict[str, float] = {}
    base_by_case = {cid: sub for cid, sub in baseline_monthly.groupby("case_id", sort=False)}
    for case_id, sub in model_monthly.groupby("case_id", sort=False):
        base = base_by_case.get(case_id)
        if base is None:
            continue
        out[case_id] = stats.active_ir(sub, base)["active_ir"]
    return out


# --------------------------------------------------------------------------
# Outcome Score
# --------------------------------------------------------------------------


def outcome_score(q_abs: float, q_uplift: float):
    return weighted_score(
        {"absolute_quality": q_abs, "baseline_uplift": q_uplift}, OUTCOME_WEIGHTS
    )


def to_100(x: float | None) -> float | None:
    if x is None or not np.isfinite(x):
        return None
    return round(100.0 * float(x), 2)

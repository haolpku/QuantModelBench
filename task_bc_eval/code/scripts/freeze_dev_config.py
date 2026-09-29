#!/usr/bin/env python3
"""Stage 2: derive every frozen threshold, scale and weight from dev, once.

Nulls are *per case*, because `Q_m` is applied per case: the question a noise
floor answers here is "what could one case reach with no signal", not "what
could the 84-case mean reach".  Draws are pooled across cases and cut at the
95th percentile.

    python task_bc_eval/code/scripts/freeze_dev_config.py
    python task_bc_eval/code/scripts/freeze_dev_config.py --refreeze-cost-scale task_b
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bceval import frozen, judge, loaders, metrics, outcome, process, runspec, stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs"
CONFIG_PATH = ROOT / "config" / "dev_frozen_v1.json"

NULL_DRAWS_PER_CASE = 200
NULL_QUANTILE = 95.0
EPSILON_FRACTION = 0.10      # practical equivalence margin, as a share of a_m
# Percentiles the cost scale used to be cut at (kept for the record; the
# scale itself is now permanently null -- see `build_cost_scale`).
COST_GOOD_Q, COST_BAD_Q = 20.0, 90.0
ACTIVE_IR_FALLBACK_TARGET = 0.50   # pre-registered MDE when no rival baseline beats the primary


def _case_macro(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return float("nan")
    return float(frame.groupby("case_id")[column].mean().mean())


# --------------------------------------------------------------------------
# tau_noise
# --------------------------------------------------------------------------


def _two_level(draws: list[np.ndarray]) -> dict:
    """Both noise floors from the same per-case null draws.

    `case_level` is the 95th percentile of what one case can reach with no
    signal; `macro` is the 95th percentile of the case-macro mean of the same
    nulls.  Scoring uses `macro`, because the reported point estimate is the
    case-macro mean; `case_level` is kept in the config so the report can say
    how far single-case evidence is from being conclusive.
    """
    if not draws:
        return {"macro": float("nan"), "case_level": float("nan"), "n_cases": 0}
    width = min(len(d) for d in draws)
    stacked = np.vstack([d[:width] for d in draws])
    return {
        "macro": stats.quantile(stacked.mean(axis=0), NULL_QUANTILE),
        "case_level": stats.quantile(stacked.reshape(-1), NULL_QUANTILE),
        "n_cases": len(draws),
    }


def _noise_rank_perm(origins: pd.DataFrame, run_key: str) -> dict:
    """Permutation null of a case's mean Rank IC."""
    sub = origins[origins["run_key"] == run_key]
    draws = []
    for _, grp in sub.groupby("case_id"):
        sizes = [int(n) for n in grp["n_predictions"].dropna()]
        d = stats.null_rank_ic(sizes, n_draws=NULL_DRAWS_PER_CASE)
        if d.size:
            draws.append(d)
    return _two_level(draws)


def _noise_binom(origins: pd.DataFrame, run_key: str) -> dict:
    sub = origins[origins["run_key"] == run_key]
    draws = []
    for _, grp in sub.groupby("case_id"):
        sizes = [int(n) for n in grp["n_predictions"].dropna()]
        d = stats.null_accuracy(sizes, n_draws=NULL_DRAWS_PER_CASE)
        if d.size:
            draws.append(d)
    return _two_level(draws)


def _noise_top3(truth: pd.DataFrame) -> dict:
    """Random-selection control: hypergeometric draws against each origin's own
    positive-return base rate (rubric 3.1.2 asks for exactly this comparison)."""
    pos = (
        truth.assign(pos=truth["ret"] > 0)
        .groupby(["case_id", "holding_month"])
        .agg(frac=("pos", "mean"), n=("pos", "size"))
        .reset_index()
    )
    draws = []
    for _, grp in pos.groupby("case_id"):
        d = stats.null_top3_hit(grp["frac"].tolist(), grp["n"].tolist(),
                                n_draws=NULL_DRAWS_PER_CASE)
        if d.size:
            draws.append(d)
    return _two_level(draws)


def _noise_random_top3(truth: pd.DataFrame, n_draws: int = 60) -> dict[str, dict]:
    rets, shps = [], []
    for _, grp in truth.groupby("case_id"):
        d = stats.null_task_b_random_top3(grp, n_draws=n_draws)
        rets.append(d["annual_return"])
        shps.append(d["sharpe"])
    return {"net_annual_return": _two_level(rets), "sharpe": _two_level(shps)}


def _noise_sign_flip(monthly: pd.DataFrame, run_key: str) -> dict[str, dict]:
    """Volatility-preserving no-edge null, calibrated on the primary baseline."""
    sub = monthly[monthly["run_key"] == run_key]
    rets, shps = [], []
    for _, grp in sub.groupby("case_id"):
        d = stats.null_sign_flip(grp["net_return"].to_numpy(), n_draws=NULL_DRAWS_PER_CASE)
        if d["annual_return"].size:
            rets.append(d["annual_return"])
            shps.append(d["sharpe"])
    return {"net_annual_return": _two_level(rets), "sharpe": _two_level(shps)}


def _noise_baseline_quantile(base_cases: pd.DataFrame, column: str) -> dict:
    """For metrics with no meaningful permutation null (MDD, coverage): the 10th
    percentile of the frozen baseline set's per-case values.  There is no
    sampling null here, so both levels are the same number."""
    value = stats.quantile(base_cases[column].tolist(), 10.0) if column in base_cases else float("nan")
    return {"macro": value, "case_level": value, "n_cases": int(len(base_cases))}


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


def build_thresholds(task, cases, origins, monthly, truth) -> tuple[dict, dict]:
    baseline_names = list(runspec.FROZEN_BASELINE_SET[task])
    base = cases[(cases["task"] == task) & (cases["kind"] == "baseline")]
    base_keys = {m: f"{task}/baseline/{m}" for m in baseline_names}
    primary_key = base_keys[runspec.PRIMARY_BASELINE[task]]
    ref_llm = cases[(cases["task"] == task) & (cases["kind"] == "llm")]["run_key"].iloc[0]

    # Per-baseline case-macro values: the target / effect anchors.
    macro = {
        m: {s.name: _case_macro(base[base["run_key"] == key], s.column)
            for s in outcome.abs_specs(task)}
        for m, key in base_keys.items()
    }

    if task == "task_b":
        econ = _noise_random_top3(truth)
        noise = {
            "rank_ic": _noise_rank_perm(origins, ref_llm),
            "accuracy": _noise_binom(origins, ref_llm),
            "top3_hit_rate": _noise_top3(truth),
            "net_annual_return": econ["net_annual_return"],
            "sharpe": econ["sharpe"],
            "max_drawdown": _noise_baseline_quantile(base, "max_drawdown"),
        }
    else:
        econ = _noise_sign_flip(monthly, primary_key)
        noise = {
            "oos_rank_ic": _noise_rank_perm(origins, ref_llm),
            "net_annual_return": econ["net_annual_return"],
            "sharpe": econ["sharpe"],
            "max_drawdown": _noise_baseline_quantile(base, "max_drawdown"),
            "coverage": _noise_baseline_quantile(base, "coverage"),
        }

    thresholds, scales = {}, {}
    for spec in outcome.abs_specs(task):
        vals = [macro[m][spec.name] for m in baseline_names
                if np.isfinite(macro[m].get(spec.name, np.nan))]
        noise_node = noise.get(spec.name) or {}
        tau_noise = float(noise_node.get("macro", np.nan))
        tau_noise_case = float(noise_node.get("case_level", np.nan))
        tau_target = float(np.max(vals)) if vals else float("nan")
        tau_effect = float(np.median(vals)) if vals else float("nan")
        # A target below the noise floor would invert the score; fall back to
        # the observed dev maximum across every run instead of silently
        # producing a decreasing scale.
        target_source = "dev_baseline_max"
        if np.isfinite(tau_noise) and np.isfinite(tau_target) and tau_target <= tau_noise:
            observed = cases[cases["task"] == task].groupby("run_key")[spec.column].mean()
            tau_target = float(max(observed.max(), tau_noise * 1.5 + 1e-6))
            target_source = ("dev_baseline_max_below_noise_floor -> "
                             "max(best dev run, 1.5 x tau_noise)")
        thresholds[spec.name] = {
            "tau_noise": tau_noise,
            "tau_noise_case_level": tau_noise_case,
            "tau_effect": tau_effect,
            "tau_target": tau_target,
            "noise_source": spec.null_method,
            "noise_unit": "p95 of the case-macro null (case-level p95 reported alongside)",
            "effect_source": "dev_baseline_median",
            "target_source": target_source,
            "baselines_used": baseline_names,
        }
        # Uplift scale: how far apart baselines are from each other on dev.
        per_case = base.groupby("case_id")[spec.column].mean() if spec.column in base else pd.Series(dtype=float)
        a_m = float(metrics.nan_iqr(per_case.tolist())) if len(per_case) else float("nan")
        scales[spec.name] = {
            "a_m": a_m,
            "epsilon": a_m * EPSILON_FRACTION if np.isfinite(a_m) else float("nan"),
            "source": "dev_frozen_baseline_case_iqr",
            "epsilon_rule": f"{EPSILON_FRACTION:g} x a_m",
        }
    return thresholds, scales, macro


def build_stability_scale(task, origins, cases) -> dict:
    """a_S: the dev spread of origin-level dispersion, cut at the 90th
    percentile so only genuinely erratic runs are pushed to S = 0."""
    metric = process.STABILITY_METRIC[task]
    keys = cases[(cases["task"] == task) & (cases["kind"] == "baseline")]["run_key"].unique()
    sub = origins[origins["run_key"].isin(keys)]
    if metric not in sub.columns or sub.empty:
        sub = origins[origins["run_key"].isin(
            cases[cases["task"] == task]["run_key"].unique())]
    iqrs = [metrics.nan_iqr(g[metric].tolist()) for _, g in sub.groupby(["run_key", "case_id"])]
    value = stats.quantile(iqrs, 90.0)
    return {"a_s": value, "metric": metric, "source": "dev_origin_iqr_p90",
            "n_cases": int(len(iqrs))}



def build_active_ir_target(task, cases, monthly) -> dict:
    """What an *existing alternative method* achieves against the primary
    baseline on dev.  Anchoring the ActiveIR score to the 90th percentile of
    that distribution keeps a full mark meaning "better than the best rival
    baseline", not an arbitrary round number."""
    primary = f"{task}/baseline/{runspec.PRIMARY_BASELINE[task]}"
    base_monthly = monthly[monthly["run_key"] == primary]
    rivals = [f"{task}/baseline/{m}" for m in runspec.FROZEN_BASELINE_SET[task]
              if f"{task}/baseline/{m}" != primary]
    values = []
    for key in rivals:
        rival = monthly[monthly["run_key"] == key]
        if rival.empty or base_monthly.empty:
            continue
        by_case = {cid: sub for cid, sub in base_monthly.groupby("case_id", sort=False)}
        for case_id, sub in rival.groupby("case_id", sort=False):
            base = by_case.get(case_id)
            if base is None:
                continue
            values.append(stats.active_ir(sub, base)["active_ir"])
    target = stats.quantile(values, 90.0)
    source = "dev_rival_baseline_active_ir_p90"
    # On Task B the primary baseline dominates every rival, so the rival
    # distribution is entirely negative and cannot anchor a "full marks"
    # target.  Fall back to a pre-registered fixed IR rather than inverting
    # the scale.
    if not np.isfinite(target) or target <= 0:
        target, source = ACTIVE_IR_FALLBACK_TARGET, "fixed_preregistered_mde"
    return {
        "tau_target": float(target),
        "tau_noise": 0.0,
        "source": source,
        "rival_p90": stats.quantile(values, 90.0),
        "n_pairs": int(len([v for v in values if np.isfinite(v)])),
    }


def build_cost_scale(task, cases, defer: bool) -> dict | None:
    """Always None: the cost term is out of the rubric (decision 2A).

    It used to be frozen per task from the dev LLM token distribution.  It is
    not any more, for two reasons that both survive more data:

    * The scale is not comparable across models.  deepseek's Task C median sits
      at 810k-1,334k tokens per case; the self-hosted qwen track sits at
      172k-307k.  A `c_good`/`c_bad` pair frozen on one model turns the other's
      cost term into a constant -- the same failure `e_retry` had.
    * Token accounting is only present on some runs.  Task B has usage on 1 of
      168 deepseek cases; a scale frozen on that is noise, and the frozen
      fallback ("score it as absent") already meant the term was usually
      dropped anyway.

    Cost is reported instead: see `render_cost_table` in `compute_rubric.py`,
    which prints tokens / tool calls / wall clock per track with no weight.
    `E` therefore renormalises over the remaining 0.55
    (valid 0.25 + retry 0.15 + diversity 0.15).
    """
    del task, cases, defer
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=CONFIG_PATH)
    parser.add_argument("--in-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--refreeze-cost-scale", default=None,
                        help="recompute only this task's c_good/c_bad in an existing config")
    args = parser.parse_args()

    cases = pd.read_parquet(args.in_dir / "case_metrics.parquet")
    origins = pd.read_parquet(args.in_dir / "origin_metrics.parquet")
    monthly = pd.read_parquet(args.in_dir / "monthly_returns.parquet")

    if args.refreeze_cost_scale:
        task = args.refreeze_cost_scale
        cfg = frozen.load(args.out, require_valid=False)
        scale = build_cost_scale(task, cases, defer=False)
        if scale is None:
            print(f"no token usage available for {task}; nothing refrozen", file=sys.stderr)
            return 1
        frozen.set_cost_scale(cfg, task, scale["c_good"], scale["c_bad"], scale["source"])
        frozen.save(cfg, args.out)
        print(f"refroze {task} cost scale: c_good={scale['c_good']:.0f} c_bad={scale['c_bad']:.0f}")
        return 0

    truth = loaders.load_task_b_truth()
    cfg = frozen.new_config()
    cfg["baselines"] = {
        task: {
            "primary": runspec.PRIMARY_BASELINE[task],
            "secondary": runspec.SECONDARY_BASELINE[task],
            "frozen_set": list(runspec.FROZEN_BASELINE_SET[task]),
        }
        for task in ("task_b", "task_c")
    }
    cfg["baselines"]["task_c"]["privileged_reference"] = ["jkp_signed_equal_weight"]

    macro_all = {}
    for task in ("task_b", "task_c"):
        th, sc, macro = build_thresholds(task, cases, origins, monthly, truth)
        cfg["thresholds"][task] = th
        metric_names = list(sc)
        cfg["epsilon"][task] = {name: sc[name]["epsilon"] for name in metric_names}
        cfg["scales"][task] = sc
        cfg["scales"][task]["stability"] = build_stability_scale(task, origins, cases)
        cost = build_cost_scale(task, cases, defer=True)
        cfg["scales"][task]["cost"] = cost
        if cost is None:
            cfg["deferred"][f"scales.{task}.cost"] = frozen.ALLOWED_DEFERRALS[f"scales.{task}.cost"]
        cfg.setdefault("uplift_scales", {})[task] = {
            name: {"a_m": sc[name]["a_m"], "epsilon_m": sc[name]["epsilon"]}
            for name in metric_names
        }
        cfg.setdefault("active_ir", {})[task] = build_active_ir_target(task, cases, monthly)
        macro_all[task] = macro

    cfg["weights"] = {
        "outcome": dict(outcome.OUTCOME_WEIGHTS),
        "uplift": dict(outcome.UPLIFT_WEIGHTS),
        "process": dict(process.PROCESS_WEIGHTS),
        "reliability": dict(process.RELIABILITY_WEIGHTS),
        "efficiency": dict(process.EFFICIENCY_WEIGHTS),
        "task_b_absolute": {s.name: s.weight for s in outcome.TASK_B_ABS},
        "task_c_absolute": {s.name: s.weight for s in outcome.TASK_C_ABS},
        "task_b_uplift_metrics": dict(outcome.TASK_B_UPLIFT_WEIGHTS),
        "task_c_uplift_metrics": dict(outcome.TASK_C_UPLIFT_WEIGHTS),
    }
    cfg["aggregation"] = {
        "point_estimate": "case_macro_mean",
        "robustness": ["median", "iqm"],
        "ci": {"method": "family_block_bootstrap", "n_boot": stats.DEFAULT_DRAWS,
               "alpha": 0.05, "seed": stats.DEFAULT_SEED},
        "family_key": {
            "task_b": ["block_id", "anchor_month", "sample_size", "subset_index"],
            "task_c": ["block_id", "anchor_month", "sample_size", "subset_index"],
        },
        "origin_rule": "origins build case metrics and time-series deltas; never weighted as cases",
        "failure_rule": "runtime failures, untradable months and invalid origins are never zero-filled",
    }
    cfg["judge"] = {
        "prompt_version": judge.JUDGE_PROMPT_VERSION,
        "dimensions": [name for name, _ in judge.DIMENSIONS],
        "scale": [judge.SCORE_MIN, judge.SCORE_MAX],
        "passes": judge.JUDGE_PASSES,
        "model": None,
        "temperature": None,
        "agreement_threshold": None,
        "audit_override": "a failed real audit overrides any judge score (rubric 4.3.3)",
        "status": "framework_frozen_not_run",
    }
    cfg["deferred"]["judge.agreement_threshold"] = frozen.ALLOWED_DEFERRALS["judge.agreement_threshold"]
    cfg["deferred"]["champion_thresholds.p_min"] = frozen.ALLOWED_DEFERRALS["champion_thresholds.p_min"]
    cfg["deferred"]["champion_thresholds.o_min"] = frozen.ALLOWED_DEFERRALS["champion_thresholds.o_min"]

    cfg["search_policy"] = {
        "max_revisions": int(pd.to_numeric(cases["max_revisions"], errors="coerce").max()),
        "selection_policy": sorted(set(cases["selection_policy"].dropna())),
        "multi_baseline": True,
        "multi_prompt": ["numeric_only", "semantic"],
        "llm_calls_during_rolling": 0,
        "note": "code is frozen before rolling; all 12 origins refit with zero LLM calls",
    }
    cfg["multiplicity"] = {
        "families": 28,
        "primary_comparisons": [
            "llm vs primary baseline (per task)",
            "semantic vs numeric_only (paired treatment effect)",
        ],
        "control": "report family-block bootstrap CIs on every headline delta; "
                   "no per-case baseline selection; no post-hoc threshold choice",
        "reference": "White (2000), Hansen (2005)",
    }
    cfg["dev_baseline_macro"] = macro_all

    frozen.save(cfg, args.out)
    report = frozen.validate(cfg)
    print(f"wrote {args.out}")
    print(f"valid={report['ok']}  errors={report['errors']}")
    for item in report["deferred"]:
        print(f"  deferred: {item}")
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

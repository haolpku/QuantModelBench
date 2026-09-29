#!/usr/bin/env python3
"""Stage 3: score every run against the frozen dev config.

Emits `outputs/rubric_scores.json` (machine-readable, every sub-component and
every dropped component named) and `outputs/rubric_tables.md` (the tables the
report embeds).  No threshold is derived here -- everything numeric comes from
`config/dev_frozen_v1.json`.

    python task_bc_eval/code/scripts/compute_rubric.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bceval import (frozen, loaders, metrics, outcome, process, report,  # noqa: E402
                    runspec, stats)

ROOT = Path(__file__).resolve().parents[2]
# Dev reads `outputs/`, test reads `outputs_test/`.  Both are read with the same
# frozen config: the test run is scored once, with dev-frozen thresholds, and
# nothing on that path may rebind a threshold (see `frozen.validate`).
OUT_DIRS = {"dev": ROOT / "outputs", "test": ROOT / "outputs_test"}
OUT_DIR = OUT_DIRS["dev"]
CONFIG_PATH = ROOT / "config" / "dev_frozen_v1.json"

HEADLINE = {
    "task_b": ["rank_ic", "accuracy", "top3_hit_rate", "net_annual_return", "sharpe", "max_drawdown"],
    "task_c": ["oos_rank_ic", "validation_rank_ic", "net_annual_return", "sharpe",
               "max_drawdown", "coverage"],
}


def macro_table(cases: pd.DataFrame, task: str) -> pd.DataFrame:
    """case-macro mean + median + IQM + family-block CI for the headline metrics."""
    rows = []
    if cases.empty:
        return pd.DataFrame(columns=["run_key", "n_cases"])
    for run_key, sub in cases.groupby("run_key", sort=True):
        row = {"run_key": run_key, "n_cases": int(sub["case_id"].nunique())}
        for col in HEADLINE[task]:
            if col not in sub.columns:
                continue
            per_case = sub.groupby("case_id")[col].mean()
            fam = sub.groupby("case_id")["family_id"].first()
            row[col] = float(per_case.mean()) if len(per_case.dropna()) else np.nan
            row[f"{col}__median"] = float(per_case.median()) if len(per_case.dropna()) else np.nan
            row[f"{col}__iqm"] = metrics.iqm(per_case.tolist())
            ci = stats.family_block_bootstrap_ci(per_case.tolist(), fam.reindex(per_case.index).tolist())
            row[f"{col}__lo"], row[f"{col}__hi"] = ci["lo"], ci["hi"]
            row[f"{col}__n"] = int(per_case.notna().sum())
        rows.append(row)
    return pd.DataFrame(rows)


def _for_run(frame: pd.DataFrame, run_key: str) -> pd.DataFrame:
    """Rows of `frame` for one run, tolerating a frame with no columns at all.

    A baselines-only split has no origin table; an empty frame has no `run_key`
    to slice on, so it is returned as-is and every consumer downstream already
    treats "no origins" as "S dropped".
    """
    if frame is None or not len(frame.columns) or "run_key" not in frame.columns:
        return pd.DataFrame()
    return frame[frame["run_key"] == run_key]


def score_run(run_key, task, cases, origins, monthly, cfg, judge_scores=None,
              attempted_case_ids=None, split="dev") -> dict:
    spec_cases = cases[cases["run_key"] == run_key]
    kind = spec_cases["kind"].iloc[0]
    thresholds = cfg["thresholds"][task]

    scored = outcome.score_absolute_quality(spec_cases, thresholds, task)
    q_abs_per_case = scored.groupby("case_id")["q_abs"].mean()
    q_abs = float(q_abs_per_case.mean()) if len(q_abs_per_case.dropna()) else np.nan
    fam = scored.groupby("case_id")["family_id"].first().reindex(q_abs_per_case.index)
    q_abs_ci = stats.family_block_bootstrap_ci(q_abs_per_case.tolist(), fam.tolist())

    # Built from the split's namespace, never by hand: on test the baseline keys
    # carry a `/test` infix, and a literal dev key silently yields an empty
    # pairing -- every uplift component drops and Q_uplift reads as absent
    # rather than as zero.
    primary = runspec.baseline_key(task, runspec.PRIMARY_BASELINE[task], split)
    secondary = runspec.baseline_key(task, runspec.SECONDARY_BASELINE[task], split)
    uplifts = {}
    for label, base_key in (("primary", primary), ("secondary", secondary)):
        if base_key == run_key:
            continue
        uplifts[label] = outcome.score_uplift(
            spec_cases,
            cases[cases["run_key"] == base_key],
            _for_run(monthly, run_key),
            _for_run(monthly, base_key),
            cfg, task,
        )
        uplifts[label]["baseline"] = base_key
        uplifts[label].pop("active_ir_by_case", None)

    q_uplift = uplifts.get("primary", {}).get("q_uplift", {}).get("value", np.nan)
    o = outcome.outcome_score(q_abs, q_uplift)

    # Reliability counts every case the track was asked for, Outcome counts
    # only the shared scored set -- see `process.reliability`.
    rel = process.reliability(spec_cases, task, attempted_case_ids=attempted_case_ids)
    stab = process.stability(_for_run(origins, run_key), spec_cases,
                             cfg["scales"][task]["stability"]["a_s"], task)
    eff = process.efficiency(spec_cases, cfg["scales"][task].get("cost"))
    jud = process.judge_component(judge_scores)
    p = process.process_score(rel["score"], stab["score"], jud.get("score"), eff["score"])

    # A frozen baseline has no generation process: no candidates, no revisions,
    # no tokens.  Scoring it on the Process axis would hand it a free 100 built
    # from Reliability alone, so the axis is marked not-applicable instead.
    process_applicable = kind == "llm"
    o_val = o.value
    p_val = p.value if process_applicable else float("nan")
    return {
        "run_key": run_key,
        "task": task,
        "kind": kind,
        "n_cases": int(spec_cases["case_id"].nunique()),
        "outcome": {
            "q_abs": outcome.to_100(q_abs),
            "q_abs_ci": [outcome.to_100(q_abs_ci["lo"]), outcome.to_100(q_abs_ci["hi"])],
            "q_abs_components": {
                s.name: outcome.to_100(scored[f"q_{s.name}"].mean()) for s in outcome.abs_specs(task)
            },
            "above_effect": outcome.above_effect_rates(scored, task),
            "q_uplift": outcome.to_100(q_uplift),
            "outcome_score": outcome.to_100(o_val),
            "dropped": list(o.dropped),
        },
        "uplift": uplifts,
        "process": {
            "reliability": outcome.to_100(rel["score"]),
            "reliability_detail": rel,
            "stability": outcome.to_100(stab["score"]),
            "stability_detail": stab,
            "judge": outcome.to_100(jud.get("score")),
            "judge_detail": jud,
            "efficiency": outcome.to_100(eff["score"]),
            "efficiency_detail": eff,
            "process_score": outcome.to_100(p_val),
            "applicable": process_applicable,
            "not_applicable_reason": None if process_applicable else "frozen_baseline_has_no_codegen_process",
            "dropped": list(p.dropped),
            "weight_used": p.weight_used,
        },
        "failure_taxonomy": process.failure_taxonomy(spec_cases, task),
        "balanced_h": outcome.to_100(float(np.sqrt(o_val * p_val)))
        if np.isfinite(o_val) and np.isfinite(p_val) and o_val >= 0 and p_val >= 0 else None,
    }


def _treatment_metrics(task: str, llm: pd.DataFrame,
                       sem: pd.DataFrame, num: pd.DataFrame) -> dict:
    """Paired semantic-minus-numeric deltas per headline metric.

    Pairs on (case_id, method): both sides can span several models, and a
    case_id-only join would emit cross-model contrasts (see `stats.paired_delta`).
    """
    per_metric: dict = {}
    for col in HEADLINE[task]:
        if col not in llm.columns:
            continue
        paired = stats.paired_delta(sem, num, col, key=["case_id", "method"])
        if paired.empty:
            continue
        fam = num.groupby("case_id")["family_id"].first().reindex(paired["case_id"]).tolist()
        ci = stats.family_block_bootstrap_ci(paired["delta"].tolist(), fam)
        deltas = paired["delta"].to_numpy()
        per_metric[col] = {
            "n_pairs": int(len(paired)),
            "mean_delta": float(np.mean(deltas)),
            "median_delta": float(np.median(deltas)),
            "win_cases": int((deltas > 0).sum()),
            "ci_lo": ci["lo"], "ci_hi": ci["hi"],
            "crosses_zero": bool(ci["lo"] <= 0 <= ci["hi"]),
        }
    return per_metric


def semantic_treatment(cases: pd.DataFrame, task: str, cfg: dict) -> dict:
    """Rubric 3.2.5: semantic - numeric_only is a separate treatment effect and
    never folded into baseline uplift."""
    llm = cases[(cases["task"] == task) & (cases["kind"] == "llm")]
    sem = llm[llm["track"] == "semantic"]
    num = llm[llm["track"] == "numeric_only"]
    if sem.empty or num.empty:
        return {"available": False}
    methods = sorted(set(sem["method"]) & set(num["method"]))
    out = {"available": True, "n_semantic": int(sem["case_id"].nunique()),
           "n_numeric": int(num["case_id"].nunique()),
           "methods": methods, "pairing_key": ["case_id", "method"],
           "per_metric": _treatment_metrics(task, llm, sem, num),
           # Pooling models hides a model x treatment interaction (dev 2026-09-28:
           # semantic helps qwen on Task B and does nothing for deepseek), so the
           # per-model breakdown is emitted alongside the pooled row, not instead
           # of it.  Read the per-model rows first when the two disagree.
           "per_method": {
               m: _treatment_metrics(task, llm, sem[sem["method"] == m], num[num["method"] == m])
               for m in methods
           }}
    return out


def _pareto_points(scores: list[dict], task: str | None = None) -> dict[str, tuple[float, float]]:
    """Only LLM tracks carry a Process score, so only they enter the frontier.

    O is normalised against task-specific thresholds, so a Task B score and a
    Task C score are not on a common ruler; the frontier is therefore computed
    within a task, never across the pooled set.
    """
    return {
        s["run_key"]: (s["outcome"]["outcome_score"], s["process"]["process_score"])
        for s in scores
        if s["kind"] == "llm"
        and (task is None or s["task"] == task)
        and s["outcome"]["outcome_score"] is not None
        and s["process"]["process_score"] is not None
    }


def render_cost_table(cases: pd.DataFrame) -> str:
    """Token / tool-call / wall-clock cost per track, reported but not scored.

    Decision 2A took cost out of `E`, so this table is the only place the
    numbers appear.  No scale, no weight, no normalisation -- just what each
    track spent, per case, on the cases it actually ran.
    """
    llm = cases[cases["kind"] == "llm"]
    if llm.empty:
        return ""
    rows = []
    for run_key, sub in llm.groupby("run_key", sort=True):
        per_case = sub.groupby("case_id")[
            ["total_tokens", "tool_call_count", "codegen_runtime_seconds"]
        ].mean()
        tok = per_case["total_tokens"]
        rows.append([
            run_key,
            f"{int(tok.notna().sum())}/{len(per_case)}",
            report.fmt(tok.median(), 0),
            report.fmt(tok.quantile(0.10), 0),
            report.fmt(tok.quantile(0.90), 0),
            report.fmt(per_case["tool_call_count"].median(), 1),
            report.fmt(per_case["codegen_runtime_seconds"].median(), 1),
            report.fmt(sub.groupby("case_id")["total_tokens"].sum().sum() / 1e6, 1),
        ])
    chunks = ["\n### 成本（仅报告，不计入 E；决策 2A）\n"]
    chunks.append(report.table(
        ["run", "有 usage 的 case", "tokens 中位", "p10", "p90",
         "tool calls 中位", "wall clock 中位 (s)", "总 tokens (M)"], rows))
    chunks.append("\n`E` 在 0.55 上重新归一（valid 0.25 + retry 0.15 + diversity 0.15），"
                  "四个轨道同一口径；token 数跨模型不可比，故不进入分数。")
    return "\n".join(chunks)


def render_tables(scores: list[dict], macro: dict[str, pd.DataFrame],
                  treatment: dict, cfg: dict, cases: pd.DataFrame | None = None) -> str:
    chunks: list[str] = []
    fmt = report.fmt

    for task in ("task_b", "task_c"):
        frame = macro[task]
        if frame.empty:
            # A split may carry only one task (test has Task B baselines and no
            # Task C run yet); render nothing rather than an empty table.
            continue
        chunks.append(f"\n### {task} case-macro 主指标\n")
        cols = [c for c in HEADLINE[task] if c in frame.columns]
        headers = ["run", "n"] + cols
        rows = []
        for _, r in frame.iterrows():
            pct = {"net_annual_return", "max_drawdown", "coverage"}
            rows.append([r["run_key"], int(r["n_cases"])] +
                        [fmt(r.get(c), 4 if "rank_ic" in c else 3, pct=c in pct) for c in cols])
        chunks.append(report.table(headers, rows))

        chunks.append(f"\n### {task} Outcome / Process 分数\n")
        rows = []
        for s in scores:
            if s["task"] != task:
                continue
            o, p = s["outcome"], s["process"]
            na = "n/a" if not p["applicable"] else None
            rows.append([
                s["run_key"], s["n_cases"],
                fmt(o["q_abs"], 1), fmt(o["q_uplift"], 1), fmt(o["outcome_score"], 1),
                fmt(p["reliability"], 1), fmt(p["stability"], 1),
                fmt(p["judge"], 1), fmt(p["efficiency"], 1), na or fmt(p["process_score"], 1),
                na or fmt(s["balanced_h"], 1),
            ])
        chunks.append(report.table(
            ["run", "n", "Q_abs", "Q_uplift", "O", "R", "S", "J", "E", "P", "H=√(O·P)"], rows))

    chunks.append("\n### LLM 轨道 vs primary baseline 的逐指标 uplift\n")
    rows = []
    for s in scores:
        if s["kind"] != "llm":
            continue
        up = (s.get("uplift") or {}).get("primary") or {}
        base = up.get("baseline", "—")
        for metric, d in (up.get("per_metric") or {}).items():
            if not d.get("n_pairs"):
                continue
            pct = metric in {"net_annual_return", "max_drawdown", "coverage"}
            rows.append([
                s["run_key"], base, metric, d["n_pairs"],
                fmt(d["mean_delta"], 4, pct=pct, signed=True),
                fmt(d["median_delta"], 4, pct=pct, signed=True),
                f"{d['win_cases']}/{d['n_pairs']}",
                fmt(d.get("improved_rate"), 3),
                fmt(d.get("u_magnitude_raw"), 3, signed=True),
            ])
        ir = up.get("active_ir") or {}
        rows.append([s["run_key"], base, "ActiveIR (年化)", ir.get("n_cases") or 0,
                     fmt(ir.get("mean_active_ir"), 3, signed=True), "—", "—", "—",
                     fmt(ir.get("score"), 3)])
    chunks.append(report.table(
        ["run", "baseline", "metric", "pairs", "mean Δ", "median Δ",
         "win cases", "ImprovedRate", "U_magnitude"], rows))

    if treatment.get("task_b", {}).get("available") or treatment.get("task_c", {}).get("available"):
        chunks.append("\n### semantic − numeric_only treatment effect\n")
        rows = []
        for task, res in treatment.items():
            for metric, d in (res.get("per_metric") or {}).items():
                pct = metric in {"net_annual_return", "max_drawdown", "coverage"}
                rows.append([
                    task, metric, d["n_pairs"],
                    fmt(d["mean_delta"], 4, pct=pct, signed=True),
                    f"{d['win_cases']}/{d['n_pairs']}",
                    report.ci_text(d["ci_lo"], d["ci_hi"], 4, pct=pct),
                    "含 0" if d["crosses_zero"] else "不含 0",
                ])
        chunks.append(report.table(
            ["task", "metric", "pairs", "mean Δ", "win cases", "95% family CI", "CI"], rows))
        chunks.append(
            "\n上表按 (case_id, method) 配对，跨模型对比不会混入。"
            "模型合并会掩盖 model×treatment 交互，故下表分模型再列一次 —— **两表不一致时以分模型为准**。\n"
        )

        chunks.append("\n### semantic − numeric_only treatment effect（分模型）\n")
        rows = []
        for task, res in treatment.items():
            for method, per_metric in (res.get("per_method") or {}).items():
                for metric, d in per_metric.items():
                    pct = metric in {"net_annual_return", "max_drawdown", "coverage"}
                    rows.append([
                        task, method, metric, d["n_pairs"],
                        fmt(d["mean_delta"], 4, pct=pct, signed=True),
                        f"{d['win_cases']}/{d['n_pairs']}",
                        report.ci_text(d["ci_lo"], d["ci_hi"], 4, pct=pct),
                        "含 0" if d["crosses_zero"] else "不含 0",
                    ])
        chunks.append(report.table(
            ["task", "model", "metric", "pairs", "mean Δ", "win cases",
             "95% family CI", "CI"], rows))

    chunks.append("\n### (O, P) Pareto（仅 LLM 轨道，按任务分别求前沿；"
                  "冻结 baseline 无 codegen 过程，不评 Process）\n")
    rows, all_points = [], {}
    for task in ("task_b", "task_c"):
        pts = _pareto_points(scores, task)
        fr = report.pareto_front(pts)
        all_points.update(pts)
        for k, v in sorted(pts.items()):
            rows.append([task, k, fmt(v[0], 1), fmt(v[1], 1), "前沿" if k in fr else "被支配"])
    chunks.append(report.table(["task", "run", "O", "P", "Pareto"], rows))
    if cases is not None:
        cost_block = render_cost_table(cases)
        if cost_block:
            chunks.append(cost_block)
    chunks.append("")
    chunks.append(report.scatter_ascii(all_points))
    chunks.append("\nO 用各任务自己的 tau 归一化，跨任务不可直接比较；散点仅作同任务内定位。")
    return "\n".join(chunks)


def _case_subset(args, cases: pd.DataFrame):
    """Return a row mask over `cases`-like frames, or None for no restriction.

    Task B and Task C have separate case_id namespaces, so run keys are
    intersected *within* a task and left independent across tasks: passing one
    Task B key and one Task C key narrows each task to that run's coverage
    instead of annihilating both.  A --case-ids file is taken as one flat set
    and applied to every task.

    Thresholds in the frozen config are untouched -- they are constants derived
    once on full dev.  This only narrows which cases feed the macro average, so
    a partial-coverage run can be compared against full-coverage ones on the
    cases they actually share.
    """
    flat: set[str] | None = None
    if args.case_ids:
        flat = {ln.strip() for ln in args.case_ids.read_text().splitlines() if ln.strip()}
        if not flat:
            raise SystemExit(f"--case-ids: {args.case_ids} lists no case ids")

    per_task: dict[str, set[str]] = {}
    for run_key in args.case_ids_from or []:
        rows = cases[cases["run_key"] == run_key]
        if rows.empty:
            raise SystemExit(f"--case-ids-from: no rows for run_key {run_key!r}")
        task = rows["task"].iloc[0]
        covered = set(rows["case_id"])
        per_task[task] = covered if task not in per_task else (per_task[task] & covered)

    for task, ids in per_task.items():
        if not ids:
            raise SystemExit(f"case subset for {task} is empty after intersection")
    if flat is None and not per_task:
        return None

    def mask(df: pd.DataFrame) -> pd.Series:
        keep = pd.Series(True, index=df.index)
        if flat is not None:
            keep &= df["case_id"].isin(flat)
        for task, ids in per_task.items():
            rows = df["task"] == task if "task" in df.columns else df["run_key"].str.startswith(task)
            keep &= ~rows | df["case_id"].isin(ids)
        return keep

    return mask


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="dev", choices=("dev", "test"),
                        help="which split to score (default dev). Selects the input "
                             "directory and the Reliability roster; thresholds stay "
                             "dev-frozen either way")
    parser.add_argument("--in-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--judge-scores", type=Path, default=None,
                        help="optional parsed judge results; absent => J dropped from P")
    parser.add_argument("--p-min", type=float, default=None)
    parser.add_argument("--o-min", type=float, default=None)
    parser.add_argument("--case-ids", type=Path, default=None,
                        help="file of case_ids (one per line); score every run on this subset only")
    parser.add_argument("--case-ids-from", nargs="*", default=None,
                        help="run_key(s) whose covered case_ids define the subset; "
                             "with several keys the intersection is used. Combines with --case-ids.")
    args = parser.parse_args()
    if args.in_dir is None:
        args.in_dir = OUT_DIRS[args.split]

    cfg = frozen.load(args.config, require_valid=False)
    cases = pd.read_parquet(args.in_dir / "case_metrics.parquet")
    # Origins are produced by LLM runs only, so a baselines-only split (test,
    # today) has no origin table.  Absent origins mean "S dropped", which is
    # the same path an LLM track with no origins already takes.
    origin_path = args.in_dir / "origin_metrics.parquet"
    origins = pd.read_parquet(origin_path) if origin_path.exists() else pd.DataFrame()
    monthly = pd.read_parquet(args.in_dir / "monthly_returns.parquet")

    subset = _case_subset(args, cases)
    if subset is not None:
        cases = cases[subset(cases)].copy()
        if len(origins.columns):
            origins = origins[subset(origins)].copy()
        monthly = monthly[subset(monthly)].copy()
        if cases.empty:
            raise SystemExit("case subset removed every row")
        kept = cases.groupby("task")["case_id"].nunique().to_dict()
        print(f"[case subset] cases per task {kept}; "
              f"{len(cases)} rows over {cases['run_key'].nunique()} runs")

    judge_scores = None
    if args.judge_scores and args.judge_scores.exists():
        judge_scores = pd.read_parquet(args.judge_scores)

    # Attempted roster per task = that split's full manifest, NOT the scored
    # subset.  `--case-ids*` narrows Outcome only; Reliability keeps this
    # denominator.  Scoring test against the dev roster would inject 84 dev
    # case ids as absent failures and 126 test cases would never be counted.
    attempted = {
        t: loaders.load_case_manifest(t, args.split)["case_id"].astype(str).tolist()
        for t in ("task_b", "task_c")
    }

    scores = []
    for run_key in sorted(cases["run_key"].unique()):
        task = cases[cases["run_key"] == run_key]["task"].iloc[0]
        js = _for_run(judge_scores, run_key) if judge_scores is not None else None
        scores.append(score_run(run_key, task, cases, origins, monthly, cfg, js,
                                attempted_case_ids=attempted[task], split=args.split))

    macro = {t: macro_table(cases[cases["task"] == t], t) for t in ("task_b", "task_c")}
    treatment = {t: semantic_treatment(cases, t, cfg) for t in ("task_b", "task_c")}

    front = sorted(k for task in ("task_b", "task_c")
                   for k in report.pareto_front(_pareto_points(scores, task)))

    payload = {
        "split": args.split,
        "config_version": cfg.get("config_version"),
        "thresholds_frozen_on": cfg.get("split"),
        "judge_available": judge_scores is not None,
        "champion_thresholds": {"p_min": args.p_min, "o_min": args.o_min},
        "champions": None if (args.p_min is None or args.o_min is None) else "see thresholds",
        "pareto_front": front,
        "scores": scores,
        "semantic_treatment": treatment,
    }
    (args.in_dir / "rubric_scores.json").write_text(
        json.dumps(payload, indent=2, default=float) + "\n")
    for task, frame in macro.items():
        frame.to_parquet(args.in_dir / f"macro_{task}.parquet", index=False)
    (args.in_dir / "rubric_tables.md").write_text(
        render_tables(scores, macro, treatment, cfg, cases) + "\n")

    print(f"split={args.split}  scored {len(scores)} runs; Pareto front: {front}")
    print(f"wrote {args.in_dir/'rubric_scores.json'} and rubric_tables.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

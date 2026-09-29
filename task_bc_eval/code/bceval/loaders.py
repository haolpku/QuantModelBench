"""Run-root readers that normalise every source into three tidy tables.

    case_metrics    one row per (run_key, case_id)   -- Outcome + Process inputs
    origin_metrics  one row per (run_key, case_id, origin) -- Stability inputs
    monthly_returns one row per (run_key, case_id, holding_month) -- ActiveIR inputs

Task B and Task C persist very different artefacts, so each layout gets its own
reader; downstream modules only ever see the three tables above.
"""

from __future__ import annotations

import json
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import runspec
from .metrics import (
    annualized_return,
    annualized_volatility,
    max_drawdown,
    net_from_gross,
    rank_ic_by_group,
    sharpe,
    spearman,
)

COST_BPS = 10.0
EXPECTED_ORIGINS = 12


# --------------------------------------------------------------------------
# Case manifests (family keys)
# --------------------------------------------------------------------------


@lru_cache(maxsize=4)
def load_case_manifest(task: str, split: str = "dev") -> pd.DataFrame:
    """Case-level manifest rows, carrying the keys that define a `family`.

    A family is the same data subset and anchor observed at several history
    lengths; those cases are not independent and must share a bootstrap block.
    """
    root = runspec.repo_root()
    dataset = runspec.dataset_for(task, split)
    path = root / dataset / "splits" / split / "manifest.jsonl"
    rows = []
    with open(path) as fh:
        for line in fh:
            rec = json.loads(line)
            if rec.get("record_type") != "case":
                continue
            if task == "task_b":
                rows.append(
                    {
                        "case_id": rec["case_id"],
                        "block_id": rec["block_id"],
                        "anchor_month": rec["anchor_month"],
                        "sample_size": rec["sample_size"],
                        "subset_index": rec["subset_index"],
                        "subset_id": rec["subset_id"],
                        "window": rec["window"],
                        "history_window_months": rec["lookback_months"],
                        "case_path": rec["path"],
                    }
                )
            else:
                rows.append(
                    {
                        "case_id": rec["case_id"],
                        "block_id": rec["block_id"],
                        "anchor_month": rec["anchor_month"],
                        "sample_size": rec["sample_size"],
                        "subset_index": rec["subset_index"],
                        "subset_id": rec["subset_id"],
                        "window": rec["history_window"],
                        "history_window_months": rec.get("window", {}).get("history_months"),
                        "case_path": rec["path"],
                    }
                )
    frame = pd.DataFrame(rows)
    # Same subset + anchor at different history lengths => one family.
    frame["family_id"] = (
        frame["block_id"] + "|" + frame["anchor_month"] + "|"
        + frame["sample_size"].astype(str) + "|" + frame["subset_index"].astype(str)
    )
    return frame


# --------------------------------------------------------------------------
# Task B evaluator truth (realised factor returns)
# --------------------------------------------------------------------------


@lru_cache(maxsize=4)
def load_task_b_truth(split: str | None = None) -> pd.DataFrame:
    """Realised held-month factor returns, long form.

    Columns: case_id, holding_month, factor_id (masked S-id), factor_name, ret.
    Needed to rebuild baseline monthly series and to calibrate Top-3 nulls.

    Task B keeps both splits under one `cases/` tree, so the default walks all
    of it (294 cases).  That is safe for the per-case uses -- every consumer
    groups by case_id first, and dev and test case_ids are disjoint, so no
    statistic mixes the two.  Pass `split` when you want the frame itself
    narrowed; the membership test comes from the split manifest rather than the
    block directory names, which are not split-aligned.
    """
    root = runspec.repo_root() / runspec.TASK_B_DATASET / "cases"
    keep: set[str] | None = None
    if split is not None:
        keep = set(load_case_manifest("task_b", split)["case_id"].astype(str))
    rows = []
    for truth_path in sorted(root.glob("*/case_*/rolling_truth.jsonl")):
        case_id = truth_path.parent.name
        if keep is not None and case_id not in keep:
            continue
        with open(truth_path) as fh:
            for line in fh:
                rec = json.loads(line)
                fut = rec["future"]
                mask = rec["masked_to_original"]
                inverse = {v: k for k, v in mask.items()}
                for idx, month in enumerate(fut["index"]):
                    for col, val in zip(fut["columns"], fut["data"][idx]):
                        rows.append(
                            {
                                "case_id": case_id,
                                "holding_month": str(month)[:7],
                                "factor_id": inverse.get(col, col),
                                "factor_name": col,
                                "ret": val,
                            }
                        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Task C evaluator panel (OOS labels)
# --------------------------------------------------------------------------


@lru_cache(maxsize=1)
def load_task_c_panel() -> pd.DataFrame:
    """(id, eom) -> ret_exc_lead1m, the forward excess return the evaluator uses.

    `ret_exc_lead1m` observed at eom t is the return of holding month t+1, which
    is exactly the label for a score formed at formation_eom t.
    """
    path = runspec.repo_root() / runspec.TASK_C_PANEL
    panel = pd.read_parquet(path, columns=["id", "eom", "ret_exc_lead1m"])
    panel["id"] = panel["id"].astype("int64").astype(str)
    panel["eom"] = _as_date_str(panel["eom"])
    return panel.dropna(subset=["ret_exc_lead1m"]).drop_duplicates(["id", "eom"])


def _as_date_str(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.dt.strftime("%Y-%m-%d")
    return series.astype(str).str.slice(0, 10)


def _as_month_str(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.dt.strftime("%Y-%m")
    return series.astype(str).str.slice(0, 7)


def _read_json(path: Path) -> dict | None:
    try:
        with open(path) as fh:
            return json.load(fh)
    except FileNotFoundError:
        return None


# --------------------------------------------------------------------------
# Task B: LLM run roots
# --------------------------------------------------------------------------


def _task_b_llm_case(spec: runspec.RunSpec, case_dir: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    case_id = case_dir.name
    summary = _read_json(case_dir / "rolling_result" / "summary.json")
    audit_path = case_dir / "rolling_result" / "rolling_audit.parquet"

    rec: dict = {
        "run_key": spec.key,
        "task": spec.task,
        "kind": spec.kind,
        "method": spec.method,
        "track": spec.track,
        "case_id": case_id,
        "block_id": case_dir.parent.name,
    }
    origins = pd.DataFrame()
    monthly = pd.DataFrame()

    if summary is None:
        rec.update({"run_complete": False, "failure_reason": "missing_summary"})
        return rec, origins, monthly

    pm = summary.get("prediction_metrics", {})
    portfolio = summary.get("portfolio", {})
    frontier = (portfolio.get("cost_frontier") or {}).get(str(int(COST_BPS)), {})

    rec.update(
        {
            "run_complete": True,
            "failure_reason": None,
            "episodes_requested": summary.get("episodes_requested"),
            "episodes_valid": summary.get("episodes_valid"),
            "completion_rate": summary.get("completion_rate"),
            "code_hash": summary.get("code_hash"),
            "llm_calls_during_rolling": summary.get("llm_calls_during_rolling"),
            "history_window_months": summary.get("history_window_months"),
            "rank_ic": pm.get("rank_ic_mean"),
            "accuracy": pm.get("accuracy_mean"),
            "top3_hit_rate": pm.get("top3_hit_rate_mean"),
            "top3_realized_return": pm.get("top3_average_realized_return_mean"),
            "net_annual_return": frontier.get("annualized_return"),
            "annual_volatility": frontier.get("annualized_volatility"),
            "sharpe": frontier.get("sharpe"),
            "max_drawdown": frontier.get("max_drawdown"),
            "average_turnover": portfolio.get("average_turnover"),
            "tradable_months": portfolio.get("months"),
        }
    )

    if audit_path.exists():
        audit = pd.read_parquet(audit_path)
        audit["holding_month"] = _as_month_str(audit["holding_month"])
        pmet = audit["prediction_metrics"].apply(lambda v: v if isinstance(v, dict) else {})
        origins = pd.DataFrame(
            {
                "run_key": spec.key,
                "case_id": case_id,
                "origin": audit["holding_month"],
                "step": audit["step"],
                "execution_valid": audit["execution_valid"].astype(bool),
                "rank_ic": pmet.apply(lambda d: d.get("rank_ic")),
                "accuracy": pmet.apply(lambda d: d.get("accuracy")),
                "top3_hit_rate": pmet.apply(lambda d: d.get("top3_hit_rate")),
                "n_predictions": pmet.apply(lambda d: d.get("n")),
                "runtime_seconds": audit["runtime_seconds"],
            }
        )
        net = net_from_gross(audit["gross_return"].to_numpy(), audit["turnover"].to_numpy(), COST_BPS)
        monthly = pd.DataFrame(
            {
                "run_key": spec.key,
                "case_id": case_id,
                "holding_month": audit["holding_month"],
                "gross_return": audit["gross_return"].astype(float),
                "turnover": audit["turnover"].astype(float),
                "net_return": net,
            }
        )
        rec["origin_valid"] = int(audit["execution_valid"].astype(bool).sum())
        rec["origin_total"] = int(len(audit))
        # Regression gate: the reconstructed series must reproduce the persisted
        # frontier, otherwise the cost convention assumed here is wrong.
        rec["net_recon_return_gap"] = abs(annualized_return(net) - (rec["net_annual_return"] or np.nan))
        rec["net_recon_sharpe_gap"] = abs(sharpe(net) - (rec["sharpe"] or np.nan))
    else:
        rec["origin_valid"] = 0
        rec["origin_total"] = EXPECTED_ORIGINS

    rec.update(_task_b_process(case_dir))
    return rec, origins, monthly


def _first_revision_valid(revisions: list[dict]) -> float:
    """1.0 / 0.0 if the *earliest* revision in the audit passed validation.

    Ordered by `round` when present (both tasks write it) so a sub-step that
    appends revisions out of order does not flip the answer.  NaN when no
    revision carries the two validation flags at all.
    """
    if not revisions:
        return np.nan
    ordered = sorted(revisions, key=lambda r: r.get("round") if r.get("round") is not None else 0)
    first = ordered[0]
    schema = first.get("schema_valid")
    execution = first.get("execution_valid")
    if schema is None and execution is None:
        return np.nan
    return float(bool(schema) and bool(execution))


def _task_b_process(case_dir: Path) -> dict:
    """Reliability / Efficiency inputs from the Task B revision audit."""
    audit = _read_json(case_dir / "revision_audit.json")
    if audit is None:
        return {"process_available": False}
    revisions = audit.get("code_revisions") or []
    calls = audit.get("llm_call_records") or []
    n_rev = len(revisions)
    valid = sum(1 for r in revisions if r.get("schema_valid") and r.get("execution_valid"))
    hashes = {r.get("code_hash") for r in revisions if r.get("code_hash")}
    return {
        "process_available": True,
        "max_revisions": audit.get("max_revisions"),
        "revision_count": n_rev,
        "valid_candidate_count": valid,
        "valid_candidate_rate": valid / n_rev if n_rev else np.nan,
        # E_retry reads this: "did the model's *first* attempt pass validation?"
        # `valid_candidate_rate` cannot answer it -- it is a within-case average
        # over revisions and already feeds E_valid.
        "first_revision_valid": _first_revision_valid(revisions),
        "distinct_code_hashes": len(hashes),
        "selection_fallback": bool(audit.get("selection_fallback")),
        "selection_policy": audit.get("selection_policy"),
        "selected_revision": audit.get("selected_revision"),
        "llm_calls_before_freeze": audit.get("llm_calls_before_freeze"),
        "provider_turns": audit.get("provider_turns_before_freeze"),
        "tool_call_count": sum(int(c.get("tool_call_count") or 0) for c in calls),
        "codegen_runtime_seconds": sum(float(r.get("runtime_seconds") or 0.0) for r in revisions),
        # Task B persisted no token usage; see decision 4 in the plan.
        "prompt_tokens": np.nan,
        "completion_tokens": np.nan,
        "total_tokens": np.nan,
        "validation_rank_ic": (audit.get("selected_validation") or {}).get("rank_ic_mean"),
        "code_chars": (audit.get("selected_validation") or {}).get("code_chars"),
    }


def load_task_b_llm(spec: runspec.RunSpec) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cases, origins, monthly = [], [], []
    for case_dir in sorted(spec.root.glob("block_*/case_*")):
        if not case_dir.is_dir():
            continue
        rec, org, mon = _task_b_llm_case(spec, case_dir)
        cases.append(rec)
        if len(org):
            origins.append(org)
        if len(mon):
            monthly.append(mon)
    return (
        pd.DataFrame(cases),
        pd.concat(origins, ignore_index=True) if origins else pd.DataFrame(),
        pd.concat(monthly, ignore_index=True) if monthly else pd.DataFrame(),
    )


# --------------------------------------------------------------------------
# Task B: baselines (results.jsonl + monthly reconstruction)
# --------------------------------------------------------------------------


@lru_cache(maxsize=2)
def _task_b_baseline_rows(path_str: str) -> tuple:
    with open(path_str) as fh:
        return tuple(json.loads(line) for line in fh)


def load_task_b_baseline(spec: runspec.RunSpec) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = [r for r in _task_b_baseline_rows(str(spec.root)) if r["method"] == spec.baseline_method]
    truth = load_task_b_truth(spec.split)
    truth_by_case = {cid: sub for cid, sub in truth.groupby("case_id", sort=False)}

    cases, monthly, origins = [], [], []
    for row in rows:
        case_id = row["case_id"]
        portfolio = row.get("portfolio") or {}
        frontier = (portfolio.get("cost_frontier") or {}).get(str(int(COST_BPS)), {})
        rec = {
            "run_key": spec.key,
            "task": spec.task,
            "kind": spec.kind,
            "method": spec.method,
            "track": None,
            "case_id": case_id,
            "block_id": row.get("block_id"),
            "run_complete": bool(row.get("complete")),
            "failure_reason": None if row.get("complete") else "incomplete_baseline",
            "episodes_requested": row.get("requested_origins"),
            "episodes_valid": row.get("valid_origins"),
            "origin_valid": row.get("valid_origins"),
            "origin_total": row.get("requested_origins"),
            "rank_ic": row.get("rank_ic_mean"),
            "accuracy": row.get("accuracy_mean"),
            "top3_hit_rate": row.get("top3_hit_rate_mean"),
            "net_annual_return": frontier.get("annualized_return", row.get("annualized_return_10bps")),
            "annual_volatility": frontier.get("annualized_volatility"),
            "sharpe": frontier.get("sharpe", row.get("sharpe_10bps")),
            "max_drawdown": frontier.get("max_drawdown"),
            "average_turnover": row.get("average_turnover"),
            "tradable_months": portfolio.get("months"),
            "process_available": False,
        }

        mon, diag = _rebuild_task_b_baseline_monthly(row, truth_by_case.get(case_id))
        rec.update(diag)
        if mon is not None:
            mon.insert(0, "case_id", case_id)
            mon.insert(0, "run_key", spec.key)
            monthly.append(mon)
        cases.append(rec)

    return (
        pd.DataFrame(cases),
        pd.concat(origins, ignore_index=True) if origins else pd.DataFrame(),
        pd.concat(monthly, ignore_index=True) if monthly else pd.DataFrame(),
    )


def _rebuild_task_b_baseline_monthly(row: dict, truth: pd.DataFrame | None):
    """Rebuild a baseline's monthly series from `weights_by_anchor` x realised returns.

    `results.jsonl` stores only aggregates, so ActiveIR would otherwise be
    impossible.  The rebuild is gated: unless it reproduces the persisted
    annualised return and turnover, the series is marked unavailable rather
    than silently used.
    """
    diag = {"monthly_series_available": False, "monthly_recon_return_gap": np.nan,
            "monthly_recon_turnover_gap": np.nan}
    weights_by_anchor = row.get("weights_by_anchor") or {}
    if truth is None or not weights_by_anchor:
        return None, diag

    # Anchor t -> holding month t+1; truth is keyed by holding month.
    anchors = sorted(weights_by_anchor)
    holding_months = sorted(truth["holding_month"].unique())
    if len(anchors) != len(holding_months):
        return None, diag

    name_to_id = truth.drop_duplicates("factor_name").set_index("factor_name")["factor_id"].to_dict()
    records, prev = [], None
    for anchor, month in zip(anchors, holding_months):
        weights = {name_to_id.get(k, k): float(v) for k, v in weights_by_anchor[anchor].items()}
        rets = truth[truth["holding_month"] == month].set_index("factor_id")["ret"].to_dict()
        gross = sum(w * rets.get(f, np.nan) for f, w in weights.items() if w != 0)
        keys = set(weights) | set(prev or {})
        turnover = (
            sum(abs(weights.get(k, 0.0) - (prev or {}).get(k, 0.0)) for k in keys) / 2.0
            if prev is not None
            else sum(abs(w) for w in weights.values()) / 2.0
        )
        records.append({"holding_month": month, "gross_return": gross, "turnover": turnover})
        prev = weights

    mon = pd.DataFrame(records)
    mon["net_return"] = net_from_gross(mon["gross_return"].to_numpy(), mon["turnover"].to_numpy(), COST_BPS)

    ref_ret = row.get("annualized_return_10bps")
    ref_turn = row.get("average_turnover")
    diag["monthly_recon_return_gap"] = abs(annualized_return(mon["net_return"].to_numpy()) - ref_ret) \
        if ref_ret is not None else np.nan
    diag["monthly_recon_turnover_gap"] = abs(float(mon["turnover"].mean()) - ref_turn) \
        if ref_turn is not None else np.nan
    diag["monthly_series_available"] = bool(
        np.isfinite(diag["monthly_recon_return_gap"]) and diag["monthly_recon_return_gap"] < 5e-3
    )
    return mon, diag


# --------------------------------------------------------------------------
# Task C: shared reader for LLM rolling dirs and baseline dirs
# --------------------------------------------------------------------------


def _task_c_case(spec: runspec.RunSpec, case_dir: Path, panel: pd.DataFrame | None):
    case_id = case_dir.name
    summary = _read_json(case_dir / "summary.json")
    rec: dict = {
        "run_key": spec.key,
        "task": spec.task,
        "kind": spec.kind,
        "method": spec.method,
        "track": spec.track,
        "case_id": case_id,
        "block_id": case_dir.parent.name,
    }
    origins = pd.DataFrame()
    monthly = pd.DataFrame()

    if summary is None:
        rec.update({"run_complete": False, "failure_reason": "missing_summary"})
        return rec, origins, monthly

    # LLM runs nest the evaluator block under "summary"; the frozen baselines
    # persist the identical block under "backtest".
    s = summary.get("summary") or summary.get("backtest") or {}
    net = s.get("net", {})
    code_block = summary.get("code") or {}
    rec.update(
        {
            "run_complete": True,
            "failure_reason": None,
            "formation_months": s.get("formation_months"),
            "tradable_months": s.get("tradable_months"),
            "unavailable_months": s.get("unavailable_months"),
            "coverage": s.get("average_coverage_min"),
            "average_turnover": s.get("average_one_way_turnover"),
            "net_annual_return": net.get("annualized_return"),
            "annual_volatility": net.get("annualized_volatility"),
            "sharpe": net.get("sharpe"),
            "max_drawdown": net.get("max_drawdown"),
            "positive_month_fraction": net.get("positive_month_fraction"),
        }
    )
    if code_block:
        rec.setdefault("validation_rank_ic", code_block.get("validation_rank_ic_mean"))
        rec["validation_rank_ic_std"] = code_block.get("validation_rank_ic_std")
        rec["schema_valid_rate"] = float(bool(code_block.get("schema_valid")))

    audit_path = case_dir / "rolling_audit.parquet"
    if audit_path.exists():
        audit = pd.read_parquet(audit_path)
        rec["origin_valid"] = int(audit["execution_valid"].astype(bool).sum())
        rec["origin_total"] = int(len(audit))
        rec["code_hash"] = audit["code_hash"].iloc[0] if len(audit) else None
        rec["llm_calls_during_rolling"] = int(audit["llm_calls"].sum()) if "llm_calls" in audit else 0
    else:
        rec["origin_valid"], rec["origin_total"] = 0, EXPECTED_ORIGINS

    monthly_path = case_dir / "backtest_monthly.parquet"
    if monthly_path.exists():
        bt = pd.read_parquet(monthly_path)
        ok = bt["status"].astype(str).str.lower().eq("ok") if "status" in bt else pd.Series(True, index=bt.index)
        monthly = pd.DataFrame(
            {
                "run_key": spec.key,
                "case_id": case_id,
                "holding_month": _as_month_str(bt["holding_month"]),
                "gross_return": bt["gross_return"].astype(float),
                "turnover": bt.get("one_way_turnover", bt.get("turnover")).astype(float),
                "net_return": bt["net_return"].astype(float),
                "tradable": ok.to_numpy(),
            }
        )
        # Untradable months are dropped, never zero-filled (rubric 1.2).
        monthly = monthly[monthly["tradable"]].drop(columns=["tradable"])

    if panel is not None:
        oos, per_month = _task_c_oos_rank_ic(case_dir, panel)
        rec.update(oos)
        if per_month is not None and len(per_month):
            origins = pd.DataFrame(
                {
                    "run_key": spec.key,
                    "case_id": case_id,
                    "origin": per_month["formation_eom"],
                    "step": range(1, len(per_month) + 1),
                    "execution_valid": True,
                    "rank_ic": per_month["rank_ic"],
                    "n_predictions": per_month["n_pairs"],
                    "join_coverage": per_month["join_coverage"],
                }
            )

    if spec.codegen_root is not None:
        rec.update(_task_c_process(spec.codegen_root / case_dir.parent.name / case_id))
    else:
        rec["process_available"] = False

    return rec, origins, monthly


def _task_c_oos_rank_ic(case_dir: Path, panel: pd.DataFrame) -> tuple[dict, pd.DataFrame | None]:
    """Rolling out-of-sample Rank IC -- the rubric's 40%-weight Task C metric.

    The rolling evaluator persists scores but no OOS IC, so it is recomputed by
    joining `predictions.parquet` to the evaluator panel's `ret_exc_lead1m`.
    """
    pred_path = case_dir / "predictions.parquet"
    if not pred_path.exists():
        return {"oos_rank_ic": np.nan, "oos_rank_ic_std": np.nan,
                "oos_join_coverage": np.nan, "oos_months": 0}, None

    pred = pd.read_parquet(pred_path, columns=["formation_eom", "id", "score"])
    pred["formation_eom"] = _as_date_str(pred["formation_eom"])
    pred["id"] = pred["id"].astype(str)
    merged = pred.merge(
        panel, left_on=["id", "formation_eom"], right_on=["id", "eom"], how="left"
    )

    rows = []
    for month, sub in merged.groupby("formation_eom", sort=True):
        ok = sub["ret_exc_lead1m"].notna()
        rows.append(
            {
                "formation_eom": month,
                "rank_ic": spearman(sub.loc[ok, "score"].to_numpy(), sub.loc[ok, "ret_exc_lead1m"].to_numpy()),
                "n_pairs": int(ok.sum()),
                "n_predictions": int(len(sub)),
                "join_coverage": float(ok.sum()) / len(sub) if len(sub) else np.nan,
            }
        )
    if not rows:
        return {"oos_rank_ic": np.nan, "oos_rank_ic_std": np.nan,
                "oos_join_coverage": np.nan, "oos_months": 0}, None
    per_month = pd.DataFrame(rows)
    return (
        {
            "oos_rank_ic": float(per_month["rank_ic"].mean(skipna=True)),
            "oos_rank_ic_std": float(per_month["rank_ic"].std(ddof=1)) if len(per_month) > 1 else np.nan,
            "oos_join_coverage": float(per_month["join_coverage"].mean(skipna=True)),
            "oos_months": int(per_month["rank_ic"].notna().sum()),
        },
        per_month,
    )


def _task_c_process(codegen_dir: Path) -> dict:
    audit = _read_json(codegen_dir / "revision_audit.json")
    if audit is None:
        return {"process_available": False}
    ev = audit.get("code_evaluation") or {}
    usage = audit.get("usage") or {}
    revisions = audit.get("code_revisions") or []
    hashes = {r.get("sha256") for r in revisions if r.get("sha256")}
    return {
        "process_available": True,
        "max_revisions": audit.get("max_revisions"),
        "revision_count": audit.get("revision_count", len(revisions)),
        "valid_candidate_count": ev.get("candidate_count_valid"),
        "valid_candidate_rate": ev.get("execution_valid_rate"),
        # See `_task_b_process`: E_retry is the first-pass validation rate.
        "first_revision_valid": _first_revision_valid(revisions),
        "schema_valid_rate": ev.get("schema_valid_rate"),
        "timeout_rate": ev.get("timeout_rate"),
        "distinct_code_hashes": len(hashes),
        "selection_fallback": bool(audit.get("selection_fallback")),
        "selection_policy": audit.get("selection_policy"),
        "selected_revision": audit.get("selected_revision"),
        "tool_call_count": audit.get("agentic_tool_call_count"),
        "codegen_runtime_seconds": sum(float(r.get("runtime_seconds") or 0.0) for r in revisions),
        "prompt_tokens": usage.get("prompt_tokens", np.nan),
        "completion_tokens": usage.get("completion_tokens", np.nan),
        "total_tokens": usage.get("total_tokens", np.nan),
        "cached_tokens": usage.get("cached_tokens", np.nan),
        "validation_rank_ic": (audit.get("selected_validation") or {}).get("rank_ic_mean"),
        "code_chars": (audit.get("selected_validation") or {}).get("code_chars"),
    }


def load_task_c_dirs(spec: runspec.RunSpec, with_oos: bool = True):
    panel = load_task_c_panel() if with_oos else None
    cases, origins, monthly = [], [], []
    for case_dir in sorted(spec.root.glob("block_*/c_case_*")):
        if not case_dir.is_dir():
            continue
        rec, org, mon = _task_c_case(spec, case_dir, panel)
        cases.append(rec)
        if len(org):
            origins.append(org)
        if len(mon):
            monthly.append(mon)
    return (
        pd.DataFrame(cases),
        pd.concat(origins, ignore_index=True) if origins else pd.DataFrame(),
        pd.concat(monthly, ignore_index=True) if monthly else pd.DataFrame(),
    )


# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------


def load_run(spec: runspec.RunSpec, with_oos: bool = True):
    if spec.layout == runspec.LAYOUT_TASK_B_LLM:
        return load_task_b_llm(spec)
    if spec.layout == runspec.LAYOUT_TASK_B_BASELINE:
        return load_task_b_baseline(spec)
    if spec.layout == runspec.LAYOUT_TASK_C_DIRS:
        return load_task_c_dirs(spec, with_oos=with_oos)
    raise ValueError(f"unknown layout: {spec.layout}")


def attach_family(cases: pd.DataFrame, task: str, split: str = "dev") -> pd.DataFrame:
    """Join the manifest family keys onto case records.

    `split` picks the manifest, which for Task C also picks the dataset build
    (see `runspec.dataset_for`).  Scoring a test run against the dev manifest
    finds no family key for any case and only warns, so pass the split.
    """
    manifest = load_case_manifest(task, split)[
        ["case_id", "family_id", "anchor_month", "sample_size", "subset_index", "window"]
    ]
    merged = cases.merge(manifest, on="case_id", how="left", suffixes=("", "_manifest"))
    missing = merged["family_id"].isna().sum()
    if missing:
        warnings.warn(f"{missing} cases in {task}/{split} have no manifest family key", stacklevel=2)
    return merged

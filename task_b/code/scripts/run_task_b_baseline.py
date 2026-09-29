#!/usr/bin/env python
"""Run the non-LLM Task B baselines on a truth JSONL file."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = CODE_ROOT.parent
ROOT = TASK_ROOT.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from factoropt.task_b import (  # noqa: E402
    BASELINE_IMPLEMENTATION_VERSION,
    baseline_portfolio_weights,
    evaluate_portfolio,
    episode_from_truth_record,
    build_features,
    run_baseline,
    score_predictions,
    turnover,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _truth_paths_from_manifest(path: Path) -> list[Path]:
    output: list[Path] = []
    for row in _read_jsonl(path):
        if row.get("record_type") != "case":
            continue
        truth_spec = (row.get("artifacts") or {}).get("truth.jsonl") or {}
        value = truth_spec.get("path")
        if not value:
            raise ValueError(f"case manifest row lacks artifacts.truth.jsonl.path: {path}")
        truth_path = Path(str(value))
        output.append(truth_path if truth_path.is_absolute() else ROOT / truth_path)
    if not output:
        raise ValueError(f"no case truth paths found in manifest: {path}")
    return output


def _realized(weights: pd.Series, target: pd.Series) -> float | None:
    aligned = pd.concat([weights.rename("weight"), target.rename("return")], axis=1).dropna()
    if aligned.empty or aligned["weight"].sum() <= 1e-12:
        return None
    weight = aligned["weight"] / aligned["weight"].sum()
    return float((weight * aligned["return"]).sum())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    truth_group = parser.add_mutually_exclusive_group(required=True)
    truth_group.add_argument(
        "--truth",
        nargs="+",
        help="One or more truth JSONL shards; duplicate base cases are deduplicated.",
    )
    truth_group.add_argument(
        "--truth-manifest",
        help="Case manifest JSONL whose case rows point to truth JSONL shards.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest-output")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "equal_weight",
            "inverse_vol",
            "momentum_12",
            "reversal_1",
            "t_stat_12",
            "t_stat_36",
            "downside_vol",
            "skip_month_momentum",
            "logistic",
            "elastic_net",
            "tree",
            "mlp",
            "mlp_deep",
            "mlp_ensemble",
        ],
        choices=[
            "equal_weight",
            "inverse_vol",
            "momentum_12",
            "reversal_1",
            "t_stat_12",
            "t_stat_36",
            "downside_vol",
            "skip_month_momentum",
            "logistic",
            "elastic_net",
            "tree",
            "mlp",
            "mlp_deep",
            "mlp_ensemble",
        ],
    )
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")

    truth_paths = (
        _truth_paths_from_manifest(Path(args.truth_manifest))
        if args.truth_manifest
        else [Path(value) for value in (args.truth or [])]
    )
    truth_rows: list[dict[str, Any]] = []
    for truth_path in truth_paths:
        truth_rows.extend(_read_jsonl(truth_path))
    unique: dict[str, dict[str, Any]] = {}
    for row in truth_rows:
        unique.setdefault(str(row.get("base_case_id", row["case_id"])), row)
    truths = list(unique.values())
    if args.max_cases:
        truths = truths[: args.max_cases]
    episodes = [episode_from_truth_record(row) for row in truths]
    episodes.sort(key=lambda item: (item.region, item.anchor_date))
    horizons = {episode.horizon_months for episode in episodes}
    if len(horizons) != 1:
        raise SystemExit(f"Truth file mixes forecast horizons: {sorted(horizons)}")
    horizon_months = next(iter(horizons), 1)
    feature_tables = {episode.case_id: build_features(episode.history) for episode in episodes}
    output_rows: list[dict[str, Any]] = []
    executor = (
        concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency)
        if args.concurrency > 1
        else None
    )
    try:
        for method in args.methods:
            gross: list[float] = []
            turnovers: list[float] = []
            previous: dict[str, pd.Series] = {}
            futures = (
                {
                    episode.case_id: executor.submit(
                        run_baseline,
                        episode,
                        method,
                        feature_tables=feature_tables,
                    )
                    for episode in episodes
                }
                if executor is not None
                else {}
            )
            for case_index, episode in enumerate(episodes, start=1):
                base_row = {
                    "case_id": episode.case_id,
                    "base_case_id": episode.case_id,
                    "method": method,
                    "baseline_implementation_version": BASELINE_IMPLEMENTATION_VERSION,
                    "region": episode.region,
                    "anchor_date": episode.anchor_date.strftime("%Y-%m-%d"),
                    "horizon_months": episode.horizon_months,
                    "target_definition": episode.target_definition,
                }
                try:
                    response = (
                        futures[episode.case_id].result()
                        if executor is not None
                        else run_baseline(
                            episode,
                            method,
                            feature_tables=feature_tables,
                        )
                    )
                    weights = baseline_portfolio_weights(
                        episode,
                        method,
                        feature_tables=feature_tables,
                        response=response,
                    )
                    previous_weights = previous.get(
                        episode.region,
                        pd.Series(0.0, index=episode.sleeve_ids),
                    )
                    current_turnover = turnover(weights, previous_weights)
                    previous[episode.region] = weights
                    realized = _realized(weights, episode.target)
                    if realized is not None:
                        gross.append(realized)
                        turnovers.append(current_turnover)
                    output_rows.append(
                        {
                            **base_row,
                            # Keep the schema-shaped deterministic baseline output so
                            # the end-to-end pipeline can expose it under the same
                            # ``parsed_response`` field as an LLM attempt.
                            "parsed_response": response,
                            "effective_method": method,
                            "prediction_metrics": score_predictions(response, episode.target),
                            "weights": {str(key): float(value) for key, value in weights.items()},
                            "gross_return": realized,
                            "turnover": current_turnover,
                            "error": None,
                        }
                    )
                    status = "succeeded"
                except Exception as exc:  # noqa: BLE001 - isolate one baseline case/method
                    output_rows.append(
                        {
                            **base_row,
                            "parsed_response": None,
                            "effective_method": None,
                            "prediction_metrics": None,
                            "weights": None,
                            "gross_return": None,
                            "turnover": None,
                            "error": str(exc),
                        }
                    )
                    status = "failed"
                print(
                    json.dumps(
                        {
                            "progress": f"{case_index}/{len(episodes)}",
                            "method": method,
                            "case_id": episode.case_id,
                            "status": status,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            output_rows.append(
                {
                    "case_id": f"SUMMARY__{method}",
                    "method": method,
                    "baseline_implementation_version": BASELINE_IMPLEMENTATION_VERSION,
                    "horizon_months": horizon_months,
                    "target_definition": episodes[0].target_definition if episodes else None,
                    "portfolio": evaluate_portfolio(
                        gross,
                        turnovers,
                        periods_per_year=12.0 / horizon_months,
                    ),
                }
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    failed = sum(1 for row in output_rows if row.get("error"))
    manifest = {
        "stage": "task_b_baselines",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "complete" if failed == 0 else "incomplete",
        "complete": failed == 0,
        "baseline_implementation_version": BASELINE_IMPLEMENTATION_VERSION,
        "training_scope_by_method": {
            str(method): (
                "case_history_rolling_origin_prior_realized_returns_no_prediction_feedback"
                if method in {"logistic", "elastic_net", "tree", "mlp", "mlp_deep", "mlp_ensemble"}
                else "rule_based_current_case_history"
            )
            for method in args.methods
        },
        "prediction_feedback": False,
        "concurrency": args.concurrency,
        "truth_files": [str(value.resolve()) for value in truth_paths],
        "cases": len(episodes),
        "methods": list(args.methods),
        "case_method_rows": len(episodes) * len(args.methods),
        "valid_case_method_rows": len(episodes) * len(args.methods) - failed,
        "failed_case_method_rows": failed,
        "output": str(output.resolve()),
        "summary": {
            str(row["method"]): row.get("portfolio")
            for row in output_rows
            if str(row.get("case_id", "")).startswith("SUMMARY__")
        },
    }
    if args.manifest_output:
        manifest_path = Path(args.manifest_output)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "cases": len(episodes),
                "truth_files": len(truth_paths),
                "methods": args.methods,
                "concurrency": args.concurrency,
                "rows": len(output_rows),
                "failed_case_method_rows": failed,
                "manifest": str(Path(args.manifest_output)) if args.manifest_output else None,
            },
            indent=2,
        )
    )
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

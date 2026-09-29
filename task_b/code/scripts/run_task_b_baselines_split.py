#!/usr/bin/env python3
"""Run deterministic Task B baselines on the frozen rolling dev protocol."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[1]
ROOT = CODE_ROOT.parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from factoropt.task_b import (  # noqa: E402
    BASELINE_IMPLEMENTATION_VERSION,
    baseline_portfolio_weights,
    build_compact_features,
    episode_from_truth_record,
    evaluate_portfolio,
    run_baseline,
    score_predictions,
    turnover,
)
from factoropt.task_b import (  # noqa: E402
    ROLLING_ML_FEATURES,
    _rolling_model_training_rows,
)

METHODS = (
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
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _truth_paths(manifest: Path) -> list[tuple[dict[str, Any], Path]]:
    rows = [row for row in _read_jsonl(manifest) if row.get("record_type") == "case"]
    output: list[tuple[dict[str, Any], Path]] = []
    for row in rows:
        path = Path(str(row["artifacts"]["rolling_truth"]["path"]))
        output.append((row, path if path.is_absolute() else ROOT / path))
    return output


def _realized(weights: pd.Series, target: Mapping[str, Any]) -> float | None:
    target_series = pd.Series(target, dtype=float)
    aligned = pd.concat([weights.rename("weight"), target_series.rename("return")], axis=1).dropna()
    if aligned.empty or float(aligned["weight"].sum()) <= 1e-12:
        return None
    normal = aligned["weight"] / aligned["weight"].sum()
    return float((normal * aligned["return"]).sum())


def _run_case(spec: Mapping[str, Any], truth_path: Path, methods: Sequence[str]) -> list[dict[str, Any]]:
    truth_rows = _read_jsonl(truth_path)
    truth_rows.sort(key=lambda row: int(row.get("sequence_step", 0) or 0))
    if not truth_rows:
        raise ValueError(f"empty rolling truth: {truth_path}")
    episodes = []
    for step, row in enumerate(truth_rows, start=1):
        episode = episode_from_truth_record(row)
        # ``episode_from_truth_record`` intentionally uses base_case_id for
        # normal evaluator joins.  Baseline feature/training caches must be
        # origin-specific because all 12 rows share that base ID.
        episode.case_id = f"{spec['case_id']}__step_{step:02d}"
        episodes.append(episode)
    feature_tables = {
        episode.case_id: build_compact_features(episode.history) for episode in episodes
    }
    training_cache = {
        episode.case_id: _rolling_model_training_rows(episode, list(ROLLING_ML_FEATURES))
        for episode in episodes
        if any(method in {"logistic", "elastic_net", "tree", "mlp", "mlp_deep", "mlp_ensemble"} for method in methods)
    }
    metadata = {
        "case_id": str(spec["case_id"]),
        "block_id": str(spec.get("block_id")),
        "window": str(spec.get("window")),
        "factor_count": int(spec.get("sample_size") or len(episodes[0].sleeve_ids)),
        "requested_origins": len(episodes),
    }
    output: list[dict[str, Any]] = []
    for method in methods:
        gross: list[float] = []
        turnovers: list[float] = []
        rank_ics: list[float] = []
        accuracies: list[float] = []
        top3_hits: list[float] = []
        errors: list[dict[str, Any]] = []
        weights_by_anchor: dict[str, dict[str, float]] = {}
        previous_weights: pd.Series | None = None
        for step, episode in enumerate(episodes, start=1):
            try:
                # Baselines use only the univariate fields; this avoids the
                # quadratic anchor correlation table while preserving values.
                response = run_baseline(
                    episode,
                    method,
                    feature_tables=feature_tables,
                    training_cache=training_cache,
                )
                weights = baseline_portfolio_weights(
                    episode, method, feature_tables=feature_tables, response=response
                )
                weights_by_anchor[str(episode.anchor_date)] = {
                    str(key): float(value) for key, value in weights.items()
                }
                metrics = score_predictions(response, episode.target)
                realized = _realized(weights, episode.target)
                current_turnover = turnover(weights, previous_weights)
                previous_weights = weights
                if realized is not None:
                    gross.append(realized)
                    turnovers.append(current_turnover)
                for values, key in ((rank_ics, "rank_ic"), (accuracies, "accuracy"), (top3_hits, "top3_hit_rate")):
                    value = metrics.get(key)
                    if value is not None and np.isfinite(float(value)):
                        values.append(float(value))
            except Exception as exc:  # isolate one method/origin
                errors.append({"step": step, "error": f"{type(exc).__name__}: {exc}"})
        portfolio = evaluate_portfolio(gross, turnovers, costs_bps=(0.0, 10.0, 25.0), periods_per_year=12.0)
        net10 = portfolio["cost_frontier"]["10"]
        output.append(
            {
                **metadata,
                "method": method,
                "baseline_implementation_version": BASELINE_IMPLEMENTATION_VERSION,
                "valid_origins": len(gross),
                "complete": len(gross) == len(episodes),
                "rank_ic_mean": float(np.mean(rank_ics)) if rank_ics else None,
                "accuracy_mean": float(np.mean(accuracies)) if accuracies else None,
                "top3_hit_rate_mean": float(np.mean(top3_hits)) if top3_hits else None,
                "portfolio": portfolio,
                "annualized_return_10bps": net10.get("annualized_return"),
                "sharpe_10bps": net10.get("sharpe"),
                "average_turnover": portfolio.get("average_turnover"),
                # Keep the deterministic monthly sleeve allocations so the
                # stock implementation can reuse exactly the same baseline
                # signal without rerunning or changing its protocol.
                "weights_by_anchor": weights_by_anchor,
                "errors": errors,
            }
        )
    return output


def _limit_blas_threads() -> None:
    """One BLAS thread per process.

    The work is single-threaded Python driving many small scikit-learn fits, so
    intra-process BLAS parallelism buys nothing and, with N processes on N
    cores, actively costs: the default is one thread per core per process.
    """
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"


def _make_pool(executor: str, workers: int):
    """Processes by default.

    `_run_case` is a long pure-Python loop over origins and methods that calls
    into scikit-learn, and those inner loops never release the GIL long enough
    to overlap.  Measured on this host: 8 threads produced 8 completed cases in
    12 minutes, i.e. no speedup at all over one process.  Threads are kept only
    as an escape hatch for hosts where pickling the truth specs is undesirable.
    """
    if executor == "thread":
        return concurrent.futures.ThreadPoolExecutor(max_workers=workers)
    return concurrent.futures.ProcessPoolExecutor(
        max_workers=workers, mp_context=mp.get_context("spawn"),
        initializer=_limit_blas_threads,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/task_b_usa_frozen_raw_rolling_294/splits/dev/manifest.jsonl"))
    parser.add_argument("--output", type=Path, default=Path(".tmp/task_b_baselines/results.jsonl"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--executor", choices=("process", "thread"), default="process",
                        help="process (default) bypasses the GIL; thread reuses ThreadPoolExecutor")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    if not manifest_path.exists():
        parser.error(f"--manifest does not exist: {manifest_path}")
    cases = _truth_paths(manifest_path)
    if not cases:
        parser.error(
            f"--manifest resolved to zero cases: {manifest_path}\n"
            "  The file exists but contains no rows with record_type == 'case'.\n"
            "  Check that the manifest belongs to this dataset root and that its "
            "'artifacts.rolling_truth.path' entries resolve."
        )
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with _make_pool(args.executor, min(args.workers, len(cases))) as pool:
        futures = {pool.submit(_run_case, spec, path, args.methods): spec for spec, path in cases}
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            result = future.result()
            rows.extend(result)
            print(json.dumps({"completed_cases": index, "total_cases": len(cases), "case_id": result[0]["case_id"]}, ensure_ascii=False), flush=True)
    rows.sort(key=lambda row: (row["case_id"], row["method"]))
    output.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows), encoding="utf-8")
    manifest = {
        "record_type": "task_b_frozen_dev_baseline_summary",
        "implementation_version": BASELINE_IMPLEMENTATION_VERSION,
        "cases": len(cases),
        "methods": list(args.methods),
        "workers": min(args.workers, len(cases)),
        "executor": args.executor,
        "rows": len(rows),
        "complete_case_method_rows": sum(bool(row["complete"]) for row in rows),
        "output": str(output.resolve()),
    }
    (output.parent / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run a frozen Task B source over a rolling one-step test year.

This script intentionally has no provider/API options.  It loads the source
selected by ``run_task_b_codegen.py`` and executes that same source once
per monthly origin.  At origin ``t`` the latest realized labels in the local
history are split into rolling train/validation windows; after the target
month settles, the next origin sees that month and the window advances by one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = CODE_ROOT.parent
ROOT = TASK_ROOT.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from factoropt.task_b import episode_from_truth_record  # noqa: E402
from factoropt.task_b_frozen import (  # noqa: E402
    DEFAULT_MIN_HISTORY_MONTHS,
    DEFAULT_TRAIN_CUTOFFS,
    DEFAULT_VALIDATION_CUTOFFS,
    FROZEN_TASK_VARIANT,
    run_frozen_rolling,
    validate_frozen_payload,
)
from factoropt.semantic import normalize_track  # noqa: E402


def _read_json(path: Path) -> Any:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError("JSON response list must contain one object")
        value = value[0]
    if not isinstance(value, dict):
        raise ValueError("JSON file must contain an object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"non-object row at {path}:{line_number}")
            rows.append(value)
    return rows


def _write_frame(frame: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_parquet(path, index=False)
        return str(path)
    except (ImportError, ModuleNotFoundError):
        csv = path.with_suffix(".csv")
        frame.to_csv(csv, index=False)
        return str(csv)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-response", required=True, type=Path, help="revision audit JSON or selected response JSON")
    parser.add_argument("--rolling-truth", required=True, type=Path, help="evaluator-only truth JSONL from build_task_b_case.py")
    parser.add_argument(
        "--payload",
        type=Path,
        help="public payload (defaults to payload.json beside rolling_truth; needed for semantic runtime cards)",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--train-months", "--train-cutoffs", dest="train_months", type=int,
        help="labelled train cutoff months per origin (defaults to the case manifest/protocol)",
    )
    parser.add_argument(
        "--validation-months", "--validation-cutoffs", dest="validation_months", type=int,
        help="labelled validation cutoff months per origin (defaults to the case manifest/protocol)",
    )
    parser.add_argument(
        "--feature-warmup-months", type=int,
        help="minimum prefix for point-in-time feature rows (defaults to the case protocol)",
    )
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--cost-bps", nargs="+", type=float, default=[0.0, 10.0, 25.0])
    # Scratch files are disposable runtime state and must stay outside the
    # paper-final run index.
    parser.add_argument("--scratch-root", type=Path, default=Path(".tmp/task_b_frozen_scratch"))
    args = parser.parse_args(argv)
    frozen = _read_json(args.frozen_response)
    payload_path = args.payload
    if payload_path is None:
        inferred_payload = args.rolling_truth.resolve().parent / "payload.json"
        payload_path = inferred_payload if inferred_payload.is_file() else None
    public_payload = _read_json(payload_path) if payload_path is not None else {}
    if payload_path is not None:
        payload_errors = validate_frozen_payload(public_payload)
        if payload_errors:
            raise SystemExit("invalid public payload: " + "; ".join(payload_errors[:10]))
    track = normalize_track(public_payload.get("track", "numeric_only"))
    factor_cards = public_payload.get("factor_cards", [])
    if not isinstance(factor_cards, list):
        raise SystemExit("payload.factor_cards must be an array")
    truth_rows = _read_jsonl(args.rolling_truth)
    if not truth_rows:
        raise SystemExit("rolling truth is empty")
    truth_rows.sort(key=lambda row: int(row.get("sequence_step", 0) or 0))
    expected_steps = list(range(1, len(truth_rows) + 1))
    observed_steps = [int(row.get("sequence_step", 0) or 0) for row in truth_rows]
    if observed_steps != expected_steps:
        raise SystemExit(f"rolling truth sequence is not contiguous: {observed_steps}")
    sequence_ids = {str(row.get("sequence_id", "")) for row in truth_rows}
    if len(sequence_ids) != 1 or "" in sequence_ids:
        raise SystemExit("rolling truth rows must share one non-empty sequence_id")
    variants = {str(row.get("task_variant", "")) for row in truth_rows}
    if variants != {FROZEN_TASK_VARIANT}:
        raise SystemExit(f"rolling truth has unsupported task_variant: {sorted(variants)}")

    def protocol_default(field: str, fallback: int) -> int:
        values: set[int] = set()
        for row in truth_rows:
            if row.get(field) is None:
                continue
            try:
                values.add(int(row[field]))
            except (TypeError, ValueError):
                raise SystemExit(f"rolling truth has a non-integer {field}") from None
        if len(values) > 1:
            raise SystemExit(f"rolling truth has inconsistent {field}: {sorted(values)}")
        return next(iter(values), int(fallback))

    train_months = int(args.train_months) if args.train_months is not None else protocol_default(
        "train_cutoffs", DEFAULT_TRAIN_CUTOFFS
    )
    validation_months = int(args.validation_months) if args.validation_months is not None else protocol_default(
        "validation_cutoffs", DEFAULT_VALIDATION_CUTOFFS
    )
    feature_warmup_months = (
        int(args.feature_warmup_months)
        if args.feature_warmup_months is not None
        else protocol_default("feature_warmup_months", DEFAULT_MIN_HISTORY_MONTHS)
    )
    if train_months < 1 or validation_months < 1 or feature_warmup_months < 1:
        parser.error("train/validation/warm-up months must be positive")
    history_windows: set[int] = set()
    for row in truth_rows:
        if row.get("history_window_months") is None:
            continue
        try:
            history_windows.add(int(row["history_window_months"]))
        except (TypeError, ValueError):
            raise SystemExit("rolling truth has a non-integer history_window_months") from None
    if len(history_windows) > 1:
        raise SystemExit(f"rolling truth has inconsistent history_window_months: {sorted(history_windows)}")

    episodes = [episode_from_truth_record(row) for row in truth_rows]
    declared_history = next(iter(history_windows), None)
    if declared_history is not None and any(len(episode.history.index) != declared_history for episode in episodes):
        raise SystemExit("rolling truth history length does not match history_window_months")
    maps = [
        {str(k): str(v) for k, v in (row.get("masked_to_original") or {}).items()}
        for row in truth_rows
    ]
    predictions, audit, summary = run_frozen_rolling(
        frozen,
        episodes,
        maps,
        train_cutoffs=train_months,
        validation_cutoffs=validation_months,
        min_history_months=feature_warmup_months,
        timeout_seconds=args.timeout_seconds,
        scratch_root=args.scratch_root,
        costs_bps=args.cost_bps,
        track=track,
        factor_cards=factor_cards,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = _write_frame(predictions, args.output_dir / "predictions.parquet")
    audit_path = _write_frame(audit, args.output_dir / "rolling_audit.parquet")
    output_summary = {
        **summary,
        "sequence_id": next(iter(sequence_ids)),
        "train_months_per_origin": train_months,
        "validation_months_per_origin": validation_months,
        "train_cutoffs_per_origin": train_months,
        "validation_cutoffs_per_origin": validation_months,
        "feature_warmup_months": feature_warmup_months,
        "history_window_months": next(iter(history_windows), None),
        "artifacts": {"predictions": prediction_path, "rolling_audit": audit_path},
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(output_summary, ensure_ascii=False, indent=2, allow_nan=False, default=str),
        encoding="utf-8",
    )
    print(json.dumps(output_summary, ensure_ascii=False, default=str))
    return 0 if summary.get("episodes_valid", 0) else 2


if __name__ == "__main__":
    raise SystemExit(main())

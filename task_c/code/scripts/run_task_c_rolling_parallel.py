#!/usr/bin/env python3
"""Run selected frozen Task C rolling backtests concurrently and resumably."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


CODE_ROOT = Path(__file__).resolve().parents[1]
ROOT = CODE_ROOT.parents[1]
SCRIPT = CODE_ROOT / "scripts" / "run_task_c_rolling.py"


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _rows(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object at {path}:{line_number}")
        values.append(value)
    return values


def _case_ids(path: Path) -> set[str]:
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def _tail(value: str, limit: int = 4000) -> str:
    return value if len(value) <= limit else "..." + value[-limit:]


def _complete(summary: Path) -> bool:
    if not summary.is_file():
        return False
    try:
        payload = json.loads(summary.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and isinstance(payload.get("summary"), dict)


def _valid_audit(audit: Path) -> bool:
    if not audit.is_file():
        return False
    try:
        payload = json.loads(audit.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and isinstance(payload.get("selected_response"), dict)


def _run_one(
    row: Mapping[str, Any],
    *,
    dataset: Path,
    codegen_root: Path,
    output_root: Path,
    panel: Path,
    daily: Path,
    nyse_cutoffs: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    block = str(row["block_id"])
    case_id = str(row["case_id"])
    case_dir = dataset / "cases" / block / case_id
    audit = codegen_root / block / case_id / "revision_audit.json"
    case_out = output_root / block / case_id
    summary = case_out / "summary.json"
    log = case_out / "runner.log"
    if args.resume and _complete(summary):
        return {
            "block_id": block,
            "case_id": case_id,
            "status": "skipped_existing",
            "returncode": 0,
            "elapsed_seconds": 0.0,
            "output_dir": str(case_out),
            "summary_path": str(summary),
            "audit": str(audit),
        }
    if not _valid_audit(audit):
        return {
            "block_id": block,
            "case_id": case_id,
            "status": "failed",
            "returncode": 2,
            "elapsed_seconds": 0.0,
            "output_dir": str(case_out),
            "audit": str(audit),
            "error": "missing or incomplete revision_audit.json",
        }
    case_out.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(SCRIPT),
        "--frozen-response",
        str(audit),
        "--case-dir",
        str(case_dir),
        "--panel",
        str(panel),
        "--daily",
        str(daily),
        "--nyse-cutoffs",
        str(nyse_cutoffs),
        "--output-dir",
        str(case_out),
        "--cost-bps",
        str(args.cost_bps),
        "--coverage-threshold",
        str(args.coverage_threshold),
        "--timeout-seconds",
        str(args.timeout_seconds),
    ]
    if args.strict_cutoffs:
        command.append("--strict-cutoffs")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(CODE_ROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=str(ROOT),
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=args.case_timeout_seconds,
        )
        elapsed = round(time.perf_counter() - started, 3)
        log.write_text(
            "COMMAND\n"
            + " ".join(command)
            + "\n\nSTDOUT\n"
            + completed.stdout
            + "\nSTDERR\n"
            + completed.stderr,
            encoding="utf-8",
        )
        status = "ok" if completed.returncode == 0 and _complete(summary) else "failed"
        return {
            "block_id": block,
            "case_id": case_id,
            "status": status,
            "returncode": int(completed.returncode),
            "elapsed_seconds": elapsed,
            "output_dir": str(case_out),
            "summary_path": str(summary),
            "audit": str(audit),
            "log": str(log),
            "stdout_tail": _tail(completed.stdout),
            "stderr_tail": _tail(completed.stderr),
        }
    except subprocess.TimeoutExpired as exc:
        elapsed = round(time.perf_counter() - started, 3)
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        log.write_text(
            "COMMAND\n"
            + " ".join(command)
            + "\n\nSTDOUT\n"
            + stdout
            + "\nSTDERR\n"
            + stderr
            + f"\nTimeoutExpired after {args.case_timeout_seconds} seconds\n",
            encoding="utf-8",
        )
        return {
            "block_id": block,
            "case_id": case_id,
            "status": "failed",
            "returncode": 124,
            "elapsed_seconds": elapsed,
            "output_dir": str(case_out),
            "audit": str(audit),
            "log": str(log),
            "error": f"case timed out after {args.case_timeout_seconds} seconds",
        }


def _write_progress(output_root: Path, results: list[dict[str, Any]], selected: int) -> None:
    ok = sum(row.get("status") in {"ok", "skipped_existing"} for row in results)
    failed = sum(row.get("status") == "failed" for row in results)
    payload = {
        "selected_case_count": selected,
        "finished_case_count": len(results),
        "completed_case_count": ok,
        "failed_case_count": failed,
        "updated_unix": time.time(),
    }
    (output_root / "progress.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _split_manifest(parser: argparse.ArgumentParser, dataset: Path, args: argparse.Namespace) -> Path:
    """Resolve the split manifest, rejecting one that belongs to another dataset.

    dev and test now live in one merged 207-case build sharing
    ``block_00..block_04`` (dev = 01/02, test = 00/03/04), so the old
    "separate builds with different block numbering" hazard is gone.  The guard
    still matters for the other pairing mistake: the numeric dataset and its
    ``_semantic`` sibling carry identical case ids and block names, so a
    manifest from one paired with ``--dataset`` pointing at the other resolves
    silently and only surfaces later as a wrong ``track`` in the summary.
    """
    if not args.split_manifest:
        return dataset / "splits" / args.split / "manifest.jsonl"
    manifest = _resolve(args.split_manifest)
    if dataset.resolve() not in manifest.resolve().parents:
        parser.error(f"--split-manifest is not inside --dataset: {manifest}")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/processed/task_c_frozen_raw_rolling_207")
    parser.add_argument("--split-manifest")
    parser.add_argument(
        "--split",
        choices=("dev", "test"),
        default="dev",
        help="split to read from --dataset; ignored when --split-manifest is given",
    )
    parser.add_argument("--codegen-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--daily", required=True)
    parser.add_argument("--nyse-cutoffs", required=True)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--case-id-file", action="append", default=[])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--coverage-threshold", type=float, default=0.95)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--case-timeout-seconds", type=float, default=7200.0)
    parser.add_argument("--strict-cutoffs", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")

    dataset = _resolve(args.dataset)
    manifest = _split_manifest(parser, dataset, args)
    codegen_root = _resolve(args.codegen_root)
    output_root = _resolve(args.output_root)
    panel = _resolve(args.panel)
    daily = _resolve(args.daily)
    nyse_cutoffs = _resolve(args.nyse_cutoffs)
    selected = [row for row in _rows(manifest) if row.get("record_type") == "case"]
    requested = set(args.case_id)
    for case_id_file in args.case_id_file:
        requested.update(_case_ids(_resolve(case_id_file)))
    if requested:
        available = {str(row.get("case_id")) for row in selected}
        missing = requested - available
        if missing:
            parser.error("requested case IDs are absent from the manifest: " + ", ".join(sorted(missing)))
        selected = [row for row in selected if str(row.get("case_id")) in requested]
    if not selected:
        parser.error("no cases selected")
    for path in (SCRIPT, panel, daily, nyse_cutoffs):
        if not path.is_file():
            parser.error(f"required file not found: {path}")
    for row in selected:
        case_dir = dataset / "cases" / str(row["block_id"]) / str(row["case_id"])
        for name in ("payload.json", "manifest.json", "rolling_truth.jsonl", "evaluator_id_map.json", "evaluator_feature_map.json"):
            if not (case_dir / name).is_file():
                parser.error(f"required evaluator file not found: {case_dir / name}")

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "selected_cases.json").write_text(
        json.dumps({"selected_case_count": len(selected), "cases": selected}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.dry_run:
        print(json.dumps({"selected_case_count": len(selected), "output_root": str(output_root)}, ensure_ascii=False))
        return 0

    results: list[dict[str, Any]] = []
    _write_progress(output_root, results, len(selected))
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                _run_one,
                row,
                dataset=dataset,
                codegen_root=codegen_root,
                output_root=output_root,
                panel=panel,
                daily=daily,
                nyse_cutoffs=nyse_cutoffs,
                args=args,
            )
            for row in selected
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            _write_progress(output_root, results, len(selected))
            print(json.dumps(result, ensure_ascii=False), flush=True)

    results.sort(key=lambda row: (str(row.get("block_id")), str(row.get("case_id"))))
    completed = sum(row.get("status") in {"ok", "skipped_existing"} for row in results)
    failed = [row for row in results if row.get("status") == "failed"]
    (output_root / "results.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results), encoding="utf-8"
    )
    summary = {
        "selected_case_count": len(selected),
        "completed_case_count": completed,
        "failed_case_count": len(failed),
        "failed_case_ids": [str(row["case_id"]) for row in failed],
        "dataset": str(dataset),
        "codegen_root": str(codegen_root),
        "panel": str(panel),
        "daily": str(daily),
        "nyse_cutoffs": str(nyse_cutoffs),
        "workers": args.workers,
        "cost_bps": args.cost_bps,
        "coverage_threshold": args.coverage_threshold,
    }
    (output_root / "batch_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: summary[key] for key in ("selected_case_count", "completed_case_count", "failed_case_count")}, ensure_ascii=False))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())

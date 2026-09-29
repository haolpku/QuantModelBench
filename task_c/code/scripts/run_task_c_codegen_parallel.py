#!/usr/bin/env python3
"""Run selected Task C dev code-generation cases concurrently and resumably."""

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
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from factoropt.dataset_overlay import resolve_artifact_root  # noqa: E402


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


def _tail(value: str, limit: int = 3000) -> str:
    return value if len(value) <= limit else "..." + value[-limit:]


def _complete(audit: Path) -> bool:
    if not audit.is_file():
        return False
    try:
        return json.loads(audit.read_text(encoding="utf-8")).get("selected_response") is not None
    except (OSError, json.JSONDecodeError):
        return False


def _audit_usage(audit: Path) -> dict[str, Any] | None:
    """Read the per-case token total back out of a written codegen audit.

    ``iterative_revision_loop`` sums each revision's usage into an audit-level
    ``usage`` field, but this runner only observes the child's return code, so
    results.jsonl previously carried no cost information.  Read the durable
    audit instead of re-deriving anything from the transcript.
    """

    try:
        payload = json.loads(audit.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    usage = payload.get("usage")
    return dict(usage) if isinstance(usage, Mapping) else None


def _run_one(row: Mapping[str, Any], dataset: Path, output_root: Path, args: argparse.Namespace, env: Mapping[str, str]) -> dict[str, Any]:
    block = str(row["block_id"])
    case_id = str(row["case_id"])
    case_out = output_root / block / case_id
    audit = case_out / "revision_audit.json"
    if args.resume and _complete(audit):
        return {
            "block_id": block, "case_id": case_id, "returncode": 0,
            "status": "skipped_existing", "output_dir": str(case_out),
            "usage": _audit_usage(audit),
        }
    case_out.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, str(CODE_ROOT / "scripts" / "run_task_c_codegen.py"),
        "--payload", str(dataset / "cases" / block / case_id / "payload.json"),
        # The semantic dataset is an overlay without artifacts/; its public
        # train/validation Parquet is the numeric build's identical bytes.
        "--artifact-root", str(resolve_artifact_root(dataset, block, case_id)),
        "--output", str(audit),
        "--model", str(args.model), "--base-url", str(args.base_url),
        "--max-revisions", str(args.max_revisions),
        "--timeout-seconds", str(args.timeout_seconds),
        "--code-timeout-seconds", str(args.code_timeout_seconds),
        "--max-tool-calls", str(args.max_tool_calls), "--max-turns", str(args.max_turns),
        "--max-output-tokens", str(args.max_output_tokens), "--repair-invalid", "--no-stop-on-final",
        "--objective", str(args.objective),
        "--selection-policy", str(args.selection_policy),
        "--scratch-root", str(case_out / ".scratch"),
    ]
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=str(ROOT), env=dict(env), capture_output=True, text=True, check=False)
    elapsed = round(time.perf_counter() - started, 3)
    log = case_out / "runner.log"
    log.write_text("COMMAND\n" + " ".join(command) + "\n\nSTDOUT\n" + completed.stdout + "\nSTDERR\n" + completed.stderr, encoding="utf-8")
    return {
        "block_id": block, "case_id": case_id, "window": row.get("history_window"), "feature_count": row.get("feature_count"),
        "returncode": int(completed.returncode), "status": "finished", "elapsed_seconds": elapsed,
        "output_dir": str(case_out), "log": str(log), "audit_complete": _complete(audit),
        "usage": _audit_usage(audit),
        "stdout_tail": _tail(completed.stdout), "stderr_tail": _tail(completed.stderr),
    }


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
    parser.add_argument("--output-root", default=".tmp/task_c_dev_remaining")
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument(
        "--case-id-file",
        action="append",
        default=[],
        help="file containing one case ID per line; may be repeated",
    )
    parser.add_argument("--exclude-case-id", action="append", default=[])
    parser.add_argument("--exclude-case-id-file")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model", default=os.environ.get("TASK_C_MODEL", "deepseek-v4-pro"))
    parser.add_argument("--base-url", default=os.environ.get("TASK_C_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api-key-env", default="LLM_API_KEY")
    parser.add_argument("--max-revisions", type=int, default=2)
    parser.add_argument("--objective", choices=("rank_ic_mean", "net_return", "mse"), default="rank_ic_mean")
    parser.add_argument("--selection-policy", choices=("llm", "host_metric"), default="llm")
    parser.add_argument("--max-tool-calls", type=int, default=8)
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--max-output-tokens", type=int, default=32768)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--code-timeout-seconds", type=float, default=1800.0, help="per-month wall clock for one candidate execution")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    if not os.environ.get(args.api_key_env) and not args.dry_run:
        parser.error(f"environment variable {args.api_key_env} is required")
    dataset = _resolve(args.dataset)
    manifest = _split_manifest(parser, dataset, args)
    selected = [row for row in _rows(manifest) if row.get("record_type") == "case"]
    requested = set(args.case_id)
    for case_id_file in args.case_id_file:
        requested.update(_case_ids(_resolve(case_id_file)))
    if requested:
        selected = [row for row in selected if str(row.get("case_id")) in requested]
    excluded = set(args.exclude_case_id)
    if args.exclude_case_id_file:
        excluded.update(_case_ids(_resolve(args.exclude_case_id_file)))
    selected = [row for row in selected if str(row.get("case_id")) not in excluded]
    if not selected:
        parser.error("no cases selected")
    output_root = _resolve(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    selection_path = output_root / "selected_cases.json"
    selection_path.write_text(json.dumps({"selected_case_count": len(selected), "excluded_case_ids": sorted(excluded), "cases": selected}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.dry_run:
        print(json.dumps({"selected_case_count": len(selected), "selection": str(selection_path)}, ensure_ascii=False))
        return 0
    env = dict(os.environ)
    env["PYTHONPATH"] = str(CODE_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["TASK_C_MODEL"] = str(args.model)
    env["TASK_C_BASE_URL"] = str(args.base_url)
    env["OPENAI_API_KEY"] = env[args.api_key_env]
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_run_one, row, dataset, output_root, args, env) for row in selected]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            with (output_root / "results.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(json.dumps(result, ensure_ascii=False), flush=True)
    completed = sum(bool(row.get("audit_complete")) or row.get("status") == "skipped_existing" for row in results)
    total_usage: dict[str, int] = {}
    for row in results:
        usage = row.get("usage")
        if not isinstance(usage, Mapping):
            continue
        for key, value in usage.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                total_usage[key] = total_usage.get(key, 0) + int(value)
    summary = {"selected_case_count": len(selected), "completed_case_count": completed, "failed_case_count": len(selected) - completed, "usage_total": (total_usage or None), "results": results}
    (output_root / "batch_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("selected_case_count", "completed_case_count", "failed_case_count", "usage_total")}, ensure_ascii=False))
    return 0 if completed == len(selected) else 1


if __name__ == "__main__":
    raise SystemExit(main())

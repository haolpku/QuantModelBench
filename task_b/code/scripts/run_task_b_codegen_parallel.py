#!/usr/bin/env python3
"""Run selected Task B frozen-code dev cases concurrently and resumably."""

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

from factoropt.dataset_overlay import resolve_case_file  # noqa: E402


def _resolve(value: str | Path, base: Path = ROOT) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def _case_id_file(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.lstrip().startswith("#")}


def _tail(value: str, limit: int = 3000) -> str:
    text = str(value or "")
    return text if len(text) <= limit else "..." + text[-limit:]


def _audit_usage(audit: Path) -> dict[str, Any] | None:
    """Read the per-case token total back out of a written revision audit.

    The agentic loop aggregates usage per revision, but the parallel runner
    only sees the child's stdout/return code, so results.jsonl carried no cost
    information.  The audit is the durable record; read it back after the
    child exits rather than re-deriving anything from the transcript.
    """

    try:
        payload = json.loads(audit.read_text(encoding="utf-8"))
    except Exception:
        return None
    usage = payload.get("usage")
    return dict(usage) if isinstance(usage, Mapping) else None


def _run_one(row: Mapping[str, Any], dataset: Path, output_root: Path, args: argparse.Namespace, env: Mapping[str, str]) -> dict[str, Any]:
    block = str(row["block_id"])
    case_id = str(row["case_id"])
    case_dir = dataset / "cases" / block / case_id
    payload = case_dir / "payload.json"
    # The semantic view is an overlay holding only payload.json; the truth file
    # is byte-identical to the numeric build's and is read from there.
    truth = resolve_case_file(dataset, block, case_id, "rolling_truth.jsonl")
    case_out = output_root / block / case_id
    audit = case_out / "revision_audit.json"
    rolling = case_out / "rolling_result"
    if args.resume and audit.is_file() and (rolling / "summary.json").is_file():
        return {
            "block_id": block,
            "case_id": case_id,
            "returncode": 0,
            "status": "skipped_existing",
            "output_dir": str(case_out),
            "usage": _audit_usage(audit),
        }
    case_out.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, str(CODE_ROOT / "scripts" / "run_task_b_codegen.py"),
        "--payload", str(payload), "--output", str(audit),
        "--rolling-truth", str(truth), "--rolling-output-dir", str(rolling),
        "--model", str(args.model), "--base-url", str(args.base_url),
        "--api-key-env", str(args.api_key_env), "--max-revisions", str(args.max_revisions),
        "--no-stop-on-final", "--repair-invalid",
        "--objective", str(args.objective),
        "--selection-policy", str(args.selection_policy),
        "--max-tool-calls", str(args.max_tool_calls), "--max-turns", str(args.max_turns),
        "--max-output-tokens", str(args.max_output_tokens), "--timeout-seconds", str(args.timeout_seconds),
        "--thinking", "provider_default", "--payload-prompt-mode", str(args.payload_prompt_mode),
        "--rolling-timeout-seconds", str(args.rolling_timeout_seconds),
        "--rolling-cost-bps", "0", "10", "25", "--scratch-root", str(case_out / ".scratch"),
    ]
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=str(ROOT), env=dict(env), capture_output=True, text=True, check=False)
    elapsed = round(time.perf_counter() - started, 3)
    log = case_out / "runner.log"
    log.write_text("COMMAND\n" + " ".join(command) + "\n\nSTDOUT\n" + completed.stdout + "\nSTDERR\n" + completed.stderr, encoding="utf-8")
    return {"block_id": block, "case_id": case_id, "window": row.get("window"), "factor_count": row.get("sample_size"), "subset_index": row.get("subset_index"), "returncode": int(completed.returncode), "status": "finished", "elapsed_seconds": elapsed, "output_dir": str(case_out), "log": str(log), "usage": _audit_usage(audit), "stdout_tail": _tail(completed.stdout), "stderr_tail": _tail(completed.stderr)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/processed/task_b_usa_frozen_raw_rolling_294", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument(
        "--split",
        choices=("dev", "test"),
        default="dev",
        help="split to read from --dataset; ignored when --split-manifest is given",
    )
    parser.add_argument("--output-root", default=".tmp/task_b_dev_remaining", type=Path)
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
    parser.add_argument("--dry-run", action="store_true", help="write the selected case list without provider calls")
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL", "deepseek-v4-pro"))
    parser.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api-key-env", default="LLM_API_KEY")
    parser.add_argument("--max-revisions", type=int, default=2)
    parser.add_argument("--max-tool-calls", type=int, default=8)
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--max-output-tokens", type=int, default=32768)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--rolling-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--objective", choices=("rank_ic_mean", "mse", "directional_accuracy"), default="rank_ic_mean")
    parser.add_argument("--selection-policy", choices=("llm", "host_metric"), default="llm")
    parser.add_argument("--payload-prompt-mode", choices=("auto", "full", "summary"), default="auto")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    dataset = _resolve(args.dataset)
    if args.split_manifest:
        manifest = _resolve(args.split_manifest)
        if dataset.resolve() not in manifest.resolve().parents:
            parser.error(f"--split-manifest is not inside --dataset: {manifest}")
    else:
        manifest = dataset / "splits" / args.split / "manifest.jsonl"
    output_root = _resolve(args.output_root)
    rows = [row for row in _read_jsonl(manifest) if row.get("record_type") == "case"]
    wanted = {str(value) for value in args.case_id}
    for case_id_file in args.case_id_file:
        wanted.update(_case_id_file(_resolve(case_id_file)))
    excluded = {str(value) for value in args.exclude_case_id}
    if args.exclude_case_id_file:
        excluded.update(_case_id_file(_resolve(args.exclude_case_id_file)))
    if wanted:
        rows = [row for row in rows if str(row.get("case_id")) in wanted]
    rows = [row for row in rows if str(row.get("case_id")) not in excluded]
    if not rows:
        parser.error("no cases selected")
    output_root.mkdir(parents=True, exist_ok=True)
    selection = output_root / "case_selection.jsonl"
    selection.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    if args.dry_run:
        summary = {"record_type": "task_b_frozen_dev_parallel_dry_run", "selected_cases": len(rows), "selection": str(selection), "output_root": str(output_root)}
        (output_root / "batch_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if not os.environ.get(args.api_key_env, ""):
        parser.error(f"missing {args.api_key_env}; export it before provider calls")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(CODE_ROOT) + ((os.pathsep + env["PYTHONPATH"]) if env.get("PYTHONPATH") else "")
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(args.workers, len(rows)), thread_name_prefix="task-b-dev") as pool:
        futures = {pool.submit(_run_one, row, dataset, output_root, args, env): row for row in rows}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps({"completed": len(results), "total": len(rows), "block_id": result.get("block_id"), "case_id": result.get("case_id"), "returncode": result.get("returncode"), "status": result.get("status"), "usage": result.get("usage")}, ensure_ascii=False), flush=True)
    results.sort(key=lambda row: (str(row.get("block_id")), str(row.get("case_id"))))
    results_path = output_root / "results.jsonl"
    results_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results), encoding="utf-8")
    total_usage: dict[str, int] = {}
    for row in results:
        usage = row.get("usage")
        if not isinstance(usage, Mapping):
            continue
        for key, value in usage.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                total_usage[key] = total_usage.get(key, 0) + int(value)
    summary = {"record_type": "task_b_frozen_dev_parallel_summary", "selected_cases": len(rows), "workers": min(args.workers, len(rows)), "succeeded": sum(row.get("returncode") == 0 for row in results), "failed": sum(row.get("returncode") != 0 for row in results), "usage_total": (total_usage or None), "output_root": str(output_root), "selection": str(selection), "results": str(results_path)}
    (output_root / "batch_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

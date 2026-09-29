#!/usr/bin/env python3
"""Generate and freeze one Task B factor-return model source.

This is the only LLM stage of the frozen-code protocol.  The default
``--max-revisions 0`` makes exactly one provider request (apart from any
within-request tool turns).  Optional revisions are still pre-test only.  If
``--rolling-truth`` is supplied, the local rolling evaluator is launched only
after a valid source is frozen; it never runs during the provider/tool loop.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

CODE_ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = CODE_ROOT.parent
ROOT = TASK_ROOT.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from factoropt.agentic import (  # noqa: E402
    AgenticConfig,
    TaskBFrozenCodeAdapter,
    normalize_base_url,
    run_agentic_case,
)
from factoropt.task_b_frozen import (  # noqa: E402
    FrozenRevisionConfig,
    frozen_code_revision_loop,
    materialize_payload_split,
    stage_public_artifacts,
    validate_frozen_payload,
)


def _read_json(path: Path) -> Any:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError("payload JSON list must contain exactly one object")
        value = value[0]
    if not isinstance(value, dict):
        raise ValueError("payload must be a JSON object")
    return value


def _read_responses(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        values = value if isinstance(value, list) else [value]
    else:
        values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not all(isinstance(item, dict) for item in values):
        raise ValueError("offline responses must be JSON objects")
    return [dict(item) for item in values]


def _sum_usage(records: Sequence[Mapping[str, Any]]) -> dict[str, int] | None:
    """Sum token accounting across every provider revision of one case.

    ``run_agentic_case`` aggregates usage per revision into ``record["usage"]``
    but the frozen audit previously kept only status/tool-call summaries, so
    the per-case token cost was dropped before the audit was written.  Keep
    both the DeepSeek-style flat cache fields and the OpenAI-style nested
    ``prompt_tokens_details.cached_tokens`` so the total is gateway-agnostic.
    """

    totals: dict[str, int] = {}
    for item in records:
        usage = item.get("usage")
        if not isinstance(usage, Mapping):
            continue
        for key in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "prompt_cache_hit_tokens",
            "prompt_cache_miss_tokens",
            "cached_tokens",
        ):
            value = usage.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
                totals[key] = totals.get(key, 0) + int(value)
    return totals or None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="frozen revision audit JSON")
    parser.add_argument("--responses", type=Path, help="offline complete code responses JSON/JSONL")
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL", ""))
    parser.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api-key-env", default="LLM_API_KEY")
    parser.add_argument("--max-revisions", type=int, default=0)
    parser.add_argument(
        "--objective", choices=("rank_ic_mean", "mse", "directional_accuracy"),
        default="rank_ic_mean", help="host-metric ranking used when --selection-policy host_metric, and always recorded as a diagnostic",
    )
    parser.add_argument(
        "--selection-policy",
        choices=("llm", "host_metric"),
        default="llm",
        help="who freezes the candidate: the LLM from the valid set, or host max of --objective",
    )
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--max-tool-calls", type=int, default=8)
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--max-output-tokens", type=int, default=16000)
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high", "max"))
    parser.add_argument("--thinking", choices=("provider_default", "enabled", "disabled"), default="provider_default")
    # Scratch files are disposable runtime state and must stay outside the
    # paper-final run index.
    parser.add_argument("--scratch-root", type=Path, default=Path(".tmp/task_b_frozen_scratch"))
    parser.add_argument(
        "--payload-prompt-mode", choices=("auto", "full", "summary"), default="auto",
        help="whether to place full public rows in the chat message (tools always retain the full payload)",
    )
    parser.add_argument("--payload-summary-threshold", type=int, default=120_000)
    parser.add_argument("--keep-artifacts", action="store_true")
    parser.add_argument("--no-stop-on-final", action="store_true")
    parser.add_argument(
        "--repair-invalid",
        action="store_true",
        help=(
            "allow one bounded no-tools repair turn when the provider's final "
            "JSON is malformed or violates the response contract"
        ),
    )
    parser.add_argument(
        "--rolling-truth",
        type=Path,
        help=(
            "optional evaluator-only rolling_truth.jsonl; when supplied, run "
            "the local rolling evaluator immediately after code freeze"
        ),
    )
    parser.add_argument(
        "--rolling-output-dir",
        type=Path,
        help="output directory for the optional synchronous rolling evaluation",
    )
    parser.add_argument(
        "--rolling-timeout-seconds",
        type=float,
        default=120.0,
        help="per-case execution timeout used by the optional rolling evaluator",
    )
    parser.add_argument(
        "--rolling-cost-bps",
        nargs="+",
        type=float,
        default=[0.0, 10.0, 25.0],
        help="transaction-cost assumptions passed to the optional rolling evaluator",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.base_url = normalize_base_url(args.base_url)
    payload = _read_json(args.payload)
    payload_errors = validate_frozen_payload(payload)
    if payload_errors:
        raise SystemExit("invalid frozen payload: " + "; ".join(payload_errors[:10]))
    train = materialize_payload_split(payload, "train")
    valid = materialize_payload_split(payload, "validation")
    # The public history is staged outside the per-revision scratch roots.  A
    # ``run_agentic_case`` call removes its own scratch root in
    # ``ScratchWorkspace.finalize``; keeping the source mount outside that
    # root lets revision 1/2 reuse the same public files after revision 0.
    scratch_root = args.scratch_root.expanduser().resolve()
    scratch_root.parent.mkdir(parents=True, exist_ok=True)
    public_mount_root = Path(
        tempfile.mkdtemp(
            prefix=f"{scratch_root.name}_public_",
            dir=str(scratch_root.parent),
        )
    )
    try:
        public_files, _ = stage_public_artifacts(
            payload,
            train,
            valid,
            directory=public_mount_root,
            feature_columns=payload.get("feature_columns"),
        )
    except Exception:
        shutil.rmtree(public_mount_root, ignore_errors=True)
        raise
    public_artifact_files = {
        str(split): str((public_mount_root / Path(relative).name).resolve())
        for split, relative in public_files.items()
    }
    # Mirror Task C: the descriptor path is relative to ``artifact_root`` and the
    # runner resolves it against the staged directory rather than the case dir.
    payload["artifact_root"] = public_mount_root.as_posix()
    responses = _read_responses(args.responses) if args.responses else []
    if not responses and not args.model:
        raise SystemExit("provide --model (or --responses for an offline run)")
    if not responses and not os.environ.get(args.api_key_env, ""):
        raise SystemExit(
            f"missing {args.api_key_env}; export it in the execution environment "
            "before making provider calls (or use --responses for an offline run)"
        )

    adapter = TaskBFrozenCodeAdapter()
    config = AgenticConfig(
        model=args.model or "offline-model",
        base_url=args.base_url,
        temperature=0.0,
        seed=int(payload.get("metadata", {}).get("seed", 1)),
        max_output_tokens=args.max_output_tokens,
        timeout_seconds=args.timeout_seconds,
        retries=1,
        max_tool_calls=args.max_tool_calls,
        max_turns=args.max_turns,
        tool_choice="auto",
        json_mode=True,
        thinking=args.thinking,
        reasoning_effort=args.reasoning_effort,
        repair_invalid=args.repair_invalid,
        keep_artifacts=args.keep_artifacts,
        payload_prompt_mode=args.payload_prompt_mode,
        payload_summary_threshold=args.payload_summary_threshold,
    )
    call_records: list[dict[str, Any]] = []

    def callback(public: Mapping[str, Any], feedback: Mapping[str, Any] | None, revision: int) -> Any:
        if responses:
            if revision >= len(responses):
                raise RuntimeError(f"offline responses has no revision {revision}")
            call_records.append({"revision": revision, "mode": "offline"})
            return responses[revision]
        request_payload = dict(public)
        # The evaluator-only mount root is a local absolute path; it is used to
        # resolve the artifact descriptors in the tool loop but must never reach
        # the provider or the prompt hash.
        request_payload.pop("artifact_root", None)
        request_payload["_revision"] = int(revision)
        if feedback is not None:
            request_payload["_revision_feedback"] = dict(feedback)
        # Each revision gets a disposable child.  The parent contains only
        # evaluator scratch state, while the public source files live in the
        # sibling temporary directory above and survive finalization.
        revision_scratch_root = scratch_root / f"revision_{int(revision):03d}"
        record = run_agentic_case(
            request_payload,
            adapter=adapter,
            config=config,
            api_key=os.environ.get(args.api_key_env, ""),
            scratch_root=revision_scratch_root,
            artifact_destination=(scratch_root / str(public.get("case_id", "case")) if args.keep_artifacts else None),
            public_artifact_files=public_artifact_files,
            truth=None,
        )
        call_records.append(
            {
                "revision": revision,
                "mode": "provider",
                "status": record.get("status"),
                "run_id": record.get("run_id"),
                "tool_call_count": record.get("tool_call_count", 0),
                "provider_turns": len(record.get("conversation_turns") or []),
                "usage": record.get("usage"),
                "validation": record.get("validation"),
                "error": record.get("error"),
                "parsed_response": record.get("parsed_response"),
            }
        )
        return record

    try:
        audit = frozen_code_revision_loop(
            payload,
            callback,
            train_df=train,
            validation_df=valid,
            config=FrozenRevisionConfig(
                max_revisions=args.max_revisions,
                timeout_seconds=args.timeout_seconds,
                objective=args.objective,
                selection_policy=args.selection_policy,
                stop_on_final=not args.no_stop_on_final,
            ),
            scratch_root=scratch_root,
        )
    finally:
        # The rolling stage does not need the public parquet mount; remove it
        # once all pre-test revisions have finished, including on exceptions.
        shutil.rmtree(public_mount_root, ignore_errors=True)
    audit["llm_calls_before_freeze"] = len(call_records)
    audit["provider_turns_before_freeze"] = int(sum(int(item.get("provider_turns", 0) or 0) for item in call_records))
    audit["llm_call_records"] = call_records
    audit["usage"] = _sum_usage(call_records)
    audit["source_payload"] = str(args.payload.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False, default=str), encoding="utf-8")
    print(json.dumps({
        "selected_revision": audit.get("selected_revision"),
        "selected_code_hash": audit.get("selected_code_hash"),
        "llm_calls_before_freeze": audit.get("llm_calls_before_freeze"),
        "provider_turns_before_freeze": audit.get("provider_turns_before_freeze"),
        "test_llm_calls_after_freeze": audit.get("test_llm_calls_after_freeze"),
        "usage": audit.get("usage"),
    }, ensure_ascii=False))

    # The hidden rolling evaluator is deliberately an opt-in post-freeze
    # stage.  The truth path is never passed to ``run_agentic_case`` or to the
    # provider; it is opened only after a valid source has been selected.
    # Keeping this orchestration here makes a case reproducible with one
    # command while preserving the public/trusted data boundary.
    if args.rolling_truth is not None:
        if not args.rolling_truth.is_file():
            print(f"rolling truth not found: {args.rolling_truth}", file=sys.stderr)
            return 2
        selected_response = audit.get("selected_response")
        if audit.get("selected_revision") is None or not isinstance(selected_response, Mapping):
            print(
                "code freeze failed; rolling evaluator was not started because "
                "selected_response is missing. Inspect code_revisions.errors and "
                "llm_call_records.",
                file=sys.stderr,
            )
            return 2

        selected_path = args.output.with_name("selected_response.json")
        selected_path.write_text(
            json.dumps(dict(selected_response), ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        rolling_output = args.rolling_output_dir or args.output.parent / "rolling_result"
        rolling_script = CODE_ROOT / "scripts" / "run_task_b_rolling.py"
        rolling_command = [
            sys.executable,
            str(rolling_script),
            "--frozen-response",
            str(selected_path),
            "--rolling-truth",
            str(args.rolling_truth),
            # Pass the payload explicitly. The rolling stage otherwise infers it
            # from the truth file's own directory, which is correct only while
            # the two sit together; under a semantic overlay the truth comes
            # from the numeric base and that inference would silently load the
            # numeric payload, dropping the factor cards and recording the run
            # as track=numeric_only.
            "--payload",
            str(args.payload),
            "--output-dir",
            str(rolling_output),
            "--timeout-seconds",
            str(args.rolling_timeout_seconds),
            "--scratch-root",
            str(scratch_root),
        ]
        rolling_command.extend(["--cost-bps", *[str(cost) for cost in args.rolling_cost_bps]])
        rolling_result = subprocess.run(rolling_command, check=False)
        if rolling_result.returncode != 0:
            return int(rolling_result.returncode)

        combined = {
            "agentic_audit": str(args.output),
            "selected_response": str(selected_path),
            "rolling_truth": str(args.rolling_truth.resolve()),
            "rolling_output_dir": str(rolling_output),
            "llm_calls_before_freeze": audit.get("llm_calls_before_freeze"),
            "selected_revision": audit.get("selected_revision"),
            "llm_calls_during_rolling": 0,
        }
        (args.output.parent / "experiment_summary.json").write_text(
            json.dumps(combined, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return 0 if audit.get("selected_revision") is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())

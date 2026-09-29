#!/usr/bin/env python3
"""Run Task C's bounded validation-revision loop.

The default callback uses the exact Task A/B-compatible Chat Completions tool
loop. ``--responses`` is provided for deterministic offline development: it is
a JSONL/JSON list of complete code-response objects, one per revision.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping


CODE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_ROOT.parent.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from factoropt.agentic import AgenticConfig, TaskCAdapter, run_task_c_codegen_case  # noqa: E402
from factoropt.task_c import materialize_split, validate_public_payload  # noqa: E402
from factoropt.task_c_iterative import RevisionConfig, iterative_revision_loop  # noqa: E402


def _read_json_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        values = raw if isinstance(raw, list) else [raw]
    else:
        values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not all(isinstance(item, dict) for item in values):
        raise ValueError(f"responses in {path} must be JSON objects")
    return [dict(item) for item in values]


def _load_payload(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        if len(raw) != 1 or not isinstance(raw[0], dict):
            raise ValueError("payload JSON must contain one object")
        raw = raw[0]
    if not isinstance(raw, dict):
        raise ValueError("payload must be a JSON object")
    errors = validate_public_payload(raw)
    if errors:
        raise ValueError("invalid public payload: " + "; ".join(errors[:10]))
    return raw


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", required=True, type=Path, help="one public Task C payload JSON")
    parser.add_argument("--output", required=True, type=Path, help="audit JSON output")
    parser.add_argument("--responses", type=Path, help="offline complete responses JSON/JSONL")
    parser.add_argument("--model", default=os.environ.get("TASK_C_MODEL", "gpt-5"))
    parser.add_argument("--base-url", default=os.environ.get("TASK_C_BASE_URL", "http://localhost:8000/v1"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--max-revisions", type=int, default=3)
    parser.add_argument(
        "--objective",
        choices=("rank_ic_mean", "net_return", "mse"),
        default="rank_ic_mean",
        help="host-metric ranking used when --selection-policy host_metric, and always recorded as a diagnostic",
    )
    parser.add_argument(
        "--selection-policy",
        choices=("llm", "host_metric"),
        default="llm",
        help="who freezes the candidate: the LLM from the valid set, or host max of --objective",
    )
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=0)
    parser.add_argument("--max-output-tokens", type=int, default=8_000)
    parser.add_argument(
        "--code-timeout-seconds",
        type=float,
        help="per monthly generated-code execution timeout (defaults to --timeout-seconds)",
    )
    parser.add_argument("--max-tool-calls", type=int, default=8)
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument(
        "--repair-invalid",
        action="store_true",
        help=(
            "after a malformed/schema-invalid response or exhausted tool budget, "
            "ask one clean no-tools finalization turn (A/B-compatible)"
        ),
    )
    # Scratch files are disposable runtime state and must stay outside the
    # paper-final run index.
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help=(
            "directory the public train/validation artifact descriptors resolve against; "
            "defaults to the payload's own directory. The semantic dataset is an overlay "
            "that does not carry artifacts/, so its driver points this at the numeric "
            "build's case directory, which holds the identical bytes the descriptors hash."
        ),
    )
    parser.add_argument("--scratch-root", type=Path, default=Path(".tmp/task_c_scratch"))
    parser.add_argument("--keep-artifacts", action="store_true")
    parser.add_argument("--no-stop-on-final", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    payload = _load_payload(args.payload)
    artifact_root = (args.artifact_root or args.payload.parent).resolve()
    if not artifact_root.is_dir():
        raise SystemExit(f"--artifact-root is not a directory: {artifact_root}")
    responses = _read_json_rows(args.responses) if args.responses else []
    adapter = TaskCAdapter()
    config = AgenticConfig(
        model=args.model,
        base_url=args.base_url,
        temperature=0.0,
        seed=int(payload.get("metadata", {}).get("seed", 1)),
        max_output_tokens=args.max_output_tokens,
        timeout_seconds=args.timeout_seconds,
        retries=args.retries,
        max_tool_calls=args.max_tool_calls,
        max_turns=args.max_turns,
        tool_choice="auto",
        json_mode=True,
        repair_invalid=args.repair_invalid,
        keep_artifacts=args.keep_artifacts,
    )

    def callback(public: Mapping[str, Any], feedback: Mapping[str, Any] | None, revision: int) -> Any:
        if responses:
            if revision >= len(responses):
                raise RuntimeError(f"offline responses has no revision {revision}")
            return responses[revision]
        request_payload = dict(public)
        request_payload["_revision"] = revision
        if feedback is not None:
            request_payload["_revision_feedback"] = dict(feedback)
        return run_task_c_codegen_case(
            request_payload,
            adapter=adapter,
            config=config,
            api_key=args.api_key,
            scratch_root=args.scratch_root,
            artifact_destination=args.scratch_root / str(public.get("case_id", "case")) if args.keep_artifacts else None,
            public_artifact_root=artifact_root,
        )

    # Artifact paths are relative to the artifact root, never the caller's
    # current working directory. This keeps a large public panel streamable
    # and prevents accidental reads from unrelated paths.
    train = materialize_split(payload, "train", include_label=True, artifact_root=artifact_root)
    valid = materialize_split(payload, "validation", include_label=True, artifact_root=artifact_root)
    audit = iterative_revision_loop(
        payload,
        callback,
        config=RevisionConfig(
            max_revisions=args.max_revisions,
            timeout_seconds=(
                args.code_timeout_seconds
                if args.code_timeout_seconds is not None
                else args.timeout_seconds
            ),
            stop_on_final=not args.no_stop_on_final,
            objective=args.objective,
            selection_policy=args.selection_policy,
            seed=int(payload.get("metadata", {}).get("seed", 1)),
        ),
        train_df=train,
        validation_df=valid,
        metadata=payload.get("metadata", {}),
        scratch_root=args.scratch_root,
        artifact_root=artifact_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"case_id": audit.get("case_id"), "selected_revision": audit.get("selected_revision"), "revision_count": audit.get("revision_count")}, ensure_ascii=False))
    return 0 if audit.get("selected_revision") is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())

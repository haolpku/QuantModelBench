#!/usr/bin/env python3
"""Evaluate a frozen Task C response locally, without any model call."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from factoropt.task_c import (  # noqa: E402
    materialize_split,
    validate_code_response,
    validate_public_payload,
)
from factoropt.semantic import normalize_track  # noqa: E402
from factoropt.task_c_iterative import evaluate_candidate_on_validation  # noqa: E402


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", required=True, type=Path)
    parser.add_argument("--response", required=True, type=Path, help="response JSON or revision audit JSON")
    parser.add_argument("--artifact-root", type=Path, help="root for public train/validation artifacts (default: payload directory)")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    args = parser.parse_args(argv)
    artifact_root = args.artifact_root or args.payload.parent
    payload = _read_json(args.payload)
    payload_errors = validate_public_payload(payload)
    if payload_errors:
        raise ValueError("invalid public payload: " + "; ".join(payload_errors[:10]))
    response_or_audit = _read_json(args.response)
    response = response_or_audit.get("selected_response") or response_or_audit.get("parsed_response") or response_or_audit
    if not isinstance(response, dict):
        raise ValueError("response file has no code response object")
    schema_errors = validate_code_response(response)
    result: dict[str, Any] = {
        "schema_valid": not schema_errors,
        "schema_errors": schema_errors,
        "code_sha256": None,
        "execution_valid": False,
        "validation": None,
    }
    if not schema_errors:
        # Artifact paths in a generated case are relative to the payload
        # directory, not the caller's current working directory.
        train = materialize_split(payload, "train", include_label=True, artifact_root=artifact_root)
        valid = materialize_split(payload, "validation", include_label=True, artifact_root=artifact_root)
        # Use the same chronological leave-future-out protocol as the
        # revision loop.  Passing the complete validation frame as ``valid_df``
        # would expose the target month's labels to the frozen source.
        runtime_metadata = dict(payload.get("metadata") or {})
        track = normalize_track(payload.get("track", "numeric_only"))
        runtime_metadata.pop("factor_cards", None)
        runtime_metadata.pop("semantic_cards", None)
        runtime_metadata["track"] = track
        if track == "semantic":
            runtime_metadata["factor_cards"] = [
                dict(card)
                for card in payload.get("factor_cards", [])
                if isinstance(card, dict)
            ]
        metrics, prediction, executions = evaluate_candidate_on_validation(
            response,
            train_df=train,
            validation_df=valid,
            metadata=runtime_metadata,
            timeout_seconds=args.timeout_seconds,
        )
        result["code_sha256"] = executions[0].code_sha256 if executions else None
        result["execution_valid"] = bool(metrics.get("execution_valid", False))
        result["execution_errors"] = list(metrics.get("errors", []))[:20]
        result["execution"] = [item.audit_dict() for item in executions]
        result["validation"] = metrics
        result["prediction_rows"] = int(len(prediction))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, default=str))
    return 0 if result["schema_valid"] and result["execution_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

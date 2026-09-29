#!/usr/bin/env python3
"""Package the minimal Task B frozen-code experiment for rclone transfer.

By default the archive contains the public generation payload only.  Add
``--include-rolling-truth`` when the destination is a trusted evaluator host;
that file contains hidden target returns and must never be mounted in the LLM
server's public/tool payload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[3]


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy(source: Path, staging: Path, relative: str) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = staging / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def build_bundle(case_dir: Path, output_dir: Path, archive: Path, include_truth: bool) -> dict[str, Any]:
    payload = case_dir / "payload.json"
    if not payload.is_file():
        raise FileNotFoundError(f"case has no payload.json: {case_dir}")
    truth = case_dir / "rolling_truth.jsonl"
    if include_truth and not truth.is_file():
        raise FileNotFoundError(f"--include-rolling-truth requested but missing: {truth}")
    if output_dir.exists() or archive.exists():
        raise FileExistsError("output directory/archive already exists; choose a new path")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    archive.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=str(output_dir.parent)))
    try:
        # Task B business code and shared agentic transport.
        files = [
            "task_b/code/factoropt/__init__.py",
            "task_b/code/factoropt/task_b.py",
            "task_b/code/factoropt/semantic.py",
            "task_b/code/factoropt/task_b_frozen.py",
            "task_b/code/factoropt/agentic/__init__.py",
            "task_b/code/factoropt/agentic/adapters.py",
            "task_b/code/factoropt/agentic/frozen_adapter.py",
            "task_b/code/factoropt/agentic/loop.py",
            "task_b/code/factoropt/agentic/tools.py",
            # Task-neutral transport imported by name from ``shared/``; it must
            # ship beside ``task_b/`` for the bundle to run standalone.
            "shared/__init__.py",
            "shared/agentic/__init__.py",
            "shared/agentic/loop.py",
            "shared/agentic/tools.py",
            "shared/agentic/protocol.py",
            "task_b/code/scripts/run_task_b_codegen.py",
            "task_b/code/scripts/run_task_b_rolling.py",
            "task_b/code/scripts/smoke_task_b.py",
            "task_b/code/schemas/task_b_code_response.schema.json",
            "task_b/code/schemas/task_b_selection_response.schema.json",
            "task_b/prompt/prompt_b_frozen_code.md",
            # The frozen Task B module imports these two files directly as a
            # standalone policy/checking dependency, avoiding Task C package
            # name collisions.
            "task_c/code/factoropt/task_c.py",
            "task_c/code/factoropt/semantic.py",
            "task_c/code/factoropt/task_c_code_runner.py",
            "task_c/code/schemas/task_c_code_response.schema.json",
        ]
        for relative in files:
            _copy(ROOT / relative, staging, relative)
        data_dir = staging / "data" / "frozen_case"
        data_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(payload, data_dir / "payload.json")
        if include_truth:
            shutil.copy2(truth, data_dir / "rolling_truth.jsonl")
        requirements = """numpy>=1.26,<3
pandas>=2.1,<3
scikit-learn>=1.3,<2
jsonschema>=4.20,<5
"""
        (staging / "requirements-frozen.txt").write_text(requirements, encoding="utf-8")
        (staging / "FROZEN_PROTOCOL.md").write_text(
            """# Task B frozen-code protocol

`run_task_b_codegen.py` is the pre-test LLM stage. It may use zero or
a bounded number of validation revisions. After a source is selected,
`run_task_b_rolling.py` executes the same code locally for each monthly
origin and makes zero provider calls.

Both the pre-test validation pass and the hidden test pass are chronological:
the current month's label is kept outside the candidate until scoring, and a
settled target is available only at the next origin.

Never send `rolling_truth.jsonl` to the model or include it in a public
payload.
""",
            encoding="utf-8",
        )
        manifest = {
            "schema_version": "task_b_frozen_bundle.v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "contains_api_key": False,
            "contains_hidden_rolling_truth": bool(include_truth),
            # Keep local absolute paths out of an archive that may be uploaded
            # to a model host.
            "case_source": case_dir.name,
            "files": [],
        }
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                manifest["files"].append({
                    "path": str(path.relative_to(staging)).replace("\\", "/"),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                })
        (staging / "bundle_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        staging.replace(output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    temporary_archive = archive.with_name(f".{archive.name}.tmp")
    try:
        with tarfile.open(temporary_archive, mode="w:gz", compresslevel=6) as handle:
            handle.add(output_dir, arcname="TaskBFrozen")
        temporary_archive.replace(archive)
    except Exception:
        temporary_archive.unlink(missing_ok=True)
        raise
    return {
        "output_dir": str(output_dir),
        "archive": str(archive),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": _sha256(archive),
        "contains_hidden_rolling_truth": bool(include_truth),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", required=True, help="directory made by build_task_b_case.py")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--archive", help="defaults to <output-dir>.tar.gz")
    parser.add_argument("--include-rolling-truth", action="store_true")
    args = parser.parse_args(argv)
    case_dir = _resolve(args.case_dir)
    output_dir = _resolve(args.output_dir)
    archive = _resolve(args.archive) if args.archive else Path(str(output_dir) + ".tar.gz")
    try:
        result = build_bundle(case_dir, output_dir, archive, args.include_rolling_truth)
    except (FileNotFoundError, FileExistsError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

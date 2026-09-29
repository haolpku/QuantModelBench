#!/usr/bin/env python3
"""Build a minimal public Task C code-generation bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


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


def _copy(source: Path, staging: Path, relative: str, records: list[dict[str, Any]]) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    target = staging / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    records.append({"path": relative, "bytes": source.stat().st_size, "sha256": _sha256(source)})


def _archive_mode(path: Path) -> str:
    if path.name.endswith((".tar.xz", ".txz")):
        return "w:xz"
    if path.name.endswith((".tar.gz", ".tgz")):
        return "w:gz"
    raise ValueError("archive must end in .tar.xz, .txz, .tar.gz, or .tgz")


# The generated code runs in an isolated ``python -I`` subprocess that only
# sees the interpreter's own site-packages, so every model family named in
# ``task_c_code_runner.ALLOWED_IMPORT_ROOTS`` (plus its transitive runtime deps)
# has to be installed there or a whitelisted import breaks at sandbox time.
# pyarrow is separately required by the parquet public-artifact reads and writes.
SERVER_REQUIREMENTS = """numpy>=1.26,<3
pandas>=2.1,<3
scipy>=1.11,<2
scikit-learn>=1.3,<2
jsonschema>=4.20,<5
pyarrow>=14,<26
torch>=2.4
xgboost>=2.0,<4
lightgbm>=4.0,<5
catboost>=1.2,<2
statsmodels>=0.14,<1
optuna>=3.6,<5
typing-extensions>=4.10
packaging>=23
filelock>=3.13
fsspec>=2024.2
PyYAML>=6.0
tqdm>=4.66
colorlog>=6.8
SQLAlchemy>=2.0
alembic>=1.13
"""


RUNTIME_FILES = (
    # Task-neutral transport imported by name from ``shared/``.
    "shared/__init__.py",
    "shared/agentic/__init__.py",
    "shared/agentic/loop.py",
    "shared/agentic/tools.py",
    "shared/agentic/protocol.py",
    "task_c/code/factoropt/__init__.py",
    "task_c/code/factoropt/task_c_backtest.py",
    "task_c/code/factoropt/task_c.py",
    "task_c/code/factoropt/semantic.py",
    "task_c/code/factoropt/task_c_code_runner.py",
    "task_c/code/factoropt/task_c_data.py",
    "task_c/code/factoropt/task_c_iterative.py",
    "task_c/code/factoropt/agentic/__init__.py",
    "task_c/code/factoropt/agentic/adapters.py",
    "task_c/code/factoropt/agentic/loop.py",
    "task_c/code/scripts/run_task_c_codegen.py",
    "task_c/code/scripts/run_task_c_codegen_parallel.py",
    "task_c/code/schemas/task_c_code_response.schema.json",
    "task_c/code/schemas/task_c_selection_response.schema.json",
    "task_c/prompt/prompt_c.md",
)


def build_bundle(
    dataset: Path,
    split_manifest: Path,
    output_dir: Path,
    archive: Path,
    excluded: set[str],
    included: set[str] | None = None,
) -> dict[str, Any]:
    if output_dir.exists() or archive.exists():
        raise FileExistsError("output directory/archive already exists; choose a new path")
    selected = [
        row for row in _rows(split_manifest)
        if row.get("record_type") == "case" and str(row.get("case_id")) not in excluded
    ]
    if included is not None:
        available = {str(row.get("case_id")) for row in selected}
        missing = included - available
        if missing:
            raise ValueError(
                "requested case IDs are absent from the selected split: "
                + ", ".join(sorted(missing))
            )
        selected = [row for row in selected if str(row.get("case_id")) in included]
    if not selected:
        raise ValueError("no cases remain after exclusions")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=str(output_dir.parent)))
    records: list[dict[str, Any]] = []
    try:
        for relative in RUNTIME_FILES:
            _copy(ROOT / relative, staging, relative, records)
        requirements = staging / "task_c" / "code" / "requirements-server.txt"
        requirements.write_text(SERVER_REQUIREMENTS, encoding="utf-8")
        records.append({"path": str(requirements.relative_to(staging)), "bytes": requirements.stat().st_size, "sha256": _sha256(requirements)})
        # Mirror the source dataset's directory name and split inside the
        # bundle.  dev and test share one merged build, so the name has to
        # carry the split (and numeric vs _semantic) that a fixed name would
        # flatten: the two tracks reuse identical case ids and block names.
        bundle_root = f"data/{dataset.name}"
        bundle_split = split_manifest.parent.name
        bundle_dataset = staging / "data" / dataset.name
        bundle_manifest = bundle_dataset / "splits" / bundle_split / "manifest.jsonl"
        bundle_manifest.parent.mkdir(parents=True, exist_ok=True)
        with bundle_manifest.open("w", encoding="utf-8", newline="\n") as handle:
            for row in selected:
                block = str(row.get("block_id") or "")
                case_id = str(row.get("case_id") or "")
                if not block or not case_id:
                    raise ValueError("case row has empty block_id or case_id")
                source_case = dataset / "cases" / block / case_id
                for name in ("payload.json", "manifest.json", "artifacts/train.parquet", "artifacts/validation.parquet"):
                    _copy(source_case / name, staging, f"{bundle_root}/cases/{block}/{case_id}/{name}", records)
                record = dict(row)
                record["path"] = f"{bundle_root}/cases/{block}/{case_id}"
                record["case_manifest_path"] = f"{bundle_root}/cases/{block}/{case_id}/manifest.json"
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        records.append({"path": str(bundle_manifest.relative_to(staging)), "bytes": bundle_manifest.stat().st_size, "sha256": _sha256(bundle_manifest)})
        manifest = {
            "schema_version": "task_c_dev_bundle.v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_dataset": str(dataset.relative_to(ROOT)).replace("\\\\", "/"),
            "source_split_manifest": str(split_manifest.relative_to(ROOT)).replace("\\\\", "/"),
            "bundle_dataset_root": bundle_root,
            "bundle_split": bundle_split,
            "selected_case_count": len(selected),
            "excluded_completed_case_ids": sorted(excluded),
            "included_case_ids": sorted(included) if included is not None else None,
            "contains_api_key": False,
            "contains_raw_source_data": False,
            "contains_hidden_rolling_truth": False,
            "files": records,
        }
        (staging / "bundle_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (staging / "SERVER_README.md").write_text(
            f"# Task C remaining {bundle_split} bundle\n\n"
            f"Contains public generation artifacts and the local validation runtime for the remaining {bundle_split} cases. "
            "It contains neither provider credentials nor hidden rolling truth/raw WRDS data.\n",
            encoding="utf-8",
        )
        staging.replace(output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    temporary = archive.with_name(f".{archive.name}.tmp")
    try:
        mode = _archive_mode(archive)
        kwargs = {"compresslevel": 6} if mode == "w:gz" else {}
        with tarfile.open(temporary, mode=mode, **kwargs) as handle:
            handle.add(output_dir, arcname="TaskCDev")
        temporary.replace(archive)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "output_dir": str(output_dir),
        "archive": str(archive),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": _sha256(archive),
        "selected_case_count": len(selected),
        "excluded_case_count": len(excluded),
    }


def _split_manifest(parser: argparse.ArgumentParser, dataset: Path, args: argparse.Namespace) -> Path:
    """Resolve the split manifest, rejecting one that belongs to another dataset.

    dev and test share one merged 207-case build, so the old "different block
    numbering" hazard is gone.  The guard still catches the live mistake: the
    numeric dataset and its ``_semantic`` sibling carry identical case ids and
    block names, so a manifest from one paired with ``--dataset`` pointing at
    the other resolves silently and only surfaces as a wrong ``track`` later.
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
    parser.add_argument("--output-dir", default=".tmp/task_c_dev_remaining_bundle")
    parser.add_argument("--archive", default=".tmp/task_c_dev_remaining_bundle.tar.gz")
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument(
        "--case-id-file",
        action="append",
        default=[],
        help="file containing one case ID per line; may be repeated",
    )
    parser.add_argument("--exclude-case-id", action="append", default=[])
    parser.add_argument("--exclude-case-id-file")
    args = parser.parse_args(argv)
    dataset = _resolve(args.dataset)
    split_manifest = _split_manifest(parser, dataset, args)
    excluded = {str(value) for value in args.exclude_case_id}
    if args.exclude_case_id_file:
        excluded.update(_case_ids(_resolve(args.exclude_case_id_file)))
    included = {str(value) for value in args.case_id}
    for case_id_file in args.case_id_file:
        included.update(_case_ids(_resolve(case_id_file)))
    try:
        result = build_bundle(
            dataset,
            split_manifest,
            _resolve(args.output_dir),
            _resolve(args.archive),
            excluded,
            included or None,
        )
    except (FileNotFoundError, FileExistsError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

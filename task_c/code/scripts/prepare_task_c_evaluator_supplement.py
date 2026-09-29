#!/usr/bin/env python3
"""Build a small evaluator-only supplement for selected Task C cases."""

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
EVALUATOR_FILES = (
    "rolling_truth.jsonl",
    "evaluator_id_map.json",
    "evaluator_feature_map.json",
)
RUNTIME_FILES = (
    "task_c/code/scripts/run_task_c_rolling.py",
    "task_c/code/scripts/run_task_c_rolling_parallel.py",
)


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


def build_bundle(
    *,
    dataset: Path,
    split_manifest: Path,
    output_dir: Path,
    archive: Path,
    included: set[str],
    nyse_cutoffs: Path,
) -> dict[str, Any]:
    if not included:
        raise ValueError("at least one --case-id or --case-id-file is required")
    if output_dir.exists() or archive.exists():
        raise FileExistsError("output directory/archive already exists; choose a new path")
    case_rows = [row for row in _rows(split_manifest) if row.get("record_type") == "case"]
    available = {str(row.get("case_id")) for row in case_rows}
    missing = included - available
    if missing:
        raise ValueError("requested case IDs are absent from the manifest: " + ", ".join(sorted(missing)))
    selected = [row for row in case_rows if str(row.get("case_id")) in included]
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=str(output_dir.parent)))
    records: list[dict[str, Any]] = []
    dataset_relative = f"data/{dataset.name}"
    try:
        for relative in RUNTIME_FILES:
            _copy(ROOT / relative, staging, relative, records)
        _copy(nyse_cutoffs, staging, "data/raw/jkp/2025-12/nyse_cutoffs.csv", records)
        for row in selected:
            block = str(row.get("block_id") or "")
            case_id = str(row.get("case_id") or "")
            if not block or not case_id:
                raise ValueError("case row has empty block_id or case_id")
            source_case = dataset / "cases" / block / case_id
            for name in EVALUATOR_FILES:
                _copy(
                    source_case / name,
                    staging,
                    f"{dataset_relative}/cases/{block}/{case_id}/{name}",
                    records,
                )
        id_path = staging / "task_c" / "config" / "task_c_bundle_case_ids.txt"
        id_path.parent.mkdir(parents=True, exist_ok=True)
        id_path.write_text("".join(f"{case_id}\n" for case_id in sorted(included)), encoding="utf-8")
        records.append({"path": str(id_path.relative_to(staging)), "bytes": id_path.stat().st_size, "sha256": _sha256(id_path)})
        manifest = {
            "schema_version": "task_c_evaluator_supplement.v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_dataset": str(dataset.relative_to(ROOT)).replace("\\", "/"),
            "source_split_manifest": str(split_manifest.relative_to(ROOT)).replace("\\", "/"),
            "selected_case_count": len(selected),
            "included_case_ids": sorted(included),
            "contains_api_key": False,
            "contains_evaluator_only_files": True,
            "contains_global_panel": False,
            "contains_global_daily_returns": False,
            "files": records,
        }
        (staging / "bundle_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (staging / "SERVER_README.md").write_text(
            "# Task C evaluator-only supplement\n\n"
            "Extract this only on the trusted evaluator. It contains rolling truth and reversible feature/ID maps. "
            "It deliberately excludes the large WRDS panel and daily-return files.\n",
            encoding="utf-8",
        )
        staging.replace(output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    temporary = archive.with_name(f".{archive.name}.tmp")
    try:
        with tarfile.open(temporary, mode=_archive_mode(archive)) as handle:
            handle.add(output_dir, arcname="TaskCEvaluatorSupplement")
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
    parser.add_argument("--output-dir", default=".tmp/task_c_evaluator_supplement")
    parser.add_argument("--archive", default=".tmp/task_c_evaluator_supplement.tar.xz")
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--case-id-file", action="append", default=[])
    parser.add_argument("--nyse-cutoffs", default="data/raw/jkp/2025-12/nyse_cutoffs.csv")
    args = parser.parse_args(argv)
    dataset = _resolve(args.dataset)
    split_manifest = _split_manifest(parser, dataset, args)
    included = set(args.case_id)
    for case_id_file in args.case_id_file:
        included.update(_case_ids(_resolve(case_id_file)))
    try:
        result = build_bundle(
            dataset=dataset,
            split_manifest=split_manifest,
            output_dir=_resolve(args.output_dir),
            archive=_resolve(args.archive),
            included=included,
            nyse_cutoffs=_resolve(args.nyse_cutoffs),
        )
    except (FileNotFoundError, FileExistsError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

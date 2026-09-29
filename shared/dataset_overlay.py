"""Resolve case files across an overlay dataset and its base.

A ``semantic`` build differs from its ``numeric_only`` sibling in a handful of
per-case files; the rest is byte-for-byte identical.  Materialising both copies
cost 10.4 GB for Task C alone, and every byte of it had to be uploaded twice.

An overlay dataset therefore stores only the files that actually differ and
declares where the rest lives::

    {
      "schema_version": "dataset_overlay.v1",
      "base_dataset": "task_c_frozen_raw_rolling_207",
      "overlay_files": ["payload.json", "manifest.json"],
      "base_files": ["artifacts/train.parquet", "artifacts/validation.parquet"]
    }

``base_dataset`` is a *sibling directory name*, resolved against the overlay's
parent.  The server root mirrors this repository, so one value is correct on
both sides with no absolute paths and no launcher flag.

``overlay_files`` and ``base_files`` are documentation for a human reading the
directory: resolution is by existence, so a file present locally always wins.
A dataset with no ``overlay.json`` resolves entirely against itself, which
keeps every existing non-overlay dataset working unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

OVERLAY_FILENAME = "overlay.json"
SCHEMA_VERSION = "dataset_overlay.v1"


def load_overlay(dataset: Path) -> dict[str, Any] | None:
    """Return the overlay spec for ``dataset``, or ``None`` when it has none."""

    spec_path = Path(dataset) / OVERLAY_FILENAME
    if not spec_path.is_file():
        return None
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read overlay spec {spec_path}: {exc}") from exc
    if not isinstance(spec, Mapping):
        raise ValueError(f"overlay spec must be an object: {spec_path}")
    version = str(spec.get("schema_version") or "")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"{spec_path}: unsupported overlay schema_version {version!r}; expected {SCHEMA_VERSION!r}"
        )
    if not str(spec.get("base_dataset") or "").strip():
        raise ValueError(f"{spec_path}: base_dataset is required")
    return dict(spec)


def base_dataset(dataset: Path) -> Path | None:
    """Resolve the base dataset directory, or ``None`` for a plain dataset.

    The base name must stay a single path segment: an overlay may only point at
    a sibling build, never at an arbitrary location on the host.
    """

    dataset = Path(dataset)
    spec = load_overlay(dataset)
    if spec is None:
        return None
    name = str(spec["base_dataset"]).strip()
    if "/" in name or "\\" in name or name in {"", ".", ".."}:
        raise ValueError(
            f"{dataset / OVERLAY_FILENAME}: base_dataset must be a sibling directory name, got {name!r}"
        )
    base = (dataset.parent / name).resolve()
    if not base.is_dir():
        raise FileNotFoundError(
            f"{dataset / OVERLAY_FILENAME}: base dataset does not exist: {base}"
        )
    if base == dataset.resolve():
        raise ValueError(f"{dataset / OVERLAY_FILENAME}: base_dataset points at the overlay itself")
    return base


def case_dir(dataset: Path, block: str, case_id: str) -> Path:
    return Path(dataset) / "cases" / str(block) / str(case_id)


def resolve_case_file(dataset: Path, block: str, case_id: str, name: str) -> Path:
    """Return the real path of a per-case file, falling back to the base build.

    ``name`` is a case-relative path such as ``payload.json`` or
    ``artifacts/train.parquet``.
    """

    local = case_dir(dataset, block, case_id) / name
    if local.is_file():
        return local
    base = base_dataset(dataset)
    if base is None:
        raise FileNotFoundError(f"case file not found: {local}")
    candidate = case_dir(base, block, case_id) / name
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        f"case file {name!r} not found for {block}/{case_id}; looked in {local} and {candidate}"
    )


def resolve_artifact_root(dataset: Path, block: str, case_id: str) -> Path:
    """Return the case directory that actually holds ``artifacts/``.

    Task C resolves the public train/validation Parquet descriptors relative to
    an artifact root, and the readers enforce that the file stays inside it.
    Pointing that root at the base case directory is what lets a slimmed
    overlay serve the same bytes the descriptors were hashed from.
    """

    local = case_dir(dataset, block, case_id)
    if (local / "artifacts").is_dir():
        return local
    base = base_dataset(dataset)
    if base is None:
        return local
    candidate = case_dir(base, block, case_id)
    if (candidate / "artifacts").is_dir():
        return candidate
    return local


__all__ = [
    "OVERLAY_FILENAME",
    "SCHEMA_VERSION",
    "base_dataset",
    "case_dir",
    "load_overlay",
    "resolve_artifact_root",
    "resolve_case_file",
]

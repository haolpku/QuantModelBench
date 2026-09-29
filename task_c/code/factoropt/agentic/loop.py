"""Task C's view of the shared provider-neutral tool loop.

The Chat Completions protocol, tool dispatcher, retries and audit logic live in
``shared/agentic`` at the repository (or bundle) root, so Task C imports them by
name instead of loading Task B's package across the filesystem.  What stays here
is Task C's own part: validating the public payload and mounting the
evaluator-only artifact files that the provider must not see.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping


def _ensure_shared_on_path() -> None:
    """Put the repository (or bundle) root on ``sys.path``.

    Task C runs both from the repository and from a generated bundle whose
    runtime files sit at ``<root>/task_c/code/...`` beside ``<root>/shared/...``.
    Walking up to the first ancestor that really holds ``shared/agentic`` keeps
    both layouts working with no hard-coded depth and no extra ``PYTHONPATH``
    entry, and it also works when this module is executed under a private
    ``spec_from_file_location`` alias.
    """

    for parent in Path(__file__).resolve().parents:
        if (parent / "shared" / "agentic" / "loop.py").is_file():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return
    raise ImportError(f"cannot locate the shared/ package above {Path(__file__).resolve()}")


_ensure_shared_on_path()


from shared.agentic.loop import (  # noqa: E402
    AgenticConfig,
    chat_endpoint,
    parse_json_object,
    run_agentic_case,
    safe_endpoint_for_record,
)

from ..task_c import validate_public_payload  # noqa: E402


def run_task_c_codegen_case(
    payload: Mapping[str, Any],
    *,
    adapter: Any,
    config: Any,
    api_key: str = "",
    scratch_root: str | Path,
    artifact_destination: str | Path | None = None,
    public_artifact_root: str | Path | None = None,
    dry_run: bool = False,
    request_fn: Any = None,
) -> dict[str, Any]:
    """Run one Task C revision through the shared generic loop.

    Large artifact-only payloads need their historical rows available to the
    evaluator-side ``python_exec`` process, while the provider should receive
    only the compact public descriptor.  We therefore pass the source files
    through the shared loop's evaluator-only mount argument instead of adding
    absolute local paths to the model payload.
    """

    payload_errors = validate_public_payload(payload)
    if payload_errors:
        raise ValueError("invalid public Task C payload: " + "; ".join(payload_errors[:10]))
    request_payload = dict(payload)
    # A caller cannot smuggle an evaluator mount through the provider-facing
    # payload.  Mounts are reconstructed from the declared train/validation
    # artifact descriptors below.
    request_payload.pop("_public_artifact_files", None)
    public_files: dict[str, str] = {}
    if not dry_run:
        root_value = public_artifact_root
        if root_value is None:
            root_value = payload.get("artifact_root") or "."
        artifact_root = Path(root_value).expanduser().resolve()
        for split in ("train", "validation"):
            section = request_payload.get(split)
            if not isinstance(section, Mapping):
                continue
            descriptor = section.get("artifact")
            if not isinstance(descriptor, Mapping):
                continue
            raw_path = descriptor.get("path")
            if not isinstance(raw_path, str) or not raw_path.strip():
                continue
            candidate = Path(raw_path).expanduser()
            if not candidate.is_absolute():
                candidate = artifact_root / candidate
            candidate = candidate.resolve()
            if candidate != artifact_root and artifact_root not in candidate.parents:
                raise ValueError(f"public {split} artifact escapes public_artifact_root")
            if not candidate.is_file():
                raise FileNotFoundError(f"public {split} artifact does not exist: {candidate}")
            public_files[split] = str(candidate)

    scratch_parent = Path(scratch_root).expanduser().resolve()
    scratch_parent.mkdir(parents=True, exist_ok=True)
    # A unique child avoids collisions when several origins/revisions are run
    # concurrently.  The shared finalizer removes it after the case; the
    # ``finally`` also cleans up failures that happen before workspace setup.
    case_scratch = Path(tempfile.mkdtemp(prefix="task_c_agentic_", dir=str(scratch_parent)))
    destination = Path(artifact_destination) if artifact_destination is not None else None
    try:
        return run_agentic_case(
            request_payload,
            adapter=adapter,
            config=config,
            api_key=api_key,
            scratch_root=case_scratch,
            artifact_destination=destination,
            public_artifact_files=public_files or None,
            dry_run=dry_run,
            request_fn=request_fn,
        )
    finally:
        if case_scratch.exists():
            shutil.rmtree(case_scratch, ignore_errors=True)


__all__ = [
    "AgenticConfig",
    "chat_endpoint",
    "parse_json_object",
    "run_agentic_case",
    "run_task_c_codegen_case",
    "safe_endpoint_for_record",
]

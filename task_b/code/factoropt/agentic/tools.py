"""Task B's view of the shared bounded tool suite.

The tool dispatcher is task-neutral and now lives in ``shared/agentic/tools.py``.
This module keeps ``factoropt.agentic.tools`` importable for existing call sites
and for bundle manifests that list it explicitly.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_shared_on_path() -> None:
    """Put the repository (or bundle) root on ``sys.path``.

    Task B runs both from the repository and from a generated frozen bundle
    whose runtime files sit at ``<root>/task_b/code/...`` beside
    ``<root>/shared/...``.  Walking up to the first ancestor that really holds
    ``shared/agentic`` keeps both layouts working with no hard-coded depth and
    no extra ``PYTHONPATH`` entry.
    """

    for parent in Path(__file__).resolve().parents:
        if (parent / "shared" / "agentic" / "loop.py").is_file():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return
    raise ImportError(f"cannot locate the shared/ package above {Path(__file__).resolve()}")


_ensure_shared_on_path()


from shared.agentic.tools import (
    ALLOWED_IMPORT_ROOTS,
    MAX_ARTIFACT_BYTES,
    MAX_ARTIFACT_COUNT,
    MAX_ARTIFACT_TOTAL_BYTES,
    MAX_CODE_CHARS,
    MAX_PURPOSE_CHARS,
    MAX_READ_CHARS,
    MAX_TIMEOUT_SECONDS,
    MAX_TOOL_OUTPUT_CHARS,
    TOOL_SCHEMAS,
    ScratchWorkspace,
    ToolDispatcher,
    runtime_metadata,
)

__all__ = [
    "ALLOWED_IMPORT_ROOTS",
    "MAX_ARTIFACT_BYTES",
    "MAX_ARTIFACT_COUNT",
    "MAX_ARTIFACT_TOTAL_BYTES",
    "MAX_CODE_CHARS",
    "MAX_PURPOSE_CHARS",
    "MAX_READ_CHARS",
    "MAX_TIMEOUT_SECONDS",
    "MAX_TOOL_OUTPUT_CHARS",
    "TOOL_SCHEMAS",
    "ScratchWorkspace",
    "ToolDispatcher",
    "runtime_metadata",
]

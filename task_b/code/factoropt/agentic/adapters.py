"""Task B's view of the task-adapter contract.

The contract itself (``TaskAdapter``, ``resolved_task_variant``) is task-neutral
and lives in ``shared/agentic/protocol.py``, next to the loop that consumes it.
This module re-exports it so the existing ``factoropt.agentic.adapters`` import
path keeps working for the runner and smoke scripts.

Task B's own adapter is :class:`factoropt.agentic.frozen_adapter.TaskBFrozenCodeAdapter`,
which implements ``validate``/``score`` against the frozen-code contract.
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


from shared.agentic.protocol import (  # noqa: E402
    TaskAdapter,
    _as_mapping,
    resolved_task_variant,
)

__all__ = ["TaskAdapter", "resolved_task_variant"]

"""Task B adapters over the shared agentic tool loop.

The provider-neutral loop and tools live in ``shared/agentic``.  What stays here
is Task B's own part: the frozen-code adapter, the legacy routing adapter, and
the re-export shims that keep ``factoropt.agentic.loop`` / ``.tools`` importable
for existing call sites and bundle manifests.
"""

from .loop import AgenticConfig, normalize_base_url, run_agentic_case
from .tools import TOOL_SCHEMAS, ToolDispatcher, runtime_metadata

try:
    from .frozen_adapter import TaskBFrozenCodeAdapter
except ImportError:  # pragma: no cover - only exercised by a private alias loader
    TaskBFrozenCodeAdapter = None  # type: ignore[assignment]

__all__ = [
    "AgenticConfig",
    "normalize_base_url",
    "TOOL_SCHEMAS",
    "ToolDispatcher",
    "run_agentic_case",
    "runtime_metadata",
    "TaskBFrozenCodeAdapter",
]

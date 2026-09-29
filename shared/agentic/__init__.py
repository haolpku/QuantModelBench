"""Provider-neutral agentic tool loop shared by Task B and Task C.

The task-specific adapter decides what to ask for and how to score the answer;
this package only speaks the Chat Completions ``tools`` protocol, runs the
bounded ``python_exec``/artifact tools, and writes one audit record per case.

Task packages reach it through ordinary imports once the repository root is on
``sys.path`` (see ``shared/sitecustomize.py``).  ``factoropt.agentic.loop`` and
``factoropt.agentic.tools`` remain as thin re-exports so existing call sites
and bundle manifests keep working unchanged.
"""

from .loop import AgenticConfig, normalize_base_url, run_agentic_case
from .protocol import TaskAdapter, resolved_task_variant
from .tools import TOOL_SCHEMAS, ToolDispatcher, runtime_metadata

__all__ = [
    "AgenticConfig",
    "normalize_base_url",
    "run_agentic_case",
    "TaskAdapter",
    "resolved_task_variant",
    "TOOL_SCHEMAS",
    "ToolDispatcher",
    "runtime_metadata",
]

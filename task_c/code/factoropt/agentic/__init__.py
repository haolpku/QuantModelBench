"""Task C adapter over the shared provider-neutral agentic loop."""

from .adapters import TaskCAdapter
from .loop import AgenticConfig, chat_endpoint, run_agentic_case, run_task_c_codegen_case

__all__ = [
    "AgenticConfig",
    "TaskCAdapter",
    "chat_endpoint",
    "run_agentic_case",
    "run_task_c_codegen_case",
]

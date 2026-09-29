"""The adapter contract the shared tool loop consumes.

Only an adapter knows what to ask a provider for and how to score the answer.
Keeping the contract here (rather than inside either task package) is what lets
Task B and Task C both import the shared loop without either one owning it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Protocol


class TaskAdapter(Protocol):
    """Interface consumed by :func:`shared.agentic.loop.run_agentic_case`."""

    task_name: str

    def prompt(self, payload: Mapping[str, Any]) -> str: ...

    def prompt_version(self, payload: Mapping[str, Any]) -> str: ...

    def schema_path(self, payload: Mapping[str, Any]) -> Path: ...

    def validate(self, payload: Mapping[str, Any], response: Any) -> list[str]: ...

    def score(
        self,
        payload: Mapping[str, Any],
        response: Mapping[str, Any] | None,
        truth: Mapping[str, Any] | None,
    ) -> dict[str, Any] | None: ...


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def resolved_task_variant(task_name: str, payload: Mapping[str, Any]) -> str:
    """Return the explicit or contract-default public task variant.

    Explicit ``task_variant`` always wins: a frozen rolling case carries its own
    variant string and the audit must record that exact value, not a guess
    derived from the horizon.
    """

    explicit = str(payload.get("task_variant") or "").strip()
    if explicit:
        return explicit
    if task_name == "factor_grouping":
        return "real_grouping"
    if task_name == "factor_routing":
        try:
            horizon = int(payload.get("horizon_months", 1))
        except (TypeError, ValueError):
            horizon = 1
        return "next_month_factor_routing" if horizon == 1 else "forward_horizon_factor_routing"
    return "unknown"


__all__ = ["TaskAdapter", "resolved_task_variant"]

"""Cross-task (Task B / Task C) rubric evaluation layer.

Implements `task_bc_metric_rubric.md`: Outcome (Absolute Quality + Baseline
Uplift), Process (Reliability / Stability / Judge / Efficiency) and the
Outcome-Process Pareto decision layer.

The package never re-runs an experiment; it only reads persisted run roots.
All thresholds and scales are frozen on dev via `scripts/freeze_dev_config.py`
and consumed as data by `scripts/compute_rubric.py`.
"""

from __future__ import annotations

__all__ = [
    "runspec",
    "loaders",
    "metrics",
    "stats",
    "outcome",
    "process",
    "judge",
    "codeshape",
    "frozen",
    "report",
]

SCHEMA_VERSION = "bceval.v1"

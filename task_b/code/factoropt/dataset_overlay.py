"""Task B's view of the shared overlay resolver.

The resolver is task-neutral and lives in ``shared/dataset_overlay.py``.  This
module keeps ``factoropt.dataset_overlay`` importable from the task package and
from a generated bundle, using the same ancestor walk as
``factoropt.agentic.tools`` so no ``PYTHONPATH`` entry for ``shared/`` is needed.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_shared_on_path() -> None:
    for parent in Path(__file__).resolve().parents:
        if (parent / "shared" / "agentic" / "loop.py").is_file():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return
    raise ImportError(f"cannot locate the shared/ package above {Path(__file__).resolve()}")


_ensure_shared_on_path()


from shared.dataset_overlay import (  # noqa: E402
    OVERLAY_FILENAME,
    SCHEMA_VERSION,
    base_dataset,
    case_dir,
    load_overlay,
    resolve_artifact_root,
    resolve_case_file,
)

__all__ = [
    "OVERLAY_FILENAME",
    "SCHEMA_VERSION",
    "base_dataset",
    "case_dir",
    "load_overlay",
    "resolve_artifact_root",
    "resolve_case_file",
]

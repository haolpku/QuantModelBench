"""Task B's view of the shared provider-neutral tool loop.

The implementation now lives in ``shared/agentic/loop.py`` so Task C can import
it by name instead of loading this package across the filesystem.  This module
stays as the Task B import path (``factoropt.agentic.loop``) and re-exports the
same objects, so runner code, tests and bundle manifests are unchanged.
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


from shared.agentic.loop import (
    AUTO_PAYLOAD_SUMMARY_THRESHOLD,
    HIDDEN_PAYLOAD_KEYS,
    AgenticConfig,
    ProviderHTTPError,
    RequestFunction,
    chat_endpoint,
    normalize_base_url,
    parse_json_object,
    run_agentic_case,
    safe_endpoint_for_record,
)

__all__ = [
    "AUTO_PAYLOAD_SUMMARY_THRESHOLD",
    "HIDDEN_PAYLOAD_KEYS",
    "AgenticConfig",
    "ProviderHTTPError",
    "RequestFunction",
    "chat_endpoint",
    "normalize_base_url",
    "parse_json_object",
    "run_agentic_case",
    "safe_endpoint_for_record",
]

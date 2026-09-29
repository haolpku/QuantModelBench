#!/usr/bin/env python3
"""Download selected authorized HF data components, never experiment credentials.

Use --dry-run to inspect paths without authentication or network access.
HF_TOKEN or the local Hugging Face login supplies authorization at runtime.
Downloaded case directories include evaluator-only material: never expose the
whole download directory to an LLM or generated-code worker.
"""
from __future__ import annotations

import argparse
from pathlib import Path

REPO_ID = "ArolaLuo/QuantModelBench"
SOURCE_REVISION = "275fa9840756e26f2aeebcbc831cf2385a728f15"
COMPONENTS = {
    "task-b": ["data/processed/task_b_usa_frozen_raw_rolling_294/**"],
    "task-b-semantic": [
        "data/processed/task_b_usa_frozen_raw_rolling_294/**",
        "data/processed/task_b_usa_frozen_raw_rolling_294_semantic/**",
    ],
    "task-c": ["data/processed/task_c_frozen_raw_rolling_207/**"],
    "task-c-semantic": [
        "data/processed/task_c_frozen_raw_rolling_207/**",
        "data/processed/task_c_frozen_raw_rolling_207_semantic/**",
    ],
    "evaluator": ["data/processed/task_c_evaluator_panel/**"],
    "jkp": ["data/processed/jkp/**"],
    "construction": ["data/manifests/**", "data/processed/factor_window_selection_plan.jsonl"],
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", nargs="+", required=True, choices=COMPONENTS)
    parser.add_argument("--revision", default=SOURCE_REVISION)
    parser.add_argument("--local-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    patterns = sorted({p for c in args.component for p in COMPONENTS[c]})
    print(f"Repository: {REPO_ID}\nRevision: {args.revision}\nDestination: {args.local_dir.resolve()}")
    print("Allowed paths:\n" + "\n".join(patterns))
    if args.dry_run:
        return 0
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        parser.error("Install the data client: python -m pip install -r requirements-data.txt")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision=args.revision,
        local_dir=args.local_dir,
        allow_patterns=patterns,
        ignore_patterns=["*.env", ".env", ".env.*", "**/runs/**"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

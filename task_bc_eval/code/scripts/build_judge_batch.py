#!/usr/bin/env python3
"""Stage 4: build the LLM-as-Judge batch. Makes no network call.

Produces one JSONL record per (frozen code, pass).  Issuing the calls is a
separate deliberate step, so a judge run can never happen as a side effect of
scoring.

    python task_bc_eval/code/scripts/build_judge_batch.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bceval import codeshape, judge, runspec  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs"

CONTRACT = {
    "task_b": {
        "entrypoint": "fit_and_predict(train_df, valid_df, predict_df, metadata)",
        "output_columns": ["factor_id", "score"],
        "requirements": "one score per candidate factor at each rolling origin; "
                        "no data outside the supplied frames; no network or disk access",
    },
    "task_c": {
        "entrypoint": "fit_and_predict(train_df, valid_df, predict_df, metadata)",
        "output_columns": ["id", "score"],
        "requirements": "one cross-sectional score per stock at each formation month; "
                        "no data outside the supplied frames; no network or disk access",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_DIR / "judge_batch.jsonl")
    parser.add_argument("--passes", type=int, default=judge.JUDGE_PASSES)
    args = parser.parse_args()

    items, missing = [], []
    counts: Counter[str] = Counter()
    for spec in runspec.build_registry():
        if not spec.is_llm:
            continue
        for case_dir in codeshape.spec_case_dirs(spec):
            code = codeshape.read_frozen_code(spec.task, case_dir)
            if not code:
                missing.append(f"{spec.key}|{case_dir.name}")
                continue
            contract = dict(CONTRACT[spec.task])
            for p in range(1, args.passes + 1):
                items.append(judge.build_item(spec.key, spec.task, spec.track,
                                              case_dir.name, code, contract, p))
            counts[spec.key] += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    judge.write_batch(items, args.out)
    meta = {
        "prompt_version": judge.JUDGE_PROMPT_VERSION,
        "dimensions": [name for name, _ in judge.DIMENSIONS],
        "passes": args.passes,
        "n_items": len(items),
        "n_codes": sum(counts.values()),
        "per_run": dict(counts),
        "missing_code": missing,
        "output_schema": judge.OUTPUT_SCHEMA,
        "network_calls": 0,
    }
    (args.out.with_suffix(".meta.json")).write_text(json.dumps(meta, indent=2) + "\n")

    for key, n in sorted(counts.items()):
        print(f"  {key:48s} codes={n}")
    print(f"\n{len(items)} judge items ({sum(counts.values())} codes x {args.passes} passes) "
          f"-> {args.out}")
    if missing:
        print(f"WARNING: {len(missing)} cases had no frozen code")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Stage 1: materialise the tidy case / origin / monthly tables.

Every later stage reads only these three parquet files, so all path and schema
knowledge is confined to `bceval.loaders`.  Run:

    python task_bc_eval/code/scripts/build_case_metrics.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bceval import codeshape, loaders, runspec  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[2] / "outputs"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--only", nargs="*", default=None,
                        help="restrict to these run keys (substring match)")
    parser.add_argument("--skip-oos", action="store_true",
                        help="skip the Task C panel join (fast smoke path)")
    parser.add_argument("--skip-codeshape", action="store_true")
    parser.add_argument("--split", default="dev", choices=("dev", "test"),
                        help="which split's runs to materialise (default dev; "
                             "thresholds are dev-frozen and are never rebound here)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    specs = runspec.build_registry(split=args.split)
    if args.only:
        specs = [s for s in specs if any(pat in s.key for pat in args.only)]

    all_cases, all_origins, all_monthly, shapes = [], [], [], []
    manifest = {"built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "split": args.split, "runs": []}

    for spec in specs:
        t0 = time.time()
        if not spec.root.exists():
            print(f"  MISSING  {spec.key}: {spec.root}")
            manifest["runs"].append({"run_key": spec.key, "status": "missing_root",
                                     "root": str(spec.root)})
            continue
        cases, origins, monthly = loaders.load_run(spec, with_oos=not args.skip_oos)
        if len(cases):
            cases = loaders.attach_family(cases, spec.task, spec.split)
            all_cases.append(cases)
        if len(origins):
            all_origins.append(origins)
        if len(monthly):
            all_monthly.append(monthly)

        if spec.is_llm and not args.skip_codeshape:
            shapes.append(codeshape.classify_spec(spec))

        elapsed = time.time() - t0
        manifest["runs"].append({
            "run_key": spec.key, "task": spec.task, "kind": spec.kind,
            "track": spec.track, "status": "ok", "n_cases": int(len(cases)),
            "n_origins": int(len(origins)), "n_monthly": int(len(monthly)),
            "coverage_note": spec.coverage_note, "seconds": round(elapsed, 1),
        })
        print(f"  ok  {spec.key:48s} cases={len(cases):3d} origins={len(origins):4d} "
              f"monthly={len(monthly):5d}  ({elapsed:.1f}s)")

    if not all_cases:
        print("no runs loaded", file=sys.stderr)
        return 1

    cases = pd.concat(all_cases, ignore_index=True)
    origins = pd.concat(all_origins, ignore_index=True) if all_origins else pd.DataFrame()
    monthly = pd.concat(all_monthly, ignore_index=True) if all_monthly else pd.DataFrame()

    cases.to_parquet(args.out_dir / "case_metrics.parquet", index=False)
    if len(origins):
        origins.to_parquet(args.out_dir / "origin_metrics.parquet", index=False)
    if len(monthly):
        monthly.to_parquet(args.out_dir / "monthly_returns.parquet", index=False)
    if shapes:
        pd.concat(shapes, ignore_index=True).to_parquet(
            args.out_dir / "code_shapes.parquet", index=False)
    (args.out_dir / "build_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"\ncases={len(cases)}  origins={len(origins)}  monthly={len(monthly)}")
    print(f"split={args.split}  runs={manifest['runs'] and len(manifest['runs'])}")
    print(f"wrote -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

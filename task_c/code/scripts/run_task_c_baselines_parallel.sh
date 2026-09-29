#!/usr/bin/env bash
# Parallel Task C baseline sweep, server side, mirrored layout.
#
# Same per-run command as the local serial driver
# (run_task_c_baselines_split.sh) -- same panel, same factor details, same
# flags, same single-threaded BLAS -- but N runs wide instead of one, because
# 123 test cases x 6 methods is ~120 core-hours and the serial driver needs
# ~2.4 days for it.
#
# OMP_NUM_THREADS=1 is part of the numerical contract, not a performance knob:
# MLP baselines are sensitive to thread count. Every run of one MLP method must
# also come from one machine -- Accelerate and OpenBLAS disagree by up to 3e-1
# on Sharpe for large cases, so a split sweep is not comparable.
#
# Usage: run_task_c_baselines_parallel.sh <split: dev|test> <output-root> <workers> [<method> ...]
# Env:   TASK_C_DATASET  TASK_C_CASE_FILE  DRY=1
set -uo pipefail

L=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
PY=$(command -v "${QUANTMODEL_PYTHON:-python3}") || { echo "Python interpreter not found" >&2; exit 2; }
cd "$L"
DATASET=${TASK_C_DATASET:-$L/data/processed/task_c_frozen_raw_rolling_207}
PANEL=${TASK_C_EVALUATOR_DIR:-$L/data/processed/task_c_evaluator_panel}
DETAILS=${TASK_C_FACTOR_DETAILS:-$L/data/processed/jkp/2025-12/factor_details.csv}
RUNNER=$L/task_c/code/scripts/run_task_c_baseline.py

die () { echo "FATAL: $*" >&2; exit 2; }

[ "$#" -ge 3 ] || die "usage: $(basename "$0") <dev|test> <output-root> <workers> [method ...]"
SPLIT=$1; shift
OUT_ROOT=$1; shift
WORKERS=$1; shift
METHODS=("$@")
if [ ${#METHODS[@]} -eq 0 ]; then
  METHODS=(rolling_ridge rolling_elastic_net jkp_signed_equal_weight
           rolling_mlp rolling_mlp_deep rolling_mlp_ensemble)
fi

case "$SPLIT" in dev|test) ;; *) die "split must be dev or test, got '$SPLIT'" ;; esac
[[ "$WORKERS" =~ ^[0-9]+$ ]] && [ "$WORKERS" -ge 1 ] || die "workers must be a positive integer"
case "$OUT_ROOT" in /*) ;; *) die "output-root must be absolute" ;; esac
case "$OUT_ROOT" in "$L"/*) ;; *) die "output-root must live under $L" ;; esac

MANIFEST=$DATASET/splits/$SPLIT/manifest.jsonl
for f in "$PY" "$RUNNER" "$MANIFEST" "$DETAILS" \
         "$PANEL/panel.parquet" "$PANEL/daily_ret.parquet" "$PANEL/nyse_cutoffs.csv"; do
  [ -f "$f" ] || die "missing input: $f"
done

export PYTHONPATH=$L/task_c/code
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

mkdir -p "$OUT_ROOT/logs"
PROGRESS=$OUT_ROOT/progress.log

# Case ids come from the split manifest, which carries one non-case header
# record, so filter on record_type rather than counting lines.
CASE_FILE=${TASK_C_CASE_FILE:-$OUT_ROOT/case_ids.txt}
if [ ! -f "$CASE_FILE" ]; then
  "$PY" -c '
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
ids = sorted(str(r["case_id"]) for r in rows if r.get("record_type") == "case")
open(sys.argv[2], "w").write("".join(i + "\n" for i in ids))
print(f"{len(ids)} cases from {sys.argv[1]}", file=sys.stderr)
' "$MANIFEST" "$CASE_FILE" || die "could not derive case ids"
fi

one () {
  local spec="$1" block case method dest start rc
  IFS='/' read -r block case method <<< "$spec"
  dest=$OUT_ROOT/$method/$block/$case
  if [ -f "$dest/summary.json" ]; then echo "cached $block/$case $method" >> "$PROGRESS"; return 0; fi
  mkdir -p "$dest"
  start=$(date +%s)
  "$PY" "$RUNNER" \
    --case-dir "$DATASET/cases/$block/$case" \
    --panel "$PANEL/panel.parquet" --daily "$PANEL/daily_ret.parquet" \
    --nyse-cutoffs "$PANEL/nyse_cutoffs.csv" --factor-details "$DETAILS" \
    --output-dir "$dest" --method "$method" \
    --cost-bps 10.0 --coverage-threshold 0.95 \
    --timeout-seconds 1800 --strict-cutoffs \
    > "$OUT_ROOT/logs/${block}__${case}__${method}.log" 2>&1
  rc=$?
  echo "done $block/$case $method rc=$rc elapsed=$(( $(date +%s) - start ))s" >> "$PROGRESS"
  return 0
}
export -f one
export PY RUNNER DATASET PANEL DETAILS OUT_ROOT PROGRESS

# One work item per (case, method) so the pool stays full even though
# rolling_mlp_ensemble takes ~15x rolling_ridge on a large case.
WORK=$OUT_ROOT/work.txt
: > "$WORK"
missing=0 cached=0 queued=0
while read -r cid; do
  [ -n "$cid" ] || continue
  block=""
  for cand in "$DATASET"/cases/*/"$cid"; do
    [ -d "$cand" ] && block=$(basename "$(dirname "$cand")") && break
  done
  if [ -z "$block" ]; then
    echo "SKIP $cid: no case dir under $DATASET/cases" >> "$PROGRESS"
    missing=$(( missing + 1 )); continue
  fi
  for m in "${METHODS[@]}"; do
    if [ -f "$OUT_ROOT/$m/$block/$cid/summary.json" ]; then
      cached=$(( cached + 1 ))
    else
      echo "$block/$cid/$m" >> "$WORK"; queued=$(( queued + 1 ))
    fi
  done
done < "$CASE_FILE"

echo "split=$SPLIT cases=$(grep -c . "$CASE_FILE") methods=${#METHODS[@]}"
echo "queued=$queued cached=$cached cases_missing=$missing workers=$WORKERS"
echo "dataset=$DATASET"
echo "output=$OUT_ROOT"
[ "$missing" -eq 0 ] || echo "WARNING: $missing case ids have no directory — check the dataset is fully deployed"

if [ "${DRY:-0}" = "1" ]; then
  echo "DRY=1, not running. First 5 work items:"; head -5 "$WORK" | sed 's/^/  /'
  exit 0
fi
[ "$queued" -gt 0 ] || { echo "nothing to do"; exit 0; }

echo "START $(date -u +%FT%TZ) items=$queued workers=$WORKERS" >> "$PROGRESS"
xargs -I{} -P "$WORKERS" bash -c 'one "$@"' _ {} < "$WORK"
echo "SWEEP COMPLETE $(date -u +%FT%TZ)" >> "$PROGRESS"
echo "SWEEP COMPLETE — $(grep -c '^done' "$PROGRESS") done, $(grep -c 'rc=0' "$PROGRESS") rc=0"

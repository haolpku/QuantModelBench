#!/usr/bin/env bash
# Drive run_task_c_baseline.py over one split, writing the layout the evaluator
# reads: <root>/<method>/<block>/<case>/{summary.json, ...}.
#
# Usage: run_task_c_baselines_split.sh <split: dev|test> <output-root> <method> [<method> ...]
#
# Differs from run_baselines.sh in two ways that matter:
#   * output is grouped by method first, so one method can be scored without
#     reading its siblings;
#   * the panel defaults to the evaluator copy, not the raw WRDS file.  The
#     server-side LLM driver already reads the evaluator panel, and the two
#     are numerically equivalent on the overlap (relative diff <= 6e-8, i.e.
#     float32 rounding), so pointing the baselines at the same file is what
#     makes "same input" a fact rather than an assumption.
#
# Set TASK_C_PANEL / TASK_C_DAILY / TASK_C_CUTOFFS to override; PANEL_OK=1
# acknowledges a deliberate use of the raw WRDS files instead.
set -u

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
PY=${QUANTMODEL_PYTHON:-python3}
[ "$#" -ge 3 ] || { echo "usage: $0 <dev|test> <output-root> <method> [method ...]" >&2; exit 2; }
cd "$ROOT"

SPLIT=$1; shift
OUT_ROOT=$1; shift
METHODS=()
while [ "$#" -gt 0 ]; do METHODS+=("$1"); shift; done
if [ ${#METHODS[@]} -eq 0 ]; then echo "no methods given" >&2; exit 2; fi
case "$SPLIT" in
  dev|test) ;;
  *) echo "split must be dev or test, got '$SPLIT'" >&2; exit 2 ;;
esac

DATASET=${TASK_C_DATASET:-$ROOT/data/processed/task_c_frozen_raw_rolling_207}
SPLIT_MANIFEST="$DATASET/splits/$SPLIT/manifest.jsonl"
CASE_FILE=${TASK_C_CASE_FILE:-}
if [ -z "$CASE_FILE" ]; then
  CASE_FILE="$OUT_ROOT/case_ids.txt"
  mkdir -p "$OUT_ROOT"
  "$PY" -c '
import json,sys
rows=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
ids=[str(r["case_id"]) for r in rows if r.get("record_type")=="case"]
open(sys.argv[2],"w").write("".join(i+"\n" for i in sorted(ids)))
print(f"{len(ids)} cases from {sys.argv[1]}", file=sys.stderr)
' "$SPLIT_MANIFEST" "$CASE_FILE"
fi

# Evaluator panel by default; raw WRDS files only on explicit acknowledgement.
if [ "${PANEL_OK:-0}" = "1" ]; then
  PANEL=${TASK_C_PANEL:-$ROOT/data/raw/wrds/wgpvj1vbztoqbloo\ \(1\).parquet}
  DAILY=${TASK_C_DAILY:-$ROOT/data/raw/wrds/ctff_daily_ret.parquet}
else
  PANEL=${TASK_C_PANEL:-$ROOT/data/processed/task_c_evaluator_panel/panel.parquet}
  DAILY=${TASK_C_DAILY:-$ROOT/data/processed/task_c_evaluator_panel/daily_ret.parquet}
fi
CUTOFFS=${TASK_C_CUTOFFS:-$ROOT/data/processed/task_c_evaluator_panel/nyse_cutoffs.csv}
DETAILS=${TASK_C_FACTOR_DETAILS:-$ROOT/data/processed/jkp/2025-12/factor_details.csv}
for f in "$PANEL" "$DAILY" "$CUTOFFS" "$DETAILS"; do
  [ -f "$f" ] || { echo "missing input: $f" >&2; exit 2; }
done

export PYTHONPATH=$ROOT/task_c/code
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

echo "split=$SPLIT panel=$PANEL"
mkdir -p "$OUT_ROOT/logs"

for CID in $(grep -v '^#' "$CASE_FILE" | grep .); do
  CASE_DIR=""
  for CAND in "$DATASET"/cases/*/"$CID"; do
    if [ -d "$CAND" ]; then CASE_DIR="$CAND"; BLOCK=$(basename "$(dirname "$CAND")"); break; fi
  done
  if [ -z "$CASE_DIR" ]; then echo "SKIP $CID: case directory not found"; continue; fi
  for M in "${METHODS[@]}"; do
    DEST="$OUT_ROOT/$M/$BLOCK/$CID"
    if [ -f "$DEST/summary.json" ]; then echo "cached $BLOCK/$CID $M"; continue; fi
    mkdir -p "$DEST"
    START=$(date +%s)
    "$PY" "$ROOT/task_c/code/scripts/run_task_c_baseline.py" \
      --case-dir "$CASE_DIR" \
      --panel "$PANEL" --daily "$DAILY" \
      --nyse-cutoffs "$CUTOFFS" --factor-details "$DETAILS" \
      --output-dir "$DEST" --method "$M" \
      --cost-bps 10.0 --coverage-threshold 0.95 \
      --timeout-seconds 1800 --strict-cutoffs \
      > "$OUT_ROOT/logs/${BLOCK}__${CID}__${M}.log" 2>&1
    echo "done $BLOCK/$CID $M rc=$? elapsed=$(( $(date +%s) - START ))s"
  done
done
echo "SWEEP COMPLETE $SPLIT"

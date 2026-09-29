#!/usr/bin/env python3
"""Build the evaluator-side Task C panel and daily-return files.

The evaluated Task C run does not read the full WRDS panel: the rolling
driver (`run_task_c_full_dev_prompt_v5.sh`) and the OOS Rank IC scorer
(`bceval.loaders.load_task_c_panel`) both read a *trimmed* copy staged at
`task_c/data/evaluator_panel/`.  Trimming is a date slice plus a column
drop plus a float32 downcast of the factor block -- nothing else.  This
script is that recipe, made reproducible.

Two files are written:

    panel.parquet      USA monthly rows, id/date/flag block + 153 factors
    daily_ret.parquet  id/date/ret_exc daily rows

Span policy.  The original copy was cut to a hand-picked window with about
a year of slack on both sides of what the dev anchors needed.  This keeps
that slack and computes it from the case manifests instead of hardcoding
dates: `--pad-months` is added before the earliest panel month the cases
require and after the latest.  Coverage is verified against the manifests
before writing, so a build that cannot serve every case fails loudly.

    python task_c/code/scripts/build_task_c_evaluator_panel.py
    python task_c/code/scripts/build_task_c_evaluator_panel.py --out-dir .tmp/evaluator_panel_v2
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]

DEFAULT_PANEL = Path("data/raw/wrds/wgpvj1vbztoqbloo (1).parquet")
DEFAULT_DAILY = Path("data/raw/wrds/ctff_daily_ret.parquet")
DEFAULT_DATASET = Path("data/processed/task_c_frozen_raw_rolling_207")
DEFAULT_CUTOFFS = Path("data/raw/jkp/2025-12/nyse_cutoffs.csv")
DEFAULT_OUT = Path(".tmp/evaluator_panel")

# Columns the full WRDS panel carries that the evaluator copy drops.  The
# remaining columns keep their original relative order.
DROPPED_COLUMNS = ("ret_exc", "ret", "curcd", "fx", "ff49", "source_crsp")

# The identifier / date / screen block, kept in its source dtype and emitted
# first.  Every other floating column is downcast to float32 -- a straight
# memory win, since the panel is read whole by every rolling case.
# The identifier / date / screen block, hoisted to the front in the deployed
# panel's order.  Below it the deployed file's column order is a near-random
# permutation of the source (133 one- and two-column runs that match no
# deliberate sort), so it is an accident of whatever generated it, not a
# contract; the rebuild keeps source order for the factors.  Parquet is read
# by name, and the only consumer (`loaders.load_task_c_panel`) projects three
# columns, so nothing downstream can observe the difference -- which was
# checked directly by comparing every overlapping value against the deployed
# file rather than by comparing schemas.
LEAD_COLUMNS = (
    "id", "permno", "gvkey", "iid", "eom", "date", "excntry",
    "me", "market_equity", "size_grp", "common", "exch_main",
    "primary_sec", "obs_main", "ret_exc_lead1m",
)

# Kept at source width.  Deliberately NOT the same set as LEAD_COLUMNS: the
# deployed file narrows `market_equity` to float32 while keeping `me` double,
# so the two lists overlap without being equal.  These are the identifier and
# screen columns -- read as keys or compared exactly, and the only ones whose
# narrowing could change a result rather than just a byte count.
WIDE_COLUMNS = (
    "id", "permno", "me", "common", "exch_main",
    "primary_sec", "obs_main", "ret_exc_lead1m",
)
COUNTRY = "USA"


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path)


def required_months(dataset: Path) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """Return (panel_start, panel_end, daily_start, daily_end) over every case.

    Read from the case manifests rather than inferred from the anchors.  The
    naming takes care: `target_start_month`/`target_end_month` are the
    *holding* months the case is scored over, and the formations that form
    the portfolios sit one month earlier.  A window of `rolling_test_months`
    therefore runs formations `start-1 .. end-1` over holdings `start .. end`.

    Panel months must cover every formation plus the train/validation history
    fitted before the first one.  Daily months must cover every holding month
    -- one month further at the far end, because a formation's return is
    realized over the month that follows it.
    """
    panel_lo = panel_hi = daily_lo = daily_hi = None
    n = 0
    for path in sorted(dataset.glob("cases/*/*/manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        window = manifest["window"]
        first_holding = pd.Period(manifest["target_start_month"], freq="M")
        last_holding = pd.Period(manifest["target_end_month"], freq="M")
        first_formation = first_holding - 1
        last_formation = last_holding - 1
        p_lo = (first_formation - window["train_months"] - window["validation_months"]).to_timestamp("M")
        p_hi = last_formation.to_timestamp("M")
        d_lo = first_holding.to_timestamp("M").replace(day=1)
        d_hi = (last_holding + 1).to_timestamp("M").replace(day=1) - pd.DateOffset(days=1)
        panel_lo = p_lo if panel_lo is None or p_lo < panel_lo else panel_lo
        panel_hi = p_hi if panel_hi is None or p_hi > panel_hi else panel_hi
        daily_lo = d_lo if daily_lo is None or d_lo < daily_lo else daily_lo
        daily_hi = d_hi if daily_hi is None or d_hi > daily_hi else daily_hi
        n += 1
    if not n:
        raise SystemExit(f"no case manifests under {dataset}")
    return panel_lo, panel_hi, daily_lo, daily_hi


def panel_schema(source: pa.Schema) -> tuple[list[str], pa.Schema]:
    """Projection and target schema for the evaluator panel.

    Column order follows the deployed file, not the source: the id/flag block
    leads and the 153 factors follow.  Parquet reads by name so order is not
    load-bearing, but a rebuild that silently reshuffles the schema makes
    every downstream `panel.columns` comparison useless as a check.
    """
    names = [n for n in LEAD_COLUMNS if n in source.names]
    names += [n for n in source.names
              if n not in DROPPED_COLUMNS and n not in LEAD_COLUMNS]
    fields = []
    for name in names:
        field = source.field(name)
        if name not in WIDE_COLUMNS and pa.types.is_floating(field.type):
            field = pa.field(name, pa.float32())
        fields.append(field)
    return names, pa.schema(fields)


def build_panel(source: Path, out: Path, lo: pd.Timestamp, hi: pd.Timestamp) -> dict:
    """Stream the USA slice out in row-group batches.

    Batched rather than one `to_table`: 167 columns of double at ~4M rows
    is a multi-GB intermediate, and only the factors are ever read as
    double -- they are narrowed per batch before anything is buffered.
    """
    dataset = ds.dataset(str(source), format="parquet")
    names, schema = panel_schema(dataset.schema)
    scanner = dataset.scanner(
        columns=names,
        filter=(ds.field("excntry") == COUNTRY)
        & (ds.field("eom") >= pa.scalar(lo.date()))
        & (ds.field("eom") <= pa.scalar(hi.date())),
    )
    rows = 0
    eom_lo = eom_hi = None
    writer = None
    try:
        for batch in scanner.to_batches():
            if not batch.num_rows:
                continue
            table = pa.Table.from_batches([batch]).cast(schema)
            if writer is None:
                writer = pq.ParquetWriter(out, schema, compression="zstd")
            writer.write_table(table)
            rows += table.num_rows
            eom = pd.to_datetime(table.column("eom").to_pandas())
            eom_lo = eom.min() if eom_lo is None or eom.min() < eom_lo else eom_lo
            eom_hi = eom.max() if eom_hi is None or eom.max() > eom_hi else eom_hi
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise SystemExit(f"no {COUNTRY} rows in {lo.date()}..{hi.date()}; nothing written")
    return {
        "rows": rows,
        "columns": len(names),
        "start": str(eom_lo.date()),
        "end": str(eom_hi.date()),
        "bytes": out.stat().st_size,
    }


def build_daily(source: Path, out: Path, lo: pd.Timestamp, hi: pd.Timestamp) -> dict:
    dataset = ds.dataset(str(source), format="parquet")
    table = dataset.to_table(
        columns=["id", "date", "ret_exc"],
        filter=(ds.field("date") >= pa.scalar(lo.date()))
        & (ds.field("date") <= pa.scalar(hi.date())),
    )
    pq.write_table(table, out, compression="zstd")
    dates = pd.to_datetime(table.column("date").to_pandas())
    return {
        "rows": table.num_rows,
        "start": str(dates.min().date()),
        "end": str(dates.max().date()),
        "bytes": out.stat().st_size,
    }


def verify(panel_path: Path, daily_path: Path, dataset: Path) -> list[str]:
    """Every case must resolve inside both files, on its own required window.

    Checked per case, not just globally: one anchor can sit comfortably
    inside the span while another is short by a single month, and the
    global check would call that build fine.
    """
    problems: list[str] = []
    panel = pq.read_table(panel_path, columns=["eom"]).column("eom").to_pandas()
    panel_months = set(pd.PeriodIndex(pd.to_datetime(panel), freq="M"))
    daily = pq.read_table(daily_path, columns=["date"]).column("date").to_pandas()
    daily_months = set(pd.PeriodIndex(pd.to_datetime(daily), freq="M"))
    for path in sorted(dataset.glob("cases/*/*/manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        window = manifest["window"]
        first_holding = pd.Period(manifest["target_start_month"], freq="M")
        last_holding = pd.Period(manifest["target_end_month"], freq="M")
        first_formation = first_holding - 1
        last_formation = last_holding - 1
        panel_first = first_formation - window["train_months"] - window["validation_months"]
        for period in pd.period_range(panel_first, last_formation, freq="M"):
            if period not in panel_months:
                problems.append(f"{manifest['case_id']}: panel missing {period}")
        for period in pd.period_range(first_holding, last_holding, freq="M"):
            if period not in daily_months:
                problems.append(f"{manifest['case_id']}: daily missing {period}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--panel", default=str(DEFAULT_PANEL))
    parser.add_argument("--daily", default=str(DEFAULT_DAILY))
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--cutoffs", default=str(DEFAULT_CUTOFFS))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--pad-months", type=int, default=12,
        help="months of slack added outside the required span (default 12, "
             "matching the slack the original dev-only copy carried)",
    )
    args = parser.parse_args(argv)

    panel_src, daily_src = _resolve(args.panel), _resolve(args.daily)
    dataset, out_dir = _resolve(args.dataset), _resolve(args.out_dir)
    for path in (panel_src, daily_src, dataset):
        if not path.exists():
            raise SystemExit(f"missing input: {path}")

    panel_lo, panel_hi, daily_lo, daily_hi = required_months(dataset)
    pad = pd.DateOffset(months=args.pad_months)
    panel_lo, panel_hi = panel_lo - pad, panel_hi + pad
    # Only the daily *lower* bound gets slack.  Its upper bound is the last
    # day of the last holding month, which is a hard requirement, not a
    # window edge -- padding it buys nothing and can only push the slice
    # past the end of the source file.
    daily_lo = daily_lo - pad
    print(f"required panel : {panel_lo.date()} .. {panel_hi.date()}")
    print(f"required daily : {daily_lo.date()} .. {daily_hi.date()}")

    out_dir.mkdir(parents=True, exist_ok=True)
    panel_out, daily_out = out_dir / "panel.parquet", out_dir / "daily_ret.parquet"
    panel_info = build_panel(panel_src, panel_out, panel_lo, panel_hi)
    print(f"wrote {panel_out}: {panel_info}")
    daily_info = build_daily(daily_src, daily_out, daily_lo, daily_hi)
    print(f"wrote {daily_out}: {daily_info}")

    problems = verify(panel_out, daily_out, dataset)
    if problems:
        for item in problems[:20]:
            print(f"COVERAGE {item}", file=sys.stderr)
        print(f"{len(problems)} coverage problems", file=sys.stderr)
        return 2

    cutoffs_src = _resolve(args.cutoffs)
    if cutoffs_src.exists():
        target = out_dir / "nyse_cutoffs.csv"
        target.write_bytes(cutoffs_src.read_bytes())
        print(f"wrote {target} ({target.stat().st_size} bytes)")

    print("coverage OK: every case resolves in both files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

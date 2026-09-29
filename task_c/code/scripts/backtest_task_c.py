#!/usr/bin/env python3
"""Run the fixed Task C JKP-compatible daily backtest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from factoropt.task_c_backtest import (  # noqa: E402
    _parse_month_series,
    backtest_predictions,
    load_daily_returns,
    load_nyse_cutoffs,
    load_stock_panel,
)


def _read_table(path: Path):
    if path.suffix.lower() in {".parquet", ".pq"}:
        import pandas as pd

        return pd.read_parquet(path)
    import pandas as pd

    return pd.read_csv(path)


def _write_frame(frame, path: Path) -> str | None:
    if frame is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_parquet(path, index=False)
        return str(path)
    except (ImportError, ModuleNotFoundError):
        csv_path = path.with_suffix(".csv")
        frame.to_csv(csv_path, index=False)
        return str(csv_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--panel", required=True, type=Path)
    parser.add_argument("--daily", required=True, type=Path)
    parser.add_argument("--nyse-cutoffs", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--features", nargs="*", default=[])
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--formation-start", help="first formation month (keeps failed months in the denominator)")
    parser.add_argument("--formation-end", help="last formation month (keeps failed months in the denominator)")
    parser.add_argument(
        "--rolling-audit",
        type=Path,
        help="optional rolling_audit.parquet/csv; failed formation months are counted as unavailable",
    )
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=0.95,
        help="minimum daily absolute-weight coverage (default 0.95; use 0.99 as a stricter sensitivity)",
    )
    parser.add_argument("--min-breakpoint-stocks", type=int, default=10)
    parser.add_argument("--min-leg-stocks", type=int, default=5)
    parser.add_argument("--strict-cutoffs", action="store_true")
    args = parser.parse_args(argv)
    predictions = _read_table(args.predictions)
    if predictions.empty and not (args.formation_start or args.formation_end or args.rolling_audit):
        raise ValueError(
            "predictions file is empty; provide formation-start/end or rolling-audit "
            "to keep failed months in the denominator"
        )
    # Avoid an accidental full 9GB WRDS read when the caller omits bounds.
    # Formation months are available in the prediction artifact and define the
    # only panel slice needed by the fixed backtester.
    formation_column = next((c for c in ("formation_eom", "eom", "time_key") if c in predictions.columns), None)
    inferred_start = inferred_end = None
    if formation_column is not None:
        parsed = _parse_month_series(predictions[formation_column]).dropna()
        if not parsed.empty:
            inferred_start = parsed.min().to_period("M").to_timestamp("M")
            inferred_end = parsed.max().to_period("M").to_timestamp("M")
    explicit_months = None
    if args.formation_start or args.formation_end:
        if not (args.formation_start and args.formation_end):
            raise ValueError("formation-start and formation-end must be supplied together")
        first_formation = pd.Period(args.formation_start, freq="M")
        last_formation = pd.Period(args.formation_end, freq="M")
        if last_formation < first_formation:
            raise ValueError("formation-end precedes formation-start")
        explicit_months = [
            period.to_timestamp("M")
            for period in pd.period_range(first_formation, last_formation, freq="M")
        ]
    elif args.rolling_audit:
        audit_frame = _read_table(args.rolling_audit)
        if "formation_eom" not in audit_frame.columns:
            raise ValueError("rolling audit must contain formation_eom")
        audit_dates = _parse_month_series(audit_frame["formation_eom"]).dropna()
        explicit_months = sorted(
            {
                value.to_period("M").to_timestamp("M")
                for value in audit_dates
            }
        )
    panel_start = args.start or (explicit_months[0] if explicit_months else inferred_start)
    panel_end = args.end or (explicit_months[-1] if explicit_months else inferred_end)
    panel = load_stock_panel(
        args.panel,
        feature_columns=args.features,
        start=panel_start,
        end=panel_end,
    )
    daily_start = args.start
    daily_end = args.end
    if explicit_months and daily_start is None:
        daily_start = (explicit_months[0] + pd.offsets.MonthEnd(1)).replace(day=1)
    if explicit_months and daily_end is None:
        daily_end = explicit_months[-1] + pd.offsets.MonthEnd(1)
    if daily_start is None and inferred_start is not None:
        daily_start = (inferred_start + pd.offsets.MonthEnd(1)).replace(day=1)
    if daily_end is None and inferred_end is not None:
        daily_end = inferred_end + pd.offsets.MonthEnd(1)
    daily = load_daily_returns(args.daily, start=daily_start, end=daily_end)
    cutoffs = load_nyse_cutoffs(args.nyse_cutoffs)
    result = backtest_predictions(
        predictions,
        panel,
        daily,
        nyse_cutoffs=cutoffs,
        formation_months=explicit_months,
        min_breakpoint_stocks=args.min_breakpoint_stocks,
        min_leg_stocks=args.min_leg_stocks,
        strict_cutoffs=args.strict_cutoffs,
        cost_bps=args.cost_bps,
        coverage_threshold=args.coverage_threshold,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "monthly": _write_frame(result["monthly"], args.output_dir / "backtest_monthly.parquet"),
        "daily": _write_frame(result["daily"], args.output_dir / "backtest_daily.parquet"),
        "weights": _write_frame(result["weights"], args.output_dir / "weights.parquet"),
    }
    weight_partitions: dict[str, str] = {}
    if not result["weights"].empty and "formation_eom" in result["weights"].columns:
        for formation, subset in result["weights"].groupby("formation_eom", sort=True):
            stamp = (
                formation.to_timestamp("M")
                if isinstance(formation, pd.Period)
                else pd.Timestamp(formation)
            ).strftime("%Y%m")
            weight_partitions[stamp] = _write_frame(
                subset,
                args.output_dir / f"weights_{stamp}.parquet",
            )
    artifacts["weight_partitions"] = weight_partitions
    (args.output_dir / "summary.json").write_text(
        json.dumps({"summary": result["summary"], "artifacts": artifacts, "diagnostics": result["diagnostics"]}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(result["summary"], ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

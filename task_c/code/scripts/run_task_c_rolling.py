#!/usr/bin/env python3
"""Execute a frozen Task C source over real WRDS months and backtest it."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from factoropt.task_c_backtest import (  # noqa: E402
    backtest_predictions,
    load_daily_returns,
    load_nyse_cutoffs,
    load_stock_panel,
)
from factoropt.task_c import validate_public_payload  # noqa: E402
from factoropt.task_c_iterative import rolling_predict  # noqa: E402
from factoropt.semantic import normalize_track  # noqa: E402


def _table(path: Path):
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _write(frame: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_parquet(path, index=False)
        return str(path)
    except (ImportError, ModuleNotFoundError):
        csv = path.with_suffix(".csv")
        frame.to_csv(csv, index=False)
        return str(csv)


def _response(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("frozen response must be an object")
    value = value.get("selected_response") or value.get("parsed_response") or value
    if not isinstance(value, dict):
        raise ValueError("frozen response file has no selected_response")
    return value


def _id_mapping(path: Path | None) -> dict[str, str] | None:
    """Read evaluator-only original->masked IDs in either map orientation."""

    if path is None:
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("id map must be a JSON object")
    direct = value.get("original_to_masked")
    if isinstance(direct, dict):
        mapping = {str(key): str(item) for key, item in direct.items()}
        if not mapping or any(not key or not item for key, item in mapping.items()):
            raise ValueError("id map contains empty original/masked IDs")
        return mapping
    reverse = value.get("masked_to_original")
    if isinstance(reverse, dict):
        mapping = {str(item): str(key) for key, item in reverse.items()}
        if not mapping or any(not key or not item for key, item in mapping.items()):
            raise ValueError("id map contains empty original/masked IDs")
        return mapping
    mapping = {str(key): str(item) for key, item in value.items()}
    if not mapping or any(not key or not item for key, item in mapping.items()):
        raise ValueError("id map must contain non-empty string pairs")
    return mapping


def _case_metadata(
    case_dir: Path,
) -> tuple[
    list[str],
    dict[str, str] | None,
    dict[str, str] | None,
    list[str],
    int,
    int,
    str,
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Read evaluator-only maps/schedule for a generated frozen case."""

    payload = json.loads((case_dir / "payload.json").read_text(encoding="utf-8"))
    public_features = [str(value) for value in payload.get("feature_columns", [])]
    payload_errors = validate_public_payload(payload)
    if payload_errors:
        raise ValueError("invalid public payload: " + "; ".join(payload_errors[:10]))
    track = normalize_track(payload.get("track", "numeric_only"))
    factor_cards = payload.get("factor_cards", [])
    if not isinstance(factor_cards, list):
        raise ValueError("payload.factor_cards must be an array")
    feature_map_path = case_dir / "evaluator_feature_map.json"
    id_map_path = case_dir / "evaluator_id_map.json"
    if not feature_map_path.is_file() or not id_map_path.is_file():
        raise ValueError("case directory is missing evaluator feature/ID maps")
    feature_value = json.loads(feature_map_path.read_text(encoding="utf-8"))
    id_value = json.loads(id_map_path.read_text(encoding="utf-8"))
    raw_to_public = feature_value.get("original_to_masked")
    original_to_masked = id_value.get("original_to_masked")
    if not isinstance(raw_to_public, dict) or not isinstance(original_to_masked, dict):
        raise ValueError("evaluator maps must contain original_to_masked objects")
    raw_features = [str(key) for key in raw_to_public]
    if [str(raw_to_public[key]) for key in raw_features] != public_features:
        raise ValueError("feature map order does not match payload feature_columns")
    truth_path = case_dir / "rolling_truth.jsonl"
    truth_rows = [json.loads(line) for line in truth_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    formation = [str(row["formation_eom"])[:7] for row in truth_rows]
    manifest_path = case_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    window = manifest.get("window", {}) if isinstance(manifest, dict) else {}
    train_months = int(window.get("train_months", 120))
    validation_months = int(window.get("validation_months", 12))
    if train_months < 1 or validation_months < 1:
        raise ValueError("case manifest has invalid train/validation window")
    return (
        raw_features,
        {str(key): str(value) for key, value in original_to_masked.items()},
        raw_to_public,
        formation,
        train_months,
        validation_months,
        track,
        [dict(card) for card in factor_cards if isinstance(card, Mapping)],
        # The public metadata the model was shown at codegen time.  `task_c.py`
        # injects `id_column` / `time_column` into it there, and the prompt
        # describes `metadata` as carrying the id/time columns without naming
        # the keys -- so a candidate that discovers them at codegen time may
        # reasonably subscript them.  Rolling has to hand back the same object;
        # narrowing it is what made the two stages disagree.
        dict(payload.get("metadata") or {}),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-response", required=True, type=Path)
    parser.add_argument("--panel", required=True, type=Path)
    parser.add_argument("--daily", required=True, type=Path)
    parser.add_argument("--nyse-cutoffs", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--features",
        nargs="+",
        help="panel feature columns; with --case-dir these are inferred from the evaluator map",
    )
    parser.add_argument("--formation-start", help="first formation month YYYY-MM (inferred with --case-dir)")
    parser.add_argument("--formation-end", help="last formation month YYYY-MM (inferred with --case-dir)")
    parser.add_argument("--train-months", type=int, help="rolling train months (inferred with --case-dir)")
    parser.add_argument("--validation-months", type=int, help="rolling validation months (inferred with --case-dir)")
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=0.95,
        help="minimum daily absolute-weight coverage (default 0.95; use 0.99 as a stricter sensitivity)",
    )
    parser.add_argument("--min-breakpoint-stocks", type=int, default=10)
    parser.add_argument("--min-leg-stocks", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--strict-cutoffs", action="store_true")
    parser.add_argument(
        "--id-map",
        type=Path,
        help="evaluator-only original_to_masked map from build_task_c_case.py",
    )
    parser.add_argument(
        "--case-dir",
        type=Path,
        help="generated Task C frozen case directory; infers anonymous features, maps and rolling months",
    )
    args = parser.parse_args(argv)

    inferred_raw_features: list[str] | None = None
    inferred_id_map: dict[str, str] | None = None
    inferred_feature_map: dict[str, str] | None = None
    inferred_formations: list[str] = []
    inferred_train_months: int | None = None
    inferred_validation_months: int | None = None
    inferred_track = "numeric_only"
    inferred_factor_cards: list[dict[str, Any]] = []
    inferred_payload_metadata: dict[str, Any] = {}
    if args.case_dir is not None:
        case_dir = args.case_dir.resolve()
        (
            inferred_raw_features,
            inferred_id_map,
            inferred_feature_map,
            inferred_formations,
            inferred_train_months,
            inferred_validation_months,
            inferred_track,
            inferred_factor_cards,
            inferred_payload_metadata,
        ) = _case_metadata(case_dir)
        if args.features:
            # A caller may explicitly pass the public (anonymous) names, but
            # raw names are the only names accepted by load_stock_panel.
            if list(args.features) == list(inferred_feature_map.values()):
                args.features = list(inferred_raw_features)
            elif list(args.features) != list(inferred_raw_features):
                raise ValueError("--features conflicts with evaluator feature map")
        else:
            args.features = list(inferred_raw_features)
        if args.id_map is None:
            args.id_map = case_dir / "evaluator_id_map.json"
        if not args.formation_start:
            args.formation_start = inferred_formations[0]
        if not args.formation_end:
            args.formation_end = inferred_formations[-1]
        if args.train_months is None:
            args.train_months = inferred_train_months
        elif int(args.train_months) != inferred_train_months:
            raise ValueError("--train-months conflicts with case manifest")
        if args.validation_months is None:
            args.validation_months = inferred_validation_months
        elif int(args.validation_months) != inferred_validation_months:
            raise ValueError("--validation-months conflicts with case manifest")
    if args.train_months is None:
        args.train_months = 120
    if args.validation_months is None:
        args.validation_months = 12
    if args.train_months < 1 or args.validation_months < 1:
        parser.error("train/validation months must be positive")
    if not args.features:
        parser.error("--features is required unless --case-dir is supplied")
    if not args.formation_start or not args.formation_end:
        parser.error("--formation-start and --formation-end are required unless --case-dir is supplied")

    first = pd.Period(args.formation_start, freq="M")
    last = pd.Period(args.formation_end, freq="M")
    if last < first:
        raise ValueError("formation-end precedes formation-start")
    formation_months = [period.to_timestamp("M") for period in pd.period_range(first, last, freq="M")]
    data_start = (first - args.train_months - args.validation_months).to_timestamp("M")
    data_end = last.to_timestamp("M")
    panel = load_stock_panel(
        args.panel,
        feature_columns=args.features,
        start=data_start,
        end=data_end,
    )
    if "label" not in panel.columns:
        if "ret_exc_lead1m" not in panel.columns:
            raise ValueError("panel must contain ret_exc_lead1m for historical labels")
        panel = panel.rename(columns={"ret_exc_lead1m": "label"})
    public_features = list(args.features)
    if inferred_feature_map is not None:
        # The model sees anonymous feature names; the evaluator loads the
        # original WRDS columns and renames them only inside this process.
        panel = panel.rename(columns=inferred_feature_map)
        public_features = [str(inferred_feature_map[str(value)]) for value in args.features]
    response = _response(args.frozen_response)
    id_mapping = _id_mapping(args.id_map)
    predictions, rolling_audit = rolling_predict(
        response,
        panel,
        formation_months,
        feature_columns=public_features,
        train_months=args.train_months,
        valid_months=args.validation_months,
        timeout_seconds=args.timeout_seconds,
        metadata={
            # Payload metadata first, then the run-level fields that override
            # it.  Without the payload half, `id_column` and `time_column`
            # vanish between codegen and rolling although codegen put them
            # there -- a frozen candidate that subscripted them passed codegen
            # and then died on the same line here.
            # `rolling_predict` re-filters this through `_model_metadata` and
            # then overwrites the window fields with the current origin's real
            # window, so nothing forward-looking can ride in on it.
            **inferred_payload_metadata,
            "seed": 1,
            "source": "wrds",
            "track": inferred_track,
            "factor_cards": inferred_factor_cards,
        },
        id_mapping=id_mapping,
    )
    # Daily data starts in the first holding month and ends in the last one;
    # this range is pushed down to Parquet row groups by the loader.
    daily_start = (first + 1).to_timestamp("M").replace(day=1)
    daily_end = (last + 1).to_timestamp("M")
    daily = load_daily_returns(args.daily, start=daily_start, end=daily_end)
    cuts = load_nyse_cutoffs(args.nyse_cutoffs)
    result = backtest_predictions(
        predictions,
        panel,
        daily,
        formation_months=formation_months,
        nyse_cutoffs=cuts,
        min_breakpoint_stocks=args.min_breakpoint_stocks,
        min_leg_stocks=args.min_leg_stocks,
        strict_cutoffs=args.strict_cutoffs,
        cost_bps=args.cost_bps,
        coverage_threshold=args.coverage_threshold,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "predictions": _write(predictions, args.output_dir / "predictions.parquet"),
        "rolling_audit": _write(rolling_audit, args.output_dir / "rolling_audit.parquet"),
        "monthly": _write(result["monthly"], args.output_dir / "backtest_monthly.parquet"),
        "daily": _write(result["daily"], args.output_dir / "backtest_daily.parquet"),
        "weights": _write(result["weights"], args.output_dir / "weights.parquet"),
    }
    # Keep a per-formation partition as well as the convenient combined table;
    # this makes it possible to audit one rebalance without loading the whole
    # OOS run and matches the naming convention in the Task C manifest.
    weight_partitions: dict[str, str] = {}
    if not result["weights"].empty and "formation_eom" in result["weights"].columns:
        for formation, subset in result["weights"].groupby("formation_eom", sort=True):
            stamp = (
                formation.to_timestamp("M")
                if isinstance(formation, pd.Period)
                else pd.Timestamp(formation)
            ).strftime("%Y%m")
            weight_partitions[stamp] = _write(
                subset,
                args.output_dir / f"weights_{stamp}.parquet",
            )
    artifacts["weight_partitions"] = weight_partitions
    summary = dict(result["summary"])
    summary.update(
        {
            "track": inferred_track,
            "semantic_card_count": (
                len(inferred_factor_cards) if inferred_track == "semantic" else 0
            ),
        }
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {"summary": summary, "artifacts": artifacts, "diagnostics": result["diagnostics"]},
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

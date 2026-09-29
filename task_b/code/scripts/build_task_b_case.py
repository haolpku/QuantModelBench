#!/usr/bin/env python3
"""Build one Task B frozen-code generation case and a 12-step test stream.

The generated ``payload.json`` is the only file needed by the one-time LLM
code-generation stage.  ``rolling_truth.jsonl`` is evaluator-only: it contains
the twelve hidden one-month targets and must not be sent to the model server.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = CODE_ROOT.parent
ROOT = TASK_ROOT.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from factoropt.task_b import (  # noqa: E402
    HISTORICAL_PANEL_FEATURE_FIELDS,
    build_router_episodes,
    episode_to_truth_record,
    load_raw_factor_returns,
    load_sleeve_cards,
    load_theme_returns,
)
from factoropt.task_b_frozen import (  # noqa: E402
    DEFAULT_MIN_HISTORY_MONTHS,
    DEFAULT_TRAIN_CUTOFFS,
    DEFAULT_VALIDATION_CUTOFFS,
    FROZEN_TASK_VARIANT,
    build_code_generation_payload,
    build_rolling_windows,
    stable_hash,
)
from factoropt.semantic import ALL_TRACKS, normalize_track, semantic_cards_payload  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, allow_nan=False, default=str) + "\n")


def _opaque_sequence_id(region: str, anchor: pd.Timestamp, factor_ids: Sequence[str]) -> str:
    raw = f"{region}|{anchor.strftime('%Y-%m-%d')}|{'|'.join(sorted(map(str, factor_ids)))}".encode()
    return "sequence_" + hashlib.sha256(raw).hexdigest()[:16]


def _select_contiguous(episodes: Sequence[Any], start_anchor: str, months: int) -> list[Any]:
    wanted = pd.Period(start_anchor, freq="M").to_timestamp("M")
    ordered = sorted(episodes, key=lambda item: pd.Timestamp(item.anchor_date))
    starts = [index for index, episode in enumerate(ordered) if pd.Timestamp(episode.anchor_date) == wanted]
    if not starts:
        available = [pd.Timestamp(item.anchor_date).strftime("%Y-%m") for item in ordered[:5]]
        raise ValueError(f"start anchor {start_anchor!r} not found; first available={available}")
    start = starts[0]
    selected = ordered[start : start + months]
    if len(selected) != months:
        raise ValueError(f"only {len(selected)} contiguous episodes remain after {start_anchor}")
    first_ids = list(map(str, selected[0].sleeve_ids))
    for index, episode in enumerate(selected, start=1):
        ids = list(map(str, episode.sleeve_ids))
        if ids != first_ids:
            raise ValueError(f"factor universe changed at sequence step {index}")
        if episode.horizon_months != 1 or len(episode.future.index) != 1:
            raise ValueError(f"sequence step {index} is not a one-month target")
        if index > 1:
            previous = selected[index - 2]
            if list(previous.history.index[1:]) != list(episode.history.index[:-1]):
                raise ValueError(f"history does not roll by one month at step {index}")
            if pd.Timestamp(previous.future.index[0]) != pd.Timestamp(episode.history.index[-1]):
                raise ValueError(f"settled target is not appended at sequence step {index}")
    return selected


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--returns",
        required=True,
        type=Path,
        help="monthly raw-factor return table (must contain factor_id; use --source-type theme only for legacy cases)",
    )
    parser.add_argument(
        "--source-type",
        choices=("raw_factor", "theme", "auto"),
        default="raw_factor",
        help="input contract; default raw_factor prevents accidental 13-theme experiments",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--region", default="USA")
    parser.add_argument("--track", choices=ALL_TRACKS, default="numeric_only")
    parser.add_argument(
        "--factor-cards",
        type=Path,
        default=Path("data/processed/jkp/2025-12/factor_cards_enriched.jsonl"),
        help="private enriched cards used by the semantic track",
    )
    parser.add_argument("--start-anchor", required=True, help="first formation month YYYY-MM")
    parser.add_argument("--months", type=int, default=12)
    parser.add_argument("--lookback", type=int, default=120)
    parser.add_argument(
        "--min-history",
        type=int,
        help="minimum observed history per factor for episode selection (defaults to lookback-12, floor 12)",
    )
    parser.add_argument(
        "--feature-warmup-months",
        type=int,
        default=DEFAULT_MIN_HISTORY_MONTHS,
        help="minimum prefix needed before creating a supervised feature row (default: 12)",
    )
    parser.add_argument(
        "--train-cutoffs",
        type=int,
        default=DEFAULT_TRAIN_CUTOFFS,
        help="labelled cutoff months used for fitting at each origin (default: 12)",
    )
    parser.add_argument(
        "--validation-cutoffs",
        type=int,
        default=DEFAULT_VALIDATION_CUTOFFS,
        help="most recent labelled cutoff months used for inner validation (default: 12)",
    )
    parser.add_argument("--min-factors", type=int, default=3)
    parser.add_argument(
        "--factor-ids", nargs="+",
        help="optional fixed factor/theme IDs (space- or comma-separated) for matching dev/test universes",
    )
    parser.add_argument("--returns-in-percent", action="store_true")
    args = parser.parse_args(argv)
    track = normalize_track(args.track)
    if args.months < 1:
        parser.error("--months must be positive")
    if args.lookback < 2:
        parser.error("--lookback must be at least 2")
    if args.feature_warmup_months < 1:
        parser.error("--feature-warmup-months must be positive")
    if args.train_cutoffs < 1 or args.validation_cutoffs < 1:
        parser.error("--train-cutoffs and --validation-cutoffs must be positive")
    if args.min_factors < 1:
        parser.error("--min-factors must be positive")
    if args.lookback - args.feature_warmup_months < args.train_cutoffs + args.validation_cutoffs:
        parser.error(
            "lookback - feature-warmup-months must cover train-cutoffs + validation-cutoffs; "
            "use a longer history (for example lookback=36 for a 24-month model window)"
        )

    episode_min_history = (
        int(args.min_history)
        if args.min_history is not None
        else int(min(args.lookback, max(12, args.lookback - 12)))
    )
    if episode_min_history < 1 or episode_min_history > args.lookback:
        parser.error("--min-history must be between 1 and --lookback")
    requested_factor_ids = None
    if args.factor_ids:
        requested_factor_ids = [
            item.strip()
            for token in args.factor_ids
            for item in str(token).split(",")
            if item.strip()
        ]
        if len(requested_factor_ids) != len(set(requested_factor_ids)):
            parser.error("--factor-ids must be unique")
    if args.source_type == "raw_factor":
        returns = load_raw_factor_returns(
            args.returns, region=args.region, returns_in_percent=args.returns_in_percent
        )
    elif args.source_type == "theme":
        returns = load_theme_returns(
            args.returns, region=args.region, returns_in_percent=args.returns_in_percent
        )
    else:
        # ``auto`` is intentionally opt-in for backwards compatibility.  A
        # direct raw-factor file is preferred; if it fails validation, fall
        # back to the legacy theme loader and record the source type below.
        try:
            returns = load_raw_factor_returns(
                args.returns, region=args.region, returns_in_percent=args.returns_in_percent
            )
            args.source_type = "raw_factor"
        except (ValueError, FileNotFoundError):
            returns = load_theme_returns(
                args.returns, region=args.region, returns_in_percent=args.returns_in_percent
            )
            args.source_type = "theme"
    available_ids = sorted(str(value) for value in returns["factor_id" if "factor_id" in returns.columns else "sleeve_id"].unique())
    cards = load_sleeve_cards(args.factor_cards if track == "semantic" else None, available_ids)
    if track == "semantic":
        incomplete_cards = [
            factor_id
            for factor_id in available_ids
            if not str(cards[factor_id].get("neutral_name", "")).strip()
            or not any(
                str(cards[factor_id].get(field, "")).strip()
                for field in ("neutral_description", "formula_or_definition", "formula_latex")
            )
        ]
        if incomplete_cards:
            parser.error(
                "semantic track requires complete name/definition/formula cards; "
                f"missing for {incomplete_cards[:10]}"
            )
    episodes = build_router_episodes(
        returns,
        cards,
        regions=[args.region],
        lookback=args.lookback,
        horizon=1,
        min_history=episode_min_history,
        min_sleeves=args.min_factors,
        anchor_step=1,
        sleeve_ids=requested_factor_ids,
    )
    selected = _select_contiguous(episodes, args.start_anchor, args.months)
    original_ids = [str(value) for value in selected[0].sleeve_ids]
    if requested_factor_ids is not None and set(original_ids) != set(requested_factor_ids):
        parser.error(
            "requested factor IDs are not all available in the selected history; "
            f"missing={sorted(set(requested_factor_ids) - set(original_ids))}"
        )
    original_to_masked = {original: f"S{index:02d}" for index, original in enumerate(original_ids, start=1)}
    masked_to_original = {masked: original for original, masked in original_to_masked.items()}
    # Keep the feature set linear in factor count so 153-factor cases are
    # usable by the code generator.  Pairwise anchor correlations belong to
    # the original prediction-JSON protocol, not this frozen-code track.
    feature_columns = [str(value) for value in HISTORICAL_PANEL_FEATURE_FIELDS]
    supervised_min_history = int(args.feature_warmup_months)
    if supervised_min_history < 1 or supervised_min_history >= len(selected[0].history.index):
        parser.error("--feature-warmup-months must be between 1 and lookback-1")
    train_df, valid_df, _predict_df, split_meta = build_rolling_windows(
        selected[0].history,
        original_to_masked=original_to_masked,
        feature_columns=feature_columns,
        train_cutoffs=args.train_cutoffs,
        validation_cutoffs=args.validation_cutoffs,
        min_history_months=supervised_min_history,
    )
    sequence_id = _opaque_sequence_id(args.region.upper(), pd.Timestamp(selected[0].anchor_date), original_ids)
    payload = build_code_generation_payload(
        train_df,
        valid_df,
        feature_columns=feature_columns,
        case_id=sequence_id,
        metadata={
            "region": str(args.region).upper(),
            "sequence_steps": int(args.months),
            "history_window_months": int(args.lookback),
            "forecast_horizon_months": 1,
            "train_cutoffs": int(args.train_cutoffs),
            "validation_cutoffs": int(args.validation_cutoffs),
            "refit_policy": "validation_select_then_train_plus_validation",
            "train_cutoff_time_range": split_meta["train_time_range"],
            "validation_cutoff_time_range": split_meta["validation_time_range"],
            "train_label_time_range": split_meta["train_label_time_range"],
            "validation_label_time_range": split_meta["validation_label_time_range"],
            "min_history_months": episode_min_history,
            "feature_warmup_months": int(args.feature_warmup_months),
            "seed": 1,
            "public_future_labels_included": False,
        },
        track=track,
        factor_cards=(
            semantic_cards_payload(cards, original_ids, original_to_masked)
            if track == "semantic"
            else None
        ),
    )

    truth_rows: list[dict[str, Any]] = []
    for step, episode in enumerate(selected, start=1):
        feature_stub = [{"factor_id": original_to_masked[str(value)]} for value in episode.sleeve_ids]
        truth_payload = {"case_id": f"{sequence_id}__step_{step:02d}", "features": feature_stub}
        truth = episode_to_truth_record(episode, truth_payload, masked_to_original)
        truth.update(
            {
                "sequence_id": sequence_id,
                "sequence_step": step,
                "task_variant": FROZEN_TASK_VARIANT,
                "public_payload_case_id": sequence_id,
                # Repeat the non-secret protocol knobs on every evaluator
                # row so a rolling run can reproduce a custom case without
                # relying on an out-of-band manifest.
                "history_window_months": int(args.lookback),
                "train_cutoffs": int(args.train_cutoffs),
                "validation_cutoffs": int(args.validation_cutoffs),
                "feature_warmup_months": int(args.feature_warmup_months),
            }
        )
        truth_rows.append(truth)

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    payload_path = output / "payload.json"
    truth_path = output / "rolling_truth.jsonl"
    _write_json(payload_path, payload)
    _write_jsonl(truth_path, truth_rows)
    manifest = {
        "schema_version": "task_b_frozen_case.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "factor_return_combination",
        "task_variant": FROZEN_TASK_VARIANT,
        "track": track,
        "sequence_id": sequence_id,
        "region": str(args.region).upper(),
        "start_anchor": pd.Timestamp(selected[0].anchor_date).strftime("%Y-%m"),
        "end_anchor": pd.Timestamp(selected[-1].anchor_date).strftime("%Y-%m"),
        "first_holding_month": pd.Timestamp(selected[0].future.index[0]).strftime("%Y-%m"),
        "last_holding_month": pd.Timestamp(selected[-1].future.index[0]).strftime("%Y-%m"),
        "sequence_steps": len(selected),
        "lookback_months": int(args.lookback),
        "forecast_horizon_months": 1,
        "factor_source_type": "raw_factor" if args.source_type == "raw_factor" else "theme",
        "factor_count": len(original_ids),
        "factor_ids_requested": requested_factor_ids,
        "factor_ids_masked": list(masked_to_original),
        "train_validation": split_meta,
        "rolling_window_definition": (
            "At each formation month t, use the latest train_cutoffs + "
            "validation_cutoffs labelled cutoff months ending at t-1; the "
            "label for cutoff u is the realised return at u+1. Shift the "
            "window one month after each target settles."
        ),
        "returns_source": str(args.returns.resolve()),
        "returns_source_type": str(args.source_type),
        "returns_sha256": _sha256(args.returns) if args.returns.is_file() else None,
        "public_future_labels_included": False,
        "evaluator_only_files": ["rolling_truth.jsonl"],
        "artifacts": {
            "payload": {"path": payload_path.name, "bytes": payload_path.stat().st_size, "sha256": _sha256(payload_path)},
            "rolling_truth": {"path": truth_path.name, "bytes": truth_path.stat().st_size, "sha256": _sha256(truth_path)},
        },
        "payload_hash": stable_hash(payload),
    }
    _write_json(output / "manifest.json", manifest)
    print(json.dumps({"payload": str(payload_path), "rolling_truth": str(truth_path), "manifest": str(output / "manifest.json"), "steps": len(selected), "factors": len(original_ids)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

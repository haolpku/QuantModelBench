#!/usr/bin/env python3
"""Build the Task C stock-level frozen-code rolling benchmark.

This builder follows the public/truth and chronological-block conventions of
``task_b_frozen_raw_rolling_294``.  The default Task C design has the same
multiple-history-window structure as Task B:

* 2Y/3Y/5Y labelled histories (24/36/60 months);
* a recent 12-month validation segment, leaving 12/24/48 training months;
* one hidden one-month forecast for twelve consecutive formation months;
* monthly from-scratch refits of one frozen source file (zero LLM calls after
  code freeze);
* evaluator-only ID/feature maps and rolling schedule;
* a 12-month purge between the last hidden target of one block and the next
  block's first training observation.

The existing ``factor_window_selection_plan.jsonl`` contains the same
feature subsets under each of the three history labels.  The default builder
keeps those labels as distinct, pre-registered model-history protocols while
reusing the subset within an anchor.  It therefore emits 14 feature-subset
cases per window and anchor (8/5/1 for 38/76/153 features), four anchors,
three windows and 168 cases in the default dataset.

``--allow-missing-sizes`` relaxes that to anchors whose eligible factor
universe cannot fill every size.  The early USA anchors added by
``build_task_c_early_selection_plan.py`` (1960-12, 1970-12) have fewer than
153 continuously populated factors, so they omit the 153-feature size and
contribute 13 subsets per window instead of 14.  ``anchor_subset_policy`` in
the manifest records each anchor's own per-window subset counts.

The public train/validation tables are Parquet artifacts because a stock-level
panel is too large and too easy to accidentally truncate in JSON.  The hidden
rolling truth is a small evaluator schedule; the trusted evaluator reloads the
source panel and daily returns when it executes the frozen response.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = CODE_ROOT.parent
ROOT = TASK_ROOT.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from factoropt.task_c import (  # noqa: E402
    build_code_generation_payload,
    validate_public_payload,
)
from factoropt.semantic import (
    ALL_TRACKS,
    SEMANTIC_CARD_SCHEMA_VERSION,
    normalize_track,
    semantic_cards_payload,
)
from factoropt.task_c_backtest import load_stock_panel  # noqa: E402


DEFAULT_PANEL = Path("data/raw/wrds/wgpvj1vbztoqbloo (1).parquet")
DEFAULT_DAILY = Path("data/raw/wrds/ctff_daily_ret.parquet")
DEFAULT_CUTOFFS = Path("data/raw/jkp/2025-12/nyse_cutoffs.csv")
DEFAULT_SELECTION_PLAN = Path("data/processed/task_c_early_selection_plan_207.jsonl")
DEFAULT_FACTOR_CARDS = Path("data/processed/jkp/2025-12/factor_cards_enriched.jsonl")
DEFAULT_OUTPUT = Path("data/processed/task_c_frozen_raw_rolling_207")
DEFAULT_ANCHORS = ("1960-12", "1980-12", "1994-12", "2008-12", "2022-12")
DEFAULT_TARGET_MONTHS = 12
DEFAULT_GAP_MONTHS = 12
DEFAULT_DEV_BLOCKS = 2
DEFAULT_SUBSETS_BY_SIZE = {38: 8, 76: 5, 153: 1}
WINDOW_MONTHS = {"2Y": 24, "3Y": 36, "5Y": 60}
DEFAULT_WINDOWS = ("2Y", "3Y", "5Y")
DEFAULT_VALIDATION_MONTHS = 12
EXPECTED_CASE_COUNT = 207
GENERATOR_VERSION = "build_task_c_dataset.v4_anchor_subset_policy"
MANIFEST_SCHEMA_VERSION = "task_c_frozen_raw_rolling_manifest.v1"
SPLIT_SCHEMA_VERSION = "task_c_frozen_raw_rolling_split.v1"
TRUTH_SCHEMA_VERSION = "task_c_frozen_raw_rolling_truth.v1"


def _load_factor_cards(path: Path | None, feature_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Load private JKP cards keyed by raw feature id.

    The loader intentionally keeps the source records private.  Public cards
    are produced later by ``semantic_cards_payload``'s strict whitelist.
    Missing records receive an empty card so numeric builds remain completely
    independent of the metadata file.
    """

    ids = [str(value) for value in feature_ids]
    cards: dict[str, dict[str, Any]] = {
        feature_id: {
            "factor_id": feature_id,
            "neutral_name": "",
            "neutral_description": "",
            "formula_or_definition": "",
            "input_fields": [],
        }
        for feature_id in ids
    }
    if path is None:
        return cards
    if not path.is_file():
        raise FileNotFoundError(f"factor cards file does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        rows = list(raw.values()) if isinstance(raw, Mapping) else raw
    else:
        rows = pd.read_csv(path).to_dict("records")
    if not isinstance(rows, list):
        raise ValueError(f"factor cards must contain a list of objects: {path}")
    for raw in rows:
        if not isinstance(raw, Mapping):
            continue
        feature_id = str(raw.get("factor_id") or raw.get("feature_id") or "").strip()
        if feature_id not in cards:
            continue
        cards[feature_id].update({str(key): _jsonable(value) for key, value in raw.items()})
    return cards


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        stamp = pd.Timestamp(value)
        return None if pd.isna(stamp) else stamp.strftime("%Y-%m-%d")
    if isinstance(value, pd.Period):
        return value.strftime("%Y-%m")
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (ROOT / candidate).resolve()


def _display_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(dict(row)), ensure_ascii=False, allow_nan=False) + "\n")
            count += 1
    return count


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def _period(value: str | pd.Period | pd.Timestamp) -> pd.Period:
    return pd.Period(value, freq="M")


def _case_id(anchor: str, window: str, subset_id: str) -> str:
    material = f"TASK_C|{anchor}|{window}|{subset_id}"
    return "c_case_" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def _safe_feature_names(raw_features: Sequence[str]) -> dict[str, str]:
    return {
        str(raw): f"f{index:03d}"
        for index, raw in enumerate(raw_features, start=1)
    }


def _artifact(path: Path, *, display_path: str, rows: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": display_path,
        "format": "parquet",
        "bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }
    if rows is not None:
        result["rows"] = int(rows)
    return result


def _load_selection_plan(
    path: Path,
    *,
    anchors: Sequence[str],
    subsets_by_size: Mapping[int, int],
    windows: Sequence[str],
    allow_missing_sizes: bool = False,
) -> tuple[list[dict[str, Any]], str]:
    """Load one case plan row for every requested history window.

    The selection plan reuses a subset within an anchor across windows.  We
    verify that invariant, while retaining separate rows because each window
    is a distinct model-history protocol.
    """

    rows = _read_jsonl(path)
    if not rows:
        raise ValueError(f"selection plan is empty: {path}")
    wanted_anchors = {str(value) for value in anchors}
    requested_windows = tuple(str(value) for value in windows)
    grouped: dict[tuple[str, str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row.get("record_type", "selection")) != "selection":
            continue
        anchor = str(row.get("anchor_month", ""))
        window = str(row.get("window", ""))
        if anchor not in wanted_anchors or window not in requested_windows:
            continue
        try:
            size = int(row["factor_count"])
            subset_index = int(row["subset_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"malformed selection row in {path}: {row}") from exc
        if size not in subsets_by_size or subset_index > int(subsets_by_size[size]):
            continue
        factors = [str(value) for value in row.get("factor_ids", [])]
        if len(factors) != size or len(set(factors)) != size:
            raise ValueError(f"selection row has wrong factor count: {row.get('subset_id')}")
        grouped[(anchor, window, size, subset_index)].append(row)

    selected: list[dict[str, Any]] = []
    for anchor in anchors:
        for window in requested_windows:
            for size in (38, 76, 153):
                expected = int(subsets_by_size.get(size, 0))
                if allow_missing_sizes and not any(
                    entries
                    for (entry_anchor, entry_window, entry_size, _), entries in grouped.items()
                    if entry_anchor == str(anchor)
                    and entry_window == window
                    and entry_size == size
                ):
                    continue
                for subset_index in range(1, expected + 1):
                    entries = grouped.get((str(anchor), window, size, subset_index), [])
                    if not entries:
                        if allow_missing_sizes:
                            break
                        raise ValueError(
                            f"selection plan has no {anchor}/{window}/{size}/s{subset_index:02d} row"
                        )
                    source = entries[0]
                    factors = tuple(sorted(str(value) for value in source.get("factor_ids", [])))
                    if len(entries) > 1 and any(
                        tuple(sorted(str(value) for value in entry.get("factor_ids", []))) != factors
                        for entry in entries[1:]
                    ):
                        raise ValueError(
                            "duplicate selection rows disagree for "
                            f"{anchor}/{window}/{size}/s{subset_index:02d}"
                        )
                    cross_window = []
                    for other_window in requested_windows:
                        other = grouped.get((str(anchor), other_window, size, subset_index), [])
                        if not other:
                            raise ValueError(
                                "selection plan is missing a paired history window for "
                                f"{anchor}/{size}/s{subset_index:02d}: {other_window}"
                            )
                        cross_window.append(
                            tuple(sorted(str(value) for value in other[0].get("factor_ids", [])))
                        )
                    if len(set(cross_window)) != 1:
                        raise ValueError(
                            "factor subset differs across history windows for "
                            f"{anchor}/{size}/s{subset_index:02d}"
                        )
                    record = dict(source)
                    record["anchor_month"] = str(anchor)
                    record["window"] = str(window)
                    record["factor_count"] = size
                    record["subset_index"] = subset_index
                    record["factor_ids"] = [str(value) for value in source["factor_ids"]]
                    record["factor_ids_sorted"] = sorted(record["factor_ids"])
                    record["source_windows"] = list(requested_windows)
                    record["deduplicated_for_task_c"] = False
                    record["same_subset_across_windows"] = True
                    selected.append(record)

    # Preserve the canonical anchor/size/subset order, independent of JSONL
    # line ordering.
    selected.sort(
        key=lambda row: (
            list(anchors).index(str(row["anchor_month"])),
            requested_windows.index(str(row["window"])),
            int(row["factor_count"]),
            int(row["subset_index"]),
        )
    )
    return selected, _sha256(path)


def _gap_audit(
    anchors: Sequence[str],
    *,
    window_protocols: Mapping[str, Mapping[str, Any]],
    target_months: int,
    required_gap: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left_index, left_label in enumerate(anchors):
        left = _period(left_label)
        for right_label in anchors[left_index + 1 :]:
            right = _period(right_label)
            for window, protocol in window_protocols.items():
                history_months = int(protocol["history_months"])
                left_target_end = left + int(target_months)
                right_history_start = right - history_months
                gap = right_history_start.ordinal - left_target_end.ordinal - 1
                row = {
                    "left_anchor": str(left_label),
                    "right_anchor": str(right_label),
                    "window": str(window),
                    "history_months": history_months,
                    "left_history_start": (left - history_months).strftime("%Y-%m"),
                    "left_hidden_target_end": left_target_end.strftime("%Y-%m"),
                    "right_history_start": right_history_start.strftime("%Y-%m"),
                    "gap_months": int(gap),
                    "required_gap_months": int(required_gap),
                    "passes": bool(gap >= required_gap),
                }
                rows.append(row)
                if gap < required_gap:
                    raise ValueError(f"purge gap failed: {row}")
    return rows


def _normalise_source_panel(panel: pd.DataFrame) -> pd.DataFrame:
    frame = panel.copy()
    if "label" not in frame.columns:
        if "ret_exc_lead1m" not in frame.columns:
            raise ValueError("stock panel must contain ret_exc_lead1m")
        frame = frame.rename(columns={"ret_exc_lead1m": "label"})
    if "eom" not in frame.columns:
        raise ValueError("stock panel must contain eom")
    frame["eom"] = pd.to_datetime(frame["eom"], errors="coerce")
    frame = frame.loc[frame["eom"].notna()].copy()
    frame["_period"] = frame["eom"].dt.to_period("M")
    frame["id"] = frame["id"].astype("string")
    return frame.sort_values(["_period", "id"], kind="mergesort").reset_index(drop=True)


def _window_frames(
    panel: pd.DataFrame,
    formation: str | pd.Period,
    *,
    raw_features: Sequence[str],
    train_months: int,
    validation_months: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Construct Task C windows without repeatedly normalising a huge panel."""

    origin = _period(formation)
    valid_start = origin - int(validation_months)
    valid_end = origin - 1
    train_start = valid_start - int(train_months)
    train_end = valid_start - 1
    required = {"id", "_period", *[str(value) for value in raw_features], "label"}
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"panel missing required columns: {missing[:10]}")

    train = panel.loc[panel["_period"].between(train_start, train_end), [
        "id", "_period", *raw_features, "label"
    ]].copy()
    valid = panel.loc[panel["_period"].between(valid_start, valid_end), [
        "id", "_period", *raw_features, "label"
    ]].copy()
    predict = panel.loc[panel["_period"].eq(origin), [
        "id", "_period", *raw_features
    ]].copy()
    if train.empty or valid.empty or predict.empty:
        raise ValueError(
            f"insufficient rows at {origin}: train={len(train)}, "
            f"validation={len(valid)}, predict={len(predict)}"
        )
    train = train.dropna(subset=["label"]).copy()
    valid = valid.dropna(subset=["label"]).copy()
    if train.empty or valid.empty:
        raise ValueError(f"training/validation labels are empty at {origin}")

    def finish(frame: pd.DataFrame, include_label: bool) -> pd.DataFrame:
        out = frame.rename(columns={"_period": "time_period"}).copy()
        out["time_key"] = out["time_period"].map(
            lambda value: f"t-{origin.ordinal - pd.Period(value, freq='M').ordinal}"
        )
        out = out.drop(columns=["time_period"])
        ordered = ["id", "time_key", *raw_features]
        if include_label:
            ordered.append("label")
        return out[ordered].reset_index(drop=True)

    train_out = finish(train, True)
    valid_out = finish(valid, True)
    predict_out = finish(predict, False)
    metadata = {
        "formation_key": "t-0",
        "train_month_range": [f"t-{int(train_months + validation_months)}", f"t-{int(validation_months + 1)}"],
        "validation_month_range": [f"t-{int(validation_months)}", "t-1"],
        "predict_month": "t-0",
        "train_months": int(train_months),
        "validation_months": int(validation_months),
        "label_column": "label",
        "future_labels_included": False,
    }
    return train_out, valid_out, predict_out, metadata


def _public_frame(
    frame: pd.DataFrame,
    *,
    raw_features: Sequence[str],
    feature_map: Mapping[str, str],
    id_map: Mapping[str, str],
    include_label: bool,
) -> pd.DataFrame:
    columns = ["id", "time_key", *raw_features] + (["label"] if include_label else [])
    out = frame[columns].copy()
    out["id"] = out["id"].astype("string").map(id_map).astype("string")
    if out["id"].isna().any():
        raise ValueError("public frame contains an ID missing from evaluator map")
    out = out.rename(columns={raw: feature_map[raw] for raw in raw_features})
    return out


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_parquet(path, index=False, engine="pyarrow", compression="zstd")
    except ImportError as exc:  # pragma: no cover - source itself is Parquet
        raise RuntimeError("Task C frozen dataset requires pyarrow") from exc


def _write_case(
    *,
    staging_root: Path,
    final_root: Path,
    case_number: int,
    block_id: str,
    block_index: int,
    anchor: str,
    window: str,
    subset: Mapping[str, Any],
    panel: pd.DataFrame,
    source_panel: Path,
    source_daily: Path,
    source_cutoffs: Path,
    target_months: int,
    train_months: int,
    validation_months: int,
    gap_months: int,
    selection_plan_hash: str,
    track: str = "numeric_only",
    factor_cards: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    track = normalize_track(track)
    raw_features = [str(value) for value in subset["factor_ids"]]
    feature_map = _safe_feature_names(raw_features)
    public_features = [feature_map[value] for value in raw_features]
    all_ids = sorted(str(value) for value in panel["id"].dropna().unique())
    width = max(6, len(str(len(all_ids))))
    id_map = {value: f"I{index:0{width}d}" for index, value in enumerate(all_ids, start=1)}
    masked_to_original = {masked: original for original, masked in id_map.items()}
    masked_to_raw_feature = {masked: raw for raw, masked in feature_map.items()}

    train, valid, _predict, window_meta = _window_frames(
        panel,
        anchor,
        raw_features=raw_features,
        train_months=train_months,
        validation_months=validation_months,
    )
    train_public = _public_frame(
        train,
        raw_features=raw_features,
        feature_map=feature_map,
        id_map=id_map,
        include_label=True,
    )
    valid_public = _public_frame(
        valid,
        raw_features=raw_features,
        feature_map=feature_map,
        id_map=id_map,
        include_label=True,
    )

    case_id = _case_id(anchor, window, str(subset["subset_id"]))
    case_dir = staging_root / "cases" / block_id / case_id
    case_dir.mkdir(parents=True, exist_ok=False)
    artifact_dir = case_dir / "artifacts"
    train_path = artifact_dir / "train.parquet"
    valid_path = artifact_dir / "validation.parquet"
    _write_parquet(train_public, train_path)
    _write_parquet(valid_public, valid_path)

    payload = build_code_generation_payload(
        train_public,
        valid_public,
        feature_columns=public_features,
        case_id=case_id,
        metadata={
            "history_window": str(window),
            "history_window_months": int(train_months + validation_months),
            "train_months": int(train_months),
            "validation_months": int(validation_months),
            "rolling_test_months": int(target_months),
            "feature_count": int(len(raw_features)),
            "feature_representation": "case_masked_numeric",
            "refit_policy": "from_scratch_each_month",
            "public_artifacts_only": True,
            "seed": 1,
        },
        track=track,
        factor_cards=(
            semantic_cards_payload(factor_cards, raw_features, feature_map)
            if track == "semantic"
            else None
        ),
        include_rows=False,
        train_artifact={
            "path": "artifacts/train.parquet",
            "format": "parquet",
            "rows": int(len(train_public)),
            "bytes": int(train_path.stat().st_size),
            "sha256": _sha256(train_path),
        },
        validation_artifact={
            "path": "artifacts/validation.parquet",
            "format": "parquet",
            "rows": int(len(valid_public)),
            "bytes": int(valid_path.stat().st_size),
            "sha256": _sha256(valid_path),
        },
    )
    payload["artifact_root"] = "."
    payload_errors = validate_public_payload(payload)
    if payload_errors:
        raise ValueError(f"generated public payload is invalid: {payload_errors[:5]}")
    payload_path = case_dir / "payload.json"
    _write_json(payload_path, payload)

    # The evaluator map is separate from the public payload.  It covers every
    # source row in the loaded rolling interval, including hidden formation
    # months, so the same map can be used for all twelve executions.
    id_map_path = case_dir / "evaluator_id_map.json"
    feature_map_path = case_dir / "evaluator_feature_map.json"
    _write_json(
        id_map_path,
        {
            "case_id": case_id,
            "original_to_masked": id_map,
            "masked_to_original": masked_to_original,
            "note": "evaluator-only; never provide this file to the LLM",
        },
    )
    _write_json(
        feature_map_path,
        {
            "case_id": case_id,
            "original_to_masked": feature_map,
            "masked_to_original": masked_to_raw_feature,
            "public_feature_columns": public_features,
            "note": "evaluator-only; feature names are anonymized in the public payload",
        },
    )

    truth_rows: list[dict[str, Any]] = []
    anchor_period = _period(anchor)
    for step in range(target_months):
        formation = anchor_period + step
        step_train, step_valid, step_predict, step_meta = _window_frames(
            panel,
            formation,
            raw_features=raw_features,
            train_months=train_months,
            validation_months=validation_months,
        )
        target_rows = panel.loc[panel["_period"].eq(formation), ["id", "label"]]
        target_rows = target_rows.dropna(subset=["label"])
        truth_rows.append(
            {
                "record_type": "rolling_truth",
                "truth_schema_version": TRUTH_SCHEMA_VERSION,
                "sequence_id": case_id,
                "sequence_step": int(step + 1),
                "history_window": str(window),
                "history_window_months": int(train_months + validation_months),
                "formation_eom": formation.to_timestamp("M").strftime("%Y-%m-%d"),
                "holding_month": (formation + 1).strftime("%Y-%m"),
                "train_month_range": step_meta["train_month_range"],
                "validation_month_range": step_meta["validation_month_range"],
                "predict_month": step_meta["predict_month"],
                "train_rows": int(len(step_train)),
                "validation_rows": int(len(step_valid)),
                "predict_rows": int(len(step_predict)),
                "realized_label_rows": int(len(target_rows)),
                "realized_label_column": "ret_exc_lead1m",
                "source_panel": _display_path(source_panel),
                "source_daily_returns": _display_path(source_daily),
                "source_nyse_cutoffs": _display_path(source_cutoffs),
                "labels_released_after_prediction": True,
                "llm_calls": 0,
            }
        )
    truth_path = case_dir / "rolling_truth.jsonl"
    _write_jsonl(truth_path, truth_rows)

    final_case_dir = final_root / "cases" / block_id / case_id
    record: dict[str, Any] = {
        "record_type": "case",
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "track": track,
        "semantic_card_schema_version": (
            SEMANTIC_CARD_SCHEMA_VERSION if track == "semantic" else None
        ),
        "case_number": int(case_number),
        "case_id": case_id,
        "sequence_id": case_id,
        "dataset_scope": "stock_multifactor_code_frozen_rolling",
        "dataset_id": _display_path(final_case_dir),
        "path": _display_path(final_case_dir),
        "case_manifest_path": _display_path(final_case_dir / "manifest.json"),
        "block_id": block_id,
        "block_index": int(block_index),
        "anchor_month": anchor,
        "history_window": str(window),
        "start_anchor_month": anchor,
        "end_anchor_month": (anchor_period + target_months - 1).strftime("%Y-%m"),
        "history_start_month": (anchor_period - train_months - validation_months).strftime("%Y-%m"),
        "history_end_month": (anchor_period - 1).strftime("%Y-%m"),
        "target_start_month": (anchor_period + 1).strftime("%Y-%m"),
        "target_end_month": (anchor_period + target_months).strftime("%Y-%m"),
        "anchor_date_is_public": False,
        "task": "stock_multifactor_model",
        "feature_count": int(len(raw_features)),
        "sample_size": int(len(raw_features)),
        "requested_sample_size": int(subset["factor_count"]),
        "selected_feature_ids": raw_features,
        "selected_feature_ids_sorted": sorted(raw_features),
        "public_feature_columns": public_features,
        "feature_names_are_masked": True,
        "subset_id": str(subset["subset_id"]),
        "subset_index": int(subset["subset_index"]),
        "selection_source_windows": list(subset.get("source_windows", [])),
        "selection_plan_sha256": selection_plan_hash,
        "sampling": {
            "method": str(subset.get("sampling_scope", "shared_factor_window_selection_plan")),
            "same_subset_across_selection_windows": True,
            "future_values_used_for_selection": False,
        },
        "window": {
            "name": str(window),
            "history_months": int(train_months + validation_months),
            "train_months": int(train_months),
            "validation_months": int(validation_months),
            "rolling_test_months": int(target_months),
            "feature_warmup_months": 0,
        },
        "train_validation": {
            "train_month_range": window_meta["train_month_range"],
            "validation_month_range": window_meta["validation_month_range"],
            "relative_time_encoding": "relative_months_from_formation",
        },
        "rolling_protocol": {
            "history_window": str(window),
            "history_window_months": int(train_months + validation_months),
            "test_steps": int(target_months),
            "one_step_horizon_months": 1,
            "train_months": int(train_months),
            "validation_months": int(validation_months),
            "refit_policy": "from_scratch_each_origin",
            "validation_policy": "select_or_early_stop_then_refit_train_plus_validation",
            "llm_calls_during_rolling": 0,
        },
        "source_panel": _display_path(source_panel),
        "source_daily_returns": _display_path(source_daily),
        "source_nyse_cutoffs": _display_path(source_cutoffs),
        "public_future_labels_included": False,
        "source_history_feature_ids_are_truth_only": True,
        "evaluator_only_files": [
            "rolling_truth.jsonl",
            "evaluator_id_map.json",
            "evaluator_feature_map.json",
        ],
        "artifacts": {
            "payload": {
                "path": _display_path(final_case_dir / "payload.json"),
                "bytes": int(payload_path.stat().st_size),
                "sha256": _sha256(payload_path),
            },
            "train": _artifact(
                train_path,
                display_path=_display_path(final_case_dir / "artifacts" / "train.parquet"),
                rows=len(train_public),
            ),
            "validation": _artifact(
                valid_path,
                display_path=_display_path(final_case_dir / "artifacts" / "validation.parquet"),
                rows=len(valid_public),
            ),
            "rolling_truth": {
                "path": _display_path(final_case_dir / "rolling_truth.jsonl"),
                "bytes": int(truth_path.stat().st_size),
                "sha256": _sha256(truth_path),
                "rows": len(truth_rows),
            },
            "evaluator_id_map": {
                "path": _display_path(final_case_dir / "evaluator_id_map.json"),
                "bytes": int(id_map_path.stat().st_size),
                "sha256": _sha256(id_map_path),
            },
            "evaluator_feature_map": {
                "path": _display_path(final_case_dir / "evaluator_feature_map.json"),
                "bytes": int(feature_map_path.stat().st_size),
                "sha256": _sha256(feature_map_path),
            },
        },
        "payload_hash": hashlib.sha256(payload_path.read_bytes()).hexdigest(),
    }
    _write_json(case_dir / "manifest.json", record)
    return record


def _split_manifests(
    *,
    staging_root: Path,
    final_root: Path,
    index: Mapping[str, Any],
    case_rows: Sequence[Mapping[str, Any]],
    dev_blocks: int,
    dev_anchors: Sequence[str] | None,
    source_manifest_hash: str,
) -> dict[str, Any]:
    blocks = sorted({str(row["block_id"]) for row in case_rows})
    if not blocks:
        raise ValueError("cannot split an empty case set")
    block_anchors = {
        str(row["block_id"]): str(row["anchor_month"])
        for row in case_rows
    }
    if dev_anchors is not None:
        requested = {str(value) for value in dev_anchors}
        available = set(block_anchors.values())
        if not requested or not requested.issubset(available):
            raise ValueError(
                f"dev_anchors must be a non-empty subset of dataset anchors; "
                f"requested={sorted(requested)}, available={sorted(available)}"
            )
        dev_set = {block for block, anchor in block_anchors.items() if anchor in requested}
        if len(dev_set) == len(blocks):
            raise ValueError("dev_anchors must leave at least one test block")
    elif len(blocks) > 1 and not 1 <= int(dev_blocks) < len(blocks):
        raise ValueError(f"dev_blocks must be between 1 and {len(blocks) - 1}")
    elif len(blocks) == 1:
        dev_set = set(blocks)
    else:
        dev_set = set(blocks[: int(dev_blocks)])
    test_set = set(blocks) - dev_set
    dev_anchor_values = sorted({block_anchors[block] for block in dev_set})
    test_anchor_values = sorted({block_anchors[block] for block in test_set})
    summaries: dict[str, Any] = {}
    for split, wanted in (("dev", dev_set), ("test", set(blocks) - dev_set)):
        selected = [dict(row) for row in case_rows if str(row["block_id"]) in wanted]
        split_index = dict(index)
        split_index.update(
            {
                "record_type": "index",
                "manifest_schema_version": SPLIT_SCHEMA_VERSION,
                "split_status": "time_block_holdout",
                "split": split,
                "split_rule": (
                    "complete anchor blocks stay together; explicit dev anchors"
                    if dev_anchors is not None
                    else "complete chronological blocks stay together; earliest blocks are dev"
                ),
                "split_blocks": sorted(wanted),
                "dev_blocks": sorted(dev_set),
                "test_blocks": sorted(test_set),
                "dev_anchors": dev_anchor_values,
                "test_anchors": test_anchor_values,
                "source_manifest": _display_path(final_root / "manifest.jsonl"),
                "source_manifest_sha256": source_manifest_hash,
                "case_count": len(selected),
                "rolling_forecast_rows": sum(int(row.get("rolling_protocol", {}).get("test_steps", 0)) for row in selected),
            }
        )
        _write_jsonl(staging_root / "splits" / split / "manifest.jsonl", [split_index, *selected])
        summaries[split] = {
            "case_count": len(selected),
            "sequence_count": len(selected),
            "rolling_forecast_rows": split_index["rolling_forecast_rows"],
            "blocks": sorted(wanted),
            "windows": dict(sorted(Counter(str(row.get("history_window")) for row in selected).items())),
            "feature_counts": dict(sorted(Counter(str(row.get("feature_count")) for row in selected).items())),
        }
    result = {
        "schema_version": "task_c_frozen_raw_rolling_split_summary.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": _display_path(final_root / "manifest.jsonl"),
        "source_manifest_sha256": source_manifest_hash,
        "all_blocks": blocks,
        "dev_blocks": sorted(dev_set),
        "test_blocks": sorted(test_set),
        "dev_anchors": dev_anchor_values,
        "test_anchors": test_anchor_values,
        "splits": summaries,
        "case_directories_reused_by_reference": True,
    }
    _write_json(staging_root / "split_manifest.json", result)
    return result


def _anchor_subset_policy(
    case_rows: Sequence[Mapping[str, Any]],
    *,
    anchors: Sequence[str],
    windows: Sequence[str],
    strict: bool = True,
) -> dict[str, dict[str, Any]]:
    """Report per-anchor subset counts on the same footing as ``subset_policy``.

    ``subset_policy`` counts feature subsets *per history window*, and an
    anchor contributes one case per window for every subset it keeps.  Counting
    raw cases here would multiply each entry by ``len(windows)`` and break the
    auditor, which compares this map against a per-anchor/per-window case
    count.  The counts are therefore taken within a window; ``case_count``
    stays the anchor's full case total.
    """

    policy: dict[str, dict[str, Any]] = {}
    for anchor in anchors:
        rows = [row for row in case_rows if str(row["anchor_month"]) == str(anchor)]
        by_window = {
            str(window): Counter(
                int(row["feature_count"])
                for row in rows
                if str(row["history_window"]) == str(window)
            )
            for window in windows
        }
        populated = [counts for counts in by_window.values() if counts]
        if strict and populated and any(counts != populated[0] for counts in populated[1:]):
            raise ValueError(
                f"{anchor} has inconsistent subset counts across history windows: "
                + json.dumps({window: dict(counts) for window, counts in by_window.items()}, sort_keys=True)
            )
        # A truncated debug build (``--max-cases``) can leave a window partly
        # filled, so fall back to the widest observed count per size.
        counts = Counter()
        for window_counts in populated:
            for size, count in window_counts.items():
                counts[size] = max(counts[size], count)
        policy[str(anchor)] = {
            "subsets_by_size": {str(size): int(counts[size]) for size in sorted(counts)},
            "subset_count_basis": "per_history_window",
            "cases_per_window": int(sum(counts.values())),
            "case_count": len(rows),
        }
    return policy


def _readme(*, index: Mapping[str, Any], split_summary: Mapping[str, Any], source_hash: str) -> str:
    dev = split_summary.get("splits", {}).get("dev", {})
    test = split_summary.get("splits", {}).get("test", {})
    protocols = index.get("window_protocols", {})
    protocol_lines = "\n".join(
        f"  - `{window}`: {spec.get('history_months')} months total = "
        f"{spec.get('train_months')} train + {spec.get('validation_months')} validation"
        for window, spec in protocols.items()
    )
    anchor_policy = index.get("anchor_subset_policy", {})
    subset_lines = []
    for anchor in index.get("anchors", []):
        policy = anchor_policy.get(str(anchor), {}) if isinstance(anchor_policy, Mapping) else {}
        counts = policy.get("subsets_by_size", {}) if isinstance(policy, Mapping) else {}
        formatted = ", ".join(
            f"{size}: {count}" for size, count in sorted(counts.items(), key=lambda item: int(item[0]))
        )
        subset_lines.append(f"  - `{anchor}`: {formatted or 'none'}")
    subset_description = "\n".join(subset_lines)
    return f"""# Task C frozen stock-factor rolling benchmark

This directory is the Task C stock-level counterpart of
`task_b_frozen_raw_rolling_294`.  It contains `{index.get('case_count')}`
non-duplicated cases and `{index.get('rolling_forecast_rows')}` hidden monthly
forecast origins.

The generated directory is for the `{index.get('track', 'numeric_only')}`
information track. `numeric_only` has no semantic cards. `semantic` keeps the
same masked `f001...` columns and adds a card per feature with only name,
definition and formula; explicit direction, theme/group, significance, source
paper and provenance fields remain excluded. Card wording may still describe
an economic category or construction framework and is not a sign/strength
prior.

## Protocol

- Source: `{index.get('source_panel')}` after USA + JKP screens.
- History protocols (the same feature subset is reused across them):
{protocol_lines}
- The LLM sees only public historical Parquet artifacts.  It returns Python
  source once; the evaluator executes that frozen source for
  `{index.get('target_months_per_sequence')}` monthly origins.  Rolling LLM
  calls are zero.
- Formation months are `{', '.join(index.get('anchors', []))}` as block starts.
  Each block's hidden target interval is followed by at least
  `{index.get('purge_policy', {}).get('minimum_gap_months')}` untouched months
  before the next block's training history.
- Feature subsets are the shared factor-window selection plan.  Canonical
  anchors use 8 subsets of 38, 5 subsets of 76 and one 153-feature census per
  history window; early anchors may omit a size when their frozen eligible
  factor universe is too small.  The same subset is reused across windows so
  comparisons isolate the amount of model history rather than changing the
  feature set.
{subset_description}
- Public IDs are masked (`I...`) and feature names are anonymous (`f001...`).
  The ID/feature maps and rolling schedule are evaluator-only.

## Files

- `manifest.jsonl`: canonical index followed by case records (SHA-256
  `{source_hash}`).
- `selection_plan.jsonl`: evaluator-side subset plan; do not mount it for the
  LLM (the case payload exposes only anonymous `f001...` names).
- `cases/<block>/<case_id>/payload.json`: public code-generation contract.
- `cases/<block>/<case_id>/artifacts/{{train,validation}}.parquet`: public
  historical rows; no predict rows or future labels.
- `cases/<block>/<case_id>/rolling_truth.jsonl`: evaluator-only monthly
  schedule and row-count audit.
- `cases/<block>/<case_id>/evaluator_id_map.json` and
  `evaluator_feature_map.json`: never mount these for the LLM.
- `splits/dev/manifest.jsonl` and `splits/test/manifest.jsonl`: chronological
  block split; case directories are referenced, not copied.

The default split has {dev.get('case_count', 0)} dev cases in blocks
{', '.join(dev.get('blocks', []))} and {test.get('case_count', 0)} test cases in
blocks {', '.join(test.get('blocks', []))}.

To generate code for one case, pass its `payload.json` to
`run_task_c_codegen.py`; then use the evaluator-owned source panel, maps and
the twelve formation months from `rolling_truth.jsonl` with
`run_task_c_rolling.py`.  The fixed backtester consumes daily `ret_exc` and
settles each holding month day by day.
"""


def _window_protocols(
    windows: Sequence[str],
    *,
    validation_months: int = DEFAULT_VALIDATION_MONTHS,
) -> tuple[tuple[str, ...], dict[str, dict[str, int]]]:
    requested = tuple(str(value) for value in windows)
    if not requested:
        raise ValueError("at least one history window is required")
    if len(set(requested)) != len(requested):
        raise ValueError("history windows must be unique")
    unknown = [value for value in requested if value not in WINDOW_MONTHS]
    if unknown:
        raise ValueError(f"unknown history windows {unknown}; choose from {sorted(WINDOW_MONTHS)}")
    if int(validation_months) < 1:
        raise ValueError("validation_months must be positive")
    protocols = {
        window: {
            "history_months": int(WINDOW_MONTHS[window]),
            "validation_months": int(validation_months),
            "train_months": int(WINDOW_MONTHS[window] - int(validation_months)),
        }
        for window in requested
    }
    if any(spec["train_months"] < 1 for spec in protocols.values()):
        raise ValueError("each history window must be longer than validation_months")
    return requested, protocols


def build_dataset(
    *,
    panel_path: Path,
    daily_path: Path,
    cutoffs_path: Path,
    selection_plan_path: Path,
    output_root: Path,
    anchors: Sequence[str] = DEFAULT_ANCHORS,
    windows: Sequence[str] = DEFAULT_WINDOWS,
    target_months: int = DEFAULT_TARGET_MONTHS,
    validation_months: int = DEFAULT_VALIDATION_MONTHS,
    gap_months: int = DEFAULT_GAP_MONTHS,
    dev_blocks: int = DEFAULT_DEV_BLOCKS,
    dev_anchors: Sequence[str] | None = None,
    subsets_by_size: Mapping[int, int] = DEFAULT_SUBSETS_BY_SIZE,
    max_cases: int | None = None,
    expected_case_count: int | None = EXPECTED_CASE_COUNT,
    track: str = "numeric_only",
    factor_cards_path: Path | None = DEFAULT_FACTOR_CARDS,
    allow_missing_sizes: bool = False,
) -> dict[str, Any]:
    track = normalize_track(track)
    if output_root.exists():
        if any(output_root.iterdir()):
            raise FileExistsError(f"refusing to overwrite non-empty output root: {output_root}")
        raise FileExistsError(f"output root already exists: {output_root}")
    for source in (panel_path, daily_path, cutoffs_path, selection_plan_path):
        if not source.is_file():
            raise FileNotFoundError(source)
    # Chronological order is part of the split contract; do not let CLI order
    # silently turn an intended future block into a training block.
    raw_anchors = tuple(str(_period(value)) for value in anchors)
    if len(set(raw_anchors)) != len(raw_anchors) or not raw_anchors:
        raise ValueError("anchors must be unique and non-empty")
    anchors = tuple(sorted(raw_anchors, key=_period))
    windows, window_specs = _window_protocols(windows, validation_months=validation_months)
    if target_months < 1 or gap_months < 0:
        raise ValueError("target_months must be positive and gap must be non-negative")
    if not 1 <= int(dev_blocks) < len(anchors):
        raise ValueError(f"dev_blocks must be between 1 and {len(anchors) - 1}")

    selections, selection_hash = _load_selection_plan(
        selection_plan_path,
        anchors=anchors,
        subsets_by_size=subsets_by_size,
        windows=windows,
        allow_missing_sizes=allow_missing_sizes,
    )
    all_selected_features = sorted(
        {str(value) for row in selections for value in row.get("factor_ids", [])}
    )
    cards = (
        _load_factor_cards(factor_cards_path, all_selected_features)
        if track == "semantic"
        else _load_factor_cards(None, all_selected_features)
    )
    if track == "semantic":
        incomplete_cards = [
            feature_id
            for feature_id in all_selected_features
            if not str(cards[feature_id].get("neutral_name", "")).strip()
            or not any(
                str(cards[feature_id].get(field, "")).strip()
                for field in ("neutral_description", "formula_or_definition", "formula_latex")
            )
        ]
        if incomplete_cards:
            raise ValueError(
                "semantic track requires complete name/definition/formula cards; "
                f"missing for {incomplete_cards[:10]}"
            )
    expected_from_plan = len(selections)
    if len(selections) != expected_from_plan:
        raise ValueError(f"selection plan yielded {len(selections)} cases, expected {expected_from_plan}")
    if expected_case_count is not None and max_cases is None and expected_from_plan != expected_case_count:
        raise ValueError(
            f"scheduled design has {expected_from_plan} cases, expected {expected_case_count}; "
            "pass --expected-case-count 0 to disable the assertion"
        )
    gap_rows = _gap_audit(
        anchors,
        window_protocols=window_specs,
        target_months=target_months,
        required_gap=gap_months,
    )

    parent = output_root.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=str(parent)))
    try:
        # Load one complete rolling interval per anchor.  The source panel is
        # projected to exactly the 153 selected fields; no raw daily data is
        # copied into this generated directory.
        case_rows: list[dict[str, Any]] = []
        selections_by_anchor: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for subset in selections:
            selections_by_anchor[str(subset["anchor_month"])].append(subset)
        case_number = 0
        all_feature_columns = sorted({str(value) for row in selections for value in row["factor_ids"]})
        for anchor in anchors:
            if max_cases is not None and case_number >= int(max_cases):
                break
            period = _period(anchor)
            max_history = max(int(spec["history_months"]) for spec in window_specs.values())
            start = (period - max_history).to_timestamp("M")
            end = (period + target_months).to_timestamp("M")
            loaded = load_stock_panel(
                panel_path,
                feature_columns=all_feature_columns,
                start=start,
                end=end,
            )
            panel = _normalise_source_panel(loaded)
            del loaded
            if panel.empty:
                raise ValueError(f"no screened stock rows for anchor {anchor}")
            block_index = anchors.index(anchor)
            for subset in selections_by_anchor[anchor]:
                if max_cases is not None and case_number >= int(max_cases):
                    break
                case_number += 1
                window = str(subset["window"])
                spec = window_specs[window]
                record = _write_case(
                    staging_root=staging,
                    final_root=output_root,
                    case_number=case_number,
                    block_id=f"block_{block_index:02d}",
                    block_index=block_index,
                    anchor=anchor,
                    window=window,
                    subset=subset,
                    panel=panel,
                    source_panel=panel_path,
                    source_daily=daily_path,
                    source_cutoffs=cutoffs_path,
                    target_months=target_months,
                    train_months=int(spec["train_months"]),
                    validation_months=int(spec["validation_months"]),
                    gap_months=gap_months,
                    selection_plan_hash=selection_hash,
                    track=track,
                    factor_cards=cards,
                )
                case_rows.append(record)
                if case_number % 5 == 0 or case_number == len(selections):
                    print(f"built {case_number}/{len(selections)} cases", file=sys.stderr, flush=True)
            del panel
            gc.collect()
        if max_cases is not None and not 1 <= int(max_cases) <= len(selections):
            raise ValueError(f"max_cases must be between 1 and {len(selections)}")
        if max_cases is None and len(case_rows) != expected_from_plan:
            raise RuntimeError(f"generated {len(case_rows)} cases, expected {expected_from_plan}")

        _write_jsonl(staging / "selection_plan.jsonl", selections)
        source_index: dict[str, Any] = {
            "record_type": "index",
            "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
            "generator_version": GENERATOR_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "task": "task_c",
            "dataset_scope": "stock_multifactor_code_frozen_rolling",
            "payload_variant": f"{track}_case_masked_v1",
            "track": track,
            "semantic_card_schema_version": (
                SEMANTIC_CARD_SCHEMA_VERSION if track == "semantic" else None
            ),
            "split_status": "unsplit",
            "source_panel": _display_path(panel_path),
            "source_panel_sha256": _sha256(panel_path),
            "source_daily_returns": _display_path(daily_path),
            "source_daily_returns_sha256": _sha256(daily_path),
            "source_nyse_cutoffs": _display_path(cutoffs_path),
            "source_nyse_cutoffs_sha256": _sha256(cutoffs_path),
            "source_selection_plan": _display_path(selection_plan_path),
            "source_selection_plan_sha256": selection_hash,
            "source_factor_cards": (
                _display_path(factor_cards_path)
                if track == "semantic" and factor_cards_path
                else None
            ),
            "source_factor_cards_sha256": (
                _sha256(factor_cards_path)
                if track == "semantic" and factor_cards_path and factor_cards_path.is_file()
                else None
            ),
            "selection_plan": {
                "path": _display_path(output_root / "selection_plan.jsonl"),
                "sha256": _sha256(staging / "selection_plan.jsonl"),
                "rows": len(selections),
                "deduplicated_for_task_c": False,
                "same_subset_across_windows": True,
            },
            "source_schema": ["id", "eom", "excntry", "me", "ret_exc_lead1m", "factor_features"],
            "source_feature_count": 153,
            "anchors": list(anchors),
            "windows": list(windows),
            "window_months": {window: int(spec["history_months"]) for window, spec in window_specs.items()},
            "window_protocols": window_specs,
            "block_ids": [f"block_{index:02d}" for index in range(len(anchors))],
            "temporal_block_count": len(anchors),
            "target_months_per_sequence": int(target_months),
            "purge_policy": {
                "minimum_gap_months": int(gap_months),
                "history_length_months": max(int(spec["history_months"]) for spec in window_specs.values()),
                "interval_definition": "history_start..hidden_target_end",
                "gap_audit_rows": gap_rows,
            },
            "split_policy": {
                "rule": "explicit_dev_anchor_holdout" if dev_anchors is not None else "chronological_time_block_holdout",
                "dev_blocks": list(dev_anchors) if dev_anchors is not None else list(anchors[:dev_blocks]),
                "test_blocks": (
                    [anchor for anchor in anchors if anchor not in set(dev_anchors)]
                    if dev_anchors is not None
                    else list(anchors[dev_blocks:])
                ),
            },
            "subset_policy": {str(key): int(value) for key, value in subsets_by_size.items()},
            "allow_missing_feature_sizes": bool(allow_missing_sizes),
            "anchor_subset_policy": _anchor_subset_policy(
                case_rows,
                anchors=anchors,
                windows=windows,
                strict=max_cases is None,
            ),
            "case_count": len(case_rows),
            "sequence_count": len(case_rows),
            "rolling_forecast_rows": len(case_rows) * int(target_months),
            "window_case_counts": dict(
                sorted(Counter(str(row["history_window"]) for row in case_rows).items())
            ),
            "feature_count_distribution": dict(sorted(Counter(str(row["feature_count"]) for row in case_rows).items())),
            "public_payload_audit": {
                "future_labels_included": False,
                "test_rows_included": False,
                "original_ids_included": False,
                "original_feature_names_included": False,
                "semantic_cards_included": track == "semantic",
                "calendar_dates_included": False,
                "rolling_llm_calls": 0,
            },
            "records_follow": "one case record per subsequent JSONL line",
        }
        _write_jsonl(staging / "manifest.jsonl", [source_index, *case_rows])
        source_manifest_hash = _sha256(staging / "manifest.jsonl")
        split_summary = _split_manifests(
            staging_root=staging,
            final_root=output_root,
            index=source_index,
            case_rows=case_rows,
            dev_blocks=dev_blocks,
            dev_anchors=dev_anchors,
            source_manifest_hash=source_manifest_hash,
        )
        _write_json(staging / "dataset_summary.json", {
            "schema_version": "task_c_frozen_raw_rolling_summary.v1",
            "generator_version": GENERATOR_VERSION,
            "case_count": len(case_rows),
            "rolling_forecast_rows": len(case_rows) * int(target_months),
            "anchors": list(anchors),
            "windows": list(windows),
            "window_protocols": window_specs,
            "split_summary": split_summary,
            "source_manifest_sha256": source_manifest_hash,
        })
        (staging / "README.md").write_text(
            _readme(index=source_index, split_summary=split_summary, source_hash=source_manifest_hash),
            encoding="utf-8",
        )
        staging.replace(output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "output_root": str(output_root),
        "manifest": str(output_root / "manifest.jsonl"),
        "case_count": len(case_rows),
        "rolling_forecast_rows": len(case_rows) * int(target_months),
        "anchors": list(anchors),
        "split_summary": split_summary,
        "source_manifest_sha256": source_manifest_hash,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--daily", type=Path, default=DEFAULT_DAILY)
    parser.add_argument("--nyse-cutoffs", type=Path, default=DEFAULT_CUTOFFS)
    parser.add_argument("--selection-plan", type=Path, default=DEFAULT_SELECTION_PLAN)
    parser.add_argument(
        "--factor-cards",
        type=Path,
        default=DEFAULT_FACTOR_CARDS,
        help="private enriched JKP cards; required for the semantic track",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--track",
        choices=ALL_TRACKS,
        default="numeric_only",
        help="public information track; semantic_numeric is accepted as a legacy alias",
    )
    parser.add_argument("--anchors", nargs="+", default=list(DEFAULT_ANCHORS))
    parser.add_argument(
        "--windows",
        nargs="+",
        choices=tuple(WINDOW_MONTHS),
        default=list(DEFAULT_WINDOWS),
        help="history protocols to generate (default: 2Y 3Y 5Y)",
    )
    parser.add_argument("--target-months", type=int, default=DEFAULT_TARGET_MONTHS)
    parser.add_argument("--validation-months", type=int, default=DEFAULT_VALIDATION_MONTHS)
    parser.add_argument("--gap-months", type=int, default=DEFAULT_GAP_MONTHS)
    parser.add_argument("--dev-blocks", type=int, default=DEFAULT_DEV_BLOCKS)
    parser.add_argument(
        "--dev-anchors",
        nargs="+",
        help="explicit dev anchors; all other anchors are held out as test blocks",
    )
    parser.add_argument("--subsets-38", type=int, default=8)
    parser.add_argument("--subsets-76", type=int, default=5)
    parser.add_argument("--subsets-153", type=int, default=1)
    parser.add_argument("--max-cases", type=int, help="debug limit; disables the full count assertion")
    parser.add_argument(
        "--expected-case-count",
        type=int,
        default=EXPECTED_CASE_COUNT,
        help="expected full-design count; use 0 to disable",
    )
    parser.add_argument(
        "--allow-missing-sizes",
        action="store_true",
        help="allow an anchor to omit a feature size absent from its selection plan",
    )
    args = parser.parse_args(argv)
    expected = None if args.expected_case_count == 0 else int(args.expected_case_count)
    try:
        result = build_dataset(
            panel_path=_resolve(args.panel),
            daily_path=_resolve(args.daily),
            cutoffs_path=_resolve(args.nyse_cutoffs),
            selection_plan_path=_resolve(args.selection_plan),
            output_root=_resolve(args.output_root),
            anchors=tuple(args.anchors),
            windows=tuple(args.windows),
            target_months=int(args.target_months),
            validation_months=int(args.validation_months),
            gap_months=int(args.gap_months),
            dev_blocks=int(args.dev_blocks),
            dev_anchors=tuple(args.dev_anchors) if args.dev_anchors else None,
            subsets_by_size={38: int(args.subsets_38), 76: int(args.subsets_76), 153: int(args.subsets_153)},
            max_cases=args.max_cases,
            expected_case_count=expected,
            track=str(args.track),
            factor_cards_path=_resolve(args.factor_cards) if args.factor_cards else None,
            allow_missing_sizes=bool(args.allow_missing_sizes),
        )
    except (FileExistsError, FileNotFoundError, OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

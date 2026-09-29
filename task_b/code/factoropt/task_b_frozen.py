"""Frozen-code, rolling-origin execution for Task B.

The original Task B agentic runner asks the model for a prediction JSON object
for every case.  This module defines the alternative *frozen code* protocol:
an LLM is asked for a ``fit_and_predict`` source once (optionally with a small
number of pre-test validation revisions), then that source is executed locally
for every month in a rolling one-step sequence.  No LLM call is made during
the rolling/test phase.

The implementation deliberately keeps the factor-return experiment separate
from the stock-level Task C package.  The Task C code runner is reused only as
the policy-checked subprocess boundary; all Task B features, targets,
portfolio rules, and metrics remain Task B's own contracts.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Sequence

import numpy as np
import pandas as pd

from .task_b import (
    HISTORICAL_PANEL_FEATURE_FIELDS,
    RouterEpisode,
    build_compact_features,
    evaluate_portfolio,
    fixed_portfolio_weights,
    score_predictions,
    turnover,
)
from .semantic import (
    ALL_TRACKS,
    CANONICAL_TRACKS,
    SEMANTIC_CARD_SCHEMA_VERSION,
    normalize_track,
    semantic_card_payload,
    validate_semantic_cards,
)


FROZEN_CODE_SCHEMA_VERSION = "task_b_factor_code_response.v1"
FROZEN_PAYLOAD_SCHEMA_VERSION = "task_b_factor_code_payload.v1"
# Bump whenever the response instructions or public payload contract changes.
# v6: after bounded revisions, the LLM (not host max-IC) chooses the frozen
# candidate from the valid set. Host metrics remain diagnostics and fallback.
FROZEN_PROMPT_VERSION = "task_b_factor_code_model_v6_llm_candidate_selection"
FROZEN_TASK_VARIANT = "factor_return_code_frozen_rolling_12step"
DEFAULT_MIN_HISTORY_MONTHS = 12
# The default rolling information set is two years of labelled cutoffs:
# 12 months for parameter fitting and the most recent 12 for inner validation.
# Longer windows (for example 120+12) remain configurable from the CLI.
DEFAULT_TRAIN_CUTOFFS = 12
DEFAULT_VALIDATION_CUTOFFS = 12
# Large enough for a hand-written deep-learning model; the limit exists to
# bound the sandbox and the revision feedback, not to discourage model families.
MAX_CODE_CHARS = 120_000


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        stamp = pd.Timestamp(value)
        return None if pd.isna(stamp) else stamp.isoformat()
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


def stable_hash(value: Any) -> str:
    encoded = json.dumps(
        _json_safe(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def code_hash(response_or_code: Mapping[str, Any] | str) -> str:
    source = (
        str(response_or_code.get("code") or "")
        if isinstance(response_or_code, Mapping)
        else str(response_or_code)
    )
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _month_offset_label(history_length: int, cutoff: int) -> str:
    # The final legal labelled cutoff is t-1.  t-0 is deliberately never
    # included in a supervised training frame because its next return is the
    # hidden target.
    return f"t-{history_length - 1 - cutoff}"


def _masked_id_sort_key(value: Any) -> tuple[int, int, str]:
    """Order masked factor IDs by their numeric index, not lexicographically.

    ``S100`` sorts before ``S11`` as a plain string, so a lexicographic sort of a
    153-factor panel disagrees with the mask assignment order produced by
    ``enumerate(factor_ids, start=1)``.  That mismatch is what made the truth
    side (numeric) and the payload side (lexicographic) disagree once a panel
    passed 99 factors.  Unparseable ids keep a deterministic trailing bucket.
    """

    text = str(value)
    if len(text) > 1 and text[0] in "Ss" and text[1:].isdigit():
        return (0, int(text[1:]), text)
    if text.isdigit():
        return (0, int(text), text)
    return (1, 0, text)


def _normalise_history(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        raise ValueError("history must contain at least one month")
    frame = history.copy()
    frame.columns = [str(value) for value in frame.columns]
    frame = frame.apply(pd.to_numeric, errors="coerce").sort_index()
    if frame.columns.empty:
        raise ValueError("history must contain at least one factor")
    return frame


def build_supervised_rows(
    history: pd.DataFrame,
    *,
    original_to_masked: Mapping[str, str] | None = None,
    feature_columns: Sequence[str] = HISTORICAL_PANEL_FEATURE_FIELDS,
    min_history_months: int = DEFAULT_MIN_HISTORY_MONTHS,
    feature_cache: MutableMapping[tuple[Any, ...], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Build factor-month features and next-month labels from one history.

    For cutoff ``u`` features use returns through ``u`` and the label is the
    same factor's return at ``u+1``.  The last row (t-0) is therefore never a
    training row.  No episode future is read by this function.
    """

    frame = _normalise_history(history)
    if min_history_months < 1:
        raise ValueError("min_history_months must be positive")
    if len(frame.index) <= min_history_months:
        raise ValueError(
            f"history needs more than min_history_months rows: {len(frame.index)} <= {min_history_months}"
        )
    mapping = {str(k): str(v) for k, v in (original_to_masked or {}).items()}
    for factor_id in frame.columns:
        mapping.setdefault(str(factor_id), str(factor_id))
    fields = [str(value) for value in feature_columns]
    missing_fields = [field for field in fields if field not in HISTORICAL_PANEL_FEATURE_FIELDS]
    if missing_fields:
        raise ValueError(f"unsupported Task B code feature fields: {missing_fields}")

    rows: list[dict[str, Any]] = []
    first_cutoff = max(int(min_history_months) - 1, 0)
    for cutoff in range(first_cutoff, len(frame.index) - 1):
        prefix = frame.iloc[: cutoff + 1]
        cache_key = (
            "prefix",
            str(prefix.index[-1]),
            tuple(frame.columns),
            tuple(fields),
        )
        table = feature_cache.get(cache_key) if feature_cache is not None else None
        if table is None:
            table = build_compact_features(prefix)
            if feature_cache is not None:
                feature_cache[cache_key] = table
        next_values = frame.iloc[cutoff + 1]
        time_key = _month_offset_label(len(frame.index), cutoff)
        for original_id in frame.columns:
            label = _finite(next_values.get(original_id))
            if label is None:
                continue
            source = table.loc[original_id]
            row: dict[str, Any] = {
                "id": mapping[str(original_id)],
                "time_key": time_key,
            }
            for field in fields:
                row[field] = _json_safe(source.get(field))
            row["label"] = label
            rows.append(row)
    if not rows:
        raise ValueError("history produced no fully observed supervised factor rows")
    return pd.DataFrame(rows, columns=["id", "time_key", *fields, "label"])


def split_supervised_rows(
    rows: pd.DataFrame,
    *,
    train_cutoffs: int = DEFAULT_TRAIN_CUTOFFS,
    validation_cutoffs: int = DEFAULT_VALIDATION_CUTOFFS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Split chronological factor-month rows into train and validation.

    ``train_cutoffs`` and ``validation_cutoffs`` count distinct historical
    cutoff months, not factor rows.  The newest validation label is the anchor
    month's realised return; the hidden next-month target is absent.
    """

    if train_cutoffs < 1 or validation_cutoffs < 1:
        raise ValueError("train_cutoffs and validation_cutoffs must be positive")
    required = {"id", "time_key", "label"}
    if not required.issubset(rows.columns):
        raise ValueError(f"supervised rows missing columns: {sorted(required - set(rows.columns))}")
    frame = rows.copy()

    def key(value: Any) -> int:
        text = str(value)
        try:
            return int(text.split("-", 1)[1])
        except (IndexError, ValueError):
            return 10**9

    # ``t-119`` is older than ``t-1``. Keep chronological order so the final
    # validation block is the newest block and no later label enters train.
    times = sorted({str(value) for value in frame["time_key"].dropna()}, key=key, reverse=True)
    required_cutoffs = int(train_cutoffs) + int(validation_cutoffs)
    if len(times) < required_cutoffs:
        raise ValueError(
            f"not enough historical cutoff months ({len(times)}) for "
            f"train/validation split ({train_cutoffs}+{validation_cutoffs})"
        )
    valid_times = times[-validation_cutoffs:]
    train_times = times[:-validation_cutoffs]
    if len(train_times) > train_cutoffs:
        train_times = train_times[-train_cutoffs:]
    train = frame.loc[frame["time_key"].astype(str).isin(train_times)].copy()
    valid = frame.loc[frame["time_key"].astype(str).isin(valid_times)].copy()
    if train.empty or valid.empty:
        raise ValueError("chronological train/validation split is empty")
    # Preserve oldest-to-newest order and deterministic factor order. Relative
    # offsets increase into the past (t-24 is older than t-1), so descending
    # numeric order is chronological.  The factor order must be the *numeric*
    # masked order (S01 < S02 < ... < S11 come before S100) because the mask was
    # assigned by ``enumerate(factor_ids, start=1)`` and the rolling truth keeps
    # that assignment order; a lexicographic sort only agrees with it while the
    # panel stays below 100 factors.
    for subset in (train, valid):
        subset["_time_order"] = subset["time_key"].map(key)
        subset["_mask_order"] = subset["id"].map(_masked_id_sort_key)
        subset.sort_values(
            ["_time_order", "_mask_order"], ascending=[False, True], inplace=True
        )
        subset.drop(columns=["_time_order", "_mask_order"], inplace=True)
        subset.reset_index(drop=True, inplace=True)

    def label_time(cutoff: str) -> str:
        # A cutoff t-k carries the next-month label at t-(k-1); t-1 therefore
        # has the already-realised anchor-month label t-0.
        offset = max(int(key(cutoff)) - 1, 0)
        return f"t-{offset}"

    metadata = {
        "train_cutoffs_requested": int(train_cutoffs),
        "validation_cutoffs_requested": int(validation_cutoffs),
        "train_cutoffs_used": int(len(train_times)),
        "validation_cutoffs_used": int(len(valid_times)),
        "train_time_range": [train_times[0], train_times[-1]],
        "validation_time_range": [valid_times[0], valid_times[-1]],
        "train_label_time_range": [label_time(train_times[0]), label_time(train_times[-1])],
        "validation_label_time_range": [label_time(valid_times[0]), label_time(valid_times[-1])],
        "train_rows": int(len(train)),
        "validation_rows": int(len(valid)),
    }
    return train, valid, metadata


def build_predict_rows(
    history: pd.DataFrame,
    *,
    original_to_masked: Mapping[str, str] | None = None,
    feature_columns: Sequence[str] = HISTORICAL_PANEL_FEATURE_FIELDS,
    feature_cache: MutableMapping[tuple[Any, ...], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Build the unlabeled t-0 factor rows for one rolling origin."""

    frame = _normalise_history(history)
    mapping = {str(k): str(v) for k, v in (original_to_masked or {}).items()}
    for factor_id in frame.columns:
        mapping.setdefault(str(factor_id), str(factor_id))
    fields = [str(value) for value in feature_columns]
    cache_key = (
        "anchor",
        str(frame.index[-1]),
        tuple(frame.columns),
        tuple(fields),
    )
    table = feature_cache.get(cache_key) if feature_cache is not None else None
    if table is None:
        table = build_compact_features(frame)
        if feature_cache is not None:
            feature_cache[cache_key] = table
    rows: list[dict[str, Any]] = []
    for original_id in frame.columns:
        source = table.loc[original_id]
        row: dict[str, Any] = {"id": mapping[str(original_id)], "time_key": "t-0"}
        for field in fields:
            row[field] = _json_safe(source.get(field))
        rows.append(row)
    return pd.DataFrame(rows, columns=["id", "time_key", *fields])


def build_rolling_windows(
    history: pd.DataFrame,
    *,
    original_to_masked: Mapping[str, str] | None = None,
    feature_columns: Sequence[str] = HISTORICAL_PANEL_FEATURE_FIELDS,
    train_cutoffs: int = DEFAULT_TRAIN_CUTOFFS,
    validation_cutoffs: int = DEFAULT_VALIDATION_CUTOFFS,
    min_history_months: int = DEFAULT_MIN_HISTORY_MONTHS,
    feature_cache: MutableMapping[tuple[Any, ...], pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Construct one point-in-time rolling ``train/valid/predict`` window.

    The returned ``train`` and ``valid`` blocks contain exactly the requested
    number of distinct cutoff months (subject to missing-label rows being
    dropped by :func:`build_supervised_rows`).  Calling this function again
    after appending the newly settled target advances both blocks by one
    month, while the source code and feature schema remain unchanged.
    """

    frame = _normalise_history(history)
    supervised = build_supervised_rows(
        frame,
        original_to_masked=original_to_masked,
        feature_columns=feature_columns,
        min_history_months=min_history_months,
        feature_cache=feature_cache,
    )
    train, valid, split_meta = split_supervised_rows(
        supervised,
        train_cutoffs=train_cutoffs,
        validation_cutoffs=validation_cutoffs,
    )
    predict = build_predict_rows(
        frame,
        original_to_masked=original_to_masked,
        feature_columns=feature_columns,
        feature_cache=feature_cache,
    )

    def calendar_range(relative_range: Sequence[str]) -> list[str]:
        dates: list[pd.Timestamp] = []
        for value in relative_range:
            offset = int(str(value).split("-", 1)[1])
            cutoff_position = len(frame.index) - 1 - offset
            label_position = cutoff_position + 1
            if 0 <= label_position < len(frame.index):
                dates.append(pd.Timestamp(frame.index[label_position]))
        if not dates:
            return []
        return [dates[0].strftime("%Y-%m"), dates[-1].strftime("%Y-%m")]

    metadata = dict(split_meta)
    metadata.update(
        {
            "formation_eom": pd.Timestamp(frame.index[-1]).strftime("%Y-%m-%d"),
            "history_window_months": int(len(frame.index)),
            "forecast_horizon_months": 1,
            "feature_warmup_months": int(min_history_months),
            "train_label_calendar_range": calendar_range(split_meta["train_time_range"]),
            "validation_label_calendar_range": calendar_range(split_meta["validation_time_range"]),
            "predict_calendar_month": (
                pd.Timestamp(frame.index[-1]) + pd.offsets.MonthEnd(1)
            ).strftime("%Y-%m"),
        }
    )
    return train, valid, predict, metadata


def validate_frozen_payload(payload: Any) -> list[str]:
    """Validate the public code-generation envelope and leakage boundary."""

    if not isinstance(payload, Mapping):
        return ["payload must be an object"]
    errors: list[str] = []
    legacy_split_fields = {"time_key_range", "label_name", "label_definition"}
    legacy_version_fields = {
        "schema_version",
        "prompt_semantic_version",
        "semantic_card_schema_version",
    }
    if "semantic_card_schema_version" in payload:
        errors.append("payload.semantic_card_schema_version is a legacy payload field")
    for key in ("case_id", "task", "task_variant", "feature_columns", "train", "validation", "predict", "metadata"):
        if key not in payload:
            errors.append(f"missing payload field: {key}")
    if payload.get("task") != "factor_return_combination":
        errors.append("task must be factor_return_combination")
    if payload.get("task_variant") != FROZEN_TASK_VARIANT:
        errors.append(f"task_variant must be {FROZEN_TASK_VARIANT}")
    raw_track = payload.get("track", "numeric_only")
    try:
        track = normalize_track(raw_track)
    except ValueError as exc:
        errors.append(str(exc))
        track = "numeric_only"
    raw_track_token = str(payload.get("track")).strip().lower()
    if "track" in payload and raw_track_token not in {track, "semantic_numeric"}:
        errors.append("payload.track must use the canonical name numeric_only or semantic")
    fields = payload.get("feature_columns")
    if not isinstance(fields, list) or not fields or any(not isinstance(x, str) for x in fields):
        errors.append("feature_columns must be a non-empty string array")
    expected_history_columns = (
        ["id", "time_key", *[str(value) for value in fields], "label"]
        if isinstance(fields, list) and all(isinstance(value, str) for value in fields)
        else None
    )
    for split in ("train", "validation"):
        section = payload.get(split)
        if not isinstance(section, Mapping):
            errors.append(f"{split} must be an object")
            continue
        for key in legacy_split_fields.intersection({str(item) for item in section}):
            errors.append(f"{split}.{key} is a legacy payload field; use rows and artifact only")
        rows = section.get("rows")
        if not isinstance(rows, list) or not rows:
            errors.append(f"{split}.rows must be a non-empty array")
            continue
        artifact = section.get("artifact")
        if isinstance(artifact, Mapping) and artifact.get("columns") is not None:
            if expected_history_columns is not None and artifact.get("columns") != expected_history_columns:
                errors.append(
                    f"{split}.artifact.columns must be exactly {expected_history_columns}"
                )
        for index, row in enumerate(rows[:10000]):
            if not isinstance(row, Mapping):
                errors.append(f"{split}.rows[{index}] must be an object")
                continue
            if not str(row.get("id", "")):
                errors.append(f"{split}.rows[{index}] has no id")
            if "label" not in row:
                errors.append(f"{split}.rows[{index}] has no label")
            if expected_history_columns is not None and list(row.keys()) != expected_history_columns:
                errors.append(
                    f"{split}.rows[{index}] columns must be exactly {expected_history_columns}"
                )
    # The semantic sidecar is keyed by masked factor IDs (the row ``id``), not
    # by the historical feature-field names.  Derive the expected order from
    # first appearance so hand-built payloads remain easy to validate.
    factor_ids: list[str] = []
    for split in ("train", "validation"):
        section = payload.get(split)
        if not isinstance(section, Mapping) or not isinstance(section.get("rows"), list):
            continue
        for row in section["rows"]:
            if isinstance(row, Mapping):
                factor_id = str(row.get("id", "")).strip()
                if factor_id and factor_id not in factor_ids:
                    factor_ids.append(factor_id)
    card_errors = validate_semantic_cards(
        payload.get("factor_cards", []),
        factor_ids,
        track=track,
    )
    errors.extend(card_errors)
    schema_marker = payload.get("semantic_card_schema_version")
    if track == "semantic" and schema_marker not in {None, SEMANTIC_CARD_SCHEMA_VERSION}:
        errors.append("semantic_card_schema_version is unsupported")
    if track == "numeric_only" and schema_marker is not None:
        errors.append("numeric_only payload must not declare semantic_card_schema_version")
    predict = payload.get("predict")
    if not isinstance(predict, Mapping):
        errors.append("predict must be an object")
    else:
        if predict.get("label_included") is not False:
            errors.append("predict.label_included must be false")
        if predict.get("rows_visible_to_llm") is not False:
            errors.append("predict.rows_visible_to_llm must be false")
        schema = predict.get("schema")
        expected_predict_columns = (
            ["id", "time_key", *[str(value) for value in fields]]
            if isinstance(fields, list) and all(isinstance(value, str) for value in fields)
            else None
        )
        if expected_predict_columns is not None and schema != expected_predict_columns:
            errors.append(f"predict.schema must be exactly {expected_predict_columns}")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        errors.append("metadata must be an object")
    elif metadata.get("future_labels_included") is not False:
        errors.append("metadata.future_labels_included must be false")
    if isinstance(metadata, Mapping):
        for key in legacy_version_fields.intersection({str(item) for item in metadata}):
            errors.append(f"metadata.{key} is a legacy payload field")
        for card_key in ("factor_cards", "semantic_cards"):
            if card_key in metadata and metadata.get(card_key) not in (None, [], {}):
                errors.append(f"metadata.{card_key} must not contain semantic cards")
    # Guard common hidden/evaluator keys recursively.  Historical labels are
    # legal only inside train/validation rows, so do not reject the literal
    # ``label`` field globally.
    hidden = {"future", "target", "truth", "masked_to_original", "original_to_masked", "anchor_date", "real_id", "real_ids"}

    def walk(value: Any, path: str = "", allow_label: bool = False) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                text = str(key).lower()
                if text in hidden:
                    errors.append(f"hidden/evaluation field is not allowed: {path + '.' if path else ''}{key}")
                if text in {"label", "labels"} and not allow_label:
                    # labels are permitted only in train/validation row maps;
                    # metadata and predict contracts must remain label-free.
                    errors.append(f"label field is not allowed here: {path + '.' if path else ''}{key}")
                walk(child, f"{path}.{key}" if path else str(key), allow_label=allow_label or text in {"train", "validation"})
        elif isinstance(value, list):
            for index, child in enumerate(value[:10000]):
                walk(child, f"{path}[{index}]", allow_label=allow_label)

    walk(payload)
    return list(dict.fromkeys(errors))


def build_code_generation_payload(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    feature_columns: Sequence[str] = HISTORICAL_PANEL_FEATURE_FIELDS,
    case_id: str = "task_b_frozen_case",
    metadata: Mapping[str, Any] | None = None,
    track: str = "numeric_only",
    factor_cards: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Create the public, masked one-time code-generation payload."""

    track = normalize_track(track)
    fields = [str(value) for value in feature_columns]
    required = {"id", "time_key", "label", *fields}
    for name, frame in (("train", train), ("validation", validation)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} frame missing columns: {missing}")

    def rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
        columns = ["id", "time_key", *fields, "label"]
        return [{str(k): _json_safe(v) for k, v in row.items()} for row in frame[columns].to_dict("records")]

    history_window = 120
    if metadata is not None:
        try:
            history_window = int(metadata.get("history_window_months", history_window))
        except (TypeError, ValueError):
            history_window = 120
    metadata_out: dict[str, Any] = {
        "id_column": "id",
        "time_column": "time_key",
        "target_is_factor_excess_return": True,
        "future_labels_included": False,
        "forecast_horizon_months": 1,
        "rolling_history_months": history_window,
        "seed": 1,
    }
    if metadata:
        # Track/cards and payload-generation provenance are envelope fields, not
        # runtime metadata.  In particular, do not let legacy version markers
        # or a second unvalidated card list reach the provider payload.
        metadata_out.update(
            {
                str(k): _json_safe(v)
                for k, v in metadata.items()
                if str(k)
                not in {
                    "track",
                    "factor_cards",
                    "semantic_cards",
                    "schema_version",
                    "prompt_semantic_version",
                    "semantic_card_schema_version",
                }
            }
        )
    factor_ids: list[str] = []
    for frame in (train, validation):
        for value in frame["id"].tolist():
            text = str(value)
            if text and text not in factor_ids:
                factor_ids.append(text)
    # First-appearance order already follows the numeric masked order because
    # ``split_supervised_rows`` sorts rows that way, but sort defensively so a
    # caller that hands over pre-sorted frames cannot flip the card order back
    # to lexicographic (S100 before S11) on panels with 100+ factors.
    factor_ids.sort(key=_masked_id_sort_key)
    public_cards = [dict(card) for card in (factor_cards or []) if isinstance(card, Mapping)]
    if track == "semantic":
        # Train/validation rows are the source of truth for the public factor
        # order.  Selection plans may retain a seeded sampling order, so do not
        # require callers to pre-sort the semantic sidecar.  Stable sorting
        # preserves duplicate or unknown cards at the end, where the validator
        # can fail closed instead of silently dropping them.
        order = {factor_id: index for index, factor_id in enumerate(factor_ids)}
        public_cards = [
            semantic_card_payload(card, str(card.get("factor_id", "")))
            for card in public_cards
        ]
        public_cards.sort(
            key=lambda card: order.get(str(card.get("factor_id", "")), len(order))
        )
    if track == "numeric_only":
        # Do not silently accept a caller-provided semantic sidecar on the
        # anonymous track.  Failing closed makes accidental cross-track
        # leakage visible at build time instead of quietly changing the
        # requested information set.
        if public_cards:
            raise ValueError("numeric_only payload must not contain factor_cards")
        public_cards = []
    elif not public_cards:
        raise ValueError("semantic track requires factor_cards")
    card_errors = validate_semantic_cards(public_cards, factor_ids, track=track)
    if card_errors:
        raise ValueError("invalid semantic factor cards: " + "; ".join(card_errors[:10]))
    payload: dict[str, Any] = {
        "case_id": str(case_id),
        "task": "factor_return_combination",
        "task_variant": FROZEN_TASK_VARIANT,
        "horizon_months": 1,
        "track": track,
        "feature_columns": fields,
        "factor_cards": public_cards,
        "train": {
            "rows": rows(train),
        },
        "validation": {
            "rows": rows(validation),
        },
        "predict": {
            "schema": ["id", "time_key", *fields],
            "label_included": False,
            "rows_visible_to_llm": False,
        },
        "metadata": metadata_out,
    }
    errors = validate_frozen_payload(payload)
    if errors:
        raise ValueError("invalid frozen code payload: " + "; ".join(errors[:10]))
    return payload


def materialize_payload_split(payload: Mapping[str, Any], split: str) -> pd.DataFrame:
    if split not in {"train", "validation"}:
        raise ValueError("split must be train or validation")
    section = payload.get(split)
    if not isinstance(section, Mapping) or not isinstance(section.get("rows"), list):
        raise ValueError(f"payload.{split}.rows is missing")
    frame = pd.DataFrame([dict(row) for row in section["rows"] if isinstance(row, Mapping)])
    if frame.empty:
        raise ValueError(f"payload.{split}.rows is empty")
    features = [str(feature) for feature in payload.get("feature_columns", [])]
    expected_columns = ["id", "time_key", *features, "label"]
    if list(map(str, frame.columns)) != expected_columns:
        raise ValueError(
            f"payload.{split}.rows columns must be exactly {expected_columns}; "
            f"got {list(map(str, frame.columns))}"
        )
    return frame.reset_index(drop=True)


def stage_public_artifacts(
    payload: MutableMapping[str, Any],
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    *,
    directory: str | Path,
    feature_columns: Sequence[str] | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Write the public history as Parquet and declare it in the payload.

    The prompt carries a size-capped copy of ``train``/``validation`` rows, so a
    candidate that wants to inspect the panel row by row cannot use the prompt
    for it.  Staging the same public rows as Parquet and advertising them through
    ``train.artifact``/``validation.artifact`` lets the evaluator-side
    ``python_exec`` mount read the real data while the provider request only
    grows by a small public descriptor.  Only public train/validation rows are
    written; no predict row, label of a future month, or truth field is staged.

    Returns the ``split -> relative path`` mount specification and the staged
    descriptor map.
    """

    fields = [str(name) for name in (feature_columns or [])] or [
        str(column)
        for column in train_df.columns
        if str(column) not in {"id", "time_key", "label"}
    ]
    columns = ["id", "time_key", *fields, "label"]
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    mounts: dict[str, str] = {}
    descriptors: dict[str, Any] = {}
    for split, frame in (("train", train_df), ("validation", validation_df)):
        missing = sorted(set(columns) - set(str(column) for column in frame.columns))
        if missing:
            raise ValueError(f"{split} frame missing columns for public artifact: {missing}")
        path = directory / f"{split}.parquet"
        frame[columns].to_parquet(path, index=False)
        mounts[split] = path.name
        descriptor = {
            "path": f"public/{split}/{path.name}",
            "format": "parquet",
            "rows": int(len(frame)),
            "columns": list(columns),
            "bytes": int(path.stat().st_size),
        }
        section = payload.get(split)
        if not isinstance(section, dict):
            raise ValueError(f"payload.{split} must be a mutable object to declare an artifact")
        section["artifact"] = descriptor
        descriptors[split] = descriptor
    return mounts, descriptors


def _load_task_c_runner_module() -> Any:
    """Load the standalone Task C subprocess runner without package conflicts."""

    alias = "_task_c_code_runner_for_task_b"
    cached = sys.modules.get(alias)
    if cached is not None:
        return cached
    path = Path(__file__).resolve().parents[2].parent / "task_c" / "code" / "factoropt" / "task_c_code_runner.py"
    # ``parents[2]`` is the task_b directory; its parent is the repository
    # root.  Resolve the standalone Task C runner from there so this module
    # does not depend on the active ``factoropt`` package name.
    if not path.is_file():
        raise ImportError(f"Task C code runner not found: {path}")
    # Give the standalone contract a package search path because it imports
    # Task C's local semantic-card module relatively.  This remains isolated
    # from whichever ``factoropt`` package is active for Task B.
    spec = importlib.util.spec_from_file_location(
        alias,
        path,
        submodule_search_locations=[str(path.parent)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load Task C code runner: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def _load_task_c_contract_module() -> Any:
    alias = "_task_c_contract_for_task_b"
    cached = sys.modules.get(alias)
    if cached is not None:
        return cached
    path = Path(__file__).resolve().parents[2].parent / "task_c" / "code" / "factoropt" / "task_c.py"
    if not path.is_file():
        raise ImportError(f"Task C contract module not found: {path}")
    spec = importlib.util.spec_from_file_location(
        alias,
        path,
        submodule_search_locations=[str(path.parent)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load Task C contract module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def validate_code_response(
    response: Any,
    *,
    expected_revision: int | None = None,
    payload: Mapping[str, Any] | None = None,
) -> list[str]:
    """Use the shared Task C code-response contract plus Task B semantics."""

    contract = _load_task_c_contract_module()
    errors = list(
        contract.validate_code_response(
            response,
            expected_revision=expected_revision,
            payload=payload,
        )
    )
    if isinstance(response, Mapping):
        # Task B's source must return factor IDs/scores through the generic
        # interface; the exact row-level output is checked by run_candidate.
        requirements = response.get("requirements")
        if isinstance(requirements, list) and any("task_c" in str(item).lower() for item in requirements):
            errors.append("requirements must not depend on task_c internals")
    return list(dict.fromkeys(str(error) for error in errors))


def run_candidate(*args: Any, **kwargs: Any) -> Any:
    return _load_task_c_runner_module().run_candidate(*args, **kwargs)


def _unwrap_response(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        selected = value.get("selected_response")
        if isinstance(selected, Mapping):
            return selected
        parsed = value.get("parsed_response")
        if isinstance(parsed, Mapping):
            return parsed
        response = value.get("response")
        if isinstance(response, Mapping):
            return response
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value.strip())
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, Mapping) else None
    return None


def _rank_corr(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 2 or len(right) != len(left):
        return None
    x = pd.Series(left, dtype=float).rank(method="average").to_numpy()
    y = pd.Series(right, dtype=float).rank(method="average").to_numpy()
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return None
    return _finite(np.corrcoef(x, y)[0, 1])


def evaluate_code_on_validation(
    response: Mapping[str, Any],
    *,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    metadata: Mapping[str, Any] | None = None,
    # Per-month wall clock; see FrozenRevisionConfig.
    timeout_seconds: float = 1_800.0,
    scratch_root: str | Path | None = None,
) -> tuple[dict[str, Any], list[Any]]:
    """Execute a candidate on each validation cutoff and aggregate metrics.

    Validation is itself evaluated as a chronological walk-forward stream.
    For validation month ``j`` the candidate receives the static training
    block plus only validation labels from months ``< j``; the current
    month's labels are retained by this evaluator solely for scoring.  This
    mirrors the information boundary used by the Task C rolling contract and
    prevents a candidate from copying the target through ``valid_df``.
    """

    months = sorted(
        {str(value) for value in validation_df["time_key"].dropna()},
        key=lambda value: -int(value.split("-", 1)[1]),
    )
    executions: list[Any] = []
    predictions: list[pd.DataFrame] = []
    ics: list[float] = []
    errors: list[str] = []
    for month_index, month in enumerate(months):
        target_rows = validation_df.loc[validation_df["time_key"].astype(str).eq(month)].copy()
        predict = target_rows.drop(columns=["label"], errors="ignore").copy()
        # At the first validation origin no validation target has settled
        # yet.  As each month is evaluated, its realised label becomes
        # available to the next origin, exactly as it does in the hidden
        # rolling/test stream.
        prior_months = set(months[:month_index])
        fit_validation = validation_df.loc[
            validation_df["time_key"].astype(str).isin(prior_months)
        ].copy()
        run_metadata = dict(metadata or {})
        # Keep the low-level evaluator fail-closed as well as the public
        # payload validator: arbitrary caller metadata must not smuggle a
        # semantic sidecar into a numeric-only execution.
        try:
            run_track = normalize_track(run_metadata.get("track", "numeric_only"))
        except ValueError:
            run_track = "numeric_only"
        supplied_cards = run_metadata.get("factor_cards")
        run_metadata.pop("factor_cards", None)
        run_metadata.pop("semantic_cards", None)
        run_metadata["track"] = run_track
        if run_track == "semantic" and isinstance(supplied_cards, list):
            run_metadata["factor_cards"] = [
                dict(card) for card in supplied_cards if isinstance(card, Mapping)
            ]
        run_metadata.update(
            {
                "seed": int(run_metadata.get("seed", 1)),
                "evaluation_month": month,
                "validation_month_index": int(month_index),
                "validation_protocol": "chronological_leave_future_out",
                "validation_rows_available": int(len(fit_validation)),
                "task": "factor_return_combination",
            }
        )
        result = run_candidate(
            response,
            train_df,
            fit_validation,
            predict,
            run_metadata,
            timeout_seconds=timeout_seconds,
            scratch_root=scratch_root,
        )
        executions.append(result)
        if not result.valid or result.prediction is None:
            errors.extend(result.errors or [result.status])
            continue
        pred = result.prediction.copy()
        pred["time_key"] = month
        predictions.append(pred)
        labels = target_rows[["id", "label"]].copy()
        labels["id"] = labels["id"].astype(str)
        pred["id"] = pred["id"].astype(str)
        merged = pred.merge(labels, on="id", how="inner", validate="one_to_one")
        merged["score"] = pd.to_numeric(merged["score"], errors="coerce")
        merged["label"] = pd.to_numeric(merged["label"], errors="coerce")
        merged = merged.replace([np.inf, -np.inf], np.nan).dropna(subset=["score", "label"])
        if len(merged) >= 2:
            ic = _rank_corr(merged["score"].tolist(), merged["label"].tolist())
            if ic is not None:
                ics.append(ic)
    combined = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(columns=["id", "score", "time_key"])
    if not combined.empty:
        label_frame = validation_df[["id", "time_key", "label"]].copy()
        label_frame["id"] = label_frame["id"].astype(str)
        label_frame["time_key"] = label_frame["time_key"].astype(str)
        joined = combined.merge(label_frame, on=["id", "time_key"], how="inner", validate="one_to_one")
        score_values = pd.to_numeric(joined["score"], errors="coerce")
        label_values = pd.to_numeric(joined["label"], errors="coerce")
        finite = np.isfinite(score_values.to_numpy(dtype=float)) & np.isfinite(label_values.to_numpy(dtype=float))
        score_values = score_values.to_numpy(dtype=float)[finite]
        label_values = label_values.to_numpy(dtype=float)[finite]
        mse = float(np.mean(np.square(score_values - label_values))) if len(score_values) else None
        directional = float(np.mean(np.sign(score_values) == np.sign(label_values))) if len(score_values) else None
    else:
        mse = None
        directional = None
    metrics = {
        "rank_ic_mean": float(np.mean(ics)) if ics else None,
        "rank_ic_std": float(np.std(ics, ddof=1)) if len(ics) > 1 else 0.0 if ics else None,
        "mse": mse,
        "directional_accuracy": directional,
        "n_months": int(len(months)),
        "n_rows": int(len(combined)),
        "execution_valid": bool(months) and len(executions) == len(months) and all(item.valid for item in executions),
        "error_count": int(len(errors)),
        "errors": errors[:20],
        "runtime_seconds": float(sum(item.runtime_seconds or 0.0 for item in executions)),
    }
    return metrics, executions


@dataclass(frozen=True)
class FrozenRevisionConfig:
    """Pre-test code-generation budget; default is exactly one LLM revision."""

    max_revisions: int = 0
    # Per-month wall clock for the frozen candidate.  A deep-learning candidate
    # needs a real budget; the value still bounds a runaway fit.
    timeout_seconds: float = 1_800.0
    objective: str = "rank_ic_mean"
    selection_policy: str = "llm"
    stop_on_final: bool = True

    def validate(self) -> None:
        if self.max_revisions < 0 or self.max_revisions > 20:
            raise ValueError("max_revisions must be between 0 and 20")
        if self.timeout_seconds <= 0 or not math.isfinite(float(self.timeout_seconds)):
            raise ValueError("timeout_seconds must be positive and finite")
        if self.objective not in {"rank_ic_mean", "mse", "directional_accuracy"}:
            raise ValueError("objective must be rank_ic_mean, mse, or directional_accuracy")
        if self.selection_policy not in {"llm", "host_metric"}:
            raise ValueError("selection_policy must be llm or host_metric")


def _objective_key(metrics: Mapping[str, Any], objective: str, code_chars: int) -> tuple[float, float, float]:
    value = metrics.get(objective)
    number = float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else -math.inf
    if objective == "mse" and math.isfinite(number):
        number = -number
    mse = metrics.get("mse")
    mse_value = float(mse) if isinstance(mse, (int, float)) and math.isfinite(float(mse)) else math.inf
    return number, -mse_value, -float(code_chars)


_PUBLIC_SELECTION_METRICS = (
    "rank_ic_mean",
    "rank_ic_std",
    "mse",
    "directional_accuracy",
    "n_months",
    "n_rows",
)


def _candidate_selection_card(
    response: Mapping[str, Any],
    metrics: Mapping[str, Any],
    revision: int,
) -> dict[str, Any]:
    return {
        "revision": int(revision),
        "code_hash": code_hash(response),
        "code_chars": int(len(str(response.get("code") or ""))),
        "revision_summary": str(response.get("revision_summary") or "")[:1000],
        "validation": {
            key: _json_safe(metrics.get(key))
            for key in _PUBLIC_SELECTION_METRICS
            if key in metrics
        },
    }


def validate_selection_response(response: Any, *, allowed_revisions: Sequence[int]) -> list[str]:
    allowed = {int(value) for value in allowed_revisions}
    if not isinstance(response, Mapping):
        return ["selection response must be a JSON object"]
    errors: list[str] = []
    extra = sorted(str(key) for key in response if key not in {"selected_revision", "selection_reason"})
    if extra:
        errors.append("selection response has unexpected fields: " + ", ".join(extra[:8]))
    raw = response.get("selected_revision")
    if not isinstance(raw, int) or isinstance(raw, bool):
        errors.append("selected_revision must be an integer")
    elif int(raw) not in allowed:
        errors.append(f"selected_revision must be one of {sorted(allowed)}")
    reason = response.get("selection_reason")
    if not isinstance(reason, str) or not reason.strip():
        errors.append("selection_reason must be a non-empty string")
    elif len(reason) > 2000:
        errors.append("selection_reason exceeds 2000 characters")
    return errors


def _parse_selection_response(
    value: Any, *, allowed_revisions: Sequence[int]
) -> tuple[int | None, str, list[str]]:
    errors = validate_selection_response(value, allowed_revisions=allowed_revisions)
    if errors or not isinstance(value, Mapping):
        return None, "", errors
    return int(value["selected_revision"]), str(value.get("selection_reason") or "").strip(), []


def _choose_host_metric_candidate(
    candidates: Sequence[tuple[Mapping[str, Any], dict[str, Any], int]],
    objective: str,
) -> tuple[Mapping[str, Any], dict[str, Any], int] | None:
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: _objective_key(item[1], objective, len(str(item[0].get("code") or ""))),
    )


def _select_frozen_candidate(
    candidates: Sequence[tuple[Mapping[str, Any], dict[str, Any], int]],
    *,
    settings: FrozenRevisionConfig,
    llm_call: Callable[[Mapping[str, Any], Mapping[str, Any] | None, int], Any],
    public_payload: Mapping[str, Any],
) -> dict[str, Any]:
    host_choice = _choose_host_metric_candidate(candidates, settings.objective)
    result: dict[str, Any] = {
        "selection_policy": settings.selection_policy,
        "host_metric_objective": settings.objective,
        "host_metric_selected_revision": None if host_choice is None else int(host_choice[2]),
        "llm_selected_revision": None,
        "selection_fallback": False,
        "selection_errors": [],
        "selection_reason": "",
        "selected_response": None,
        "selected_metrics": {},
        "selected_revision": None,
    }
    if not candidates:
        result["selection_reason"] = "no candidate passed schema and execution checks"
        return result
    if settings.selection_policy == "host_metric" or len(candidates) == 1:
        chosen = host_choice if settings.selection_policy == "host_metric" else candidates[0]
        assert chosen is not None
        result["selected_response"] = dict(chosen[0])
        result["selected_metrics"] = dict(chosen[1])
        result["selected_revision"] = int(chosen[2])
        result["selection_reason"] = (
            f"max validation {settings.objective}; deterministic MSE/complexity tie-break"
            if settings.selection_policy == "host_metric"
            else "single_valid_candidate"
        )
        return result
    envelope = {
        "selection_request": True,
        "allowed_revisions": [int(item[2]) for item in candidates],
        "candidates": [_candidate_selection_card(*item) for item in candidates],
        "instruction": (
            "Choose exactly one allowed revision to freeze. Do not emit new source. "
            "Validation numbers are diagnostics, not a required ranking rule."
        ),
    }
    try:
        raw = llm_call(public_payload, envelope, int(settings.max_revisions) + 1)
        parsed = _unwrap_response(raw)
        picked, reason, errors = _parse_selection_response(
            parsed, allowed_revisions=[int(item[2]) for item in candidates]
        )
    except Exception as exc:
        picked, reason, errors = None, "", [f"{type(exc).__name__}: {exc}"]
    if picked is None:
        chosen = candidates[-1]
        result["selection_fallback"] = True
        result["selection_errors"] = errors[:20]
        result["selection_reason"] = "fallback_last_valid" + (f": {errors[0]}" if errors else "")
    else:
        chosen = next(item for item in candidates if int(item[2]) == picked)
        result["llm_selected_revision"] = picked
        result["selection_reason"] = reason or "llm_choice"
    result["selected_response"] = dict(chosen[0])
    result["selected_metrics"] = dict(chosen[1])
    result["selected_revision"] = int(chosen[2])
    return result


def _public_candidate_for_feedback(response: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Keep the previous generated source available to the next revision.

    Revision calls are intentionally stateless provider requests.  Sending
    only aggregate validation errors made a model regenerate the envelope
    from scratch and frequently lose already-correct fields.  The previous
    candidate is public (it contains no evaluator truth), bounded by the same
    source-size limit as the response contract, and is included only as
    repair context—not as a test signal.
    """

    if not isinstance(response, Mapping):
        return None
    candidate: dict[str, Any] = {}
    for key in (
        "revision",
        "status",
        "language",
        "entrypoint",
        "requirements",
        "input_contract",
        "output_contract",
        "explanation",
        "revision_summary",
    ):
        if key in response:
            candidate[key] = _json_safe(response.get(key))
    if "code" in response:
        code = str(response.get("code") or "")
        candidate["code"] = code[:MAX_CODE_CHARS]
    return candidate


def frozen_code_revision_loop(
    public_payload: Mapping[str, Any],
    llm_call: Callable[[Mapping[str, Any], Mapping[str, Any] | None, int], Any],
    *,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    config: FrozenRevisionConfig | None = None,
    scratch_root: str | Path | None = None,
) -> dict[str, Any]:
    """Generate/validate code before test and freeze one candidate."""

    settings = config or FrozenRevisionConfig()
    settings.validate()
    payload_errors = validate_frozen_payload(public_payload)
    if payload_errors:
        raise ValueError("invalid public payload: " + "; ".join(payload_errors[:10]))
    revisions: list[dict[str, Any]] = []
    candidates: list[tuple[Mapping[str, Any], dict[str, Any], int]] = []
    feedback: Mapping[str, Any] | None = None
    for revision in range(settings.max_revisions + 1):
        started = time.perf_counter()
        response: Mapping[str, Any] | None = None
        errors: list[str] = []
        metrics: dict[str, Any] = {}
        executions: list[Any] = []
        try:
            response = _unwrap_response(llm_call(public_payload, feedback, revision))
            if response is None:
                errors.append("LLM callback did not return a JSON object")
            else:
                errors.extend(
                    validate_code_response(
                        response,
                        expected_revision=revision,
                        payload=public_payload,
                    )
                )
                if not errors:
                    runtime_metadata = (
                        dict(public_payload.get("metadata") or {})
                        if isinstance(public_payload.get("metadata"), Mapping)
                        else {}
                    )
                    if normalize_track(public_payload.get("track", "numeric_only")) == "semantic":
                        runtime_metadata["track"] = "semantic"
                        runtime_metadata["factor_cards"] = [
                            dict(card)
                            for card in public_payload.get("factor_cards", [])
                            if isinstance(card, Mapping)
                        ]
                    metrics, executions = evaluate_code_on_validation(
                        response,
                        train_df=train_df,
                        validation_df=validation_df,
                        metadata=runtime_metadata,
                        timeout_seconds=settings.timeout_seconds,
                        scratch_root=scratch_root,
                    )
                    errors.extend(str(item) for item in metrics.get("errors", []))
                    if not metrics.get("execution_valid"):
                        errors.append("candidate execution was not valid for every validation month")
        except Exception as exc:  # keep failed revisions auditable
            errors.append(f"{type(exc).__name__}: {exc}")
        errors = list(dict.fromkeys(errors))
        valid = response is not None and not errors
        if response is not None:
            metrics["code_chars"] = len(str(response.get("code") or ""))
        if valid and response is not None:
            candidates.append((response, metrics, revision))
        elapsed = round(time.perf_counter() - started, 6)
        revisions.append(
            {
                "round": revision,
                "code_hash": code_hash(response) if response is not None else None,
                "schema_valid": response is not None
                and not validate_code_response(
                    response,
                    expected_revision=revision,
                    payload=public_payload,
                ),
                "execution_valid": bool(valid),
                "validation": {key: value for key, value in metrics.items() if key != "errors"},
                "errors": errors[:20],
                "runtime_seconds": elapsed,
                "execution": [item.audit_dict() for item in executions],
                "revision_summary": str(response.get("revision_summary", ""))[:1000] if response else "",
            }
        )
        feedback = {
            "revision": revision + 1,
            "candidate_status": "valid" if valid else "invalid",
            "validation": {
                key: _json_safe(metrics.get(key))
                for key in ("rank_ic_mean", "rank_ic_std", "mse", "directional_accuracy", "n_months", "n_rows")
                if key in metrics
            },
            "io_checks": {"execution_valid": bool(valid)},
            "runtime": {"seconds": elapsed},
            "errors": errors[:20],
            "allowed_next_revision": revision < settings.max_revisions,
        }
        previous_candidate = _public_candidate_for_feedback(response)
        if previous_candidate is not None:
            feedback["previous_candidate"] = previous_candidate
        if valid and response is not None and response.get("status") == "final" and settings.stop_on_final:
            break
    choice = _select_frozen_candidate(
        candidates,
        settings=settings,
        llm_call=llm_call,
        public_payload=public_payload,
    )
    selected_response = choice["selected_response"]
    return {
        "record_type": "task_b_frozen_code_revision_audit",
        "schema_version": FROZEN_CODE_SCHEMA_VERSION,
        "case_id": public_payload.get("case_id"),
        "payload_hash": stable_hash(public_payload),
        "prompt_semantic_version": f"{FROZEN_PROMPT_VERSION}_{normalize_track(public_payload.get('track', 'numeric_only'))}",
        "max_revisions": settings.max_revisions,
        "code_revisions": revisions,
        "selected_revision": choice["selected_revision"],
        "selected_code_hash": code_hash(selected_response) if selected_response else None,
        "selected_response": dict(selected_response) if selected_response else None,
        "selected_validation": choice["selected_metrics"],
        "validation_protocol": "chronological_leave_future_out",
        "selection_policy": choice["selection_policy"],
        "selection_objective": (
            settings.objective if settings.selection_policy == "host_metric" else "llm_choice"
        ),
        "selection_reason": choice["selection_reason"],
        "selection_fallback": choice["selection_fallback"],
        "selection_errors": choice["selection_errors"],
        "llm_selected_revision": choice["llm_selected_revision"],
        "host_metric_objective": choice["host_metric_objective"],
        "host_metric_selected_revision": choice["host_metric_selected_revision"],
        "test_locked": bool(selected_response),
        "test_llm_calls_after_freeze": 0,
    }


def _prediction_rows_from_scores(ids: Sequence[str], scores: Mapping[str, Any]) -> list[dict[str, Any]]:
    clean = {str(key): _finite(value) for key, value in scores.items()}
    valid = [float(value) for value in clean.values() if value is not None]
    mean = float(np.mean(valid)) if valid else 0.0
    std = float(np.std(valid)) if len(valid) > 1 else 0.0
    zscores = {
        key: ((float(value) - mean) / std if value is not None and std > 1e-12 else 0.0)
        for key, value in clean.items()
    }
    order = sorted((str(key) for key in ids), key=lambda key: (-zscores.get(key, 0.0), key))
    ranks = {key: index + 1 for index, key in enumerate(order)}
    rows: list[dict[str, Any]] = []
    for key in ids:
        text = str(key)
        z = zscores.get(text, 0.0)
        rows.append(
            {
                "factor_id": text,
                "probability_positive": float(1.0 / (1.0 + math.exp(-float(np.clip(z, -30.0, 30.0))))),
                "expected_return_z": float(z),
                "rank": int(ranks[text]),
                "confidence": float(min(0.99, 0.5 + 0.15 * abs(float(z)))),
                "abstain": clean.get(text) is None,
            }
        )
    return rows


def _target_masked(episode: RouterEpisode, masked_to_original: Mapping[str, str]) -> pd.Series:
    target = episode.target
    return pd.Series(
        {str(masked): _finite(target.get(str(original))) for masked, original in masked_to_original.items()},
        dtype=float,
    )


def run_frozen_rolling(
    frozen_response: Mapping[str, Any],
    episodes: Sequence[RouterEpisode],
    masked_maps: Sequence[Mapping[str, str]],
    *,
    feature_columns: Sequence[str] = HISTORICAL_PANEL_FEATURE_FIELDS,
    train_cutoffs: int = DEFAULT_TRAIN_CUTOFFS,
    validation_cutoffs: int = DEFAULT_VALIDATION_CUTOFFS,
    min_history_months: int = DEFAULT_MIN_HISTORY_MONTHS,
    # Per-episode wall clock for the frozen candidate (see FrozenRevisionConfig).
    timeout_seconds: float = 1_800.0,
    scratch_root: str | Path | None = None,
    costs_bps: Sequence[float] = (0.0, 10.0, 25.0),
    track: str = "numeric_only",
    factor_cards: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Execute one frozen source on a sequence of one-month episodes."""

    track = normalize_track(track)
    response = _unwrap_response(frozen_response)
    if response is None:
        raise ValueError("frozen response is not a JSON object")
    if train_cutoffs < 1 or validation_cutoffs < 1 or min_history_months < 1:
        raise ValueError("train_cutoffs, validation_cutoffs, and min_history_months must be positive")
    if timeout_seconds <= 0 or not math.isfinite(float(timeout_seconds)):
        raise ValueError("timeout_seconds must be positive and finite")
    errors = validate_code_response(response)
    if errors:
        raise ValueError("invalid frozen response: " + "; ".join(errors[:10]))
    if len(episodes) != len(masked_maps) or not episodes:
        raise ValueError("episodes and masked_maps must be non-empty and aligned")
    first_originals = [str(value) for value in episodes[0].sleeve_ids]
    first_masked = {str(key): str(value) for key, value in masked_maps[0].items()}
    if set(first_masked.values()) != set(first_originals):
        raise ValueError("first masked map does not cover the episode factor universe")
    public_cards = [dict(card) for card in (factor_cards or [])]
    masked_by_original = {original: masked for masked, original in first_masked.items()}
    expected_masked_ids = [masked_by_original[original] for original in first_originals]
    card_errors = validate_semantic_cards(
        public_cards,
        expected_masked_ids,
        track=track,
    )
    if card_errors:
        raise ValueError("invalid rolling semantic cards: " + "; ".join(card_errors[:10]))
    for step, (episode, mapping_value) in enumerate(zip(episodes, masked_maps), start=1):
        mapping_check = {str(k): str(v) for k, v in mapping_value.items()}
        if set(mapping_check.values()) != set(map(str, episode.sleeve_ids)):
            raise ValueError(f"masked map does not cover factor universe at step {step}")
        if set(mapping_check) != set(first_masked) or any(
            mapping_check.get(masked) != original for masked, original in first_masked.items()
        ):
            raise ValueError(f"masked ID mapping changed at rolling step {step}")
    for step, (previous, current) in enumerate(zip(episodes, episodes[1:]), start=2):
        if list(map(str, previous.history.columns)) != list(map(str, current.history.columns)):
            raise ValueError(f"factor universe changed at rolling step {step}")
        if list(previous.history.index[1:]) != list(current.history.index[:-1]):
            raise ValueError(f"history did not roll by one month at rolling step {step}")
        if pd.Timestamp(previous.future.index[0]) != pd.Timestamp(current.history.index[-1]):
            raise ValueError(f"previous target was not appended at rolling step {step}")
    expected_hash = code_hash(response)
    prediction_frames: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    gross_returns: list[float] = []
    turnovers: list[float] = []
    previous_weights: pd.Series | None = None
    # Consecutive origins share almost all historical prefixes.  Reusing the
    # deterministic compact feature tables keeps a 12-month/153-factor run
    # practical while preserving the exact same point-in-time calculations.
    feature_cache: dict[tuple[Any, ...], pd.DataFrame] = {}
    for index, (episode, mapping_value) in enumerate(zip(episodes, masked_maps)):
        mapping = {str(k): str(v) for k, v in mapping_value.items()}
        started = time.perf_counter()
        audit: dict[str, Any] = {
            "step": index + 1,
            "formation_eom": pd.Timestamp(episode.anchor_date).strftime("%Y-%m-%d"),
            "holding_month": (pd.Timestamp(episode.anchor_date) + pd.offsets.MonthEnd(1)).strftime("%Y-%m"),
            "code_hash": expected_hash,
            "llm_calls": 0,
            "fit_mode": "from_scratch",
            "window_policy": "rolling_train_then_validation_refit_contract",
            "refit_policy": "validation_select_then_train_plus_validation",
            "label_cutoff_policy": "cutoff_u_features_through_u_label_is_return_u_plus_1",
            "history_months": int(len(episode.history.index)),
            "forecast_horizon_months": int(episode.horizon_months),
        }
        try:
            if episode.horizon_months != 1:
                raise ValueError("frozen rolling protocol requires horizon=1 episodes")
            train, valid, predict, split_meta = build_rolling_windows(
                episode.history,
                original_to_masked={original: masked for masked, original in mapping.items()},
                feature_columns=feature_columns,
                train_cutoffs=train_cutoffs,
                validation_cutoffs=validation_cutoffs,
                min_history_months=min_history_months,
                feature_cache=feature_cache,
            )
            runtime_metadata: dict[str, Any] = {
                "seed": 1,
                "id_column": "id",
                "time_column": "time_key",
                "target_is_factor_excess_return": True,
                "future_labels_included": False,
                "task": "factor_return_combination",
                "formation_eom": pd.Timestamp(episode.anchor_date).strftime("%Y-%m-%d"),
                "history_months": len(episode.history.index),
                "forecast_horizon_months": 1,
                "fit_mode": "from_scratch",
                "window_policy": "rolling_train_then_validation_refit_contract",
                "refit_policy": "validation_select_then_train_plus_validation",
                "label_cutoff_policy": "cutoff_u_features_through_u_label_is_return_u_plus_1",
                "track": track,
                **split_meta,
            }
            if track == "semantic":
                runtime_metadata["factor_cards"] = public_cards
            result = run_candidate(
                response,
                train,
                valid,
                predict,
                runtime_metadata,
                timeout_seconds=timeout_seconds,
                scratch_root=scratch_root,
            )
            audit.update(
                {
                    "execution_status": result.status,
                    "execution_valid": bool(result.valid),
                    "runtime_seconds": result.runtime_seconds,
                    "errors": list(result.errors),
                    "train_rows": int(len(train)),
                    "validation_rows": int(len(valid)),
                    "predict_rows": int(len(predict)),
                    "train_time_range": split_meta.get("train_time_range"),
                    "validation_time_range": split_meta.get("validation_time_range"),
                    "train_label_time_range": split_meta.get("train_label_time_range"),
                    "validation_label_time_range": split_meta.get("validation_label_time_range"),
                    "train_label_calendar_range": split_meta.get("train_label_calendar_range"),
                    "validation_label_calendar_range": split_meta.get("validation_label_calendar_range"),
                    "predict_calendar_month": split_meta.get("predict_calendar_month"),
                    "train_cutoffs_used": split_meta.get("train_cutoffs_used"),
                    "validation_cutoffs_used": split_meta.get("validation_cutoffs_used"),
                    "rolling_window_cutoffs": int(train_cutoffs + validation_cutoffs),
                }
            )
            if not result.valid or result.prediction is None:
                audit["prediction_rows_out"] = 0
                audit_rows.append(audit)
                continue
            pred = result.prediction.copy()
            pred["id"] = pred["id"].astype(str)
            scores = {str(row["id"]): float(row["score"]) for row in pred.to_dict("records")}
            prediction_rows = _prediction_rows_from_scores(list(predict["id"].astype(str)), scores)
            target_masked = _target_masked(episode, mapping)
            score_response = {"predictions": prediction_rows}
            metrics = score_predictions(score_response, target_masked)
            weights = fixed_portfolio_weights(
                episode.history.rename(columns={original: masked for masked, original in mapping.items()}),
                prediction_rows,
            )
            target_original = episode.target
            weight_original = pd.Series(
                {mapping.get(str(key), str(key)): float(value) for key, value in weights.items()},
                dtype=float,
            )
            aligned = pd.concat([weight_original.rename("weight"), target_original.rename("return")], axis=1).dropna()
            gross = None
            if not aligned.empty and float(aligned["weight"].sum()) > 1e-12:
                normal = aligned["weight"] / aligned["weight"].sum()
                gross = _finite(float((normal * aligned["return"]).sum()))
            current_turnover = turnover(weights, previous_weights)
            previous_weights = weights
            if gross is not None:
                gross_returns.append(gross)
                turnovers.append(current_turnover)
            output = pred[["id", "score"]].copy()
            output.insert(0, "formation_eom", pd.Timestamp(episode.anchor_date))
            output.insert(1, "holding_month", pd.Timestamp(episode.anchor_date) + pd.offsets.MonthEnd(1))
            output["code_hash"] = expected_hash
            output["execution_status"] = "success"
            prediction_frames.append(output)
            audit.update(
                {
                    "prediction_rows_out": int(len(output)),
                    "prediction_metrics": metrics,
                    "gross_return": gross,
                    "turnover": current_turnover,
                    "weights": {str(k): _finite(v) for k, v in weights.items()},
                }
            )
        except Exception as exc:  # keep one bad month auditable
            audit.update(
                {
                    "execution_status": "window_error",
                    "execution_valid": False,
                    "errors": [f"{type(exc).__name__}: {exc}"],
                    "prediction_rows_out": 0,
                }
            )
        audit["elapsed_seconds"] = round(time.perf_counter() - started, 6)
        audit_rows.append(audit)
    predictions = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else pd.DataFrame(columns=["formation_eom", "holding_month", "id", "score", "code_hash", "execution_status"])
    )
    portfolio = evaluate_portfolio(gross_returns, turnovers, costs_bps=costs_bps, periods_per_year=12.0)
    audit_frame = pd.DataFrame(audit_rows)

    def metric_mean(name: str) -> float | None:
        values = [
            _finite(item.get(name))
            for row in audit_rows
            for item in [row.get("prediction_metrics")]
            if isinstance(item, Mapping)
        ]
        clean = [value for value in values if value is not None]
        return float(np.mean(clean)) if clean else None

    summary = {
        "protocol": FROZEN_TASK_VARIANT,
        "track": track,
        "semantic_card_count": len(public_cards),
        "code_hash": expected_hash,
        "episodes_requested": int(len(episodes)),
        "episodes_valid": int(sum(bool(row.get("execution_valid")) for row in audit_rows)),
        "completion_rate": float(sum(bool(row.get("execution_valid")) for row in audit_rows) / len(episodes)),
        "llm_calls_during_rolling": 0,
        "fit_mode": "from_scratch",
        "window_policy": "rolling_train_then_validation_refit_contract",
        "refit_policy": "validation_select_then_train_plus_validation",
        "train_cutoffs_per_origin": int(train_cutoffs),
        "validation_cutoffs_per_origin": int(validation_cutoffs),
        "forecast_horizon_months": 1,
        "prediction_metrics": {
            "rank_ic_mean": metric_mean("rank_ic"),
            "accuracy_mean": metric_mean("accuracy"),
            "top3_hit_rate_mean": metric_mean("top3_hit_rate"),
            "top3_average_realized_return_mean": metric_mean("top3_average_realized_return"),
        },
        "portfolio": portfolio,
    }
    return predictions, audit_frame, summary


__all__ = [
    "DEFAULT_MIN_HISTORY_MONTHS",
    "DEFAULT_TRAIN_CUTOFFS",
    "DEFAULT_VALIDATION_CUTOFFS",
    "FROZEN_CODE_SCHEMA_VERSION",
    "FROZEN_PAYLOAD_SCHEMA_VERSION",
    "FROZEN_PROMPT_VERSION",
    "FROZEN_TASK_VARIANT",
    "FrozenRevisionConfig",
    "build_code_generation_payload",
    "build_rolling_windows",
    "build_predict_rows",
    "build_supervised_rows",
    "code_hash",
    "evaluate_code_on_validation",
    "frozen_code_revision_loop",
    "materialize_payload_split",
    "run_candidate",
    "run_frozen_rolling",
    "split_supervised_rows",
    "validate_code_response",
    "validate_frozen_payload",
    "validate_selection_response",
]

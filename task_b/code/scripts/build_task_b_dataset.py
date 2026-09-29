#!/usr/bin/env python3
"""Build the Task B raw-factor frozen-code rolling benchmark.

The benchmark is intentionally separate from the legacy 13-theme and the
single-anchor Task B datasets.  It creates one frozen-code case for every
``(time block, history window, factor subset)`` and stores a twelve-month
one-step evaluator stream with each case:

* 2Y/3Y/5Y histories (24/36/60 months);
* eight 38-factor subsets, five 76-factor subsets, and one 153-factor census;
* the same subset is reused across the three history windows inside a block;
* temporal blocks are purged so the union of each history and its twelve hidden
  targets is followed by at least twelve untouched months before the next block.

With the frozen JKP 2025-12 file filtered to ``--region USA`` (the live build)
this gives seven blocks and therefore ``7 * 3 * (8 + 5 + 1) = 294`` cases.  The
``--returns`` CSV carries every region, so the region is a filter, not a file
choice.  The LLM sees only ``payload.json``.
``rolling_truth.jsonl`` contains the evaluator-only rolling histories/targets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
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

from factoropt.task_b import (  # noqa: E402
    HISTORICAL_PANEL_FEATURE_FIELDS,
    RouterEpisode,
    build_compact_features,
    episode_to_truth_record,
    load_raw_factor_returns,
)
from factoropt.task_b_frozen import (  # noqa: E402
    DEFAULT_MIN_HISTORY_MONTHS,
    FROZEN_TASK_VARIANT,
    build_code_generation_payload,
    split_supervised_rows,
    stable_hash,
)
from factoropt.semantic import (
    ALL_TRACKS,
    SEMANTIC_CARD_SCHEMA_VERSION,
    normalize_track,
    semantic_cards_payload,
)


WINDOW_MONTHS: dict[str, int] = {"2Y": 24, "3Y": 36, "5Y": 60}
DEFAULT_WINDOWS = ("2Y", "3Y", "5Y")
DEFAULT_SUBSETS_BY_SIZE = {38: 8, 76: 5, 153: 1}
DEFAULT_TARGET_MONTHS = 12
DEFAULT_GAP_MONTHS = 12
DEFAULT_MASTER_SEED = 20260904
EXPECTED_CASE_COUNT = 294
GENERATOR_VERSION = "build_task_b_dataset.v1_294"
MANIFEST_SCHEMA_VERSION = "task_b_frozen_raw_rolling_manifest.v1"
SPLIT_SCHEMA_VERSION = "task_b_frozen_raw_rolling_split.v1"


def _jsonable(value: Any) -> Any:
    """Convert pandas/numpy values and non-finite numbers for JSON."""

    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        stamp = pd.Timestamp(value)
        return None if pd.isna(stamp) else stamp.strftime("%Y-%m-%d")
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


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
            handle.write(
                json.dumps(_jsonable(dict(row)), ensure_ascii=False, allow_nan=False) + "\n"
            )
            count += 1
    return count


def _stable_seed(master_seed: int, *parts: object) -> int:
    material = "|".join([str(master_seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big")


def _load_cards(path: Path | None, factor_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Load private factor metadata used only for reproducible stratification."""

    ids = [str(value) for value in factor_ids]
    cards: dict[str, dict[str, Any]] = {
        factor_id: {
            "factor_id": factor_id,
            # Never fall back to the raw source mnemonic in a public card.
            # ``semantic_card_payload`` will use the masked Sxx id when a
            # metadata record is unavailable.
            "neutral_name": "",
            "neutral_description": "",
            "formula_or_definition": "",
            "input_fields": [],
            "source_paper": "",
            "theme_for_selection": "",
        }
        for factor_id in ids
    }
    if path is None:
        return cards
    if not path.is_file():
        raise FileNotFoundError(f"factor cards file does not exist: {path}")
    if path.suffix.lower() == ".jsonl":
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        rows = list(raw.values()) if isinstance(raw, Mapping) else raw
    else:
        rows = pd.read_csv(path).to_dict("records")
    if not isinstance(rows, list):
        raise ValueError(f"factor cards must contain a list of objects: {path}")
    for raw in rows:
        if not isinstance(raw, Mapping):
            continue
        factor_id = str(raw.get("factor_id", "")).strip()
        if factor_id not in cards:
            continue
        card = cards[factor_id]
        card.update({str(key): _jsonable(value) for key, value in raw.items()})
        # The enriched JKP cards call this field
        # ``jkp_group_for_selection_only``.  Keep the semantic label private
        # and normalize it to the selector's common name.
        if not str(card.get("theme_for_selection", "")).strip():
            card["theme_for_selection"] = str(
                card.get("jkp_group_for_selection_only", "")
                or card.get("jkp_theme_for_evaluation_only", "")
                or ""
            ).strip()
    return cards


def _theme(cards: Mapping[str, Mapping[str, Any]], factor_id: str) -> str:
    value = str(cards.get(str(factor_id), {}).get("theme_for_selection", "")).strip()
    return value or "_unlabelled"


def _allocate_theme_quotas(
    eligible: Sequence[str],
    size: int,
    cards: Mapping[str, Mapping[str, Any]],
    seed: int,
) -> dict[str, int]:
    if size < 1 or size > len(eligible):
        raise ValueError("sample size must be between one and the eligible factor count")
    counts = Counter(_theme(cards, factor_id) for factor_id in eligible)
    themes = sorted(counts)
    raw = {theme: size * counts[theme] / len(eligible) for theme in themes}
    quotas = {theme: min(counts[theme], int(np.floor(raw[theme]))) for theme in themes}
    remainder = size - sum(quotas.values())
    while remainder:
        candidates = [theme for theme in themes if quotas[theme] < counts[theme]]
        if not candidates:
            raise RuntimeError("unable to allocate theme-stratified quotas")
        candidates.sort(
            key=lambda theme: (
                -(raw[theme] - np.floor(raw[theme])),
                _stable_seed(seed, "quota", theme),
            )
        )
        for theme in candidates:
            if quotas[theme] >= counts[theme]:
                continue
            quotas[theme] += 1
            remainder -= 1
            if remainder == 0:
                break
    return quotas


def _sample_subset(
    eligible: Sequence[str],
    size: int,
    cards: Mapping[str, Mapping[str, Any]],
    seed: int,
) -> tuple[list[str], dict[str, int]]:
    eligible = [str(value) for value in eligible]
    quotas = _allocate_theme_quotas(eligible, size, cards, seed)
    by_theme: dict[str, list[str]] = defaultdict(list)
    for factor_id in sorted(eligible):
        by_theme[_theme(cards, factor_id)].append(factor_id)
    selected: list[str] = []
    for theme in sorted(by_theme):
        quota = quotas.get(theme, 0)
        if quota:
            picker = random.Random(_stable_seed(seed, "theme", theme))
            selected.extend(picker.sample(by_theme[theme], quota))
    random.Random(_stable_seed(seed, "order")).shuffle(selected)
    if len(selected) != size or len(set(selected)) != size:
        raise RuntimeError("theme-stratified subset contains duplicate/wrong IDs")
    return selected, dict(sorted(Counter(_theme(cards, value) for value in selected).items()))


def _matrix_for_region(returns: pd.DataFrame, region: str) -> pd.DataFrame:
    subset = returns.loc[returns["region"].astype(str).str.upper().eq(region.upper())]
    if subset.empty:
        raise ValueError(f"no raw-factor returns found for region={region}")
    matrix = (
        subset.pivot_table(index="date", columns="factor_id", values="return", aggfunc="mean")
        .sort_index()
    )
    matrix.columns = [str(value) for value in matrix.columns]
    full_index = pd.date_range(matrix.index.min(), matrix.index.max(), freq="ME")
    matrix = matrix.reindex(full_index)
    matrix.index.name = "date"
    return matrix


def _period_index(matrix: pd.DataFrame, month: pd.Period | str) -> int | None:
    target = pd.Period(month, freq="M")
    locations = np.flatnonzero(matrix.index.to_period("M") == target)
    return int(locations[0]) if len(locations) == 1 else None


def _slice_window(
    matrix: pd.DataFrame,
    anchor: pd.Period | str,
    lookback: int,
    target_months: int,
    factor_ids: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    end = _period_index(matrix, anchor)
    if end is None or end - lookback + 1 < 0 or end + target_months >= len(matrix.index):
        return None
    history = matrix.iloc[end - lookback + 1 : end + 1].copy()
    future = matrix.iloc[end + 1 : end + target_months + 1].copy()
    expected_history = pd.date_range(history.index[0], periods=lookback, freq="ME")
    expected_future = pd.date_range(future.index[0], periods=target_months, freq="ME")
    if not history.index.equals(expected_history) or not future.index.equals(expected_future):
        return None
    if factor_ids is not None:
        ids = [str(value) for value in factor_ids]
        if any(value not in history.columns for value in ids):
            return None
        history = history.loc[:, ids]
        future = future.loc[:, ids]
    return history, future


def _common_anchors(
    matrix: pd.DataFrame,
    *,
    windows: Sequence[str],
    target_months: int,
) -> list[str]:
    """Find year-end anchors with a complete 153-factor history/target panel."""

    if not matrix.columns.size:
        raise ValueError("raw-factor matrix has no columns")
    max_factors = len(matrix.columns)
    if max_factors != 153:
        raise ValueError(
            f"raw-factor matrix has {max_factors} factors; the 294-case design requires exactly 153"
        )
    first = matrix.index[0].to_period("M")
    last = matrix.index[-1].to_period("M")
    earliest = first + max(WINDOW_MONTHS[name] for name in windows) - 1
    latest = last - target_months
    candidates: list[str] = []
    for period in pd.period_range(earliest, latest, freq="M"):
        if period.month != 12:
            continue
        usable = True
        for window in windows:
            sliced = _slice_window(matrix, period, WINDOW_MONTHS[window], target_months)
            if sliced is None:
                usable = False
                break
            history, future = sliced
            if history.isna().any().any() or future.isna().any().any():
                usable = False
                break
        if usable:
            candidates.append(period.strftime("%Y-%m"))
    return candidates


def _schedule_anchors(
    available: Sequence[str],
    *,
    max_history: int,
    target_months: int,
    gap_months: int,
    policy: str,
) -> list[str]:
    if not available:
        raise ValueError("no complete year-end anchors are available")
    first = pd.Period(available[0], freq="M")
    last = pd.Period(available[-1], freq="M")
    spacing = int(max_history + target_months + gap_months)
    span = last.ordinal - first.ordinal
    count = span // spacing + 1
    if policy == "earliest_regular":
        selected = [first + index * spacing for index in range(count)]
    elif policy == "latest_regular":
        start = last - (count - 1) * spacing
        selected = [start + index * spacing for index in range(count)]
    elif policy == "earliest_with_latest":
        if count == 1:
            selected = [first]
        else:
            selected = [first + index * spacing for index in range(count - 1)] + [last]
    else:
        raise ValueError(f"unknown anchor policy: {policy}")
    available_set = set(available)
    labels = [period.strftime("%Y-%m") for period in selected]
    missing = [label for label in labels if label not in available_set]
    if missing:
        raise ValueError(
            "scheduled anchor is not a complete year-end anchor: "
            + ", ".join(missing)
        )
    if len(set(labels)) != len(labels):
        raise ValueError("scheduled anchors are not unique")
    return labels


def _gap_audit(
    anchors: Sequence[str],
    *,
    windows: Sequence[str],
    target_months: int,
    required_gap: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left_index, left_label in enumerate(anchors):
        left = pd.Period(left_label, freq="M")
        for right_label in anchors[left_index + 1 :]:
            right = pd.Period(right_label, freq="M")
            for window in windows:
                lookback = WINDOW_MONTHS[window]
                left_target_end = left + target_months
                right_history_start = right - lookback + 1
                gap = right_history_start.ordinal - left_target_end.ordinal - 1
                row = {
                    "left_anchor": left_label,
                    "right_anchor": right_label,
                    "window": window,
                    "left_target_end": left_target_end.strftime("%Y-%m"),
                    "right_history_start": right_history_start.strftime("%Y-%m"),
                    "gap_months": int(gap),
                    "required_gap_months": int(required_gap),
                    "passes": bool(gap >= required_gap),
                }
                rows.append(row)
                if gap < required_gap:
                    raise ValueError(f"purge gap failed: {row}")
    return rows


def _feature_cache(history: pd.DataFrame, min_history_months: int) -> dict[str, pd.DataFrame]:
    """Compute compact point-in-time features once for all subsets in a cell."""

    if min_history_months < 1 or min_history_months >= len(history.index):
        raise ValueError("feature warm-up must be between 1 and history length-1")
    cache: dict[str, pd.DataFrame] = {}
    first_cutoff = max(min_history_months - 1, 0)
    for cutoff in range(first_cutoff, len(history.index) - 1):
        prefix = history.iloc[: cutoff + 1]
        cache[pd.Timestamp(prefix.index[-1]).strftime("%Y-%m-%d")] = build_compact_features(prefix)
    return cache


def _supervised_from_cache(
    history: pd.DataFrame,
    factor_ids: Sequence[str],
    original_to_masked: Mapping[str, str],
    cache: Mapping[str, pd.DataFrame],
    *,
    min_history_months: int,
) -> pd.DataFrame:
    fields = [str(value) for value in HISTORICAL_PANEL_FEATURE_FIELDS]
    ids = [str(value) for value in factor_ids]
    rows: list[dict[str, Any]] = []
    first_cutoff = max(min_history_months - 1, 0)
    for cutoff in range(first_cutoff, len(history.index) - 1):
        prefix_key = pd.Timestamp(history.index[cutoff]).strftime("%Y-%m-%d")
        table = cache[prefix_key]
        next_values = pd.to_numeric(history.iloc[cutoff + 1], errors="coerce")
        time_key = f"t-{len(history.index) - 1 - cutoff}"
        for original_id in ids:
            label = next_values.get(original_id)
            try:
                label_float = float(label)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(label_float):
                continue
            source = table.loc[original_id]
            row: dict[str, Any] = {
                "id": str(original_to_masked[original_id]),
                "time_key": time_key,
            }
            for field in fields:
                row[field] = _jsonable(source.get(field))
            row["label"] = label_float
            rows.append(row)
    if not rows:
        raise ValueError("history produced no supervised rows")
    return pd.DataFrame(rows, columns=["id", "time_key", *fields, "label"])


def _calendar_range(history: pd.DataFrame, relative_range: Sequence[str]) -> list[str]:
    dates: list[pd.Timestamp] = []
    for value in relative_range:
        offset = int(str(value).split("-", 1)[1])
        cutoff_position = len(history.index) - 1 - offset
        label_position = cutoff_position + 1
        if 0 <= label_position < len(history.index):
            dates.append(pd.Timestamp(history.index[label_position]))
    return [dates[0].strftime("%Y-%m"), dates[-1].strftime("%Y-%m")] if dates else []


def _case_id(region: str, block_id: str, anchor: str, window: str, subset_id: str) -> str:
    material = f"{region.upper()}|{block_id}|{anchor}|{window}|{subset_id}"
    return "case_" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def _build_selection_plan(
    *,
    anchors: Sequence[str],
    matrix: pd.DataFrame,
    cards: Mapping[str, Mapping[str, Any]],
    windows: Sequence[str],
    target_months: int,
    master_seed: int,
    subsets_by_size: Mapping[int, int],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    all_factor_ids = sorted(str(value) for value in matrix.columns)
    plans_by_block: dict[str, list[dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    for block_index, anchor in enumerate(anchors, start=1):
        block_id = f"block_{block_index:02d}"
        # Selection eligibility is a point-in-time data-availability rule.  We
        # require complete history/target cells for all requested windows so a
        # 153-factor census and every paired subset share one universe; no
        # performance statistic is consulted.
        for window in windows:
            sliced = _slice_window(matrix, anchor, WINDOW_MONTHS[window], target_months)
            if sliced is None:
                raise ValueError(f"missing slice for {block_id}/{window}/{anchor}")
            history, future = sliced
            if history[all_factor_ids].isna().any().any() or future[all_factor_ids].isna().any().any():
                raise ValueError(f"incomplete 153-factor panel for {block_id}/{window}/{anchor}")
        subsets: list[dict[str, Any]] = []
        for size in (38, 76, 153):
            count = 1 if size == 153 else int(subsets_by_size.get(size, 1))
            signatures: set[tuple[str, ...]] = set()
            for subset_index in range(1, count + 1):
                if size == 153:
                    selected = list(all_factor_ids)
                    theme_counts = dict(sorted(Counter(_theme(cards, value) for value in selected).items()))
                    seed = _stable_seed(master_seed, "raw_factor_subset", block_id, size, subset_index)
                    attempt = 0
                else:
                    attempt = 0
                    while True:
                        seed = _stable_seed(
                            master_seed,
                            "raw_factor_subset",
                            block_id,
                            anchor,
                            size,
                            subset_index,
                            attempt,
                        )
                        selected, theme_counts = _sample_subset(all_factor_ids, size, cards, seed)
                        signature = tuple(sorted(selected))
                        if signature not in signatures:
                            break
                        attempt += 1
                        if attempt > 1000:
                            raise RuntimeError(f"could not obtain a distinct subset for {block_id}/{size}")
                signature = tuple(sorted(selected))
                signatures.add(signature)
                subset_id = f"{block_id}_{anchor.replace('-', '')}_{size}_s{subset_index:02d}"
                subset = {
                    "block_id": block_id,
                    "anchor_month": anchor,
                    "subset_id": subset_id,
                    "subset_index": int(subset_index),
                    "factor_count": int(size),
                    "factor_ids": list(selected),
                    "factor_ids_sorted": list(signature),
                    "theme_counts": theme_counts,
                    "seed": int(seed),
                    "sampling_attempt": int(attempt),
                    "sampling_method": "proportional_theme_stratified_random",
                    "sampling_within_subset": "without_replacement",
                    "sampling_across_subsets": "with_replacement",
                    "selection_scope": "time_block_reused_across_history_windows",
                    "same_factor_subset_across_windows": True,
                    "future_values_used_for_selection": False,
                }
                subsets.append(subset)
                rows.append(
                    {
                        "record_type": "selection",
                        "block_id": block_id,
                        "anchor_month": anchor,
                        "subset_id": subset_id,
                        "subset_index": int(subset_index),
                        "factor_count": int(size),
                        "factor_ids": list(selected),
                        "factor_ids_sorted": list(signature),
                        "theme_counts": theme_counts,
                        "seed": int(seed),
                        "sampling_attempt": int(attempt),
                        "sampling_scope": "time_block_reused_across_history_windows",
                        "same_factor_subset_across_windows": True,
                        "future_values_used_for_selection": False,
                    }
                )
        plans_by_block[block_id] = subsets
    return rows, plans_by_block


def _write_case(
    *,
    output_root: Path,
    case_number: int,
    block_id: str,
    block_index: int,
    anchor: str,
    window: str,
    subset: Mapping[str, Any],
    matrix: pd.DataFrame,
    region: str,
    target_months: int,
    train_cutoffs: int,
    validation_cutoffs: int,
    feature_warmup_months: int,
    plan_hash: str,
    track: str = "numeric_only",
    cards: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    factor_ids = [str(value) for value in subset["factor_ids"]]
    lookback = WINDOW_MONTHS[window]
    first_slice = _slice_window(matrix, anchor, lookback, target_months, factor_ids)
    if first_slice is None:
        raise ValueError(f"cannot build first episode for {block_id}/{window}/{subset['subset_id']}")
    first_history, _first_future = first_slice
    original_to_masked = {
        original: f"S{index:02d}" for index, original in enumerate(factor_ids, start=1)
    }
    masked_to_original = {masked: original for original, masked in original_to_masked.items()}
    # Build compact features once per (block, window) in the caller and pass a
    # cache through metadata; the cache itself is intentionally not persisted.
    cache = _feature_cache(first_history, feature_warmup_months)
    supervised = _supervised_from_cache(
        first_history,
        factor_ids,
        original_to_masked,
        cache,
        min_history_months=feature_warmup_months,
    )
    train_df, valid_df, split_meta = split_supervised_rows(
        supervised,
        train_cutoffs=train_cutoffs,
        validation_cutoffs=validation_cutoffs,
    )
    case_id = _case_id(region, block_id, anchor, window, str(subset["subset_id"]))
    metadata = {
        "region": region.upper(),
        "sequence_steps": int(target_months),
        "history_window_months": int(lookback),
        "forecast_horizon_months": 1,
        "factor_count": int(len(factor_ids)),
        "factor_source_type": "raw_factor",
        "train_cutoffs": int(train_cutoffs),
        "validation_cutoffs": int(validation_cutoffs),
        "refit_policy": "validation_select_then_train_plus_validation",
        "train_cutoff_time_range": split_meta["train_time_range"],
        "validation_cutoff_time_range": split_meta["validation_time_range"],
        "train_label_time_range": split_meta["train_label_time_range"],
        "validation_label_time_range": split_meta["validation_label_time_range"],
        "min_history_months": int(feature_warmup_months),
        "feature_warmup_months": int(feature_warmup_months),
        "rolling_gap_months": int(DEFAULT_GAP_MONTHS),
        "public_future_labels_included": False,
        "seed": 1,
    }
    payload = build_code_generation_payload(
        train_df,
        valid_df,
        feature_columns=list(HISTORICAL_PANEL_FEATURE_FIELDS),
        case_id=case_id,
        metadata=metadata,
        track=track,
        factor_cards=semantic_cards_payload(
            cards,
            factor_ids,
            original_to_masked,
        ) if normalize_track(track) == "semantic" else None,
    )

    truth_rows: list[dict[str, Any]] = []
    anchor_period = pd.Period(anchor, freq="M")
    for step in range(target_months):
        step_anchor = anchor_period + step
        sliced = _slice_window(matrix, step_anchor, lookback, 1, factor_ids)
        if sliced is None:
            raise ValueError(f"missing rolling step {step + 1} for {case_id}")
        history, future = sliced
        episode = RouterEpisode(
            case_id=f"{region.upper()}_{step_anchor.strftime('%Y%m')}_{window}_{subset['subset_id']}",
            region=region.upper(),
            anchor_date=pd.Timestamp(history.index[-1]),
            history=history,
            future=future,
            # Cards are generation-time context only.  The evaluator does not
            # need to carry the enriched objects twelve times in truth.
            sleeve_cards={},
        )
        feature_stub = [{"factor_id": original_to_masked[value]} for value in factor_ids]
        truth_payload = {"case_id": f"{case_id}__step_{step + 1:02d}", "features": feature_stub}
        truth = episode_to_truth_record(episode, truth_payload, masked_to_original)
        truth.update(
            {
                "sequence_id": case_id,
                "sequence_step": int(step + 1),
                "task_variant": FROZEN_TASK_VARIANT,
                "public_payload_case_id": case_id,
                "block_id": block_id,
                "block_index": int(block_index),
                "window": window,
                "anchor_month": anchor,
                "subset_id": str(subset["subset_id"]),
                "subset_index": int(subset["subset_index"]),
                "factor_source_type": "raw_factor",
                "history_window_months": int(lookback),
                "train_cutoffs": int(train_cutoffs),
                "validation_cutoffs": int(validation_cutoffs),
                "feature_warmup_months": int(feature_warmup_months),
                "rolling_gap_months": int(DEFAULT_GAP_MONTHS),
            }
        )
        truth_rows.append(truth)

    case_dir = output_root / "cases" / block_id / case_id
    case_dir.mkdir(parents=True, exist_ok=False)
    payload_path = case_dir / "payload.json"
    truth_path = case_dir / "rolling_truth.jsonl"
    selection_path = case_dir / "selection.json"
    _write_json(payload_path, payload)
    _write_jsonl(truth_path, truth_rows)
    _write_json(
        selection_path,
        {
            "selection_plan_sha256": plan_hash,
            "block_id": block_id,
            "anchor_month": anchor,
            "window": window,
            "subset_id": subset["subset_id"],
            "subset_index": int(subset["subset_index"]),
            "factor_count": len(factor_ids),
            "factor_ids": factor_ids,
            "factor_ids_sorted": list(subset["factor_ids_sorted"]),
            "theme_counts": dict(subset["theme_counts"]),
            "source_ids_are_private": True,
        },
    )
    record: dict[str, Any] = {
        "record_type": "case",
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "track": normalize_track(track),
        "case_number": int(case_number),
        "case_id": case_id,
        "sequence_id": case_id,
        "dataset_id": _display_path(case_dir),
        "path": _display_path(case_dir),
        "block_id": block_id,
        "block_index": int(block_index),
        "region": region.upper(),
        "window": window,
        "lookback_months": int(lookback),
        "forecast_horizon_months": 1,
        "sequence_steps": int(target_months),
        "anchor_month": anchor,
        "start_anchor_month": anchor,
        "end_anchor_month": (anchor_period + target_months - 1).strftime("%Y-%m"),
        "history_start_month": first_history.index[0].strftime("%Y-%m"),
        "history_end_month": first_history.index[-1].strftime("%Y-%m"),
        "target_start_month": (anchor_period + 1).strftime("%Y-%m"),
        "target_end_month": (anchor_period + target_months).strftime("%Y-%m"),
        "anchor_date_is_public": False,
        "factor_source_type": "raw_factor",
        "sample_size": len(factor_ids),
        "requested_sample_size": int(subset["factor_count"]),
        "subset_id": str(subset["subset_id"]),
        "subset_index": int(subset["subset_index"]),
        "selected_factor_ids": factor_ids,
        "selected_factor_ids_sorted": list(subset["factor_ids_sorted"]),
        "selected_theme_counts": dict(subset["theme_counts"]),
        "selection_plan_sha256": plan_hash,
        "sampling": {
            "method": "proportional_theme_stratified_random",
            "within_subset": "without_replacement",
            "across_subsets": "with_replacement",
            "scope": "time_block_reused_across_history_windows",
            "same_factor_subset_across_history_windows": True,
            "seed": int(subset["seed"]),
            "sampling_attempt": int(subset["sampling_attempt"]),
        },
        "rolling_protocol": {
            "test_steps": int(target_months),
            "one_step_horizon_months": 1,
            "train_cutoffs_per_origin": int(train_cutoffs),
            "validation_cutoffs_per_origin": int(validation_cutoffs),
            "feature_warmup_months": int(feature_warmup_months),
            "refit_policy": "from_scratch_each_origin",
            "llm_calls_during_rolling": 0,
        },
        "train_validation": split_meta,
        "public_future_labels_included": False,
        "source_history_factor_ids_are_truth_only": True,
        "evaluator_only_files": ["rolling_truth.jsonl"],
        "artifacts": {
            "payload": {
                "path": _display_path(payload_path),
                "bytes": int(payload_path.stat().st_size),
                "sha256": _sha256(payload_path),
            },
            "rolling_truth": {
                "path": _display_path(truth_path),
                "bytes": int(truth_path.stat().st_size),
                "sha256": _sha256(truth_path),
                "rows": len(truth_rows),
            },
            "selection": {
                "path": _display_path(selection_path),
                "bytes": int(selection_path.stat().st_size),
                "sha256": _sha256(selection_path),
            },
        },
        "payload_hash": stable_hash(payload),
    }
    _write_json(case_dir / "manifest.json", record)
    record["case_manifest_path"] = _display_path(case_dir / "manifest.json")
    # Rewrite once with the self-reference field included; the manifest hash
    # itself is intentionally not embedded to avoid a recursive digest.
    _write_json(case_dir / "manifest.json", record)
    return record


def _split_manifests(
    *,
    output_root: Path,
    index: Mapping[str, Any],
    case_rows: Sequence[Mapping[str, Any]],
    dev_blocks: int,
    source_manifest_hash: str,
) -> dict[str, Any]:
    blocks = sorted({str(row["block_id"]) for row in case_rows})
    if not blocks:
        raise ValueError("cannot split an empty case set")
    if len(blocks) == 1:
        dev_set = set(blocks)
    elif dev_blocks < 1 or dev_blocks >= len(blocks):
        raise ValueError(f"dev_blocks must be between 1 and {len(blocks) - 1}")
    else:
        dev_set = set(blocks[:dev_blocks])
    summaries: dict[str, Any] = {}
    for split, wanted in (("dev", dev_set), ("test", set(blocks) - dev_set)):
        selected: list[dict[str, Any]] = []
        for source in case_rows:
            if str(source["block_id"]) not in wanted:
                continue
            row = dict(source)
            row.update(
                {
                    "split": split,
                    "split_rule": "chronological_time_block_holdout",
                    "dev_blocks": blocks[:dev_blocks] if len(blocks) > 1 else blocks,
                    "test_blocks": blocks[dev_blocks:] if len(blocks) > 1 else [],
                    "source_manifest": _display_path(output_root / "manifest.jsonl"),
                    "source_manifest_sha256": source_manifest_hash,
                    "source_case_path": source.get("path"),
                }
            )
            selected.append(row)
        # ``--max-cases`` smoke builds can stop inside the first block.  Keep
        # an index-only test manifest in that debug-only shape; full designs
        # still require both chronological splits to be non-empty.
        if not selected and len(blocks) > 1:
            raise ValueError(f"{split} split is empty")
        summaries[split] = {
            "case_count": len(selected),
            "sequence_count": len(selected),
            "rolling_forecast_rows": sum(int(row.get("sequence_steps", 0)) for row in selected),
            "blocks": sorted(wanted),
            "windows": sorted({str(row["window"]) for row in selected}),
            "factor_counts": dict(sorted(Counter(str(row["sample_size"]) for row in selected).items())),
        }
        split_index = dict(index)
        split_index.update(
            {
                "record_type": "index",
                "manifest_schema_version": SPLIT_SCHEMA_VERSION,
                "split_status": "chronological_time_block_holdout",
                "split": split,
                "split_rule": "complete purged time blocks stay together; earliest blocks are dev",
                "split_blocks": sorted(wanted),
                "dev_blocks": blocks[:dev_blocks],
                "test_blocks": blocks[dev_blocks:],
                "source_manifest": _display_path(output_root / "manifest.jsonl"),
                "source_manifest_sha256": source_manifest_hash,
                "case_count": len(selected),
                "sequence_count": len(selected),
                "rolling_forecast_rows": summaries[split]["rolling_forecast_rows"],
            }
        )
        _write_jsonl(output_root / "splits" / split / "manifest.jsonl", [split_index, *selected])
    result = {
        "schema_version": "task_b_frozen_raw_rolling_split_summary.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": _display_path(output_root / "manifest.jsonl"),
        "source_manifest_sha256": source_manifest_hash,
        "all_blocks": blocks,
        "dev_blocks": blocks[:dev_blocks] if len(blocks) > 1 else blocks,
        "test_blocks": blocks[dev_blocks:] if len(blocks) > 1 else [],
        "splits": summaries,
        "case_directories_reused_by_reference": True,
    }
    _write_json(output_root / "split_manifest.json", result)
    return result


def build_dataset(
    *,
    returns_path: Path,
    cards_path: Path | None,
    output_root: Path,
    region: str,
    windows: Sequence[str],
    target_months: int,
    gap_months: int,
    anchor_policy: str,
    dev_blocks: int,
    master_seed: int,
    subsets_by_size: Mapping[int, int],
    returns_in_percent: bool = False,
    max_cases: int | None = None,
    expected_case_count: int | None = EXPECTED_CASE_COUNT,
    track: str = "numeric_only",
) -> dict[str, Any]:
    track = normalize_track(track)
    if output_root.exists():
        if any(output_root.iterdir()):
            raise FileExistsError(f"refusing to overwrite non-empty output root: {output_root}")
        raise FileExistsError(f"output root already exists: {output_root}")
    if not windows or any(window not in WINDOW_MONTHS for window in windows):
        raise ValueError(f"windows must be drawn from {sorted(WINDOW_MONTHS)}")
    if len(set(windows)) != len(windows):
        raise ValueError("windows must be unique")
    if target_months < 1 or gap_months < 0:
        raise ValueError("target_months must be positive and gap_months non-negative")
    returns = load_raw_factor_returns(
        returns_path,
        region=region,
        returns_in_percent=returns_in_percent,
    )
    matrix = _matrix_for_region(returns, region)
    all_factor_ids = sorted(str(value) for value in matrix.columns)
    # The numeric-only view must remain buildable without a metadata file.  If
    # a card file is available we still read it privately for the deterministic
    # subset sampler, so separately generated numeric/semantic datasets keep
    # the same factor subsets; the file is never copied into numeric payloads.
    if track == "semantic" or (cards_path is not None and cards_path.is_file()):
        cards = _load_cards(cards_path, all_factor_ids)
    else:
        cards = _load_cards(None, all_factor_ids)
    if track == "semantic":
        incomplete_cards = [
            factor_id
            for factor_id in all_factor_ids
            if not str(cards[factor_id].get("neutral_name", "")).strip()
            or not any(
                str(cards[factor_id].get(field, "")).strip()
                for field in ("neutral_description", "formula_or_definition", "formula_latex")
            )
        ]
        if incomplete_cards:
            raise ValueError(
                "semantic track requires complete name/definition/formula cards; "
                f"missing for {incomplete_cards[:10]}"
            )
    available = _common_anchors(matrix, windows=windows, target_months=target_months)
    anchors = _schedule_anchors(
        available,
        max_history=max(WINDOW_MONTHS[window] for window in windows),
        target_months=target_months,
        gap_months=gap_months,
        policy=anchor_policy,
    )
    gap_rows = _gap_audit(
        anchors,
        windows=windows,
        target_months=target_months,
        required_gap=gap_months,
    )
    selection_rows, plans = _build_selection_plan(
        anchors=anchors,
        matrix=matrix,
        cards=cards,
        windows=windows,
        target_months=target_months,
        master_seed=master_seed,
        subsets_by_size=subsets_by_size,
    )
    selection_hash = hashlib.sha256(
        json.dumps(_jsonable(selection_rows), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    expected_total = len(anchors) * len(windows) * sum(
        1 if int(size) == 153 else int(subsets_by_size.get(size, 1)) for size in (38, 76, 153)
    )
    if expected_case_count is not None and max_cases is None and expected_total != expected_case_count:
        raise ValueError(
            f"scheduled design has {expected_total} cases, expected {expected_case_count}; "
            "check coverage/gap settings or pass --expected-case-count 0"
        )
    output_root.mkdir(parents=True, exist_ok=False)
    case_rows: list[dict[str, Any]] = []
    case_number = 0
    for block_index, anchor in enumerate(anchors, start=1):
        block_id = f"block_{block_index:02d}"
        for window in windows:
            lookback = WINDOW_MONTHS[window]
            # Exact 2Y histories contain only 12 usable transitions after the
            # 12-month feature warm-up, so use 6+6 labelled cutoffs there.
            # 3Y/5Y retain the standard 12+12 inner split.
            train_cutoffs = 6 if lookback == 24 else 12
            validation_cutoffs = 6 if lookback == 24 else 12
            feature_warmup = DEFAULT_MIN_HISTORY_MONTHS
            source_slice = _slice_window(matrix, anchor, lookback, target_months)
            if source_slice is None:
                raise ValueError(f"missing source slice for {block_id}/{window}")
            full_history, _ = source_slice
            # Compute once for the full 153-factor history; each subset simply
            # selects rows from these tables.  This avoids 294 repeated
            # feature calculations while retaining exact point-in-time values.
            cache = _feature_cache(full_history, feature_warmup)
            for subset in plans[block_id]:
                if max_cases is not None and case_number >= max_cases:
                    break
                # _write_case builds its own compact cache for correctness in
                # isolation.  Replace the helper's cache through a lightweight
                # temporary monkey-free path below by constructing the case
                # with the same full-history cache and then writing artifacts.
                # The explicit implementation keeps this script easy to audit;
                # the full cache is reused by the local assembly function.
                # Build the case from the shared full-factor feature cache;
                # only the selected rows are serialized into this case.
                case_number += 1
                record = _write_case_cached(
                    output_root=output_root,
                    case_number=case_number,
                    block_id=block_id,
                    block_index=block_index,
                    anchor=anchor,
                    window=window,
                    subset=subset,
                    matrix=matrix,
                    region=region,
                    target_months=target_months,
                    gap_months=gap_months,
                    train_cutoffs=train_cutoffs,
                    validation_cutoffs=validation_cutoffs,
                    feature_warmup_months=feature_warmup,
                    plan_hash=selection_hash,
                    full_history=full_history,
                    full_feature_cache=cache,
                    track=track,
                    cards=cards,
                )
                case_rows.append(record)
                if case_number % 10 == 0 or case_number == expected_total:
                    print(f"built {case_number}/{expected_total} cases", file=sys.stderr, flush=True)
            if max_cases is not None and case_number >= max_cases:
                break
        if max_cases is not None and case_number >= max_cases:
            break

    if max_cases is not None and case_number != max_cases:
        raise ValueError(f"max_cases={max_cases} exceeds available cases ({expected_total})")
    if max_cases is None and case_number != expected_total:
        raise RuntimeError(f"generated {case_number} cases, expected {expected_total}")

    _write_jsonl(output_root / "selection_plan.jsonl", selection_rows)
    source_index: dict[str, Any] = {
        "record_type": "index",
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "task_b",
        "dataset_scope": "raw_factor_frozen_code_rolling",
        "payload_variant": f"{track}_frozen_code_rolling_v1",
        "track": track,
        "semantic_card_schema_version": (
            SEMANTIC_CARD_SCHEMA_VERSION if track == "semantic" else None
        ),
        "split_status": "unsplit",
        "factor_source_type": "raw_factor",
        "region": region.upper(),
        "source_returns": _display_path(returns_path),
        "source_returns_sha256": _sha256(returns_path),
        # The card file is an evaluator-side selector input for matched
        # subsets, but it is part of the public provenance only for the
        # semantic build.  Keep numeric-only manifests self-describing as an
        # anonymous information set.
        "source_factor_cards": (
            _display_path(cards_path) if track == "semantic" and cards_path else None
        ),
        "source_factor_cards_sha256": (
            _sha256(cards_path) if track == "semantic" and cards_path else None
        ),
        "source_schema": ["date", "region", "factor_id", "return"],
        "source_factor_count": len(all_factor_ids),
        "source_date_range": [matrix.index[0].strftime("%Y-%m"), matrix.index[-1].strftime("%Y-%m")],
        "selection_plan": {
            "path": _display_path(output_root / "selection_plan.jsonl"),
            "sha256": _sha256(output_root / "selection_plan.jsonl"),
            "rows": len(selection_rows),
        },
        "windows": list(windows),
        "window_months": {window: WINDOW_MONTHS[window] for window in windows},
        "target_months_per_sequence": int(target_months),
        "anchors_available_complete": list(available),
        "anchors": list(anchors),
        "anchor_policy": anchor_policy,
        "temporal_block_count": len(anchors),
        "block_ids": [f"block_{index:02d}" for index in range(1, len(anchors) + 1)],
        "purge_policy": {
            "minimum_gap_months": int(gap_months),
            "longest_history_months": max(WINDOW_MONTHS[window] for window in windows),
            "required_anchor_spacing_months": int(
                max(WINDOW_MONTHS[window] for window in windows) + target_months + gap_months
            ),
            "interval_definition": "history_start..hidden_target_end",
            "gap_audit_rows": gap_rows,
        },
        "inner_split_policy": {
            "2Y": {"train_cutoffs": 6, "validation_cutoffs": 6, "reason": "24-month history minus 12-month feature warm-up leaves 12 labels"},
            "3Y": {"train_cutoffs": 12, "validation_cutoffs": 12},
            "5Y": {"train_cutoffs": 12, "validation_cutoffs": 12},
            "feature_warmup_months": DEFAULT_MIN_HISTORY_MONTHS,
        },
        "subset_policy": {str(key): int(value) for key, value in subsets_by_size.items()},
        "case_count": len(case_rows),
        "sequence_count": len(case_rows),
        "payload_view_count": len(case_rows),
        "rolling_forecast_rows": len(case_rows) * target_months,
        "canonical_group_count": len(selection_rows),
        "case_count_formula": "time_blocks * history_windows * (8*38 + 5*76 + 1*153)",
        "factor_count_distribution": dict(sorted(Counter(str(row["sample_size"]) for row in case_rows).items())),
        "public_payload_audit": {
            "future_labels_included": False,
            "test_rows_included": False,
            "original_factor_ids_included": False,
            "calendar_dates_included": False,
            "rolling_llm_calls": 0,
            "semantic_cards_included": track == "semantic",
        },
        "records_follow": "one case record per subsequent JSONL line",
    }
    _write_jsonl(output_root / "manifest.jsonl", [source_index, *case_rows])
    source_manifest_hash = _sha256(output_root / "manifest.jsonl")
    split_summary = _split_manifests(
        output_root=output_root,
        index=source_index,
        case_rows=case_rows,
        dev_blocks=dev_blocks,
        source_manifest_hash=source_manifest_hash,
    )
    # A concise human-readable index is kept next to the machine manifests.
    readme = _readme(
        output_root=output_root,
        source_index=source_index,
        split_summary=split_summary,
        source_manifest_hash=source_manifest_hash,
    )
    (output_root / "README.md").write_text(readme, encoding="utf-8")
    summary = {
        "output_root": str(output_root),
        "manifest": str(output_root / "manifest.jsonl"),
        "selection_plan": str(output_root / "selection_plan.jsonl"),
        "track": track,
        "case_count": len(case_rows),
        "sequence_count": len(case_rows),
        "rolling_forecast_rows": len(case_rows) * target_months,
        "anchors": anchors,
        "split_summary": split_summary,
        "source_manifest_sha256": source_manifest_hash,
    }
    _write_json(output_root / "dataset_summary.json", summary)
    return summary


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def _write_case_cached(
    *,
    output_root: Path,
    case_number: int,
    block_id: str,
    block_index: int,
    anchor: str,
    window: str,
    subset: Mapping[str, Any],
    matrix: pd.DataFrame,
    region: str,
    target_months: int,
    gap_months: int,
    train_cutoffs: int,
    validation_cutoffs: int,
    feature_warmup_months: int,
    plan_hash: str,
    full_history: pd.DataFrame,
    full_feature_cache: Mapping[str, pd.DataFrame],
    track: str = "numeric_only",
    cards: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Cache-aware variant of ``_write_case`` used by the full generator."""

    factor_ids = [str(value) for value in subset["factor_ids"]]
    lookback = WINDOW_MONTHS[window]
    first_history = full_history.loc[:, factor_ids]
    original_to_masked = {
        original: f"S{index:02d}" for index, original in enumerate(factor_ids, start=1)
    }
    subset_cache = {key: table.loc[factor_ids] for key, table in full_feature_cache.items()}
    supervised = _supervised_from_cache(
        first_history,
        factor_ids,
        original_to_masked,
        subset_cache,
        min_history_months=feature_warmup_months,
    )
    train_df, valid_df, split_meta = split_supervised_rows(
        supervised,
        train_cutoffs=train_cutoffs,
        validation_cutoffs=validation_cutoffs,
    )
    case_id = _case_id(region, block_id, anchor, window, str(subset["subset_id"]))
    metadata = {
        "region": region.upper(),
        "sequence_steps": int(target_months),
        "history_window_months": int(lookback),
        "forecast_horizon_months": 1,
        "factor_count": int(len(factor_ids)),
        "factor_source_type": "raw_factor",
        "train_cutoffs": int(train_cutoffs),
        "validation_cutoffs": int(validation_cutoffs),
        "refit_policy": "validation_select_then_train_plus_validation",
        "train_cutoff_time_range": split_meta["train_time_range"],
        "validation_cutoff_time_range": split_meta["validation_time_range"],
        "train_label_time_range": split_meta["train_label_time_range"],
        "validation_label_time_range": split_meta["validation_label_time_range"],
        "min_history_months": int(feature_warmup_months),
        "feature_warmup_months": int(feature_warmup_months),
        "rolling_gap_months": int(gap_months),
        "public_future_labels_included": False,
        "seed": 1,
    }
    payload = build_code_generation_payload(
        train_df,
        valid_df,
        feature_columns=list(HISTORICAL_PANEL_FEATURE_FIELDS),
        case_id=case_id,
        metadata=metadata,
        track=track,
        factor_cards=(
            semantic_cards_payload(cards, factor_ids, original_to_masked)
            if normalize_track(track) == "semantic"
            else None
        ),
    )
    masked_to_original = {masked: original for original, masked in original_to_masked.items()}
    truth_rows: list[dict[str, Any]] = []
    anchor_period = pd.Period(anchor, freq="M")
    for step in range(target_months):
        step_anchor = anchor_period + step
        sliced = _slice_window(matrix, step_anchor, lookback, 1, factor_ids)
        if sliced is None:
            raise ValueError(f"missing rolling step {step + 1} for {case_id}")
        history, future = sliced
        episode = RouterEpisode(
            case_id=f"{region.upper()}_{step_anchor.strftime('%Y%m')}_{window}_{subset['subset_id']}",
            region=region.upper(),
            anchor_date=pd.Timestamp(history.index[-1]),
            history=history,
            future=future,
            sleeve_cards={},
        )
        feature_stub = [{"factor_id": original_to_masked[value]} for value in factor_ids]
        truth = episode_to_truth_record(
            episode,
            {"case_id": f"{case_id}__step_{step + 1:02d}", "features": feature_stub},
            masked_to_original,
        )
        truth.update(
            {
                "sequence_id": case_id,
                "sequence_step": int(step + 1),
                "task_variant": FROZEN_TASK_VARIANT,
                "public_payload_case_id": case_id,
                "block_id": block_id,
                "block_index": int(block_index),
                "window": window,
                "anchor_month": anchor,
                "subset_id": str(subset["subset_id"]),
                "subset_index": int(subset["subset_index"]),
                "factor_source_type": "raw_factor",
                "history_window_months": int(lookback),
                "train_cutoffs": int(train_cutoffs),
                "validation_cutoffs": int(validation_cutoffs),
                "feature_warmup_months": int(feature_warmup_months),
                "rolling_gap_months": int(gap_months),
            }
        )
        truth_rows.append(truth)

    case_dir = output_root / "cases" / block_id / case_id
    case_dir.mkdir(parents=True, exist_ok=False)
    payload_path = case_dir / "payload.json"
    truth_path = case_dir / "rolling_truth.jsonl"
    selection_path = case_dir / "selection.json"
    _write_json(payload_path, payload)
    _write_jsonl(truth_path, truth_rows)
    _write_json(
        selection_path,
        {
            "selection_plan_sha256": plan_hash,
            "block_id": block_id,
            "anchor_month": anchor,
            "window": window,
            "subset_id": subset["subset_id"],
            "subset_index": int(subset["subset_index"]),
            "factor_count": len(factor_ids),
            "factor_ids": factor_ids,
            "factor_ids_sorted": list(subset["factor_ids_sorted"]),
            "theme_counts": dict(subset["theme_counts"]),
            "source_ids_are_private": True,
        },
    )
    record: dict[str, Any] = {
        "record_type": "case",
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "track": normalize_track(track),
        "case_number": int(case_number),
        "case_id": case_id,
        "sequence_id": case_id,
        "dataset_id": _display_path(case_dir),
        "path": _display_path(case_dir),
        "block_id": block_id,
        "block_index": int(block_index),
        "region": region.upper(),
        "window": window,
        "lookback_months": int(lookback),
        "forecast_horizon_months": 1,
        "sequence_steps": int(target_months),
        "anchor_month": anchor,
        "start_anchor_month": anchor,
        "end_anchor_month": (anchor_period + target_months - 1).strftime("%Y-%m"),
        "history_start_month": first_history.index[0].strftime("%Y-%m"),
        "history_end_month": first_history.index[-1].strftime("%Y-%m"),
        "target_start_month": (anchor_period + 1).strftime("%Y-%m"),
        "target_end_month": (anchor_period + target_months).strftime("%Y-%m"),
        "anchor_date_is_public": False,
        "factor_source_type": "raw_factor",
        "sample_size": len(factor_ids),
        "requested_sample_size": int(subset["factor_count"]),
        "subset_id": str(subset["subset_id"]),
        "subset_index": int(subset["subset_index"]),
        "selected_factor_ids": factor_ids,
        "selected_factor_ids_sorted": list(subset["factor_ids_sorted"]),
        "selected_theme_counts": dict(subset["theme_counts"]),
        "selection_plan_sha256": plan_hash,
        "sampling": {
            "method": "proportional_theme_stratified_random",
            "within_subset": "without_replacement",
            "across_subsets": "with_replacement",
            "scope": "time_block_reused_across_history_windows",
            "same_factor_subset_across_history_windows": True,
            "seed": int(subset["seed"]),
            "sampling_attempt": int(subset["sampling_attempt"]),
        },
        "rolling_protocol": {
            "test_steps": int(target_months),
            "one_step_horizon_months": 1,
            "train_cutoffs_per_origin": int(train_cutoffs),
            "validation_cutoffs_per_origin": int(validation_cutoffs),
            "feature_warmup_months": int(feature_warmup_months),
            "refit_policy": "from_scratch_each_origin",
            "llm_calls_during_rolling": 0,
        },
        "train_validation": split_meta,
        "public_future_labels_included": False,
        "source_history_factor_ids_are_truth_only": True,
        "evaluator_only_files": ["rolling_truth.jsonl"],
        "artifacts": {
            "payload": {
                "path": _display_path(payload_path),
                "bytes": int(payload_path.stat().st_size),
                "sha256": _sha256(payload_path),
            },
            "rolling_truth": {
                "path": _display_path(truth_path),
                "bytes": int(truth_path.stat().st_size),
                "sha256": _sha256(truth_path),
                "rows": len(truth_rows),
            },
            "selection": {
                "path": _display_path(selection_path),
                "bytes": int(selection_path.stat().st_size),
                "sha256": _sha256(selection_path),
            },
        },
        "payload_hash": stable_hash(payload),
    }
    record["case_manifest_path"] = _display_path(case_dir / "manifest.json")
    _write_json(case_dir / "manifest.json", record)
    return record


def _readme(
    *,
    output_root: Path,
    source_index: Mapping[str, Any],
    split_summary: Mapping[str, Any],
    source_manifest_hash: str,
) -> str:
    splits = split_summary.get("splits", {})
    dev = splits.get("dev", {})
    test = splits.get("test", {})
    try:
        output_display = output_root.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        output_display = str(output_root.resolve())
    return f"""# Task B raw-factor frozen rolling benchmark

This directory is the Task B raw-factor, no-monthly-LLM-call experiment for
the `{source_index.get('track')}` track.  Build a second directory with
`--track semantic` to obtain the matched semantic view.

## Design

- Source: `{source_index.get('source_returns')}` (`factor_id` monthly returns,
  not the 13-theme table), region `{source_index.get('region')}`.
- Windows: 2Y/3Y/5Y = 24/36/60 months.
- Each case contains one frozen-code payload and a 12-step one-month rolling
  evaluator stream. The same source code is executed locally at every origin;
  `llm_calls_during_rolling` is zero.
- Each time block has 8 distinct 38-factor subsets, 5 distinct 76-factor
  subsets and one 153-factor census. The subset is identical across 2Y/3Y/5Y
  inside that block.
- History + the 12 hidden target months are followed by at least 12 untouched
  months before the next block. The longest 5Y interval therefore requires
  84-month anchor spacing. The scheduler found {source_index.get('temporal_block_count')}
  blocks: {', '.join(source_index.get('anchors', []))}.
- Exact 2Y histories have only 12 labelled transitions after the 12-month
  feature warm-up, so their inner split is 6 train cutoffs + 6 validation
  cutoffs. 3Y/5Y use 12 + 12.

## Files

- `manifest.jsonl`: unsplit index followed by {source_index.get('case_count')} case records.
- `selection_plan.jsonl`: private, reproducible factor-subset plan (98 rows for
  the full design).
- `cases/<block>/<case_id>/payload.json`: public input for one LLM code
  generation call.
- On the `semantic` track, `payload.json.factor_cards` is keyed by the masked
  `Sxx` ids and contains only neutral name, definition and formula fields.
  Explicit direction, theme/group, significance and citation/provenance fields
  are intentionally absent.  Card names/formulas may still describe an
  economic category or construction framework; they are not sign/strength
  priors.  The `numeric_only` payload has no cards.
- `cases/<block>/<case_id>/rolling_truth.jsonl`: evaluator-only histories and
  hidden one-month targets; never send it to the model server.
- `splits/dev/manifest.jsonl` and `splits/test/manifest.jsonl`: chronological
  block manifests. Case directories are reused by reference; large payloads
  are not copied. Dev has {dev.get('case_count')} cases and test has
  {test.get('case_count')} cases.

The canonical manifest SHA-256 is `{source_manifest_hash}`.

## One case: generate once, roll for twelve months

```bash
python3 task_b/code/scripts/run_task_b_codegen.py \\
  --payload {output_display}/cases/block_01/<case_id>/payload.json \\
  --output {output_display}/cases/block_01/<case_id>/revision_audit.json \\
  --max-revisions 0

python3 task_b/code/scripts/run_task_b_rolling.py \\
  --frozen-response {output_display}/cases/block_01/<case_id>/revision_audit.json \\
  --rolling-truth {output_display}/cases/block_01/<case_id>/rolling_truth.jsonl \\
  --output-dir {output_display}/cases/block_01/<case_id>/rolling_result
```

Use the split manifests to schedule only dev or test cases. The hidden truth is
required only on the trusted evaluator host.
"""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--returns",
        default="data/processed/jkp/2025-12/factor_returns_monthly_vw_cap.csv",
        type=Path,
        help="raw monthly factor-return CSV (must contain factor_id)",
    )
    parser.add_argument(
        "--factor-cards",
        default="data/processed/jkp/2025-12/factor_cards_enriched.jsonl",
        type=Path,
        help="private metadata used for subset sampling and (semantic) cards",
    )
    parser.add_argument(
        "--output-root",
        default="data/processed/task_b_usa_frozen_raw_rolling_294",
        type=Path,
    )
    parser.add_argument("--region", default="USA")
    parser.add_argument(
        "--track",
        choices=ALL_TRACKS,
        default="numeric_only",
        help="public information track; semantic_numeric is accepted as a legacy alias",
    )
    parser.add_argument("--windows", nargs="+", choices=tuple(WINDOW_MONTHS), default=list(DEFAULT_WINDOWS))
    parser.add_argument("--target-months", type=int, default=DEFAULT_TARGET_MONTHS)
    parser.add_argument("--gap-months", type=int, default=DEFAULT_GAP_MONTHS)
    parser.add_argument(
        "--anchor-policy",
        choices=("earliest_with_latest", "earliest_regular", "latest_regular"),
        default="earliest_with_latest",
    )
    parser.add_argument("--dev-blocks", type=int, default=2)
    parser.add_argument("--master-seed", type=int, default=DEFAULT_MASTER_SEED)
    parser.add_argument("--subsets-38", type=int, default=8)
    parser.add_argument("--subsets-76", type=int, default=5)
    parser.add_argument("--subsets-153", type=int, default=1)
    parser.add_argument("--returns-in-percent", action="store_true")
    parser.add_argument("--max-cases", type=int, help="debug limit; omits the 294-case assertion")
    parser.add_argument(
        "--expected-case-count",
        type=int,
        default=EXPECTED_CASE_COUNT,
        help="expected full-design count; use 0 to disable the assertion",
    )
    args = parser.parse_args(argv)
    expected = None if args.expected_case_count == 0 else int(args.expected_case_count)
    if args.max_cases is not None and args.max_cases < 1:
        parser.error("--max-cases must be positive")
    try:
        result = build_dataset(
            returns_path=args.returns.resolve(),
            cards_path=args.factor_cards.resolve() if args.factor_cards else None,
            output_root=args.output_root.resolve(),
            region=str(args.region).upper(),
            windows=tuple(args.windows),
            target_months=int(args.target_months),
            gap_months=int(args.gap_months),
            anchor_policy=str(args.anchor_policy),
            dev_blocks=int(args.dev_blocks),
            master_seed=int(args.master_seed),
            subsets_by_size={38: int(args.subsets_38), 76: int(args.subsets_76), 153: int(args.subsets_153)},
            returns_in_percent=bool(args.returns_in_percent),
            max_cases=args.max_cases,
            expected_case_count=expected,
            track=str(args.track),
        )
    except (FileExistsError, FileNotFoundError, OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

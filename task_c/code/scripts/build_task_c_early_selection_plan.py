#!/usr/bin/env python3
"""Extend the Task C selection plan with early USA anchors.

The existing plan is kept byte-for-byte for the canonical four anchors.  For
new early anchors, a factor is eligible only when it has at least
``min_nonnull_stocks`` observations in every month of the longest requested
history plus hidden-target interval.  The resulting factor universe is frozen
before case generation; unavailable sizes (153 here) are omitted rather than
filled or silently imputed at dataset construction time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PANEL = ROOT / "data/raw/wrds/wgpvj1vbztoqbloo (1).parquet"
DEFAULT_CARDS = ROOT / "data/processed/jkp/2025-12/factor_cards_enriched.jsonl"
DEFAULT_BASE_PLAN = ROOT / "data/processed/factor_window_selection_plan.jsonl"
DEFAULT_OUTPUT = ROOT / "data/processed/task_c_early_selection_plan_246.jsonl"
WINDOW_MONTHS = {"2Y": 24, "3Y": 36, "5Y": 60}


def _period(value: str) -> pd.Period:
    return pd.Period(value, freq="M")


def _stable_seed(*parts: object) -> int:
    material = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _cards(path: Path) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[str(row.get("factor_id"))] = row
    return out


def _theme(cards: Mapping[str, Mapping[str, Any]], factor_id: str) -> str:
    card = cards.get(factor_id, {})
    return str(card.get("jkp_group_for_selection_only") or card.get("theme_for_selection") or "_unlabelled")


def _sample_subset(
    eligible: Sequence[str],
    size: int,
    cards: Mapping[str, Mapping[str, Any]],
    *,
    seed: int,
) -> list[str]:
    if len(eligible) < size:
        raise ValueError(f"only {len(eligible)} eligible factors for requested size {size}")
    # Preserve the existing plan's theme-stratified spirit while allowing an
    # anchor-specific universe.  Quotas are proportional to eligible themes;
    # deterministic seeded sampling makes the plan reproducible.
    by_theme: dict[str, list[str]] = {}
    for factor_id in sorted(eligible):
        by_theme.setdefault(_theme(cards, factor_id), []).append(factor_id)
    themes = sorted(by_theme)
    raw = {theme: size * len(by_theme[theme]) / len(eligible) for theme in themes}
    quotas = {theme: min(len(by_theme[theme]), int(raw[theme])) for theme in themes}
    remainder = size - sum(quotas.values())
    while remainder:
        candidates = [theme for theme in themes if quotas[theme] < len(by_theme[theme])]
        candidates.sort(key=lambda theme: (-(raw[theme] - int(raw[theme])), _stable_seed(seed, theme)))
        for theme in candidates:
            if remainder == 0:
                break
            quotas[theme] += 1
            remainder -= 1
    selected: list[str] = []
    for theme in themes:
        picker = random.Random(_stable_seed(seed, "theme", theme))
        selected.extend(picker.sample(by_theme[theme], quotas[theme]))
    random.Random(_stable_seed(seed, "order")).shuffle(selected)
    if len(selected) != size or len(set(selected)) != size:
        raise RuntimeError("failed to construct a unique factor subset")
    return selected


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_plan(
    *,
    panel: Path,
    cards_path: Path,
    base_plan: Path,
    output: Path,
    early_anchors: Sequence[str],
    windows: Sequence[str] = ("2Y", "3Y", "5Y"),
    target_months: int = 12,
    min_nonnull_stocks: int = 30,
    subsets_by_size: Mapping[int, int] = {38: 8, 76: 5, 153: 1},
) -> dict[str, Any]:
    rows = _read_rows(base_plan)
    existing_anchors = {str(row.get("anchor_month")) for row in rows if row.get("record_type", "selection") == "selection"}
    anchors = tuple(str(_period(value)) for value in early_anchors)
    if len(set(anchors)) != len(anchors) or any(anchor in existing_anchors for anchor in anchors):
        raise ValueError("early anchors must be unique and absent from the base plan")
    if not anchors:
        raise ValueError("at least one early anchor is required")
    cards = _cards(cards_path)
    factor_ids = sorted(cards)
    max_history = max(WINDOW_MONTHS[str(window)] for window in windows)
    bounds = []
    for anchor in anchors:
        period = _period(anchor)
        # Match build_task_c_dataset.py exactly: the history starts
        # at ``period - max_history`` (not ``+ 1``), so a 5Y protocol with a
        # 12-month validation segment uses a 73-month inclusive interval.
        bounds.append(((period - max_history).to_timestamp("M"), (period + target_months).to_timestamp("M")))
    start = min(value[0] for value in bounds).date()
    end = max(value[1] for value in bounds).date()
    table = pq.read_table(
        panel,
        columns=["eom", "id", "excntry", "me", "common", "exch_main", "primary_sec", "obs_main", *factor_ids],
        filters=[
            ("excntry", "==", "USA"),
            ("eom", ">=", start),
            ("eom", "<=", end),
            ("common", "==", 1.0),
            ("exch_main", "==", 1.0),
            ("primary_sec", "==", 1.0),
            ("obs_main", "==", 1.0),
        ],
    )
    frame = table.to_pandas()
    frame["me"] = pd.to_numeric(frame["me"], errors="coerce")
    frame = frame.loc[frame["me"].gt(0)].copy()
    frame["month"] = pd.to_datetime(frame["eom"]).dt.to_period("M")
    generated: list[dict[str, Any]] = []
    coverage: dict[str, Any] = {}
    for anchor in sorted(anchors, key=_period):
        period = _period(anchor)
        months = pd.period_range(period - max_history, period + target_months, freq="M")
        scoped = frame.loc[frame["month"].isin(months)]
        counts = scoped.groupby("month")[factor_ids].count().reindex(months)
        eligible = [factor for factor in factor_ids if bool((counts[factor] >= min_nonnull_stocks).all())]
        available_sizes = [size for size in (38, 76, 153) if len(eligible) >= size and int(subsets_by_size.get(size, 0)) > 0]
        if not available_sizes:
            raise ValueError(f"{anchor} has only {len(eligible)} eligible factors")
        coverage[anchor] = {
            "interval": [str(months[0]), str(months[-1])],
            "eligible_factor_count": len(eligible),
            "min_nonnull_stocks": int(min_nonnull_stocks),
            "available_sizes": available_sizes,
            "rows_by_month_min": int(scoped.groupby("month").size().reindex(months).min()),
        }
        for window in windows:
            for size in available_sizes:
                selected_count = int(subsets_by_size[size])
                for subset_index in range(1, selected_count + 1):
                    selected = _sample_subset(
                        eligible,
                        size,
                        cards,
                        seed=_stable_seed("task_c_early", anchor, size, subset_index),
                    )
                    generated.append({
                        "record_type": "selection",
                        "window": str(window),
                        "anchor_month": anchor,
                        "lookback_months": int(WINDOW_MONTHS[str(window)]),
                        "subset_id": f"{window}_{anchor.replace('-', '')}_{size}_s{subset_index:02d}",
                        "subset_index": subset_index,
                        "factor_count": size,
                        "factor_ids": selected,
                        "factor_ids_sorted": sorted(selected),
                        "theme_counts": {
                            theme: sum(_theme(cards, factor) == theme for factor in selected)
                            for theme in sorted({_theme(cards, factor) for factor in selected})
                        },
                        "seed": _stable_seed("task_c_early", anchor, size, subset_index),
                        "sampling_scope": "early_anchor_eligible_factors",
                        "source_windows": list(windows),
                        "deduplicated_for_task_c": False,
                        "same_subset_across_windows": True,
                        "eligibility": {
                            "min_nonnull_stocks": int(min_nonnull_stocks),
                            "eligible_factor_count": len(eligible),
                        },
                    })
    combined = rows + generated
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n" for row in combined),
        encoding="utf-8",
    )
    return {
        "output": str(output),
        "base_rows": len(rows),
        "generated_rows": len(generated),
        "total_rows": len(combined),
        "coverage": coverage,
        "anchors": sorted(existing_anchors | set(anchors), key=_period),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--factor-cards", type=Path, default=DEFAULT_CARDS)
    parser.add_argument("--base-plan", type=Path, default=DEFAULT_BASE_PLAN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--early-anchors", nargs="+", default=["1960-12", "1970-12"])
    parser.add_argument("--min-nonnull-stocks", type=int, default=30)
    args = parser.parse_args()
    result = build_plan(
        panel=args.panel.resolve(), cards_path=args.factor_cards.resolve(), base_plan=args.base_plan.resolve(),
        output=args.output.resolve(), early_anchors=args.early_anchors,
        min_nonnull_stocks=int(args.min_nonnull_stocks),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

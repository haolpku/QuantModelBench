#!/usr/bin/env python3
"""Audit a generated Task B raw-factor frozen rolling directory.

The audit is local and read-only.  It checks the case/selection counts, the
purged time-block rule, public-payload leakage guards, rolling truth alignment,
artifact hashes, and dev/test disjointness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[1]
ROOT = CODE_ROOT.parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from factoropt.task_b import episode_from_truth_record  # noqa: E402
from factoropt.task_b_frozen import validate_frozen_payload  # noqa: E402
from factoropt.semantic import normalize_track  # noqa: E402


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _case_path(row: Mapping[str, Any]) -> Path:
    value = row.get("path") or row.get("dataset_id") or row.get("case_manifest_path")
    if not value:
        raise ValueError("case row has no path")
    return _resolve(str(value))


def _contains_raw_factor_token(value: Any, factor_id: str) -> bool:
    """Check an identifier token, not an arbitrary natural-language substring."""

    text = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
    escaped = re.escape(str(factor_id))
    return bool(
        re.search(
            rf"(?<![A-Za-z0-9]){escaped}(?=(?:_|[^A-Za-z0-9]|$))",
            text,
            flags=re.IGNORECASE,
        )
    )


def _raw_factor_pattern(factor_ids: Sequence[str]) -> re.Pattern[str] | None:
    ids = sorted({str(value) for value in factor_ids if str(value).strip()}, key=len, reverse=True)
    if not ids:
        return None
    return re.compile(
        r"(?<![A-Za-z0-9])(" + "|".join(re.escape(value) for value in ids) + r")"
        r"(?=(?:_|[^A-Za-z0-9]|$))",
        flags=re.IGNORECASE,
    )


def _audit_truth(case_dir: Path, case: Mapping[str, Any]) -> None:
    truth_path = case_dir / "rolling_truth.jsonl"
    if not truth_path.is_file():
        raise ValueError(f"missing truth: {truth_path}")
    rows = _read_jsonl(truth_path)
    expected_steps = list(range(1, int(case.get("sequence_steps", 0)) + 1))
    observed = [int(row.get("sequence_step", 0)) for row in rows]
    if observed != expected_steps:
        raise ValueError(f"non-contiguous truth steps in {truth_path}: {observed}")
    if not rows:
        raise ValueError(f"empty truth: {truth_path}")
    sequence_ids = {str(row.get("sequence_id", "")) for row in rows}
    if sequence_ids != {str(case.get("case_id"))}:
        raise ValueError(f"truth sequence ID mismatch in {truth_path}")
    maps = []
    episodes = []
    for row in rows:
        if row.get("task_variant") != "factor_return_code_frozen_rolling_12step":
            raise ValueError(f"unsupported truth task variant in {truth_path}")
        maps.append({str(k): str(v) for k, v in (row.get("masked_to_original") or {}).items()})
        episodes.append(episode_from_truth_record(row))
    first_map = maps[0]
    for index, (episode, mapping) in enumerate(zip(episodes, maps), start=1):
        if set(mapping.values()) != set(map(str, episode.sleeve_ids)):
            raise ValueError(f"truth ID map does not cover factors at step {index}: {truth_path}")
        if mapping != first_map:
            raise ValueError(f"truth ID map changed at step {index}: {truth_path}")
        if episode.horizon_months != 1:
            raise ValueError(f"truth horizon is not one month: {truth_path}")
        if len(episode.history.index) != int(case["lookback_months"]):
            raise ValueError(f"truth history length mismatch: {truth_path}")
    for index, (left, right) in enumerate(zip(episodes, episodes[1:]), start=2):
        if list(left.history.index[1:]) != list(right.history.index[:-1]):
            raise ValueError(f"history does not roll at step {index}: {truth_path}")
        if pd.Timestamp(left.future.index[0]) != pd.Timestamp(right.history.index[-1]):
            raise ValueError(f"settled target is not appended at step {index}: {truth_path}")
    artifact = case.get("artifacts", {}).get("rolling_truth", {})
    if isinstance(artifact, Mapping) and artifact.get("sha256") and _sha256(truth_path) != artifact["sha256"]:
        raise ValueError(f"truth hash mismatch: {truth_path}")


def audit_dataset(root: Path, *, expected_cases: int | None = 294, min_gap: int = 12) -> dict[str, Any]:
    manifest = root / "manifest.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    rows = _read_jsonl(manifest)
    indices = [row for row in rows if row.get("record_type") == "index"]
    cases = [row for row in rows if row.get("record_type") == "case"]
    if len(indices) != 1:
        raise ValueError(f"expected one index row, found {len(indices)}")
    index = indices[0]
    if expected_cases is not None and len(cases) != expected_cases:
        raise ValueError(f"expected {expected_cases} cases, found {len(cases)}")
    if int(index.get("case_count", len(cases))) != len(cases):
        raise ValueError("index case_count disagrees with manifest")
    if str(index.get("factor_source_type", "raw_factor")) == "theme":
        raise ValueError("manifest claims a theme source")
    counts = Counter(
        (str(row.get("block_id")), str(row.get("window")), int(row.get("sample_size", 0)))
        for row in cases
    )
    expected_per_block = {(window, size): (8 if size == 38 else 5 if size == 76 else 1) for window in ("2Y", "3Y", "5Y") for size in (38, 76, 153)}
    block_ids = sorted({str(row.get("block_id")) for row in cases})
    for block in block_ids:
        for (window, size), expected in expected_per_block.items():
            if counts[(block, window, size)] != expected:
                raise ValueError(f"count mismatch for {block}/{window}/{size}: {counts[(block, window, size)]}")
    subset_fingerprints: dict[tuple[str, int], tuple[str, ...]] = {}
    for row in cases:
        key = (str(row.get("block_id")), int(row.get("subset_index", 0)))
        # A subset index is only comparable within factor count; include it in
        # the key below when checking each block/window pair.
        key = (f"{key[0]}|{row.get('sample_size')}", key[1])
        fingerprint = tuple(sorted(str(value) for value in row.get("selected_factor_ids", [])))
        previous = subset_fingerprints.get(key)
        if previous is not None and previous != fingerprint:
            raise ValueError(f"factor subset changed across windows: {key}")
        subset_fingerprints[key] = fingerprint

        case_dir = _case_path(row)
        payload_path = case_dir / "payload.json"
        if not payload_path.is_file():
            raise ValueError(f"missing payload: {payload_path}")
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        errors = validate_frozen_payload(payload)
        if errors:
            raise ValueError(f"invalid payload {payload_path}: {errors[:3]}")
        if payload.get("case_id") != row.get("case_id"):
            raise ValueError(f"payload case_id mismatch: {payload_path}")
        payload_text = payload_path.read_text(encoding="utf-8")
        if re.search(r"(?:19|20)\d{2}-(?:0[1-9]|1[0-2])", payload_text):
            raise ValueError(f"public payload contains a calendar month: {payload_path}")
        try:
            payload_track = normalize_track(payload.get("track", "numeric_only"))
        except ValueError:
            payload_track = "numeric_only"
        if payload_track == "numeric_only":
            for original_id in row.get("selected_factor_ids", []):
                if f'"{str(original_id)}"' in payload_text:
                    raise ValueError(f"public payload contains original factor ID {original_id}: {payload_path}")
        else:
            # Semantic cards are allowed to contain human-readable names and
            # definitions (for example, ``age`` inside ``Average``).  Check
            # the non-card envelope and the technical card fields using whole
            # identifier tokens so natural language is not a false positive.
            envelope = dict(payload)
            cards = envelope.pop("factor_cards", [])
            envelope_text = json.dumps(envelope, ensure_ascii=False)
            selected_ids = [str(value) for value in row.get("selected_factor_ids", [])]
            token_pattern = _raw_factor_pattern(selected_ids)
            envelope_match = token_pattern.search(envelope_text) if token_pattern else None
            if envelope_match:
                raise ValueError(
                    f"public payload contains original factor ID {envelope_match.group(1)}: {payload_path}"
                )
            for card in cards if isinstance(cards, list) else []:
                if not isinstance(card, Mapping):
                    continue
                technical = {
                    key: value
                    for key, value in card.items()
                    if key not in {"neutral_name", "neutral_description"}
                }
                technical_text = json.dumps(technical, ensure_ascii=False)
                card_match = token_pattern.search(technical_text) if token_pattern else None
                if card_match:
                    raise ValueError(
                        f"public semantic card contains original factor ID {card_match.group(1)}: {payload_path}"
                    )
        artifact = row.get("artifacts", {}).get("payload", {})
        if isinstance(artifact, Mapping) and artifact.get("sha256") and _sha256(payload_path) != artifact["sha256"]:
            raise ValueError(f"payload hash mismatch: {payload_path}")
        _audit_truth(case_dir, row)

    purge = index.get("purge_policy", {})
    gap_rows = purge.get("gap_audit_rows", []) if isinstance(purge, Mapping) else []
    if not gap_rows:
        raise ValueError("manifest has no gap audit")
    if any(not bool(row.get("passes")) or int(row.get("gap_months", -1)) < min_gap for row in gap_rows):
        raise ValueError("purged gap audit failed")

    split_counts: dict[str, int] = {}
    split_case_ids: dict[str, set[str]] = {}
    for split in ("dev", "test"):
        path = root / "splits" / split / "manifest.jsonl"
        if not path.is_file():
            raise ValueError(f"missing split manifest: {path}")
        split_rows = _read_jsonl(path)
        selected = [row for row in split_rows if row.get("record_type") == "case"]
        split_counts[split] = len(selected)
        split_case_ids[split] = {str(row.get("case_id")) for row in selected}
    if split_case_ids["dev"] & split_case_ids["test"]:
        raise ValueError("dev/test case overlap")
    if split_case_ids["dev"] | split_case_ids["test"] != {str(row.get("case_id")) for row in cases}:
        raise ValueError("dev/test split does not cover canonical cases")
    return {
        "audit_ok": True,
        "case_count": len(cases),
        "block_count": len(block_ids),
        "blocks": block_ids,
        "case_counts_by_window_factor": {
            f"{window}/{size}": sum(value for (block, w, s), value in counts.items() if w == window and s == size)
            for window in ("2Y", "3Y", "5Y")
            for size in (38, 76, 153)
        },
        "split_counts": split_counts,
        "minimum_gap_months": min(int(row["gap_months"]) for row in gap_rows),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/processed/task_b_usa_frozen_raw_rolling_294", type=Path)
    parser.add_argument("--expected-cases", type=int, default=294)
    parser.add_argument("--min-gap-months", type=int, default=12)
    args = parser.parse_args(argv)
    try:
        result = audit_dataset(
            _resolve(args.dataset),
            expected_cases=None if args.expected_cases == 0 else args.expected_cases,
            min_gap=args.min_gap_months,
        )
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

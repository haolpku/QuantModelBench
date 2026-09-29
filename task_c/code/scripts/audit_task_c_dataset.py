#!/usr/bin/env python3
"""Audit a generated Task C frozen stock rolling directory.

The audit is read-only.  It validates public payload contracts and artifact
hashes, confirms that evaluator-only maps/truth are not mounted in the public
payload, checks each manifest-declared rolling schedule, and verifies
chronological dev/test block disjointness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[1]
ROOT = CODE_ROOT.parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from factoropt.task_c import materialize_split, validate_public_payload  # noqa: E402
from factoropt.semantic import normalize_track  # noqa: E402


# Hashes are present in artifact descriptors, so scanning every six-digit
# substring would create false positives.  The public builder never emits
# compact calendar tokens; the structural payload validator separately checks
# metadata/paths.  Here we only reject human-readable date forms.
_CALENDAR_RE = re.compile(
    r"(?<!\d)(?:19|20)\d{2}[-_/]\d{1,2}(?:[-_/]\d{1,2})?(?!\d)"
)


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not an object")
    return value


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


def _case_dir(root: Path, row: Mapping[str, Any]) -> Path:
    raw = row.get("path") or row.get("dataset_id")
    if not raw:
        raise ValueError("case has no path")
    path = Path(str(raw))
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _check_artifact(
    case_dir: Path,
    descriptor: Mapping[str, Any],
    label: str,
    *,
    dataset_root: Path | None = None,
) -> Path:
    raw = str(descriptor.get("path") or "")
    if not raw:
        raise ValueError(f"{label} has unsafe path")
    raw_path = Path(raw)
    if raw_path.is_absolute():
        path = raw_path.resolve()
        if dataset_root is None or dataset_root.resolve() not in path.parents:
            raise ValueError(f"{label} has unsafe absolute path")
    elif ".." in raw_path.parts:
        raise ValueError(f"{label} has unsafe traversal path")
    else:
        case_candidate = (case_dir / raw_path).resolve()
        root_candidate = (ROOT / raw_path).resolve()
        if case_candidate.is_file():
            path = case_candidate
            if case_dir.resolve() not in path.parents:
                raise ValueError(f"{label} escapes case directory")
        elif root_candidate.is_file() and (dataset_root is None or dataset_root.resolve() in root_candidate.parents):
            # Case manifests use repository-relative display paths for their
            # own artifacts; public payload descriptors remain case-relative.
            path = root_candidate
        else:
            path = case_candidate
    if not path.is_file():
        raise ValueError(f"missing {label}: {path}")
    if descriptor.get("sha256") and _sha256(path) != str(descriptor["sha256"]):
        raise ValueError(f"{label} sha256 mismatch: {path}")
    if descriptor.get("bytes") is not None and int(descriptor["bytes"]) != path.stat().st_size:
        raise ValueError(f"{label} byte count mismatch: {path}")
    return path


def _contains_raw_feature_token(value: Any, feature_id: str) -> bool:
    """Match a complete identifier token rather than a word substring."""

    text = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
    escaped = re.escape(str(feature_id))
    return bool(
        re.search(
            rf"(?<![A-Za-z0-9]){escaped}(?=(?:_|[^A-Za-z0-9]|$))",
            text,
            flags=re.IGNORECASE,
        )
    )


def _raw_feature_pattern(feature_ids: Sequence[str]) -> re.Pattern[str] | None:
    ids = sorted({str(value) for value in feature_ids if str(value).strip()}, key=len, reverse=True)
    if not ids:
        return None
    return re.compile(
        r"(?<![A-Za-z0-9])(" + "|".join(re.escape(value) for value in ids) + r")"
        r"(?=(?:_|[^A-Za-z0-9]|$))",
        flags=re.IGNORECASE,
    )


def _audit_case(root: Path, row: Mapping[str, Any], *, target_steps: int) -> None:
    case_dir = _case_dir(root, row)
    payload_path = case_dir / "payload.json"
    if not payload_path.is_file():
        raise ValueError(f"missing payload: {payload_path}")
    payload_text = payload_path.read_text(encoding="utf-8")
    payload = json.loads(payload_text)
    errors = validate_public_payload(payload)
    if errors:
        raise ValueError(f"invalid public payload {payload_path}: {errors[:5]}")
    if payload.get("case_id") != row.get("case_id"):
        raise ValueError(f"payload/case ID mismatch: {payload_path}")
    if _CALENDAR_RE.search(payload_text):
        raise ValueError(f"public payload contains a calendar literal: {payload_path}")
    try:
        payload_track = normalize_track(payload.get("track", "numeric_only"))
    except ValueError:
        payload_track = "numeric_only"
    if payload_track == "numeric_only":
        if any(str(feature) in payload_text for feature in row.get("selected_feature_ids", [])):
            raise ValueError(f"public payload contains an original feature name: {payload_path}")
    else:
        # Neutral names/definitions may legitimately contain a raw token as a
        # normal English substring (e.g. ``age`` in ``Average``).  Exclude the
        # card text from the broad check, then enforce complete-token masking
        # for the envelope and technical formula fields.
        envelope = dict(payload)
        cards = envelope.pop("factor_cards", [])
        envelope_text = json.dumps(envelope, ensure_ascii=False)
        selected_features = [str(value) for value in row.get("selected_feature_ids", [])]
        token_pattern = _raw_feature_pattern(selected_features)
        envelope_match = token_pattern.search(envelope_text) if token_pattern else None
        if envelope_match:
            raise ValueError(
                f"public payload contains an original feature name {envelope_match.group(1)}: {payload_path}"
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
                    f"public semantic card contains an original feature name {card_match.group(1)}: {payload_path}"
                )
    if payload.get("predict", {}).get("rows") not in (None, []):
        raise ValueError(f"predict rows are present in public payload: {payload_path}")

    artifacts = row.get("artifacts", {})
    if not isinstance(artifacts, Mapping):
        raise ValueError(f"case has no artifact manifest: {case_dir}")
    payload_descriptor = artifacts.get("payload")
    if isinstance(payload_descriptor, Mapping):
        _check_artifact(case_dir, payload_descriptor, "payload", dataset_root=root)
    train_descriptor = payload.get("train", {}).get("artifact")
    valid_descriptor = payload.get("validation", {}).get("artifact")
    if not isinstance(train_descriptor, Mapping) or not isinstance(valid_descriptor, Mapping):
        raise ValueError(f"public payload must use train/validation artifacts: {payload_path}")
    train_path = _check_artifact(case_dir, train_descriptor, "train artifact")
    valid_path = _check_artifact(case_dir, valid_descriptor, "validation artifact")
    train = materialize_split(payload, "train", include_label=True, artifact_root=case_dir)
    valid = materialize_split(payload, "validation", include_label=True, artifact_root=case_dir)
    features = [str(value) for value in payload["feature_columns"]]
    expected_train = {"id", "time_key", "label", *features}
    if set(train.columns) != expected_train or set(valid.columns) != expected_train:
        raise ValueError(f"public artifact columns do not match payload: {case_dir}")
    if not train["id"].astype(str).str.startswith("I").all() or not valid["id"].astype(str).str.startswith("I").all():
        raise ValueError(f"public IDs are not masked: {case_dir}")
    if not all(re.fullmatch(r"f\d{3}", feature) for feature in features):
        raise ValueError(f"public feature names are not anonymous: {case_dir}")
    if int(train_descriptor.get("rows", len(train))) != len(train) or int(valid_descriptor.get("rows", len(valid))) != len(valid):
        raise ValueError(f"artifact row count mismatch: {case_dir}")
    window_meta = row.get("window", {})
    if not isinstance(window_meta, Mapping):
        raise ValueError(f"case window metadata is missing: {case_dir}")
    train_months = int(window_meta.get("train_months", 0))
    validation_months = int(window_meta.get("validation_months", 0))
    if train_months < 1 or validation_months < 1:
        raise ValueError(f"case has invalid train/validation window: {case_dir}")
    history_months = int(window_meta.get("history_months", train_months + validation_months))
    if history_months != train_months + validation_months:
        raise ValueError(f"case history window does not equal train+validation: {case_dir}")
    if row.get("history_window") and str(row.get("history_window")) != str(window_meta.get("name")):
        raise ValueError(f"case history window name mismatch: {case_dir}")

    truth_path = case_dir / "rolling_truth.jsonl"
    truth = _read_jsonl(truth_path)
    if [int(item.get("sequence_step", 0)) for item in truth] != list(range(1, target_steps + 1)):
        raise ValueError(f"rolling truth steps are not contiguous: {truth_path}")
    if any(str(item.get("sequence_id")) != str(row.get("case_id")) for item in truth):
        raise ValueError(f"rolling truth sequence mismatch: {truth_path}")
    dates = [pd.Period(str(item["formation_eom"]), freq="M") for item in truth]
    if dates != list(pd.period_range(dates[0], periods=target_steps, freq="M")):
        raise ValueError(f"rolling formation months are not contiguous: {truth_path}")
    if any(int(item.get("llm_calls", -1)) != 0 for item in truth):
        raise ValueError(f"rolling truth reports an LLM call: {truth_path}")
    truth_descriptor = artifacts.get("rolling_truth")
    if isinstance(truth_descriptor, Mapping):
        _check_artifact(case_dir, truth_descriptor, "rolling truth", dataset_root=root)

    id_map = _read_json(case_dir / "evaluator_id_map.json")
    feature_map = _read_json(case_dir / "evaluator_feature_map.json")
    if set(str(key) for key in id_map.get("original_to_masked", {})) != set(
        str(value) for value in id_map.get("masked_to_original", {}).values()
    ):
        raise ValueError(f"ID map is not bijective: {case_dir}")
    if set(str(key) for key in feature_map.get("original_to_masked", {})) != set(
        str(value) for value in feature_map.get("masked_to_original", {}).values()
    ):
        raise ValueError(f"feature map is not bijective: {case_dir}")
    public_text = payload_text + train_path.name + valid_path.name
    if "evaluator_id_map" in public_text or "evaluator_feature_map" in public_text:
        raise ValueError(f"public payload references evaluator maps: {case_dir}")


def audit_dataset(root: Path, *, expected_cases: int | None = None, min_gap_months: int = 12) -> dict[str, Any]:
    manifest_path = root / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    rows = _read_jsonl(manifest_path)
    index_rows = [row for row in rows if row.get("record_type") == "index"]
    cases = [row for row in rows if row.get("record_type") == "case"]
    if len(index_rows) != 1:
        raise ValueError(f"expected one index row, found {len(index_rows)}")
    index = index_rows[0]
    if expected_cases is not None and len(cases) != expected_cases:
        raise ValueError(f"expected {expected_cases} cases, found {len(cases)}")
    if int(index.get("case_count", len(cases))) != len(cases):
        raise ValueError("index case_count disagrees with manifest")
    target_steps = int(index.get("target_months_per_sequence", 12))
    windows = [str(value) for value in index.get("windows", [])]
    if not windows:
        windows = sorted({str(row.get("history_window", "fixed")) for row in cases})
    counts = Counter(
        (str(row.get("anchor_month")), str(row.get("history_window", "fixed")), int(row.get("feature_count", 0)))
        for row in cases
    )
    if expected_cases is not None:
        default_policy = {
            int(size): int(count)
            for size, count in dict(index.get("subset_policy", {38: 8, 76: 5, 153: 1})).items()
        }
        anchor_policy = index.get("anchor_subset_policy", {})
        for anchor in index.get("anchors", []):
            configured = anchor_policy.get(str(anchor), {}) if isinstance(anchor_policy, Mapping) else {}
            expected_per_anchor = {
                int(size): int(count)
                for size, count in dict(configured.get("subsets_by_size", default_policy)).items()
            }
            for window in windows:
                for size, expected in expected_per_anchor.items():
                    actual = counts[(str(anchor), str(window), size)]
                    if actual != expected:
                        raise ValueError(f"count mismatch for {anchor}/{window}/{size}: {actual}")
                actual_sizes = {
                    size
                    for (case_anchor, case_window, size), count in counts.items()
                    if case_anchor == str(anchor) and case_window == str(window) and count
                }
                if actual_sizes != set(expected_per_anchor):
                    raise ValueError(
                        f"feature-size mismatch for {anchor}/{window}: "
                        f"actual={sorted(actual_sizes)}, expected={sorted(expected_per_anchor)}"
                    )
    for row in cases:
        _audit_case(root, row, target_steps=target_steps)

    purge = index.get("purge_policy", {})
    gap_rows = purge.get("gap_audit_rows", []) if isinstance(purge, Mapping) else []
    if not gap_rows:
        raise ValueError("manifest has no purge gap audit")
    if any(not bool(item.get("passes")) or int(item.get("gap_months", -1)) < min_gap_months for item in gap_rows):
        raise ValueError("purge gap audit failed")

    split_ids: dict[str, set[str]] = {}
    split_counts: dict[str, int] = {}
    for split in ("dev", "test"):
        path = root / "splits" / split / "manifest.jsonl"
        if not path.is_file():
            raise ValueError(f"missing split manifest: {path}")
        selected = [row for row in _read_jsonl(path) if row.get("record_type") == "case"]
        split_ids[split] = {str(row.get("case_id")) for row in selected}
        split_counts[split] = len(selected)
    if split_ids["dev"] & split_ids["test"]:
        raise ValueError("dev/test case overlap")
    if split_ids["dev"] | split_ids["test"] != {str(row.get("case_id")) for row in cases}:
        raise ValueError("dev/test manifests do not cover canonical cases")
    return {
        "audit_ok": True,
        "case_count": len(cases),
        "block_count": len(set(str(row.get("block_id")) for row in cases)),
        "anchors": list(index.get("anchors", [])),
        "case_counts_by_feature_count": dict(sorted(Counter(str(row.get("feature_count")) for row in cases).items())),
        "split_counts": split_counts,
        "minimum_gap_months": min(int(item["gap_months"]) for item in gap_rows),
        "windows": windows,
        "window_counts": dict(sorted(Counter(str(row.get("history_window", "fixed")) for row in cases).items())),
        "target_steps": target_steps,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default="data/processed/task_c_frozen_raw_rolling_207")
    parser.add_argument("--expected-cases", type=int, default=0, help="fixed expected case count; use 0 to read manifest")
    parser.add_argument("--min-gap-months", type=int, default=12)
    args = parser.parse_args(argv)
    try:
        result = audit_dataset(
            _resolve(args.dataset),
            expected_cases=None if args.expected_cases == 0 else int(args.expected_cases),
            min_gap_months=int(args.min_gap_months),
        )
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

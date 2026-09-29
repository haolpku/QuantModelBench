#!/usr/bin/env python3
"""Build the Task B semantic view as an overlay on the numeric build.

The semantic track differs from ``numeric_only`` in exactly one per-case file:
``payload.json`` carries the factor cards and the semantic row order.
``rolling_truth.jsonl`` is byte-identical in both views, so this builder writes
only the payloads and leaves the truth to the base dataset, which
``shared/dataset_overlay.py`` resolves at run time.

The payload transform is lifted verbatim from
``.tmp/build_full_dev_frozen_track_datasets.py`` (the script that produced the
dev-only 84-case semantic build on 2026-09-20) so that a payload generated here
is byte-identical to one generated there.  ``--verify-against`` asserts exactly
that and is the gate that must pass before any new split is written.

Usage::

    # gate: reproduce the existing dev build without writing it
    python3 task_b/code/scripts/build_task_b_semantic_overlay.py \
        --split dev --output /tmp/gate \
        --verify-against data/processed/task_b_usa_frozen_raw_rolling_294_semantic

    # build a split into the overlay dataset
    python3 task_b/code/scripts/build_task_b_semantic_overlay.py \
        --split test --output data/processed/task_b_usa_frozen_raw_rolling_294_semantic
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from factoropt.semantic import semantic_cards_payload as b_cards_payload  # noqa: E402
from factoropt.task_b_frozen import validate_frozen_payload  # noqa: E402

DEFAULT_SOURCE = ROOT / "data/processed/task_b_usa_frozen_raw_rolling_294"
DEFAULT_OUTPUT = ROOT / "data/processed/task_b_usa_frozen_raw_rolling_294_semantic"
DEFAULT_CARDS = ROOT / "data/processed/jkp/2025-12/factor_cards_enriched.jsonl"


# --- helpers lifted verbatim from the 2026-09-20 staging builder -------------


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values),
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cards(cards_path: Path) -> dict[str, dict[str, Any]]:
    cards: dict[str, dict[str, Any]] = {}
    for line in cards_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        factor_id = str(row.get("factor_id") or "").strip()
        if factor_id:
            cards[factor_id] = row
    if not cards:
        raise ValueError(f"no factor cards found in {cards_path}")
    return cards


def masked_id_sort_key(value: Any) -> tuple[int, int, str]:
    text = str(value)
    if len(text) > 1 and text[0] in "Ss" and text[1:].isdigit():
        return (0, int(text[1:]), text)
    if text.isdigit():
        return (0, int(text), text)
    return (1, 0, text)


def reorder_b_rows(payload: dict[str, Any], masked_to_original: Mapping[str, str]) -> list[str]:
    rank = {
        masked: index
        for index, masked in enumerate(sorted(masked_to_original, key=masked_id_sort_key))
    }
    order: list[str] = []
    for split in ("train", "validation"):
        rows = payload[split]["rows"]
        rows.sort(
            key=lambda row: (str(row["time_key"]), -rank[str(row["id"])]),
            reverse=True,
        )
        for row in rows:
            masked = str(row["id"])
            if masked not in order:
                order.append(masked)
    return order


def semantic_b_payload(
    payload: dict[str, Any], truth: list[dict[str, Any]], cards: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    masked_to_original = {str(k): str(v) for k, v in truth[0]["masked_to_original"].items()}
    original_to_masked = {original: masked for masked, original in masked_to_original.items()}
    masked_order = reorder_b_rows(payload, masked_to_original)
    factor_ids = [masked_to_original[masked] for masked in masked_order]
    payload["track"] = "semantic"
    payload["factor_cards"] = b_cards_payload(cards, factor_ids, original_to_masked)
    payload.pop("semantic_card_schema_version", None)
    errors = validate_frozen_payload(payload)
    if errors:
        raise ValueError(f"Task B {payload.get('case_id')}: {errors[:8]}")
    return payload


# --- overlay-specific manifest handling -------------------------------------


def _rel(path: Path) -> str:
    """Repository-relative display path, matching the source manifests.

    A gate run writes to a scratch directory outside the repository; there is
    no relative form for that, and the gate writes no manifest anyway, so fall
    back to the absolute path instead of failing.
    """

    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def overlay_manifest_row(
    row: dict[str, Any],
    *,
    source_root: Path,
    output_root: Path,
    block: str,
    case_id: str,
    payload_file: Path,
) -> dict[str, Any]:
    """Rewrite a numeric case row for the overlay.

    Every substantive field is carried over untouched.  What changes is where
    the row says its files are: the payload moves into the overlay, while
    ``rolling_truth`` keeps pointing at the base build that still holds it.
    ``selection`` is dropped, matching the 2026-09-20 semantic convention.
    """

    source_case = source_root / "cases" / block / case_id
    target_case = output_root / "cases" / block / case_id
    result = dict(row)
    result["track"] = "semantic"
    result["dataset_id"] = _rel(target_case)
    result["path"] = _rel(target_case)
    result["case_manifest_path"] = None
    result["source_case_path"] = _rel(source_case)
    result["source_manifest"] = _rel(source_root / "manifest.jsonl")
    result["payload_hash"] = sha256(payload_file)

    artifacts = result.get("artifacts")
    if isinstance(artifacts, Mapping):
        transformed: dict[str, Any] = {}
        payload_descriptor = artifacts.get("payload")
        if isinstance(payload_descriptor, Mapping):
            descriptor = dict(payload_descriptor)
            descriptor["path"] = _rel(payload_file)
            descriptor["bytes"] = int(payload_file.stat().st_size)
            descriptor["sha256"] = result["payload_hash"]
            transformed["payload"] = descriptor
        truth_descriptor = artifacts.get("rolling_truth")
        if isinstance(truth_descriptor, Mapping):
            # The overlay does not hold the truth file; the descriptor points
            # at the base build, whose bytes the original hash was taken from.
            descriptor = dict(truth_descriptor)
            base_truth = source_case / "rolling_truth.jsonl"
            if not base_truth.is_file():
                raise FileNotFoundError(base_truth)
            descriptor["path"] = _rel(base_truth)
            transformed["rolling_truth"] = descriptor
        result["artifacts"] = transformed
    return result


def build_split(
    *,
    source_root: Path,
    output_root: Path,
    split: str,
    cards: Mapping[str, Mapping[str, Any]],
    write_manifest: bool,
) -> list[Path]:
    source_manifest = source_root / "splits" / split / "manifest.jsonl"
    if not source_manifest.is_file():
        raise SystemExit(f"missing source manifest: {source_manifest}")
    values = read_jsonl(source_manifest)
    rows = [row for row in values if row.get("record_type") == "case"]
    if not rows:
        raise SystemExit(f"no case records in {source_manifest}")

    written: list[Path] = []
    manifest_out: list[dict[str, Any]] = []
    for value in values:
        if value.get("record_type") != "case":
            header = dict(value)
            header["track"] = "semantic"
            header["dataset_root"] = _rel(output_root)
            header["source_manifest"] = _rel(source_root / "manifest.jsonl")
            plan = header.get("selection_plan")
            if isinstance(plan, Mapping) and plan.get("path"):
                plan = dict(plan)
                plan["path"] = _rel(source_root / "selection_plan.jsonl")
                header["selection_plan"] = plan
            manifest_out.append(header)
            continue

        block = str(value["block_id"])
        case_id = str(value["case_id"])
        source_case = source_root / "cases" / block / case_id
        payload = read_json(source_case / "payload.json")
        truth = read_jsonl(source_case / "rolling_truth.jsonl")
        payload = semantic_b_payload(payload, truth, cards)
        payload_file = output_root / "cases" / block / case_id / "payload.json"
        write_json(payload_file, payload)
        written.append(payload_file)
        manifest_out.append(
            overlay_manifest_row(
                value,
                source_root=source_root,
                output_root=output_root,
                block=block,
                case_id=case_id,
                payload_file=payload_file,
            )
        )

    if write_manifest:
        write_jsonl(output_root / "splits" / split / "manifest.jsonl", manifest_out)
    return written


def write_overlay_spec(output_root: Path, source_root: Path) -> None:
    write_json(
        output_root / "overlay.json",
        {
            "schema_version": "dataset_overlay.v1",
            "base_dataset": source_root.name,
            "overlay_files": ["payload.json"],
            "base_files": ["rolling_truth.jsonl", "manifest.json", "selection.json"],
            "note": (
                "The semantic view differs from the numeric build only in payload.json. "
                "Every other per-case file is byte-identical and is read from base_dataset."
            ),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="numeric build to derive from")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="overlay dataset root")
    parser.add_argument("--split", required=True, choices=("dev", "test"))
    parser.add_argument("--cards", type=Path, default=DEFAULT_CARDS)
    parser.add_argument(
        "--verify-against",
        type=Path,
        help="compare generated payloads byte-for-byte against this dataset and write no manifest",
    )
    args = parser.parse_args()

    source_root = args.source if args.source.is_absolute() else ROOT / args.source
    output_root = args.output if args.output.is_absolute() else ROOT / args.output
    cards_path = args.cards if args.cards.is_absolute() else ROOT / args.cards
    source_root = source_root.resolve()
    cards = load_cards(cards_path)

    verify_root = None
    if args.verify_against is not None:
        verify_root = (
            args.verify_against if args.verify_against.is_absolute() else ROOT / args.verify_against
        ).resolve()

    output_root.mkdir(parents=True, exist_ok=True)
    written = build_split(
        source_root=source_root,
        output_root=output_root,
        split=args.split,
        cards=cards,
        write_manifest=verify_root is None,
    )

    if verify_root is not None:
        same = 0
        differing: list[str] = []
        missing: list[str] = []
        for path in written:
            relative = path.relative_to(output_root)
            reference = verify_root / relative
            if not reference.is_file():
                missing.append(str(relative))
                continue
            if path.read_bytes() == reference.read_bytes():
                same += 1
            else:
                differing.append(str(relative))
        report = {
            "split": args.split,
            "generated": len(written),
            "identical": same,
            "differing": len(differing),
            "missing_in_reference": len(missing),
            "differing_sample": differing[:5],
            "missing_sample": missing[:5],
            "reference": str(verify_root),
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if (differing or missing) == [] and same == len(written) else 1

    write_overlay_spec(output_root, source_root)
    print(
        json.dumps(
            {
                "split": args.split,
                "payloads_written": len(written),
                "output_root": _rel(output_root),
                "base_dataset": source_root.name,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

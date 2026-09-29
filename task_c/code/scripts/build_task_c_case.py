#!/usr/bin/env python3
"""Build one masked Task C code-generation case from the supplied WRDS panel."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from factoropt.task_c import build_code_generation_payload, build_rolling_windows  # noqa: E402
from factoropt.semantic import ALL_TRACKS, normalize_track, semantic_card_payload  # noqa: E402
from factoropt.task_c_backtest import load_stock_panel  # noqa: E402


def _load_cards(path: Path | None, feature_ids: list[str]) -> dict[str, dict[str, Any]]:
    cards = {str(feature): {"factor_id": str(feature)} for feature in feature_ids}
    if path is None:
        return cards
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        rows = list(raw.values()) if isinstance(raw, Mapping) else raw
    else:
        rows = pd.read_csv(path).to_dict("records")
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, Mapping):
            continue
        feature = str(row.get("factor_id") or row.get("feature_id") or "").strip()
        if feature in cards:
            cards[feature].update(dict(row))
    return cards


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="public payload JSON")
    parser.add_argument("--formation-month", required=True, help="formation month YYYY-MM")
    parser.add_argument("--features", nargs="+", required=True)
    parser.add_argument("--train-months", type=int, default=120)
    parser.add_argument("--validation-months", type=int, default=12)
    parser.add_argument("--case-id", default="task_c_wrds_case")
    parser.add_argument(
        "--track",
        choices=ALL_TRACKS,
        default="numeric_only",
        help="public information track; semantic_numeric is a legacy alias",
    )
    parser.add_argument(
        "--factor-cards",
        type=Path,
        default=Path("data/processed/jkp/2025-12/factor_cards_enriched.jsonl"),
        help="private enriched factor cards used by the semantic track",
    )
    parser.add_argument("--truth-output", type=Path, help="evaluator-only masked ID mapping")
    parser.add_argument(
        "--manifest-output",
        type=Path,
        help="case manifest path (default: <payload-stem>_manifest.json)",
    )
    args = parser.parse_args(argv)
    track = normalize_track(args.track)
    formation = pd.Period(args.formation_month, freq="M")
    start = (formation - args.train_months - args.validation_months).to_timestamp("M")
    end = formation.to_timestamp("M")
    panel = load_stock_panel(args.panel, feature_columns=args.features, start=start, end=end)
    if "label" not in panel.columns:
        if "ret_exc_lead1m" not in panel.columns:
            raise ValueError("panel must contain ret_exc_lead1m")
        panel = panel.rename(columns={"ret_exc_lead1m": "label"})
    train, valid, predict, metadata = build_rolling_windows(
        panel,
        formation.to_timestamp("M"),
        feature_columns=args.features,
        train_months=args.train_months,
        valid_months=args.validation_months,
    )
    # Mask IDs in the public history. The evaluator keeps the mapping outside
    # the payload; the frozen source is later run on evaluator-owned IDs.
    # Include the hidden formation-month IDs in the evaluator map as well. The
    # map itself remains evaluator-only, but retaining these entries lets the
    # rolling runner present the same masked-ID representation at test time.
    identifiers = sorted(
        set(train["id"].astype(str))
        | set(valid["id"].astype(str))
        | set(predict["id"].astype(str))
    )
    mapping = {original: f"I{index:06d}" for index, original in enumerate(identifiers, start=1)}
    # The original panel is no longer needed after the windows are built.
    # Release it before constructing the public artifacts (the WRDS slice can
    # otherwise briefly occupy several hundred MB).
    del panel, predict
    gc.collect()
    raw_features = [str(value) for value in args.features]
    feature_map = {raw: f"f{index:03d}" for index, raw in enumerate(raw_features, start=1)}
    public_features = [feature_map[raw] for raw in raw_features]
    cards = _load_cards(args.factor_cards if track == "semantic" else None, raw_features)
    if track == "semantic":
        incomplete_cards = [
            raw
            for raw in raw_features
            if not str(cards[raw].get("neutral_name", "")).strip()
            or not any(
                str(cards[raw].get(field, "")).strip()
                for field in ("neutral_description", "formula_or_definition", "formula_latex")
            )
        ]
        if incomplete_cards:
            raise ValueError(
                "semantic track requires complete name/definition/formula cards; "
                f"missing for {incomplete_cards[:10]}"
            )
    train_public = train.rename(columns=feature_map)
    valid_public = valid.rename(columns=feature_map)
    train_public["id"] = train_public["id"].astype(str).map(mapping)
    valid_public["id"] = valid_public["id"].astype(str).map(mapping)
    # Keep the chat payload small. Public historical rows are mounted as
    # read-only JSONL artifacts and can be materialized by the local evaluator
    # with --artifact-root; no hidden/test data is written here.
    artifact_dir = args.output.parent / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    train_path = artifact_dir / f"{args.case_id}_train.jsonl"
    valid_path = artifact_dir / f"{args.case_id}_validation.jsonl"
    train_public.to_json(train_path, orient="records", lines=True, force_ascii=False)
    valid_public.to_json(valid_path, orient="records", lines=True, force_ascii=False)

    def digest(path: Path) -> str:
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    payload = build_code_generation_payload(
        train_public,
        valid_public,
        feature_columns=public_features,
        case_id=args.case_id,
        metadata={
            "train_months": args.train_months,
            "validation_months": args.validation_months,
            "seed": 1,
            "source": "WRDS JKP USA",
            "public_artifacts_only": True,
        },
        track=track,
        factor_cards=(
            [
                semantic_card_payload(card, feature_map[raw])
                for raw, card in cards.items()
            ]
            if track == "semantic"
            else None
        ),
        include_rows=False,
        train_artifact={
            "path": f"artifacts/{train_path.name}",
            "format": "jsonl",
            "rows": int(len(train_public)),
            "bytes": int(train_path.stat().st_size),
            "sha256": digest(train_path),
        },
        validation_artifact={
            "path": f"artifacts/{valid_path.name}",
            "format": "jsonl",
            "rows": int(len(valid_public)),
            "bytes": int(valid_path.stat().st_size),
            "sha256": digest(valid_path),
        },
    )
    payload["artifact_root"] = "."
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    manifest_path = args.manifest_output or args.output.with_name(f"{args.output.stem}_manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    def relative_path(path: Path) -> str:
        try:
            return path.resolve().relative_to(manifest_path.parent.resolve()).as_posix()
        except ValueError:
            return str(path.resolve())

    payload_bytes = args.output.read_bytes()
    manifest = {
        "schema_version": "task_c_case_manifest.v1",
        "case_id": args.case_id,
        "task": "stock_multifactor_model",
        "source_panel": str(args.panel),
        "formation_eom": formation.to_timestamp("M").strftime("%Y-%m-%d"),
        "features": [str(value) for value in args.features],
        "public_features": public_features,
        "track": track,
        "window": {
            "train_months": int(args.train_months),
            "validation_months": int(args.validation_months),
            "train_range": metadata["train_month_range"],
            "validation_range": metadata["validation_month_range"],
            "predict_month": metadata["predict_month"],
        },
        "public_payload": {
            "path": relative_path(args.output),
            "bytes": int(len(payload_bytes)),
            "sha256": hashlib.sha256(payload_bytes).hexdigest(),
        },
        "public_artifacts": [
            {
                "split": "train",
                "path": relative_path(train_path),
                "format": "jsonl",
                "rows": int(len(train_public)),
                "bytes": int(train_path.stat().st_size),
                "sha256": digest(train_path),
            },
            {
                "split": "validation",
                "path": relative_path(valid_path),
                "format": "jsonl",
                "rows": int(len(valid_public)),
                "bytes": int(valid_path.stat().st_size),
                "sha256": digest(valid_path),
            },
        ],
        "masking": {
            "method": "stable sorted identifier mapping",
            "identifier_count": int(len(mapping)),
            "evaluator_mapping_path": relative_path(args.truth_output) if args.truth_output else None,
            "mapping_must_not_be_sent_to_llm": True,
        },
        "point_in_time": {
            "future_labels_included": False,
            "predict_rows_visible_to_llm": False,
            "test_labels_available_during_generation": False,
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.truth_output:
        args.truth_output.parent.mkdir(parents=True, exist_ok=True)
        args.truth_output.write_text(
            json.dumps(
                {
                    "case_id": args.case_id,
                    "original_to_masked": mapping,
                    "masked_to_original": {masked: original for original, masked in mapping.items()},
                    "original_feature_to_masked": feature_map,
                    "masked_feature_to_original": {masked: raw for raw, masked in feature_map.items()},
                    "note": "evaluator-only; never provide this file to the LLM",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    print(json.dumps({"case_id": args.case_id, "train_rows": len(train_public), "validation_rows": len(valid_public), "payload": str(args.output), "manifest": str(manifest_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

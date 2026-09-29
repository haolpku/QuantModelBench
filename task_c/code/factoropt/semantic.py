"""Public semantic-card protocol for Task C.

Task C keeps feature columns masked (``f001`` ...), even on the semantic
track.  A card is a sidecar keyed by that masked id and contains only neutral
name/definition/formula information.  Selection themes, expected direction,
significance statistics, citations and provenance are deliberately excluded.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any


CANONICAL_TRACKS = ("numeric_only", "semantic")
LEGACY_TRACK_ALIASES = {"semantic_numeric": "semantic"}
ALL_TRACKS = CANONICAL_TRACKS + tuple(LEGACY_TRACK_ALIASES)
SEMANTIC_CARD_FIELDS = frozenset(
    {
        "factor_id",
        "neutral_name",
        "neutral_description",
        "formula_or_definition",
        "input_fields",
        "formula_latex",
        "formula_latex_instantiated",
        "formula_inputs",
    }
)
SEMANTIC_CARD_SCHEMA_VERSION = "semantic_factor_card.v1"
_CALENDAR_LITERAL_RE = re.compile(
    r"(?<!\d)(?:19|20)\d{2}[-/](?:0[1-9]|1[0-2])"
    r"(?:[-/](?:0[1-9]|[12]\d|3[01]))?(?!\d)"
)


def _mask_formula_identifiers(
    value: str,
    *,
    identifier_map: Mapping[str, str] | None = None,
    known_identifiers: Sequence[str] | None = None,
) -> str:
    """Translate raw factor tokens in technical text to public ids."""

    text = str(value)
    mapping = {str(key).casefold(): str(val) for key, val in (identifier_map or {}).items()}
    identifiers = [str(item) for item in (known_identifiers or ()) if str(item).strip()]
    identifiers.extend(str(key) for key in (identifier_map or {}) if str(key).strip())
    unique: list[str] = []
    seen: set[str] = set()
    for identifier in sorted(identifiers, key=len, reverse=True):
        folded = identifier.casefold()
        if folded in seen or re.fullmatch(r"(?:S|f)\d+", identifier):
            continue
        seen.add(folded)
        unique.append(identifier)
    if not unique:
        return text
    pattern = re.compile(
        r"(?<![A-Za-z0-9])(" + "|".join(re.escape(item) for item in unique) + r")"
        r"(?=(?:_|[^A-Za-z0-9]|$))",
        flags=re.IGNORECASE,
    )
    return pattern.sub(
        lambda match: mapping.get(match.group(1).casefold(), "<input>"),
        text,
    )


def normalize_track(track: Any) -> str:
    value = str(track or "").strip().lower()
    value = LEGACY_TRACK_ALIASES.get(value, value)
    if value not in CANONICAL_TRACKS:
        raise ValueError(
            f"unknown track {track!r}; expected one of {', '.join(CANONICAL_TRACKS)}"
        )
    return value


def _text(value: Any, *, limit: int = 16_000) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, dict)):
        try:
            value = json.dumps(value, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            value = str(value)
    text = str(value).strip()
    if text.casefold() in {"", "nan", "<na>", "nat", "none"}:
        return ""
    # Date thresholds belong to private construction/provenance context, not
    # the semantic treatment.  Redact the literal but retain the formula text
    # so the public payload remains readable and passes the no-calendar audit.
    text = _CALENDAR_LITERAL_RE.sub("<date>", text)
    return text[:limit]


def _list_of_text(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        raw = value.strip()
        if raw.startswith("["):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    value = parsed
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
        if isinstance(value, str):
            return [part.strip() for part in re.split(r"[,;|]", value) if part.strip()]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_text(item, limit=512) for item in value if _text(item, limit=512)]
    return [_text(value, limit=512)]


def _formula_inputs(
    value: Any,
    *,
    identifier_map: Mapping[str, str] | None = None,
    known_identifiers: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    # JSONL keeps this field as an array, while CSV/Excel sources often leave
    # it as a JSON-encoded string.  Normalize both before applying the public
    # semantic whitelist.
    if isinstance(value, str):
        raw = value.strip()
        if raw.startswith("["):
            try:
                value = json.loads(raw)
            except (TypeError, ValueError, json.JSONDecodeError):
                return []
        else:
            return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    output: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        record: dict[str, Any] = {}
        for key in ("symbol", "meaning", "construction_latex", "definition", "source_fields"):
            if key not in item:
                continue
            if key == "source_fields":
                fields = [
                    _mask_formula_identifiers(
                        field,
                        identifier_map=identifier_map,
                        known_identifiers=known_identifiers,
                    )
                    for field in _list_of_text(item.get(key))
                ]
                if fields:
                    record[key] = fields
            else:
                text = _text(item.get(key), limit=8_000)
                if key in {"symbol", "construction_latex", "source_fields"}:
                    text = _mask_formula_identifiers(
                        text,
                        identifier_map=identifier_map,
                        known_identifiers=known_identifiers,
                    )
                if text:
                    record[key] = text
        if record:
            output.append(record)
    return output


def semantic_card_payload(
    card: Mapping[str, Any] | None,
    masked_id: str,
    *,
    identifier_map: Mapping[str, str] | None = None,
    known_identifiers: Sequence[str] | None = None,
) -> dict[str, Any]:
    source = card if isinstance(card, Mapping) else {}
    local_identifier_map = {str(key): str(value) for key, value in (identifier_map or {}).items()}
    local_known_identifiers = [str(value) for value in (known_identifiers or ())]
    for key in ("factor_id", "_raw_factor_id", "feature_id", "sleeve_id", "mnemonic", "abbreviation"):
        raw_identifier = _text(source.get(key), limit=1_000)
        if raw_identifier:
            local_known_identifiers.append(raw_identifier)
            local_identifier_map.setdefault(raw_identifier, str(masked_id))
    name = _text(
        source.get("neutral_name") or source.get("factor_name") or source.get("name") or masked_id,
        limit=1_000,
    ) or str(masked_id)
    # Legacy metadata loaders sometimes use the raw mnemonic as a fallback
    # name.  Do not let that fallback leak an identifier into a masked public
    # semantic payload.
    raw_identifiers = {
        _text(source.get(key), limit=1_000).casefold()
        for key in ("factor_id", "_raw_factor_id", "feature_id", "sleeve_id", "mnemonic", "abbreviation")
        if _text(source.get(key), limit=1_000)
    }
    if name.casefold() in raw_identifiers:
        name = str(masked_id)
    definition = _text(
        source.get("neutral_description")
        or source.get("definition")
        or source.get("description")
        or source.get("formula_or_definition"),
        limit=16_000,
    )
    formula_latex = _mask_formula_identifiers(
        _text(source.get("formula_latex"), limit=16_000),
        identifier_map=local_identifier_map,
        known_identifiers=local_known_identifiers,
    )
    formula_instantiated = _mask_formula_identifiers(
        _text(source.get("formula_latex_instantiated"), limit=16_000),
        identifier_map=local_identifier_map,
        known_identifiers=local_known_identifiers,
    )
    formula_or_definition = _mask_formula_identifiers(
        _text(source.get("formula_or_definition"), limit=16_000),
        identifier_map=local_identifier_map,
        known_identifiers=local_known_identifiers,
    )
    formula = formula_instantiated or formula_latex or formula_or_definition or definition
    if not formula_or_definition:
        formula_or_definition = formula
    if not definition:
        definition = formula_or_definition or formula
    output: dict[str, Any] = {
        "factor_id": str(masked_id),
        "neutral_name": name,
        "neutral_description": definition,
        "formula_or_definition": formula_or_definition,
        "input_fields": [
            _mask_formula_identifiers(
                item,
                identifier_map=local_identifier_map,
                known_identifiers=local_known_identifiers,
            )
            for item in _list_of_text(source.get("input_fields"))
        ],
        "formula_latex": formula_latex or formula,
    }
    if formula_instantiated:
        output["formula_latex_instantiated"] = formula_instantiated
    inputs = _formula_inputs(
        source.get("formula_inputs"),
        identifier_map=local_identifier_map,
        known_identifiers=local_known_identifiers,
    )
    if inputs:
        output["formula_inputs"] = inputs
    return output


def semantic_cards_payload(
    cards: Mapping[str, Mapping[str, Any]] | None,
    feature_ids: Sequence[str],
    original_to_masked: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    source = {str(key): value for key, value in (cards or {}).items()}
    mapping = {str(key): str(value) for key, value in (original_to_masked or {}).items()}
    known_identifiers = list(source)
    known_identifiers.extend(
        str(value.get("factor_id"))
        for value in source.values()
        if isinstance(value, Mapping) and str(value.get("factor_id", "")).strip()
    )
    result: list[dict[str, Any]] = []
    for raw in feature_ids:
        raw_id = str(raw)
        masked_id = mapping.get(raw_id, raw_id)
        card = source.get(raw_id)
        if card is None:
            card = source.get(masked_id)
        result.append(
            semantic_card_payload(
                card,
                masked_id,
                identifier_map=mapping,
                known_identifiers=known_identifiers,
            )
        )
    return result


def validate_semantic_cards(cards: Any, feature_ids: Sequence[str], *, track: str = "semantic") -> list[str]:
    canonical = normalize_track(track)
    if canonical == "numeric_only":
        return [] if cards in (None, [], ()) else ["numeric_only payload must not contain factor_cards"]
    errors: list[str] = []
    if not isinstance(cards, list):
        return ["semantic payload.factor_cards must be an array"]
    expected = [str(value) for value in feature_ids]
    observed: list[str] = []
    forbidden_tokens = (
        "source_paper", "citation", "reference", "direction", "theme", "group",
        "significance", "p_value", "t_stat", "in_sample", "provenance",
        "source_url", "source_line", "commit", "category",
    )
    for index, card in enumerate(cards):
        if not isinstance(card, Mapping):
            errors.append(f"factor_cards[{index}] must be an object")
            continue
        unexpected = sorted(str(key) for key in card if str(key) not in SEMANTIC_CARD_FIELDS)
        if unexpected:
            errors.append(f"factor_cards[{index}] contains non-semantic fields: {unexpected}")
        for key in card:
            normalized = re.sub(r"[^a-z0-9]+", "_", str(key).lower()).strip("_")
            if any(token in normalized for token in forbidden_tokens):
                errors.append(f"factor_cards[{index}].{key} is evaluator/provenance metadata")
        factor_id = str(card.get("factor_id", "")).strip()
        if not factor_id:
            errors.append(f"factor_cards[{index}] has no factor_id")
        observed.append(factor_id)
        for field in ("neutral_name", "neutral_description", "formula_or_definition", "formula_latex"):
            if field not in card or not isinstance(card.get(field), str):
                errors.append(f"factor_cards[{index}].{field} must be a string")
        if "input_fields" in card and not isinstance(card.get("input_fields"), list):
            errors.append(f"factor_cards[{index}].input_fields must be an array")
        if "formula_inputs" in card:
            inputs = card.get("formula_inputs")
            if not isinstance(inputs, list) or any(not isinstance(item, Mapping) for item in inputs):
                errors.append(f"factor_cards[{index}].formula_inputs must be an array of objects")
            elif any(
                any(str(key) not in {"symbol", "meaning", "construction_latex", "definition", "source_fields"} for key in item)
                for item in inputs
            ):
                errors.append(f"factor_cards[{index}].formula_inputs contains non-semantic fields")
    if observed != expected:
        errors.append("factor_cards must cover feature IDs exactly in feature order")
    if len(observed) != len(set(observed)):
        errors.append("factor_cards contain duplicate factor_id values")
    return list(dict.fromkeys(errors))


__all__ = [
    "ALL_TRACKS", "CANONICAL_TRACKS", "LEGACY_TRACK_ALIASES", "SEMANTIC_CARD_FIELDS",
    "SEMANTIC_CARD_SCHEMA_VERSION", "normalize_track", "semantic_card_payload",
    "semantic_cards_payload", "validate_semantic_cards",
]

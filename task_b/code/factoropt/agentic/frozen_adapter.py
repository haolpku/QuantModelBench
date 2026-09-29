"""Agentic adapter for Task B's one-time frozen-code protocol."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ..task_b_frozen import (
    FROZEN_PROMPT_VERSION,
    validate_code_response,
    validate_selection_response,
)


def _normalize_track(value: Any) -> str:
    text = str(value or "numeric_only").strip().lower()
    if text == "semantic_numeric":
        text = "semantic"
    if text not in {"numeric_only", "semantic"}:
        raise ValueError(f"unknown track: {value!r}")
    return text


REPO_ROOT = Path(__file__).resolve().parents[4]
PROMPT_PATH = REPO_ROOT / "task_b" / "prompt" / "prompt_b_frozen_code.md"
SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schemas" / "task_b_code_response.schema.json"
SELECTION_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schemas" / "task_b_selection_response.schema.json"


def _is_selection_request(payload: Mapping[str, Any]) -> bool:
    if payload.get("_selection_request"):
        return True
    feedback = payload.get("_revision_feedback")
    return isinstance(feedback, Mapping) and bool(feedback.get("selection_request"))


class TaskBFrozenCodeAdapter:
    """Adapter whose final response is executable source, not predictions."""

    task_name = "factor_routing_code"

    def prompt(self, payload: Mapping[str, Any]) -> str:
        if PROMPT_PATH.is_file():
            prompt = PROMPT_PATH.read_text(encoding="utf-8").strip()
        else:
            prompt = _fallback_prompt()
        track = _normalize_track(payload.get("track", "numeric_only"))
        if track == "semantic":
            prompt += "\n\n" + """

SEMANTIC TRACK
`payload["factor_cards"]` is aligned to the masked factor ids in the rows.
Each card contains only a neutral factor name, definition, documented formula
and formula-input descriptions. Use this context when it supports a modelling
choice, but estimate all signs, persistence and weights from the supplied
historical labels. No explicit direction, theme/group, significance,
source-paper, or provenance fields are supplied as priors. Names and formulas
may still naturally describe an economic category or construction framework;
do not treat that wording as a known predictive sign or strength. The
executable function receives the same sanitized cards as
`metadata["factor_cards"]`; keep using masked ids and do not infer or emit raw
source identifiers. If the user message is a compact payload summary, the
public cards are still shown under `fields.factor_cards`; the complete copy is
also available in `payload` inside `python_exec`.
""".strip("\n")
        else:
            prompt += "\n\n" + """

NUMERIC-ONLY TRACK
Factor ids are fully anonymous and no semantic cards are available. Infer all
relationships from the supplied numeric history only.
""".strip("\n")
        revision = payload.get("_revision")
        feedback = payload.get("_revision_feedback")
        if _is_selection_request(payload):
            allowed = []
            if isinstance(feedback, Mapping):
                raw_allowed = feedback.get("allowed_revisions")
                if isinstance(raw_allowed, list):
                    allowed = [int(item) for item in raw_allowed if isinstance(item, int) and not isinstance(item, bool)]
            prompt += (
                "\n\nCANDIDATE SELECTION TURN\n"
                "Revisions are finished. Choose exactly one already-scored, "
                "execution-valid candidate to freeze. Do not write new source, "
                "do not call tools, and do not return the ten-key code object. "
                "Validation metrics are diagnostics; you may weigh stability, "
                "errors, complexity and economic plausibility, not only Rank IC.\n"
                "PUBLIC CANDIDATE TABLE (no hidden labels):\n"
                + json.dumps(dict(feedback) if isinstance(feedback, Mapping) else {}, ensure_ascii=False, sort_keys=True)
                + "\n\nMANDATORY RESPONSE INSTANCE FOR THIS TURN:\n"
                + json.dumps(
                    {
                        "selected_revision": allowed[0] if allowed else 0,
                        "selection_reason": "<why this candidate is frozen>",
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n`selected_revision` MUST be one of "
                + json.dumps(allowed)
                + ". Return exactly those two keys."
            )
            return prompt
        if revision is not None:
            prompt += (
                f"\n\nREVISION PROTOCOL\nCURRENT REVISION: {int(revision)}. "
                "The JSON field `revision` MUST be "
                f"the integer {int(revision)} (not the previous or next round). "
                "Return a complete replacement source, even when the prior "
                "candidate was invalid."
            )
        if isinstance(feedback, Mapping):
            prompt += "\n\nPUBLIC VALIDATION FEEDBACK (aggregate only; no hidden data):\n" + json.dumps(
                dict(feedback), ensure_ascii=False, sort_keys=True
            )
            if isinstance(feedback.get("previous_candidate"), Mapping):
                prompt += (
                    "\nThe feedback includes the previous public candidate. "
                    "Preserve correct source and contract fields where possible, "
                    "fix every listed error, and still return one complete JSON "
                    "object—not a diff and not the feedback envelope."
                )
        # The original frozen-code prompt described the field names but did
        # not show the nested contract *instance*.  Reasoning models commonly
        # filled that gap with a generic JSON-Schema object (``type`` /
        # ``properties``) or with prose.  Put a concrete, payload-specific
        # instance at the end of every prompt so it is the most recent and
        # unambiguous instruction.  Feature names are public and are safe to
        # include here; labels/future rows are never added to this hint.
        prompt += "\n\nMANDATORY RESPONSE INSTANCE FOR THIS TURN:\n" + _response_instance_hint(
            payload, expected_revision=revision
        )
        prompt += (
            "\nReplace every angle-bracket placeholder in that instance before returning. "
            "The returned object must have exactly the ten top-level keys shown."
        )
        return prompt

    def prompt_version(self, payload: Mapping[str, Any]) -> str:
        return f"{FROZEN_PROMPT_VERSION}_{_normalize_track(payload.get('track', 'numeric_only'))}"

    def schema_path(self, payload: Mapping[str, Any]) -> Path:
        if _is_selection_request(payload):
            return SELECTION_SCHEMA_PATH
        return SCHEMA_PATH

    def validate(self, payload: Mapping[str, Any], response: Any) -> list[str]:
        if _is_selection_request(payload):
            feedback = payload.get("_revision_feedback")
            allowed = []
            if isinstance(feedback, Mapping) and isinstance(feedback.get("allowed_revisions"), list):
                allowed = [
                    int(item)
                    for item in feedback.get("allowed_revisions")
                    if isinstance(item, int) and not isinstance(item, bool)
                ]
            return validate_selection_response(response, allowed_revisions=allowed)
        expected = payload.get("_revision")
        expected_revision = (
            int(expected) if isinstance(expected, int) and not isinstance(expected, bool) else None
        )
        return validate_code_response(
            response,
            expected_revision=expected_revision,
            payload=payload,
        )

    def score(
        self,
        payload: Mapping[str, Any],
        response: Mapping[str, Any] | None,
        truth: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        errors = self.validate(payload, response or {})
        return {"schema_valid": not errors, "validation_errors": errors}


def _fallback_prompt() -> str:
    return (
        "Generate complete Python source defining "
        "fit_and_predict(train_df, valid_df, predict_df, metadata). "
        "Return exactly the JSON object required by task_b_code_response.schema.json."
    )


def _response_instance_hint(
    payload: Mapping[str, Any], *, expected_revision: Any = None
) -> str:
    """Return a concrete response instance for the current public payload.

    This is deliberately an instance rather than the JSON Schema itself.  It
    prevents a model from returning schema declarations as the values of
    ``input_contract``/``output_contract`` while still showing every public
    feature name that must be declared.
    """

    raw_features = payload.get("feature_columns")
    if isinstance(raw_features, list) and raw_features and all(
        isinstance(item, str) and item.strip() for item in raw_features
    ):
        features = [str(item) for item in raw_features]
    else:
        # The fallback is only used for a malformed/incomplete prompt during
        # diagnostics; normal validated payloads always carry feature names.
        features = ["<feature>"]
    try:
        revision = int(expected_revision) if expected_revision is not None else 0
    except (TypeError, ValueError):
        revision = 0
    train_columns = ["id", "time_key", *features, "label"]
    predict_columns = ["id", "time_key", *features]
    instance = {
        "revision": revision,
        "status": "candidate",
        "language": "python",
        "code": "<complete Python source defining fit_and_predict>",
        "entrypoint": "fit_and_predict",
        "requirements": ["numpy", "pandas", "scikit-learn"],
        "input_contract": {
            "train_columns": train_columns,
            "validation_columns": train_columns,
            "predict_columns": predict_columns,
        },
        "output_contract": {
            "columns": ["id", "score"],
            "one_row_per_id": True,
        },
        "explanation": "<short public explanation>",
        "revision_summary": "<short change summary>",
    }
    return json.dumps(instance, ensure_ascii=False, indent=2)


__all__ = ["TaskBFrozenCodeAdapter"]

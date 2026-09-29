"""Task C adapter for the shared A/B agentic Chat Completions loop."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ..task_c import PROMPT_SEMANTIC_VERSION, validate_code_response
from ..semantic import normalize_track


REPO_ROOT = Path(__file__).resolve().parents[4]
PROMPT_PATH = REPO_ROOT / "task_c" / "prompt" / "prompt_c.md"
SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schemas" / "task_c_code_response.schema.json"
SELECTION_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schemas" / "task_c_selection_response.schema.json"


def _is_selection_request(payload: Mapping[str, Any]) -> bool:
    if payload.get("_selection_request"):
        return True
    feedback = payload.get("_revision_feedback")
    return isinstance(feedback, Mapping) and bool(feedback.get("selection_request"))


class TaskCAdapter:
    task_name = "stock_multifactor_model"

    def prompt(self, payload: Mapping[str, Any]) -> str:
        if PROMPT_PATH.exists():
            prompt = PROMPT_PATH.read_text(encoding="utf-8").strip()
        else:
            prompt = _fallback_prompt()
        track = normalize_track(payload.get("track", "numeric_only"))
        if track == "semantic":
            prompt += "\n\n" + """

SEMANTIC TRACK
The payload includes `factor_cards` aligned to the anonymous feature columns.
Each card contains only a neutral factor name, definition, documented formula
and formula-input descriptions. Use this context when it improves feature
engineering or regularization, while estimating predictive direction and
strength from historical labels. There are no explicit direction, theme/group,
significance, source-paper, or provenance fields supplied as priors. Names and
formulas may still naturally describe an economic category or construction
framework; do not treat that wording as a known predictive sign or strength.
The same sanitized cards are available as `metadata["factor_cards"]` at
runtime. Keep the feature columns and stock ids masked; never reconstruct raw
identifiers. If the user message is a compact payload summary, the public
cards are still shown under `fields.factor_cards`; the complete copy is also
available in `payload` inside `python_exec`.
""".strip("\n")
        else:
            prompt += "\n\nNUMERIC-ONLY TRACK\nFeature names and ids are fully anonymous; infer relationships from numeric history only."
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
                "Return a complete replacement source, even when the prior code failed."
            )
        if isinstance(feedback, Mapping):
            prompt += "\n\nPUBLIC VALIDATION FEEDBACK (no hidden test data):\n" + json.dumps(
                feedback, ensure_ascii=False, sort_keys=True
            )
            if isinstance(feedback.get("previous_candidate"), Mapping):
                prompt += (
                    "\nThe feedback includes the previous public candidate. "
                    "Keep the contract fields and the source that were already "
                    "correct, fix every listed error, and still return one "
                    "complete JSON object—not a diff and not the feedback envelope."
                )
        # Put a concrete, payload-specific response instance last.  The static
        # markdown example is intentionally generic, while this tail contains
        # the exact feature names and revision expected by the validator.
        prompt += "\n\nMANDATORY RESPONSE INSTANCE FOR THIS TURN:\n" + _response_instance_hint(
            payload, expected_revision=revision
        )
        prompt += (
            "\nReplace every angle-bracket placeholder in that instance before returning. "
            "Return exactly the ten top-level keys shown; do not return the schema "
            "definition or a diff."
        )
        return prompt

    def prompt_version(self, payload: Mapping[str, Any]) -> str:
        return f"{PROMPT_SEMANTIC_VERSION}_{normalize_track(payload.get('track', 'numeric_only'))}"

    def schema_path(self, payload: Mapping[str, Any]) -> Path:
        if _is_selection_request(payload):
            return SELECTION_SCHEMA_PATH
        return SCHEMA_PATH

    def validate(self, payload: Mapping[str, Any], response: Any) -> list[str]:
        if _is_selection_request(payload):
            from ..task_c_iterative import validate_selection_response

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
        expected_revision = int(expected) if isinstance(expected, int) and not isinstance(expected, bool) else None
        return validate_code_response(response, expected_revision=expected_revision, payload=payload)

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
        "Generate a complete Python source defining "
        "fit_and_predict(train_df, valid_df, predict_df, metadata). "
        "Return only the JSON object required by task_c_code_response.schema.json."
    )


def _response_instance_hint(
    payload: Mapping[str, Any], *, expected_revision: Any = None
) -> str:
    """Return a concrete response instance for the current public payload.

    Showing an instance (rather than copying the JSON Schema) prevents models
    from putting ``type``/``properties`` declarations into the contract values.
    Only public feature names are included; labels and future rows are not.
    """

    raw_features = payload.get("feature_columns")
    if isinstance(raw_features, (list, tuple)) and raw_features and all(
        isinstance(item, str) and item.strip() for item in raw_features
    ):
        features = [str(item) for item in raw_features]
    else:
        # Valid production payloads always carry feature names.  Keep a clear
        # placeholder for diagnostics if an adapter is called with malformed
        # input before public-payload validation runs.
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


__all__ = ["TaskCAdapter"]

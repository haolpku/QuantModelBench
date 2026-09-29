#!/usr/bin/env python3
"""Offline smoke test for Task B's frozen-code rolling protocol."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from factoropt.task_b import build_router_episodes, make_smoke_fixture  # noqa: E402
from factoropt.task_b_frozen import (  # noqa: E402
    FrozenRevisionConfig,
    build_code_generation_payload,
    build_rolling_windows,
    frozen_code_revision_loop,
    run_frozen_rolling,
    validate_frozen_payload,
)


def _response(revision: int) -> dict:
    code = """
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

def fit_and_predict(train_df, valid_df, predict_df, metadata):
    features = [c for c in train_df.columns if c not in ('id', 'time_key', 'label')]
    fit_df = pd.concat([train_df, valid_df], ignore_index=True)
    x = fit_df[features].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    y = pd.to_numeric(fit_df['label'], errors='coerce').fillna(0.0)
    model = Ridge(alpha=1.0)
    model.fit(x, y)
    xp = predict_df[features].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    return pd.DataFrame({'id': predict_df['id'].astype(str), 'score': model.predict(xp)})
""".strip()
    return {
        "revision": revision,
        "status": "final",
        "language": "python",
        "code": code,
        "entrypoint": "fit_and_predict",
        "requirements": ["numpy", "pandas", "scikit-learn"],
        "input_contract": {
            "train_columns": [
                "id", "time_key", "return_1m", "return_12m", "volatility_12m", "label"
            ],
            "validation_columns": [
                "id", "time_key", "return_1m", "return_12m", "volatility_12m", "label"
            ],
            "predict_columns": ["id", "time_key", "return_1m", "return_12m", "volatility_12m"],
        },
        "output_contract": {"columns": ["id", "score"], "one_row_per_id": True},
        "explanation": "Deterministic ridge smoke candidate.",
        "revision_summary": "Initial frozen source.",
    }


def main() -> int:
    fixture = make_smoke_fixture(months=180, sleeves=6)
    episodes = build_router_episodes(
        fixture,
        regions=["WORLD"],
        lookback=120,
        horizon=1,
        min_history=108,
        anchor_step=1,
    )
    selected = episodes[:12]
    if len(selected) != 12:
        raise AssertionError(f"expected 12 episodes, got {len(selected)}")
    ids = [str(value) for value in selected[0].sleeve_ids]
    mapping = {original: f"S{index:02d}" for index, original in enumerate(ids, start=1)}
    train, valid, _predict, split_meta = build_rolling_windows(
        selected[0].history,
        original_to_masked=mapping,
        feature_columns=["return_1m", "return_12m", "volatility_12m"],
        train_cutoffs=12,
        validation_cutoffs=12,
        min_history_months=12,
    )
    if split_meta["train_time_range"] != ["t-24", "t-13"] or split_meta["validation_time_range"] != ["t-12", "t-1"]:
        raise AssertionError(split_meta)
    if split_meta["train_label_time_range"] != ["t-23", "t-12"] or split_meta["validation_label_time_range"] != ["t-11", "t-0"]:
        raise AssertionError(split_meta)
    payload = build_code_generation_payload(
        train,
        valid,
        feature_columns=["return_1m", "return_12m", "volatility_12m"],
        case_id="smoke_frozen",
    )
    # The semantic sidecar is aligned to masked factor ids and is rebuilt from
    # a strict whitelist; evaluator priors must not survive the boundary.
    semantic_cards = [
        {
            "factor_id": mapping[original],
            "neutral_name": f"Fixture signal {index}",
            "neutral_description": "A synthetic historical signal.",
            "formula_or_definition": "x_t - x_{t-1}",
            "formula_latex": r"x_t-x_{t-1}",
            "input_fields": ["x"],
            "source_paper": "must be stripped",
            "official_direction": 1,
            "theme": "must be stripped",
        }
        for index, original in enumerate(ids, start=1)
    ]
    semantic_payload = build_code_generation_payload(
        train,
        valid,
        feature_columns=["return_1m", "return_12m", "volatility_12m"],
        case_id="smoke_frozen_semantic",
        track="semantic",
        factor_cards=semantic_cards,
    )
    semantic_errors = validate_frozen_payload(semantic_payload)
    if semantic_errors:
        raise AssertionError(semantic_errors)
    if [card["factor_id"] for card in semantic_payload["factor_cards"]] != [mapping[original] for original in ids]:
        raise AssertionError("semantic cards are not aligned to masked ids")
    if any("source_paper" in card or "official_direction" in card or "theme" in card for card in semantic_payload["factor_cards"]):
        raise AssertionError("private semantic metadata leaked into public Task B payload")
    try:
        build_code_generation_payload(
            train,
            valid,
            feature_columns=["return_1m", "return_12m", "volatility_12m"],
            case_id="smoke_frozen_numeric_leak",
            track="numeric_only",
            factor_cards=semantic_cards,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("numeric_only accepted factor cards")
    audit = frozen_code_revision_loop(
        payload,
        lambda public, feedback, revision: _response(revision),
        train_df=train,
        validation_df=valid,
        config=FrozenRevisionConfig(max_revisions=0, timeout_seconds=20),
    )
    if audit["selected_revision"] != 0 or audit["test_llm_calls_after_freeze"] != 0:
        raise AssertionError(audit)
    selection_calls = []

    def _selecting_callback(public, feedback, revision):
        selection_calls.append({"revision": revision, "selection": isinstance(feedback, dict) and bool(feedback.get("selection_request"))})
        if selection_calls[-1]["selection"]:
            allowed = list((feedback or {}).get("allowed_revisions") or [])
            pick = 0 if 0 in allowed else allowed[-1]
            return {"selected_revision": int(pick), "selection_reason": "prefer the first valid source"}
        item = _response(revision)
        item["status"] = "candidate"
        item["revision_summary"] = f"smoke candidate {revision}"
        return item

    selected_audit = frozen_code_revision_loop(
        payload,
        _selecting_callback,
        train_df=train,
        validation_df=valid,
        config=FrozenRevisionConfig(max_revisions=1, timeout_seconds=20, stop_on_final=False, selection_policy="llm"),
    )
    if selected_audit.get("selected_revision") != 0:
        raise AssertionError(selected_audit)
    if selected_audit.get("selection_policy") != "llm" or selected_audit.get("llm_selected_revision") != 0:
        raise AssertionError(selected_audit)
    if not any(item["selection"] for item in selection_calls):
        raise AssertionError(selection_calls)
    maps = [{mapping[original]: original for original in ids} for _ in selected]
    predictions, rolling_audit, summary = run_frozen_rolling(
        audit["selected_response"],
        selected,
        maps,
        feature_columns=["return_1m", "return_12m", "volatility_12m"],
        train_cutoffs=12,
        validation_cutoffs=12,
        min_history_months=12,
        timeout_seconds=20,
    )
    if len(predictions) != 12 * len(ids):
        raise AssertionError({"prediction_rows": len(predictions), "summary": summary})
    if not bool(rolling_audit["execution_valid"].all()):
        raise AssertionError(rolling_audit.to_dict("records"))
    print(json.dumps({"SMOKE_OK": True, "rolling_steps": len(selected), "prediction_rows": len(predictions), "llm_calls_during_rolling": summary["llm_calls_during_rolling"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

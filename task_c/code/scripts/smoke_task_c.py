#!/usr/bin/env python3
"""Small deterministic end-to-end Task C smoke test.

The fixture exercises schema/policy checks, two validation revisions, frozen
monthly execution, JKP weights, daily compounding, drift-based turnover, and
the one-time transaction-cost deduction. It does not touch the multi-GB WRDS
files.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from factoropt.agentic import AgenticConfig, TaskCAdapter, run_task_c_codegen_case  # noqa: E402
from factoropt.task_c import build_code_generation_payload, validate_public_payload  # noqa: E402
from factoropt.task_c_backtest import backtest_predictions  # noqa: E402
from factoropt.task_c_iterative import RevisionConfig, iterative_revision_loop, rolling_predict  # noqa: E402


def _response(revision: int, *, good: bool, final: bool = False) -> dict:
    if good:
        code = """import numpy as np\nimport pandas as pd\ndef fit_and_predict(train_df, valid_df, predict_df, metadata):\n    x = pd.to_numeric(predict_df['f1'], errors='coerce').fillna(0.0)\n    return pd.DataFrame({'id': predict_df['id'].astype(str), 'score': x.astype(float)})\n"""
        summary = "Use the first standardized numeric signal with deterministic missing-value handling."
    else:
        code = """import numpy as np\nimport pandas as pd\ndef fit_and_predict(train_df, valid_df, predict_df, metadata):\n    return pd.DataFrame({'id': predict_df['id'].astype(str), 'score': np.zeros(len(predict_df))})\n"""
        summary = "Initial constant-score baseline."
    return {
        "revision": revision,
        "status": "final" if final else "candidate",
        "language": "python",
        "code": code,
        "entrypoint": "fit_and_predict",
        "requirements": ["numpy", "pandas"],
        "input_contract": {
            "train_columns": ["id", "time_key", "f1", "label"],
            "validation_columns": ["id", "time_key", "f1", "label"],
            "predict_columns": ["id", "time_key", "f1"],
        },
        "output_contract": {"columns": ["id", "score"], "one_row_per_id": True},
        "explanation": "Deterministic smoke candidate.",
        "revision_summary": summary,
    }


def _historical_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for month_index, month in enumerate(pd.period_range("2020-01", periods=5, freq="M")):
        for stock in range(12):
            f1 = float(stock + month_index * 0.01)
            rows.append({"id": f"I{stock:03d}", "time_key": month.strftime("%Y-%m"), "f1": f1, "label": f1 / 100.0})
    frame = pd.DataFrame(rows)
    return frame.iloc[:36].reset_index(drop=True), frame.iloc[36:].reset_index(drop=True)


def _rolling_panel() -> pd.DataFrame:
    rows = []
    months = pd.period_range("2020-01", periods=9, freq="M")
    for month_index, month in enumerate(months):
        for stock in range(30):
            f1 = float(stock - 15) / 15.0 + month_index * 0.001
            rows.append(
                {
                    "id": f"I{stock:03d}",
                    "eom": month.to_timestamp("M"),
                    "f1": f1,
                    "ret_next": f1 * 0.01,
                    "me": float(10 + stock),
                    "excntry": "USA",
                    "size_grp": "big",
                    "crsp_exchcd": 1,
                    "common": 1,
                    "exch_main": 1,
                    "primary_sec": 1,
                    "obs_main": 1,
                }
            )
    return pd.DataFrame(rows)


def _daily_returns(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for holding in sorted(panel["eom"].drop_duplicates())[1:]:
        month = (pd.Timestamp(holding) + pd.offsets.MonthEnd(1)).to_period("M")
        first_day = pd.Timestamp(month.start_time)
        for day in (first_day + pd.DateOffset(days=1), first_day + pd.DateOffset(days=2)):
            for row in panel.loc[panel["eom"].eq(holding)].itertuples(index=False):
                rows.append({"id": row.id, "date": day, "ret_exc": row.f1 * 0.002})
    return pd.DataFrame(rows)


def _agentic_artifact_mount_smoke(train: pd.DataFrame, valid: pd.DataFrame) -> int:
    """Verify the shared A/B tool loop can mount artifact-only history.

    The fake provider asks ``python_exec`` for the row count.  The count is
    produced inside the evaluator subprocess, which catches regressions where
    a large artifact descriptor is sent to the model but never materialized.
    """

    with tempfile.TemporaryDirectory(prefix="task_c_artifact_smoke_") as temp:
        root = Path(temp)
        train_path = root / "train.jsonl"
        valid_path = root / "validation.jsonl"
        # Artifact-only public cases use the same relative time encoding as
        # the real WRDS builder; infer the origin as the month after the last
        # validation row for this tiny fixture.
        train_public = train.copy()
        valid_public = valid.copy()
        origin = max(pd.to_datetime(valid_public["time_key"]).dt.to_period("M")) + 1
        for frame in (train_public, valid_public):
            periods = pd.to_datetime(frame["time_key"]).dt.to_period("M")
            frame["time_key"] = periods.map(lambda value: f"t-{origin.ordinal - value.ordinal}")
        train_public.to_json(train_path, orient="records", lines=True)
        valid_public.to_json(valid_path, orient="records", lines=True)
        payload = build_code_generation_payload(
            train_public,
            valid_public,
            feature_columns=["f1"],
            case_id="smoke_c_artifact_mount",
            include_rows=False,
            train_artifact={"path": train_path.name, "format": "jsonl", "rows": len(train_public)},
            validation_artifact={"path": valid_path.name, "format": "jsonl", "rows": len(valid_public)},
        )
        payload["artifact_root"] = "."
        state = {"calls": 0}

        def response(message: dict) -> dict:
            return {
                "id": "task-c-artifact-smoke",
                "object": "chat.completion",
                "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            }

        def request_fn(body: dict) -> dict:
            state["calls"] += 1
            if any(message.get("role") == "tool" for message in body.get("messages", [])):
                return response({"role": "assistant", "content": json.dumps(_response(0, good=True, final=True))})
            arguments = {
                "code": "print(len(payload['train']['rows']))",
                "timeout_seconds": 10,
                "purpose": "artifact materialization smoke",
            }
            return response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "artifact-call",
                            "type": "function",
                            "function": {"name": "python_exec", "arguments": json.dumps(arguments)},
                        }
                    ],
                }
            )

        result = run_task_c_codegen_case(
            payload,
            adapter=TaskCAdapter(),
            config=AgenticConfig(
                model="fake-model",
                base_url="http://offline.invalid/v1",
                max_output_tokens=2_000,
                max_tool_calls=1,
                max_turns=3,
                json_mode=True,
                keep_artifacts=False,
            ),
            scratch_root=root / "scratch",
            public_artifact_root=root,
            request_fn=request_fn,
        )
        if result.get("status") != "success" or result.get("tool_call_count") != 1:
            raise AssertionError(result)
        trace = result.get("tool_trace") or []
        stdout = str(trace[0].get("result", {}).get("stdout", "")) if trace else ""
        if str(len(train_public)) not in stdout:
            raise AssertionError({"expected_train_rows": len(train_public), "stdout": stdout, "result": result})
        if "public_artifact_files" in json.dumps(result.get("request", {}), ensure_ascii=False):
            raise AssertionError("evaluator-only mount key leaked into provider request")
        return state["calls"]


def run() -> dict:
    train, valid = _historical_frames()
    artifact_agentic_calls = _agentic_artifact_mount_smoke(train, valid)
    payload = build_code_generation_payload(train, valid, feature_columns=["f1"], case_id="smoke_c")
    semantic_payload = build_code_generation_payload(
        train,
        valid,
        feature_columns=["f1"],
        case_id="smoke_c_semantic",
        track="semantic",
        factor_cards=[
            {
                "factor_id": "f1",
                "neutral_name": "Fixture signal",
                "neutral_description": "A synthetic cross-sectional signal.",
                "formula_or_definition": "x_t - x_{t-1}",
                "formula_latex": r"x_t-x_{t-1}",
                "input_fields": ["x"],
                "source_paper": "must be stripped",
                "official_direction": -1,
                "theme": "must be stripped",
            }
        ],
    )
    semantic_errors = validate_public_payload(semantic_payload)
    if semantic_errors:
        raise AssertionError(semantic_errors)
    if semantic_payload["factor_cards"][0]["factor_id"] != "f1":
        raise AssertionError("semantic card is not aligned to the masked feature")
    if any(key in semantic_payload["factor_cards"][0] for key in ("source_paper", "official_direction", "theme")):
        raise AssertionError("private semantic metadata leaked into public Task C payload")
    try:
        build_code_generation_payload(
            train,
            valid,
            feature_columns=["f1"],
            case_id="smoke_c_numeric_leak",
            track="numeric_only",
            factor_cards=semantic_payload["factor_cards"],
        )
    except ValueError:
        pass
    else:
        raise AssertionError("numeric_only accepted factor cards")
    calls = []

    def callback(public, feedback, revision):
        if isinstance(feedback, dict) and feedback.get("selection_request"):
            allowed = list(feedback.get("allowed_revisions") or [])
            calls.append({"revision": "selection", "has_feedback": True, "allowed": allowed})
            pick = 1 if 1 in allowed else allowed[-1]
            return {"selected_revision": int(pick), "selection_reason": "prefer the non-constant signal"}
        calls.append({"revision": revision, "has_feedback": feedback is not None})
        return _response(revision, good=revision > 0, final=revision > 0)

    audit = iterative_revision_loop(
        payload,
        callback,
        config=RevisionConfig(max_revisions=1, stop_on_final=True, timeout_seconds=20, selection_policy="llm"),
        train_df=train,
        validation_df=valid,
        metadata={"seed": 7},
    )
    assert audit["selected_revision"] == 1, audit
    assert audit["selection_policy"] == "llm", audit
    assert calls[1]["has_feedback"], calls
    assert len(calls) in {2, 3}, calls

    panel = _rolling_panel()
    frozen = audit["selected_response"]
    predictions, rolling_audit = rolling_predict(
        frozen,
        panel,
        [pd.Timestamp("2020-05-31"), pd.Timestamp("2020-06-30")],
        feature_columns=["f1"],
        train_months=2,
        valid_months=2,
        generate_lead_label_from="ret_next",
        timeout_seconds=20,
    )
    assert len(predictions) == 60, rolling_audit.to_dict("records")
    assert rolling_audit["execution_valid"].all()
    case_id_map = {str(identifier): f"M{index:03d}" for index, identifier in enumerate(sorted(panel["id"].unique()), 1)}
    masked_predictions, masked_audit = rolling_predict(
        frozen,
        panel,
        [pd.Timestamp("2020-05-31")],
        feature_columns=["f1"],
        train_months=2,
        valid_months=2,
        generate_lead_label_from="ret_next",
        timeout_seconds=20,
        id_mapping=case_id_map,
    )
    assert masked_audit["execution_valid"].all()
    assert set(masked_predictions["id"]) <= set(panel["id"])

    # Backtest the two generated score months against two-day holding periods.
    daily = _daily_returns(panel)
    result_zero = backtest_predictions(
        predictions,
        panel,
        daily,
        min_breakpoint_stocks=10,
        min_leg_stocks=5,
        cost_bps=0,
        coverage_threshold=1.0,
    )
    result_cost = backtest_predictions(
        predictions,
        panel,
        daily,
        min_breakpoint_stocks=10,
        min_leg_stocks=5,
        cost_bps=25,
        coverage_threshold=1.0,
    )
    assert (result_zero["monthly"]["status"] == "ok").all(), result_zero["monthly"]
    assert (result_cost["monthly"]["net_return"] <= result_zero["monthly"]["gross_return"]).all()
    assert (result_zero["monthly"]["turnover"] > 0).all()
    return {
        "selected_revision": audit["selected_revision"],
        "revision_count": audit["revision_count"],
        "prediction_rows": int(len(predictions)),
        "rolling_months": int(len(rolling_audit)),
        "backtest_months": int(len(result_zero["monthly"])),
        "backtest_summary": result_cost["summary"],
        "calls": calls,
        "artifact_agentic_calls": artifact_agentic_calls,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = run()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

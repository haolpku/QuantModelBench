"""Human-like validation revision loop and frozen-source rolling execution."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
import warnings
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .task_c import (
    PROMPT_SEMANTIC_VERSION,
    build_rolling_windows,
    code_response_hash,
    _parse_month,
    public_candidate_for_feedback,
    stable_hash,
    validation_feedback,
    validation_metrics,
    validate_code_response,
    validate_public_payload,
)
from .task_c_data import _canonical_identifier, point_in_time_audit
from .task_c_code_runner import CodeExecutionResult, run_candidate
from .semantic import normalize_track


LLMCall = Callable[[Mapping[str, Any], Mapping[str, Any] | None, int], Any]
ValidationBacktest = Callable[[pd.DataFrame], Mapping[str, Any]]


@dataclass(frozen=True)
class RevisionConfig:
    max_revisions: int = 3
    # Per-month wall clock for one candidate execution.  Deep-learning
    # candidates need a real budget; the value still bounds a runaway fit.
    timeout_seconds: float = 1_800.0
    stop_on_final: bool = True
    objective: str = "rank_ic_mean"
    selection_policy: str = "llm"
    seed: int = 1

    def validate(self) -> None:
        if self.max_revisions < 0 or self.max_revisions > 20:
            raise ValueError("max_revisions must be between 0 and 20")
        if self.timeout_seconds <= 0 or not math.isfinite(self.timeout_seconds):
            raise ValueError("timeout_seconds must be positive")
        if self.objective not in {"rank_ic_mean", "net_return", "mse"}:
            raise ValueError("objective must be rank_ic_mean, net_return, or mse")
        if self.selection_policy not in {"llm", "host_metric"}:
            raise ValueError("selection_policy must be llm or host_metric")


def _unwrap_response(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        parsed = value.get("parsed_response")
        if isinstance(parsed, Mapping):
            return parsed
        response = value.get("response")
        if isinstance(response, Mapping):
            return response
        return value
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if "\n" in text:
                text = text.split("\n", 1)[1]
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, Mapping) else None
    return None


_AGENTIC_AUDIT_EXCLUDED_KEYS = frozenset(
    {
        # The generated source is stored once in ``response`` below.  These
        # fields can also contain a full prompt/payload or provider-private
        # reasoning, so do not copy them into the revision summary.
        "payload",
        "messages",
        "api_parameters",
        "raw_turns",
        "raw_response",
        "parsed_response",
        "reasoning",
    }
)


def _record_usage(value: Any) -> dict[str, Any] | None:
    """Extract a provider-neutral token total from an agentic loop record.

    ``run_agentic_case`` aggregates every turn into ``record["usage"]``; that
    value is JSON-safe (counts plus small nested count breakdowns) and carries
    no prompt text, so it can be persisted in the case audit directly.  An
    offline callback returns a bare response object and yields ``None``.
    """

    if not isinstance(value, Mapping):
        return None
    usage = value.get("usage")
    if not isinstance(usage, Mapping):
        return None
    return {str(key): item for key, item in usage.items()}


def _sum_usage(revisions: Sequence[Mapping[str, Any]]) -> dict[str, int] | None:
    """Sum the per-revision token counts into one audit-level total."""

    totals: dict[str, int] = {}
    for revision in revisions:
        usage = revision.get("usage")
        if not isinstance(usage, Mapping):
            continue
        for key in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "prompt_cache_hit_tokens",
            "prompt_cache_miss_tokens",
            "cached_tokens",
        ):
            value = usage.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
                totals[key] = totals.get(key, 0) + int(value)
    return totals or None


def _safe_agentic_audit(value: Any) -> dict[str, Any] | None:
    """Keep the shared A/B loop audit without duplicating private inputs.

    The production callback returns the complete provider-neutral record from
    ``run_agentic_case``.  The revision loop must retain its tool diagnostics
    (including bounded stdout/stderr and source hashes), but copying the whole
    record would duplicate the public payload and could carry request messages
    or private reasoning into the persistent Task C audit.  This helper keeps
    only JSON-safe, bounded fields and recursively drops those private keys.
    Offline callbacks that return a bare response object simply yield ``None``.
    """

    if not isinstance(value, Mapping):
        return None
    markers = {"tool_trace", "conversation_turns", "tool_call_count", "run_id"}
    if not any(marker in value for marker in markers):
        return None

    def clean(item: Any, *, key: str = "", depth: int = 0) -> Any:
        if key in _AGENTIC_AUDIT_EXCLUDED_KEYS:
            return None
        if depth > 6:
            return "[truncated]"
        if item is None or isinstance(item, (bool, int, float, str)):
            if isinstance(item, str):
                return item[:12_000]
            return item
        if isinstance(item, (np.integer, np.floating, np.bool_)):
            return item.item()
        if isinstance(item, Mapping):
            output: dict[str, Any] = {}
            for child_key, child_value in list(item.items())[:128]:
                child_name = str(child_key)
                if child_name in _AGENTIC_AUDIT_EXCLUDED_KEYS:
                    continue
                cleaned = clean(child_value, key=child_name, depth=depth + 1)
                if cleaned is not None:
                    output[child_name] = cleaned
            return output
        if isinstance(item, (list, tuple)):
            return [clean(child, depth=depth + 1) for child in list(item)[:128]]
        return str(item)[:12_000]

    allowed = {
        "run_id",
        "status",
        "model_id",
        "prompt_semantic_version",
        "prompt_hash",
        "payload_hash",
        "settings_hash",
        "schema_hash",
        "model_settings",
        "runtime",
        "tool_limits",
        "request",
        "conversation_turns",
        "tool_trace",
        "public_artifacts",
        "tool_call_count",
        "tool_success_count",
        "tool_failure_count",
        "repair_count",
        "retry_count",
        "latency_ms",
        "artifact_summary",
        "error",
        "warnings",
        "dry_run",
    }
    output: dict[str, Any] = {}
    for key in sorted(allowed):
        if key not in value or key in _AGENTIC_AUDIT_EXCLUDED_KEYS:
            continue
        cleaned = clean(value[key], key=key)
        if cleaned is not None:
            output[key] = cleaned
    return output or None


def _month_values(frame: pd.DataFrame, time_column: str = "time_key") -> list[Any]:
    if time_column in frame.columns:
        values = frame[time_column].dropna().astype(str).unique().tolist()

        def sort_key(value: str) -> tuple[int, Any]:
            match = re.fullmatch(r"t-(\d+)", value.strip(), flags=re.IGNORECASE)
            if match:
                # ``t-132`` is chronologically earlier than ``t-1``.
                return (0, -int(match.group(1)))
            try:
                parsed = _calendar_period_series(pd.Series([value])).iloc[0]
            except (TypeError, ValueError):
                parsed = pd.NaT
            if not pd.isna(parsed):
                return (1, parsed.ordinal)
            return (2, value)

        return sorted(values, key=sort_key)
    return [None]


def _without_label(frame: pd.DataFrame, label_column: str = "label") -> pd.DataFrame:
    def normalize(value: Any) -> str:
        return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")

    aliases = {
        str(column)
        for column in frame.columns
        if normalize(column)
        in {
            normalize(label_column),
            "label",
            "labels",
            "target",
            "y",
            "next_return",
            "next_month_return",
            "ret_exc_lead1m",
            "ret_exc_lead_1m",
            "lead_return",
            "return_next_month",
        }
        or normalize(column).startswith(("future_", "target_"))
    }
    return frame.drop(columns=list(aliases), errors="ignore").copy()


def _runtime_mask_frames(
    frames: Sequence[pd.DataFrame],
    original_to_masked: Mapping[str, str],
) -> tuple[list[pd.DataFrame], dict[str, str]]:
    """Apply the evaluator's case ID map before invoking frozen source code.

    Public code-generation artifacts use masked IDs.  Reusing that map during
    hidden rolling execution avoids an accidental train/test representation
    shift if a candidate (despite the prompt) uses ``id`` as a categorical
    key.  New securities that were not present in the public history receive
    a deterministic one-way token; their reverse mapping is kept evaluator-
    side so the backtester still joins on the original ID.
    """

    mapping = {
        _canonical_identifier(original): str(masked).strip()
        for original, masked in original_to_masked.items()
    }
    inverse = {masked: original for original, masked in mapping.items()}
    if len(inverse) != len(mapping) or any(not key or not value for key, value in mapping.items()):
        raise ValueError("id_mapping must be a one-to-one map of non-empty IDs")
    used = set(inverse)
    output: list[pd.DataFrame] = []
    for frame in frames:
        if "id" not in frame.columns:
            raise ValueError("rolling frame requires id for evaluator masking")
        copied = frame.copy()
        tokens: dict[str, str] = {}
        for raw in copied["id"].astype("string").tolist():
            original = _canonical_identifier(raw)
            if not original:
                raise ValueError("rolling frames contain a null/empty id")
            if original in mapping:
                tokens[original] = mapping[original]
                continue
            digest = hashlib.sha256(original.encode("utf-8")).hexdigest()[:12]
            candidate = f"I_H{digest}"
            suffix = 1
            while candidate in used and inverse.get(candidate) != original:
                candidate = f"I_H{digest}_{suffix}"
                suffix += 1
            mapping[original] = candidate
            inverse[candidate] = original
            used.add(candidate)
            tokens[original] = candidate
        copied["id"] = copied["id"].map(_canonical_identifier).map(tokens).astype("string")
        output.append(copied)
    return output, inverse


def _model_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """Keep evaluator runtime metadata free of calendar/truth side channels."""

    blocked_exact = {
        "formation_eom",
        "formation_period",
        "formation_date",
        "formation_month",
        "origin_month",
        "as_of_month",
        "anchor_date",
        "calendar_date",
        "test_start",
        "test_end",
        "truth",
        "ground_truth",
        "original_to_masked",
        "masked_to_original",
        "id_mapping",
    }
    date_literal = re.compile(
        r"(?<!\d)(?:(?:19|20)\d{2}[-_/]\d{1,2}(?:[-_/]\d{1,2})?|(?:19|20)\d{4})(?!\d)"
    )

    def clean(key: Any, value: Any) -> tuple[bool, Any]:
        normalized = re.sub(r"[^a-z0-9]+", "_", str(key).lower()).strip("_")
        if normalized in blocked_exact or any(
            token in normalized for token in ("path", "filename", "mapping", "secret", "token")
        ):
            return False, None
        if isinstance(value, (pd.Timestamp, pd.Period, np.datetime64, datetime, date)):
            return False, None
        if isinstance(value, str) and date_literal.search(value):
            return False, None
        if isinstance(value, Mapping):
            nested: dict[str, Any] = {}
            for child_key, child_value in value.items():
                keep, cleaned = clean(child_key, child_value)
                if keep:
                    nested[str(child_key)] = cleaned
            return True, nested
        if isinstance(value, (list, tuple)):
            nested_values: list[Any] = []
            for index, child_value in enumerate(value):
                keep, cleaned = clean(f"item_{index}", child_value)
                if keep:
                    nested_values.append(cleaned)
            return True, nested_values
        return True, value

    result: dict[str, Any] = {}
    for key, value in (metadata or {}).items():
        keep, cleaned = clean(key, value)
        if keep:
            result[str(key)] = cleaned
    return result


def _calendar_period_series(source: pd.Series) -> pd.Series:
    """Parse calendar-like values, including compact YYYYMM encodings."""

    raw = source.astype("string").str.strip()
    compact = raw.str.fullmatch(r"\d{6}", na=False)
    parsed = pd.Series(pd.NaT, index=source.index, dtype="datetime64[ns]")
    if bool(compact.any()):
        parsed.loc[compact] = pd.to_datetime(raw.loc[compact], format="%Y%m", errors="coerce")
    remaining = raw.notna() & ~compact
    if bool(remaining.any()):
        values = raw.loc[remaining]
        # ``format='mixed'`` avoids pandas' format-inference warning for a
        # heterogeneous month/date column (warnings are treated as errors in
        # the reproducibility checks).  Keep a compatibility fallback for
        # older pandas versions that do not expose the argument.
        try:
            converted = pd.to_datetime(values, format="mixed", errors="coerce", utc=True)
        except (TypeError, ValueError):  # pragma: no cover - old pandas
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                converted = pd.to_datetime(values, errors="coerce", utc=True)
        if getattr(converted.dt, "tz", None) is not None:
            converted = converted.dt.tz_convert("UTC").dt.tz_localize(None)
        parsed.loc[remaining] = converted
    periods = parsed.dt.to_period("M")
    if periods.isna().any():
        raise ValueError("frame contains unparseable calendar time values")
    return periods


def _relative_offset_series(source: pd.Series) -> pd.Series:
    """Extract canonical integer offsets without invoking date parsing."""

    raw = source.astype("string").str.strip()
    extracted = raw.str.extract(r"^t-(\d+)$", expand=False)
    return pd.to_numeric(extracted, errors="coerce")


def _infer_model_formation_period(
    frames: Sequence[pd.DataFrame],
    metadata: Mapping[str, Any] | None,
) -> pd.Period | None:
    for key in ("formation_eom", "formation_period", "anchor_date"):
        value = (metadata or {}).get(key)
        if value is not None:
            try:
                parsed = _calendar_period_series(pd.Series([value])).iloc[0]
            except (TypeError, ValueError):
                parsed = pd.NaT
            if not pd.isna(parsed):
                return parsed
    observed: list[pd.Period] = []
    saw_relative = False
    saw_calendar = False
    for frame in frames:
        source_column = next((column for column in ("time_key", "eom", "date", "month") if column in frame.columns), None)
        if source_column is None or frame.empty:
            continue
        offsets = _relative_offset_series(frame[source_column])
        if bool(offsets.notna().any()):
            if not bool(offsets.notna().all()):
                raise ValueError("model frames mix relative and calendar time encodings")
            saw_relative = True
            continue
        saw_calendar = True
        observed.extend(_calendar_period_series(frame[source_column]).dropna().tolist())
    if saw_relative and saw_calendar:
        raise ValueError("model frames mix relative and calendar time encodings")
    return max(observed) + 1 if observed else None


def _model_time_frame(
    frame: pd.DataFrame,
    *,
    formation_period: pd.Period | None,
) -> pd.DataFrame:
    """Hide evaluator calendar columns and expose canonical relative keys."""

    output = frame.copy(deep=True)
    source_column = next((column for column in ("time_key", "eom", "date", "month") if column in output.columns), None)
    if source_column is None:
        return output
    offsets = _relative_offset_series(output[source_column])
    if bool(offsets.notna().any()):
        if not bool(offsets.notna().all()):
            raise ValueError("model frame mixes relative and calendar time encodings")
        offsets = offsets.astype("int64")
    else:
        if formation_period is None:
            raise ValueError("calendar model frame requires an inferred formation month")
        periods = _calendar_period_series(output[source_column])
        offsets = periods.map(lambda value: int(formation_period.ordinal - value.ordinal))
        if (offsets < 0).any():
            raise ValueError("model frame contains rows after the formation month")
    output["time_key"] = offsets.map(lambda value: f"t-{int(value)}").astype("string")
    drop_columns = [
        column
        for column in output.columns
        if column != "time_key" and re.sub(r"[^a-z0-9]+", "_", str(column).lower()).strip("_")
        in {"eom", "date", "month", "anchor_date", "formation_eom"}
    ]
    return output.drop(columns=drop_columns, errors="ignore")


def _prepare_model_frames(
    frames: Sequence[pd.DataFrame],
    metadata: Mapping[str, Any] | None,
) -> list[pd.DataFrame]:
    formation_period = _infer_model_formation_period(frames, metadata)
    return [_model_time_frame(frame, formation_period=formation_period) for frame in frames]


def evaluate_candidate_on_validation(
    response: Mapping[str, Any],
    *,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    outer_validation_df: pd.DataFrame | None = None,
    metadata: Mapping[str, Any] | None = None,
    # Per-month wall clock; see RevisionConfig.
    timeout_seconds: float = 1_800.0,
    scratch_root: str | Path | None = None,
    validation_backtest_fn: ValidationBacktest | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, list[CodeExecutionResult]]:
    """Run a candidate month-by-month and score it outside generated code.

    ``outer_validation_df`` is optional.  When supplied, its labels are used
    only by this evaluator; the generated function receives feature-only rows
    as ``predict_df``.  This gives a strict iteration-validation/selection-
    holdout track without changing the candidate interface.
    """

    model_frames = [train_df, validation_df]
    if outer_validation_df is not None:
        model_frames.append(outer_validation_df)
    prepared_frames = _prepare_model_frames(model_frames, metadata)
    train_df = prepared_frames[0]
    validation_df = prepared_frames[1]
    target = prepared_frames[2] if outer_validation_df is not None else prepared_frames[1]
    # Enforce the same chronological split for direct evaluator callers as
    # for payload-built cases.  The public validator checks embedded rows,
    # while artifact/fixture callers reach this function with DataFrames and
    # therefore need an equivalent guard here.
    for frame_name, frame in (("train", train_df), ("validation", validation_df)):
        if "time_key" not in frame.columns:
            continue
        offsets = _relative_offset_series(frame["time_key"])
        if offsets.isna().any():
            raise ValueError(f"{frame_name} contains an invalid relative time key")
        if (offsets <= 0).any():
            raise ValueError(f"{frame_name} contains formation-month/future labels")
    if "time_key" in train_df.columns and "time_key" in validation_df.columns:
        train_offsets = _relative_offset_series(train_df["time_key"])
        valid_offsets = _relative_offset_series(validation_df["time_key"])
        if not train_offsets.empty and not valid_offsets.empty and int(train_offsets.min()) <= int(valid_offsets.max()):
            raise ValueError("train and validation rows overlap or are not chronological")
    if outer_validation_df is not None and "time_key" in target.columns:
        outer_offsets = _relative_offset_series(target["time_key"])
        if outer_offsets.isna().any() or (outer_offsets <= 0).any():
            raise ValueError("outer validation contains formation-month/future labels")
    time_column = "time_key" if "time_key" in target.columns else "eom" if "eom" in target.columns else None
    prediction_frames: list[pd.DataFrame] = []
    executions: list[CodeExecutionResult] = []
    errors: list[str] = []
    months = _month_values(target, time_column or "time_key")
    for month_index, month in enumerate(months):
        if time_column is None:
            predict = _without_label(target)
            fit_validation = validation_df.iloc[0:0].copy()
        else:
            predict = target.loc[target[time_column].astype(str).eq(str(month))].copy()
            predict = _without_label(predict)
            if outer_validation_df is not None:
                # The explicit outer-validation track keeps all inner
                # validation labels available to the candidate, but never
                # passes outer rows/labels as ``valid_df``.
                fit_validation = validation_df.copy(deep=True)
            elif time_column in validation_df.columns:
                # In the default track, score one validation month at a time
                # and expose only *earlier* validation labels to the candidate.
                # This prevents a model from copying the target month's label
                # through ``valid_df`` while preserving a chronological
                # walk-forward development protocol.
                prior_months = set(months[:month_index])
                fit_validation = validation_df.loc[validation_df[time_column].astype(str).isin(prior_months)].copy()
            else:
                fit_validation = validation_df.iloc[0:0].copy()
        if predict.empty:
            continue
        run_metadata = _model_metadata(metadata)
        # ``metadata`` is a public low-level API, so apply the same track
        # boundary here as in the payload-driven revision loop.  Cards may be
        # passed only when the declared track is semantic; numeric-only runs
        # must remain fully anonymous even for direct evaluator callers.
        try:
            run_track = normalize_track(run_metadata.get("track", "numeric_only"))
        except ValueError:
            run_track = "numeric_only"
        supplied_cards = run_metadata.get("factor_cards")
        run_metadata.pop("factor_cards", None)
        run_metadata.pop("semantic_cards", None)
        run_metadata["track"] = run_track
        if run_track == "semantic" and isinstance(supplied_cards, list):
            run_metadata["factor_cards"] = [
                dict(card) for card in supplied_cards if isinstance(card, Mapping)
            ]
        declared_features = (metadata or {}).get("feature_columns") if isinstance(metadata, Mapping) else None
        if isinstance(declared_features, (list, tuple)) and declared_features:
            model_features = [str(column) for column in declared_features]
        else:
            model_features = [
                str(column)
                for column in train_df.columns
                if str(column) not in {"id", "time_key", "label"}
            ]
        run_metadata.update(
            {
                "seed": int(run_metadata.get("seed", 1)),
                "feature_columns": model_features,
                "evaluation_month": month,
                "validation_protocol": (
                    "explicit_outer_holdout" if outer_validation_df is not None else "chronological_leave_future_out"
                ),
                "validation_month_index": int(month_index),
                "refit_policy": "select_on_valid_then_refit_train_plus_valid",
            }
        )
        try:
            result = run_candidate(
                response,
                train_df,
                fit_validation,
                predict,
                run_metadata,
                timeout_seconds=timeout_seconds,
                scratch_root=scratch_root,
            )
        except Exception as exc:  # keep one bad month from cancelling a revision
            errors.append(f"{type(exc).__name__}: {exc}")
            continue
        executions.append(result)
        if not result.valid or result.prediction is None:
            errors.extend(result.errors or [result.status])
            continue
        prediction = result.prediction.copy()
        if time_column is not None:
            prediction[time_column] = month
        prediction_frames.append(prediction)
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame(columns=["id", "score"])
    try:
        metrics = validation_metrics(
            predictions,
            target,
            time_column=time_column or "time_key",
            label_column="label",
        )
    except Exception as exc:
        metrics = {
            "rank_ic_mean": None,
            "rank_ic_std": None,
            "n_months": 0,
            "n_rows": 0,
            "mse": None,
            "directional_accuracy": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
        errors.append(metrics["error"])
    if validation_backtest_fn is not None and not predictions.empty:
        try:
            backtest_value = validation_backtest_fn(predictions.copy(deep=True))
            if not isinstance(backtest_value, Mapping):
                raise TypeError("validation_backtest_fn must return a mapping")
            # Only aggregate, finite fields may cross into the revision
            # feedback envelope.  The callback itself remains evaluator-only;
            # row-level weights, IDs, dates and daily paths are never copied.
            allowed_scalars = {
                "net_return",
                "gross_return",
                "turnover",
                "coverage",
                "coverage_min",
                "tradable_months",
                "unavailable_months",
                "n_months",
                "sharpe",
                "annualized_volatility",
                "cumulative_return",
            }
            aggregate: dict[str, Any] = {}
            for key in allowed_scalars:
                value = backtest_value.get(key)
                if isinstance(value, bool):
                    aggregate[key] = bool(value)
                elif isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(float(value)):
                    aggregate[key] = float(value) if isinstance(value, (float, np.floating)) else int(value)
            if "status" in backtest_value and isinstance(backtest_value.get("status"), str):
                status_text = str(backtest_value["status"]).strip().lower()
                if status_text in {"ok", "valid", "partial", "unavailable", "error"}:
                    aggregate["status"] = status_text
            metrics["validation_backtest"] = aggregate
            # Make the common objective fields available to the frozen
            # selection rule without trusting arbitrary callback keys.
            for key in ("net_return", "gross_return", "turnover", "coverage"):
                if key in aggregate:
                    metrics[key] = aggregate[key]
        except Exception as exc:
            errors.append(f"validation backtest error: {type(exc).__name__}: {exc}")
    all_valid = bool(executions) and len(executions) == len(months) and all(item.valid for item in executions)
    metrics["execution_valid"] = all_valid
    metrics["validation_protocol"] = (
        "explicit_outer_holdout" if outer_validation_df is not None else "chronological_leave_future_out"
    )
    metrics["fit_validation_rows"] = int(
        len(validation_df)
        if outer_validation_df is not None
        else sum(
            len(validation_df.loc[validation_df[time_column].astype(str).isin(set(months[:index]))])
            for index in range(len(months))
        )
        if time_column is not None and time_column in validation_df.columns
        else 0
    )
    metrics["runtime_seconds"] = float(sum(item.runtime_seconds or 0.0 for item in executions))
    metrics["error_count"] = int(len(errors))
    metrics["errors"] = errors[:20]
    return metrics, predictions, executions


def _objective_key(metrics: Mapping[str, Any], objective: str) -> tuple[float, float, float]:
    value = metrics.get(objective)
    number = float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else -math.inf
    # Rank IC/net return are maximized; MSE is minimized. Missing primary
    # values are always worse than a finite candidate. Runtime/complexity are
    # deterministic secondary keys.
    primary = -number if objective == "mse" and math.isfinite(number) else number
    mse = metrics.get("mse")
    mse_value = float(mse) if isinstance(mse, (int, float)) and math.isfinite(float(mse)) else math.inf
    complexity = float(metrics.get("code_chars", math.inf))
    return (primary, -mse_value, -complexity)


_PUBLIC_SELECTION_METRICS = (
    "rank_ic_mean",
    "rank_ic_std",
    "mse",
    "directional_accuracy",
    "net_return",
    "n_months",
    "n_rows",
)


def validate_selection_response(response: Any, *, allowed_revisions: Sequence[int]) -> list[str]:
    allowed = {int(value) for value in allowed_revisions}
    if not isinstance(response, Mapping):
        return ["selection response must be a JSON object"]
    errors: list[str] = []
    extra = sorted(str(key) for key in response if key not in {"selected_revision", "selection_reason"})
    if extra:
        errors.append("selection response has unexpected fields: " + ", ".join(extra[:8]))
    raw = response.get("selected_revision")
    if not isinstance(raw, int) or isinstance(raw, bool):
        errors.append("selected_revision must be an integer")
    elif int(raw) not in allowed:
        errors.append(f"selected_revision must be one of {sorted(allowed)}")
    reason = response.get("selection_reason")
    if not isinstance(reason, str) or not reason.strip():
        errors.append("selection_reason must be a non-empty string")
    elif len(reason) > 2000:
        errors.append("selection_reason exceeds 2000 characters")
    return errors


def _candidate_selection_card(
    response: Mapping[str, Any],
    metrics: Mapping[str, Any],
    revision: int,
) -> dict[str, Any]:
    from .task_c import _json_safe

    return {
        "revision": int(revision),
        "code_hash": code_response_hash(response),
        "code_chars": int(len(str(response.get("code") or ""))),
        "revision_summary": str(response.get("revision_summary") or "")[:1000],
        "validation": {
            key: _json_safe(metrics.get(key))
            for key in _PUBLIC_SELECTION_METRICS
            if key in metrics
        },
    }


def _parse_selection_response(
    value: Any, *, allowed_revisions: Sequence[int]
) -> tuple[int | None, str, list[str]]:
    errors = validate_selection_response(value, allowed_revisions=allowed_revisions)
    if errors or not isinstance(value, Mapping):
        return None, "", errors
    return int(value["selected_revision"]), str(value.get("selection_reason") or "").strip(), []


def _choose_host_metric_candidate(
    candidates: Sequence[tuple[Mapping[str, Any], dict[str, Any], int]],
    objective: str,
) -> tuple[Mapping[str, Any], dict[str, Any], int] | None:
    if not candidates:
        return None
    return max(candidates, key=lambda item: _objective_key(item[1], objective))


def _select_frozen_candidate(
    candidates: Sequence[tuple[Mapping[str, Any], dict[str, Any], int]],
    *,
    settings: RevisionConfig,
    llm_call: LLMCall,
    public_payload: Mapping[str, Any],
) -> dict[str, Any]:
    host_choice = _choose_host_metric_candidate(candidates, settings.objective)
    result: dict[str, Any] = {
        "selection_policy": settings.selection_policy,
        "host_metric_objective": settings.objective,
        "host_metric_selected_revision": None if host_choice is None else int(host_choice[2]),
        "llm_selected_revision": None,
        "selection_fallback": False,
        "selection_errors": [],
        "selection_reason": "",
        "selected_response": None,
        "selected_metrics": {},
        "selected_revision": None,
    }
    if not candidates:
        result["selection_reason"] = "no candidate passed schema and execution checks"
        return result
    if settings.selection_policy == "host_metric" or len(candidates) == 1:
        chosen = host_choice if settings.selection_policy == "host_metric" else candidates[0]
        assert chosen is not None
        result["selected_response"] = dict(chosen[0])
        result["selected_metrics"] = dict(chosen[1])
        result["selected_revision"] = int(chosen[2])
        result["selection_reason"] = (
            f"max validation {settings.objective}; deterministic MSE/complexity tie-break"
            if settings.selection_policy == "host_metric"
            else "single_valid_candidate"
        )
        return result
    envelope = {
        "selection_request": True,
        "allowed_revisions": [int(item[2]) for item in candidates],
        "candidates": [_candidate_selection_card(*item) for item in candidates],
        "instruction": (
            "Choose exactly one allowed revision to freeze. Do not emit new source. "
            "Validation numbers are diagnostics, not a required ranking rule."
        ),
    }
    try:
        raw = llm_call(public_payload, envelope, int(settings.max_revisions) + 1)
        parsed = _unwrap_response(raw)
        picked, reason, errors = _parse_selection_response(
            parsed, allowed_revisions=[int(item[2]) for item in candidates]
        )
    except Exception as exc:
        picked, reason, errors = None, "", [f"{type(exc).__name__}: {exc}"]
    if picked is None:
        chosen = candidates[-1]
        result["selection_fallback"] = True
        result["selection_errors"] = errors[:20]
        result["selection_reason"] = "fallback_last_valid" + (f": {errors[0]}" if errors else "")
    else:
        chosen = next(item for item in candidates if int(item[2]) == picked)
        result["llm_selected_revision"] = picked
        result["selection_reason"] = reason or "llm_choice"
    result["selected_response"] = dict(chosen[0])
    result["selected_metrics"] = dict(chosen[1])
    result["selected_revision"] = int(chosen[2])
    return result


def iterative_revision_loop(
    public_payload: Mapping[str, Any],
    llm_call: LLMCall,
    *,
    config: RevisionConfig | None = None,
    train_df: pd.DataFrame | None = None,
    validation_df: pd.DataFrame | None = None,
    outer_validation_df: pd.DataFrame | None = None,
    metadata: Mapping[str, Any] | None = None,
    scratch_root: str | Path | None = None,
    artifact_root: str | Path | None = None,
    validation_backtest_fn: ValidationBacktest | None = None,
) -> dict[str, Any]:
    """Give an LLM bounded validation-driven chances and freeze one source.

    ``llm_call`` is deliberately injectable: the production adapter uses the
    shared Task A/B Chat Completions loop, while tests can provide a deterministic
    callback.  It receives only public payload, previous aggregate feedback,
    and the integer revision; it never receives test rows or truth.
    """

    settings = config or RevisionConfig()
    settings.validate()
    payload_errors = validate_public_payload(public_payload)
    if payload_errors:
        raise ValueError("invalid public payload: " + "; ".join(payload_errors[:10]))
    if train_df is None or validation_df is None:
        from .task_c import materialize_split

        train_df = materialize_split(
            public_payload,
            "train",
            include_label=True,
            artifact_root=artifact_root,
        )
        validation_df = materialize_split(
            public_payload,
            "validation",
            include_label=True,
            artifact_root=artifact_root,
        )
    history_train = train_df.copy(deep=True)
    history_valid = validation_df.copy(deep=True)
    # Semantic cards are public generation context.  Carry the same sanitized
    # sidecar into the generated function's runtime metadata so a candidate
    # can use it consistently during validation and the later frozen roll.
    runtime_metadata = dict(metadata or {})
    try:
        payload_track = normalize_track(public_payload.get("track", "numeric_only"))
    except ValueError:
        payload_track = "numeric_only"
    # The validated public envelope is authoritative.  Never let an auxiliary
    # caller-supplied metadata mapping add cards to ``numeric_only`` or replace
    # the cards that were actually shown during semantic code generation.
    runtime_metadata.pop("factor_cards", None)
    runtime_metadata.pop("semantic_cards", None)
    runtime_metadata["track"] = payload_track
    if payload_track == "semantic":
        runtime_metadata["factor_cards"] = [
            dict(card)
            for card in public_payload.get("factor_cards", [])
            if isinstance(card, Mapping)
        ]
    revisions: list[dict[str, Any]] = []
    valid_candidates: list[tuple[Mapping[str, Any], dict[str, Any], int]] = []
    feedback: Mapping[str, Any] | None = None
    for revision in range(settings.max_revisions + 1):
        started = time.perf_counter()
        agentic_audit: dict[str, Any] | None = None
        try:
            raw_response = llm_call(public_payload, feedback, revision)
            agentic_audit = _safe_agentic_audit(raw_response)
            response = _unwrap_response(raw_response)
            if response is None:
                raise ValueError("LLM callback did not return a JSON object")
            schema_errors = validate_code_response(response, expected_revision=revision, payload=public_payload)
        except Exception as exc:  # keep a failed revision auditable
            response = None
            schema_errors = [f"LLM call error: {type(exc).__name__}: {exc}"]
        metrics: dict[str, Any] = {}
        executions: list[CodeExecutionResult] = []
        prediction_rows = 0
        if response is not None and not schema_errors:
            try:
                metrics, predictions, executions = evaluate_candidate_on_validation(
                    response,
                    train_df=history_train,
                    validation_df=history_valid,
                    outer_validation_df=outer_validation_df,
                    metadata=runtime_metadata,
                    timeout_seconds=settings.timeout_seconds,
                    scratch_root=scratch_root,
                    validation_backtest_fn=validation_backtest_fn,
                )
            except Exception as exc:
                metrics = {
                    "rank_ic_mean": None,
                    "rank_ic_std": None,
                    "n_months": 0,
                    "n_rows": 0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                predictions = pd.DataFrame()
                executions = []
                schema_errors.append(metrics["error"])
            prediction_rows = int(len(predictions))
            metrics["code_chars"] = len(str(response.get("code", "")))
            if not metrics.get("execution_valid"):
                schema_errors.extend([str(e) for e in metrics.get("errors", [])])
        else:
            predictions = pd.DataFrame()
        elapsed = round(time.perf_counter() - started, 6)
        all_errors = list(dict.fromkeys(schema_errors))
        execution_valid = bool(response is not None and not all_errors and metrics.get("execution_valid", False))
        if response is not None and execution_valid:
            valid_candidates.append((response, metrics, revision))
        execution_audit = [item.audit_dict() for item in executions]
        revision_audit = {
            "round": revision,
            "feedback_in": dict(feedback) if isinstance(feedback, Mapping) else None,
            "sha256": code_response_hash(response) if response is not None else None,
            "schema_valid": response is not None and not validate_code_response(
                response, expected_revision=revision, payload=public_payload
            ),
            "execution_valid": execution_valid,
            # Keep the complete candidate source in the local audit so a
            # revision can be independently reproduced.  It contains no
            # future labels or provider-private reasoning (schema validation
            # has already run above).
            "response": dict(response) if response is not None else None,
            "validation": {key: value for key, value in metrics.items() if key != "errors"},
            "validation_backtest_enabled": validation_backtest_fn is not None,
            "errors": all_errors[:20],
            "prediction_rows": prediction_rows,
            "runtime_seconds": elapsed,
            "execution": execution_audit,
            "revision_summary": str(response.get("revision_summary", ""))[:1000] if response else "",
            # Preserve the shared A/B transport/tool audit separately from the
            # candidate response.  The helper has already removed full
            # request payloads and provider-private reasoning.
            "agentic_audit": agentic_audit,
            # Token accounting for this revision, read from the shared loop's
            # aggregated ``record["usage"]``.  Kept at the revision level so a
            # per-case cost can be attributed without replaying transcripts.
            "usage": _record_usage(raw_response),
        }
        revisions.append(revision_audit)
        feedback = validation_feedback(
            revision=revision + 1,
            validation=metrics,
            io_checks={
                "rows_ok": execution_valid,
                "finite_score": all(item.checks.get("finite_score", False) for item in executions) if executions else False,
                "execution_valid": execution_valid,
            },
            runtime={
                "seconds": elapsed,
                "timed_out": any(item.timed_out for item in executions),
            },
            errors=all_errors,
            allowed_next_revision=revision < settings.max_revisions,
            previous_candidate=public_candidate_for_feedback(response),
        )
        if response is not None and response.get("status") == "final" and execution_valid and settings.stop_on_final:
            break
    choice = _select_frozen_candidate(
        valid_candidates,
        settings=settings,
        llm_call=llm_call,
        public_payload=public_payload,
    )
    selected_response = choice["selected_response"]
    selected_metrics = choice["selected_metrics"]
    selected_revision = choice["selected_revision"]
    selection_reason = choice["selection_reason"]
    schema_count = sum(bool(item.get("schema_valid")) for item in revisions)
    execution_count = sum(bool(item.get("execution_valid")) for item in revisions)
    execution_runs = [run for item in revisions for run in item.get("execution", [])]
    execution_valid_count = sum(bool(run.get("valid")) for run in execution_runs)
    timeout_count = sum(bool(run.get("timed_out")) for run in execution_runs)
    syntax_count = sum(bool((run.get("checks") or {}).get("syntax_valid")) for run in execution_runs)
    policy_count = sum(bool((run.get("checks") or {}).get("policy_valid")) for run in execution_runs)
    import_count = sum(
        bool((run.get("checks") or {}).get("imports_valid", (run.get("checks") or {}).get("policy_valid")))
        for run in execution_runs
    )
    # ``runtime_valid_rate`` is a success rate, not merely a count of child
    # processes that were launched.  A candidate that raises, times out, or
    # serializes an invalid result must remain in the denominator and count as
    # a runtime failure; otherwise the code-evaluation report would make
    # broken sources look successful simply because they reached the runner.
    runtime_count = sum(str(run.get("status")) == "success" for run in execution_runs)
    code_evaluation = {
        "revision_count": int(len(revisions)),
        "schema_valid_rate": float(schema_count / len(revisions)) if revisions else 0.0,
        "syntax_valid_rate": float(syntax_count / len(execution_runs)) if execution_runs else 0.0,
        "import_valid_rate": float(import_count / len(execution_runs)) if execution_runs else 0.0,
        "policy_valid_rate": float(policy_count / len(execution_runs)) if execution_runs else 0.0,
        "runtime_valid_rate": float(runtime_count / len(execution_runs)) if execution_runs else 0.0,
        "execution_valid_rate": float(execution_count / len(revisions)) if revisions else 0.0,
        "output_valid_rate": float(execution_valid_count / len(execution_runs)) if execution_runs else 0.0,
        "timeout_rate": float(timeout_count / len(execution_runs)) if execution_runs else 0.0,
        "candidate_count_valid": int(len(valid_candidates)),
        "execution_count": int(len(execution_runs)),
    }
    return {
        "record_type": "task_c_code_revision_audit",
        "case_id": public_payload.get("case_id"),
        "payload_hash": stable_hash(public_payload),
        # Keep the information track in the immutable prompt/audit identity so
        # a numeric-only and a semantic run cannot be accidentally aggregated
        # as if they had received the same public context.
        "track": payload_track,
        "semantic_card_count": int(
            len(public_payload.get("factor_cards", []))
            if payload_track == "semantic" and isinstance(public_payload.get("factor_cards", []), list)
            else 0
        ),
        "prompt_semantic_version": f"{PROMPT_SEMANTIC_VERSION}_{payload_track}",
        "max_revisions": settings.max_revisions,
        "selection_policy": choice["selection_policy"],
        "selection_objective": (
            settings.objective if settings.selection_policy == "host_metric" else "llm_choice"
        ),
        "selection_reason": selection_reason,
        "selection_fallback": choice["selection_fallback"],
        "selection_errors": choice["selection_errors"],
        "llm_selected_revision": choice["llm_selected_revision"],
        "host_metric_objective": choice["host_metric_objective"],
        "host_metric_selected_revision": choice["host_metric_selected_revision"],
        "validation_backtest_enabled": validation_backtest_fn is not None,
        "code_revisions": revisions,
        "selected_revision": selected_revision,
        "selected_code_hash": code_response_hash(selected_response) if selected_response else None,
        "selected_response": dict(selected_response) if selected_response else None,
        "selected_validation": selected_metrics,
        "code_evaluation": code_evaluation,
        "agentic_revision_count": int(sum(item.get("agentic_audit") is not None for item in revisions)),
        "agentic_tool_call_count": int(
            sum(
                int((item.get("agentic_audit") or {}).get("tool_call_count", 0) or 0)
                for item in revisions
            )
        ),
        "revision_count": len(revisions),
        "usage": _sum_usage(revisions),
        "test_locked": bool(selected_response),
        "test_llm_calls_after_freeze": 0,
    }


def rolling_predict(
    frozen_response: Mapping[str, Any],
    panel: pd.DataFrame,
    formation_months: Sequence[str | pd.Timestamp],
    *,
    feature_columns: Sequence[str],
    train_months: int = 120,
    valid_months: int = 12,
    label_column: str = "label",
    generate_lead_label_from: str | None = None,
    # Per-month wall clock; see RevisionConfig.
    timeout_seconds: float = 1_800.0,
    scratch_root: str | Path | None = None,
    metadata: Mapping[str, Any] | None = None,
    id_mapping: Mapping[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the frozen source once per formation month; no LLM calls occur.

    If ``id_mapping`` is supplied, it is an evaluator-owned
    ``original_id -> masked_id`` map applied only to the model inputs. The
    returned predictions are mapped back to original IDs for the backtester.
    """

    response_errors = validate_code_response(frozen_response)
    if response_errors:
        raise ValueError("cannot run invalid frozen response: " + "; ".join(response_errors[:10]))
    expected_hash = code_response_hash(frozen_response)
    prediction_frames: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    for formation in formation_months:
        # Treat YYYY-MM, month starts, and month ends identically.  Window
        # boundaries are monthly information sets, not day-level origins.
        formation_timestamp = _parse_month(formation)
        try:
            train, valid, predict, window_meta = build_rolling_windows(
                panel,
                formation_timestamp,
                feature_columns=feature_columns,
                train_months=train_months,
                valid_months=valid_months,
                label_column=label_column,
                generate_lead_label_from=generate_lead_label_from,
            )
            pit_audit = point_in_time_audit(
                train,
                valid,
                predict,
                formation_month=formation_timestamp,
            )
            if not pit_audit.get("point_in_time_valid", False):
                raise ValueError("rolling point-in-time audit failed: " + "; ".join(pit_audit.get("errors", [])))
            reverse_ids: dict[str, str] = {}
            if id_mapping:
                model_frames, reverse_ids = _runtime_mask_frames(
                    [train, valid, predict], id_mapping
                )
                model_train, model_valid, model_predict = model_frames
            else:
                model_train, model_valid, model_predict = train, valid, predict
            run_meta = _model_metadata(metadata)
            run_meta.update(window_meta)
            try:
                run_track = normalize_track(run_meta.get("track", "numeric_only"))
            except ValueError:
                run_track = "numeric_only"
            # The public payload's track is the authority.  A low-level
            # caller can pass arbitrary metadata to ``rolling_predict``;
            # never let that side channel add semantic cards to the anonymous
            # track (or let a legacy ``semantic_cards`` alias survive).
            supplied_cards = run_meta.get("factor_cards")
            run_meta.pop("factor_cards", None)
            run_meta.pop("semantic_cards", None)
            run_meta["track"] = run_track
            if run_track == "semantic" and isinstance(supplied_cards, list):
                run_meta["factor_cards"] = [
                    dict(card) for card in supplied_cards if isinstance(card, Mapping)
                ]
            run_meta["refit_policy"] = "select_on_valid_then_refit_train_plus_valid"
            run_meta["fit_mode"] = "from_scratch"
            if id_mapping:
                run_meta["id_representation"] = "case_masked"
            result = run_candidate(
                frozen_response,
                model_train,
                model_valid,
                model_predict,
                run_meta,
                timeout_seconds=timeout_seconds,
                scratch_root=scratch_root,
            )
            audit = {
                "formation_eom": formation_timestamp.strftime("%Y-%m-%d"),
                "holding_month": (formation_timestamp + pd.offsets.MonthEnd(1)).strftime("%Y-%m"),
                "code_hash": expected_hash,
                "llm_calls": 0,
                "fit_mode": "from_scratch",
                "window_policy": "rolling_train_then_validation_refit_contract",
                "execution_status": result.status,
                "execution_valid": result.valid,
                "train_rows": int(len(train)),
                "validation_rows": int(len(valid)),
                "predict_rows": int(len(predict)),
                "train_month_range": window_meta.get("train_month_range"),
                "validation_month_range": window_meta.get("validation_month_range"),
                "predict_month": window_meta.get("predict_month"),
                "train_months": int(window_meta.get("train_months", train_months)),
                "validation_months": int(window_meta.get("validation_months", valid_months)),
                "label_column": window_meta.get("label_column", label_column),
                "time_key_encoding": window_meta.get("time_key_encoding"),
                "runtime_seconds": result.runtime_seconds,
                "errors": result.errors,
                "checks": dict(result.checks),
                "return_code": result.return_code,
                "timed_out": bool(result.timed_out),
                "stdout": result.stdout,
                "stderr": result.stderr,
                "point_in_time": pit_audit,
            }
            if id_mapping:
                audit["id_representation"] = "case_masked"
            if result.valid and result.prediction is not None:
                prediction = result.prediction.copy()
                if reverse_ids:
                    model_ids = prediction["id"].astype("string")
                    prediction["id"] = model_ids.map(reverse_ids).fillna(model_ids)
                # Evaluator-owned timing columns overwrite (rather than
                # ``insert`` beside) any optional columns a candidate may
                # return, so a harmless extra column cannot crash a valid run.
                prediction["formation_eom"] = formation_timestamp
                prediction["holding_month"] = formation_timestamp + pd.offsets.MonthEnd(1)
                ordered = ["formation_eom", "holding_month"] + [
                    column for column in prediction.columns if column not in {"formation_eom", "holding_month"}
                ]
                prediction = prediction[ordered]
                prediction["code_hash"] = expected_hash
                prediction["execution_status"] = "success"
                prediction_frames.append(prediction)
                audit["prediction_rows_out"] = int(len(prediction))
            else:
                audit["prediction_rows_out"] = 0
            audit_rows.append(audit)
        except Exception as exc:
            audit_rows.append(
                {
                    "formation_eom": formation_timestamp.strftime("%Y-%m-%d"),
                    "holding_month": (formation_timestamp + pd.offsets.MonthEnd(1)).strftime("%Y-%m"),
                    "code_hash": expected_hash,
                    "llm_calls": 0,
                    "fit_mode": "from_scratch",
                    "execution_status": "window_error",
                    "execution_valid": False,
                    "errors": [f"{type(exc).__name__}: {exc}"],
                    "prediction_rows_out": 0,
                }
            )
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame(columns=["formation_eom", "holding_month", "id", "score", "code_hash", "execution_status"])
    return predictions, pd.DataFrame(audit_rows)


__all__ = [
    "LLMCall",
    "RevisionConfig",
    "ValidationBacktest",
    "evaluate_candidate_on_validation",
    "iterative_revision_loop",
    "rolling_predict",
    "validate_selection_response",
]

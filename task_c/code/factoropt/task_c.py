"""Contracts and point-in-time data helpers for Task C.

Nothing in this module calls an LLM or reads hidden test data.  It is the
small, auditable boundary shared by the agentic runner, the local code runner,
and the rolling evaluator.  In particular, labels are accepted only in the
historical ``train``/``validation`` splits and are stripped from ``predict``
before a candidate is executed.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import warnings
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .semantic import (
    ALL_TRACKS,
    CANONICAL_TRACKS,
    SEMANTIC_CARD_SCHEMA_VERSION,
    normalize_track,
    semantic_card_payload,
    validate_semantic_cards,
)


PAYLOAD_SCHEMA_VERSION = "task_c_code_payload.v1"
CODE_RESPONSE_SCHEMA_VERSION = "task_c_code_response.v1"
# v4 removes legacy split metadata, standardizes public row columns, and makes
# the train/validation/predict roles explicit in the revision protocol. Keep
# old audits replayable; new runs are tagged with this version.
PROMPT_SEMANTIC_VERSION = "task_c_code_model_v5_llm_candidate_selection"
DEFAULT_TRAIN_MONTHS = 120
DEFAULT_VALID_MONTHS = 12
# Large enough for a hand-written deep-learning model; the limit exists to
# bound the sandbox and the revision feedback, not to discourage model families.
MAX_CODE_CHARS = 120_000

# Canonical public information tracks.  ``semantic_numeric`` remains a
# backwards-compatible input alias, while generated payloads use ``semantic``.
TRACKS = CANONICAL_TRACKS
LEGACY_TRACKS = tuple(value for value in ALL_TRACKS if value not in TRACKS)

_HIDDEN_KEYS = {
    "truth",
    "ground_truth",
    "future",
    "future_return",
    "future_returns",
    "target_returns",
    "test_returns",
    "masked_to_original",
    "original_to_masked",
    "id_mapping",
    "real_id",
    "real_ids",
    "public_artifact_files",
}
_LABEL_NAMES = {
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
_PROTECTED_COLUMN_NAMES = {
    "truth",
    "ground_truth",
    "test",
    "test_rows",
    "test_labels",
    "test_returns",
    "real_id",
    "real_ids",
    "id_mapping",
    "masked_to_original",
    "original_to_masked",
    "future_labels_included",
    "public_artifact_files",
}
_RELATIVE_TIME_RE = re.compile(r"^t-(\d+)$", flags=re.IGNORECASE)
_CALENDAR_LITERAL_RE = re.compile(
    r"(?<!\d)(?:(?:19|20)\d{2}[-_/]\d{1,2}(?:[-_/]\d{1,2})?|(?:19|20)\d{4})(?!\d)"
)


def _normalise_column(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _unsafe_relative_path(value: Any) -> bool:
    """Recognize POSIX and Windows absolute/traversal paths on every host."""

    text = str(value).replace("\\", "/")
    path = Path(text)
    return bool(
        path.is_absolute()
        or re.match(r"^[A-Za-z]:/", text)
        or text.startswith("//")
        or any(part == ".." for part in path.parts)
    )


def _canonical_identifier(value: Any) -> str:
    """Normalize numeric IDs while preserving meaningful leading zeros."""

    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    text = str(value).strip()
    if text.lower() in {"", "nan", "<na>", "nat", "none"}:
        return ""
    if re.fullmatch(r"[+-]?(?:0|[1-9]\d*)\.0+", text):
        try:
            return str(int(float(text)))
        except (OverflowError, ValueError):
            pass
    return text


def _relative_time_offset(value: Any) -> int | None:
    match = _RELATIVE_TIME_RE.fullmatch(str(value).strip())
    return int(match.group(1)) if match else None


def _relative_time_key(value: Any) -> str:
    offset = _relative_time_offset(value)
    if offset is None:
        raise ValueError(f"time key must use relative form t-N, got {value!r}")
    return f"t-{offset}"


def _json_safe(value: Any) -> Any:
    """Convert pandas/numpy scalars and missing values to JSON primitives."""

    if value is None:
        return None
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (pd.Timestamp, np.datetime64, datetime, date)):
        timestamp = pd.Timestamp(value)
        return None if pd.isna(timestamp) else timestamp.isoformat()
    if isinstance(value, pd.Period):
        return value.strftime("%Y-%m")
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def stable_hash(value: Any) -> str:
    encoded = json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    """Hash a public artifact without loading the whole file in memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_response_hash(response: Mapping[str, Any] | str) -> str:
    """Hash source code (rather than prose) for the immutable test audit."""

    if isinstance(response, Mapping):
        value = str(response.get("code") or "")
    else:
        value = str(response)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _schema_path() -> Path:
    return Path(__file__).resolve().parents[1] / "schemas" / "task_c_code_response.schema.json"


def _schema_errors(value: Mapping[str, Any]) -> list[str]:
    """Validate against JSON Schema when available, with a useful fallback."""

    try:
        import jsonschema  # type: ignore

        schema = json.loads(_schema_path().read_text(encoding="utf-8"))
        validator = jsonschema.Draft202012Validator(schema)
        errors = sorted(validator.iter_errors(value), key=lambda item: list(item.path))
        return [f"{'.'.join(str(p) for p in error.path) or '<root>'}: {error.message}" for error in errors]
    except ImportError:
        # The project requirements include jsonschema, but keeping a small
        # fallback makes contract inspection possible in a minimal bootstrap
        # environment and never weakens the explicit checks below.
        return []
    except Exception as exc:  # pragma: no cover - malformed local schema
        return [f"schema validator error: {type(exc).__name__}: {exc}"]


def validate_code_response(
    response: Any,
    *,
    expected_revision: int | None = None,
    require_status: str | None = None,
    payload: Mapping[str, Any] | None = None,
) -> list[str]:
    """Return deterministic validation errors for one LLM code response."""

    if not isinstance(response, Mapping):
        return ["response must be a JSON object"]
    errors = _schema_errors(response)
    required = {
        "revision",
        "status",
        "language",
        "code",
        "entrypoint",
        "requirements",
        "input_contract",
        "output_contract",
        "explanation",
        "revision_summary",
    }
    response_keys = {str(key) for key in response}
    missing = sorted(required - response_keys)
    errors.extend(f"missing required field: {key}" for key in missing)
    extras = sorted(response_keys - required)
    errors.extend(f"unknown response field: {key}" for key in extras)
    if response.get("language") != "python":
        errors.append("language must be python")
    if response.get("entrypoint") != "fit_and_predict":
        errors.append("entrypoint must be fit_and_predict")
    status = response.get("status")
    if status not in {"candidate", "final"}:
        errors.append("status must be candidate or final")
    if require_status is not None and status != require_status:
        errors.append(f"status must be {require_status}")
    revision = response.get("revision")
    if isinstance(revision, bool) or not isinstance(revision, (int, np.integer)):
        errors.append("revision must be an integer")
    elif expected_revision is not None and int(revision) != int(expected_revision):
        errors.append(f"revision must equal {expected_revision}")
    elif isinstance(revision, (int, np.integer)) and not 0 <= int(revision) <= 20:
        errors.append("revision must be between 0 and 20")
    code = response.get("code")
    if not isinstance(code, str):
        errors.append("code must be a string")
    elif not code.strip():
        errors.append("code must not be empty")
    elif len(code) < 20:
        errors.append("code must contain at least 20 characters")
    elif len(code) > MAX_CODE_CHARS:
        errors.append(f"code exceeds {MAX_CODE_CHARS} characters")
    elif "def fit_and_predict" not in code:
        errors.append("code must define fit_and_predict")
    elif _CALENDAR_LITERAL_RE.search(code):
        errors.append("code must not contain calendar date literals")
    for field in ("explanation", "revision_summary"):
        if not isinstance(response.get(field), str):
            errors.append(f"{field} must be a string")
        elif len(response[field]) > (2000 if field == "explanation" else 1000):
            errors.append(f"{field} is too long")
    requirements = response.get("requirements")
    if isinstance(requirements, list):
        if len(requirements) > 20:
            errors.append("requirements must contain at most 20 items")
        for item in requirements:
            if not isinstance(item, str) or not item.strip() or not re.fullmatch(r"[A-Za-z0-9_.-]+", item):
                errors.append("requirements must contain package-like non-empty strings")
    contract = response.get("input_contract")
    if not isinstance(contract, Mapping):
        errors.append("input_contract must be an object")
    else:
        for key in ("train_columns", "validation_columns", "predict_columns"):
            if not isinstance(contract.get(key), list):
                errors.append(f"input_contract.{key} must be an array")
            else:
                if any(not isinstance(column, str) for column in contract.get(key, [])):
                    errors.append(f"input_contract.{key} must contain strings")
                if len(set(str(column) for column in contract.get(key, []))) != len(contract.get(key, [])):
                    errors.append(f"input_contract.{key} must not contain duplicate columns")
                declared_columns = [str(column) for column in contract.get(key, [])]
                protected_contract_columns = _find_protected_columns(declared_columns)
                calendar_contract_columns = [
                    column
                    for column in declared_columns
                    if _normalise_column(column)
                    in {"eom", "date", "month", "anchor_date", "formation_eom", "formation_date", "calendar_date"}
                ]
                label_contract_columns = [
                    column
                    for column in declared_columns
                    if _normalise_column(column) in _LABEL_NAMES
                    or _normalise_column(column).startswith(("future_", "target_"))
                ]
                if protected_contract_columns:
                    errors.append(
                        f"input_contract.{key} contains protected columns: "
                        f"{sorted(protected_contract_columns)}"
                    )
                if calendar_contract_columns:
                    errors.append(
                        f"input_contract.{key} contains evaluator-only calendar columns: "
                        f"{sorted(calendar_contract_columns)}"
                    )
                if key == "predict_columns" and label_contract_columns:
                    errors.append(
                        f"input_contract.predict_columns contains label/target columns: "
                        f"{sorted(label_contract_columns)}"
                    )
        contract_extras = sorted(
            {str(key) for key in contract}
            - {"train_columns", "validation_columns", "predict_columns"}
        )
        errors.extend(f"input_contract has unknown field: {key}" for key in contract_extras)
    output = response.get("output_contract")
    if not isinstance(output, Mapping):
        errors.append("output_contract must be an object")
    else:
        columns = output.get("columns")
        if not isinstance(columns, list) or "id" not in columns or "score" not in columns:
            errors.append("output_contract.columns must include id and score")
        elif any(not isinstance(column, str) for column in columns):
            errors.append("output_contract.columns must contain strings")
        if output.get("one_row_per_id") is not True:
            errors.append("output_contract.one_row_per_id must be true")
        if isinstance(columns, list) and len({str(column) for column in columns}) != len(columns):
            errors.append("output_contract.columns must not contain duplicates")
        output_extras = sorted(
            {str(key) for key in output} - {"columns", "one_row_per_id"}
        )
        errors.extend(f"output_contract has unknown field: {key}" for key in output_extras)
    if isinstance(payload, Mapping) and isinstance(contract, Mapping):
        features = payload.get("feature_columns")
        if isinstance(features, list) and all(isinstance(item, str) for item in features):
            required_columns = {
                "train_columns": {"id", "time_key", "label", *[str(item) for item in features]},
                "validation_columns": {"id", "time_key", "label", *[str(item) for item in features]},
                "predict_columns": {"id", "time_key", *[str(item) for item in features]},
            }
            for key, required in required_columns.items():
                declared = contract.get(key)
                if isinstance(declared, list):
                    missing = sorted(required - {str(item) for item in declared})
                    if missing:
                        errors.append(f"input_contract.{key} omits required columns: {missing}")
    # De-duplicate errors so a missing schema dependency does not produce a
    # confusingly repeated message.
    return list(dict.fromkeys(str(error) for error in errors))


def _find_label_columns(columns: Iterable[Any]) -> set[str]:
    return {
        str(column)
        for column in columns
        if _normalise_column(column) in _LABEL_NAMES
        or _normalise_column(column).startswith("future_")
        or _normalise_column(column).startswith("target_")
    }


def _find_protected_columns(columns: Iterable[Any]) -> set[str]:
    """Find hidden-target, mapping, and test-routing columns in a table.

    Embedded rows are inspected by :func:`validate_public_payload`, but large
    artifact-only payloads cannot be traversed until they are materialized.
    Keep this check shared by both paths so an artifact cannot bypass the
    public-payload boundary simply by moving a protected field into its
    header.
    """

    protected: set[str] = set()
    for column in columns:
        normalized = _normalise_column(column)
        if (
            normalized in _PROTECTED_COLUMN_NAMES
            or normalized.startswith(("truth_", "ground_truth", "future_", "target_"))
        ):
            protected.add(str(column))
    return protected


def _time_source(frame: pd.DataFrame) -> pd.Series:
    """Return the model-facing time series, preferring ``time_key``.

    A few evaluator-owned tables retain an original ``eom`` column alongside
    the anonymised key.  The public contract must audit the key that the model
    actually receives, not accidentally fall back to that calendar column.
    """

    if "time_key" in frame.columns:
        return frame["time_key"]
    source_column = next((column for column in ("eom", "date", "month") if column in frame.columns), None)
    if source_column is None:
        raise ValueError("frame requires time_key/eom/date/month")
    return frame[source_column]


def _parse_calendar_series(values: pd.Series) -> pd.Series:
    """Parse heterogeneous calendar encodings without inference warnings."""

    raw = values.astype("string").str.strip()
    compact = raw.str.fullmatch(r"\d{6}", na=False)
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    if bool(compact.any()):
        parsed.loc[compact] = pd.to_datetime(raw.loc[compact], format="%Y%m", errors="coerce")
    remaining = raw.notna() & ~compact
    if bool(remaining.any()):
        try:
            converted = pd.to_datetime(
                raw.loc[remaining], format="mixed", errors="coerce", utc=True
            )
        except (TypeError, ValueError):  # pragma: no cover - old pandas
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                converted = pd.to_datetime(raw.loc[remaining], errors="coerce", utc=True)
        if getattr(converted.dt, "tz", None) is not None:
            converted = converted.dt.tz_convert("UTC").dt.tz_localize(None)
        parsed.loc[remaining] = converted
    return parsed


def _relative_offsets(
    source: pd.Series,
    *,
    formation_period: pd.Period | None = None,
) -> pd.Series:
    """Convert a time series to integer months before the formation origin."""

    raw = source.astype("string")
    offsets = raw.map(_relative_time_offset)
    if bool(offsets.notna().any()) and not bool(offsets.notna().all()):
        raise ValueError("time column mixes relative and calendar encodings")
    if bool(offsets.notna().all()) and len(raw):
        return offsets.astype("int64")
    periods = _parse_calendar_series(source).dt.to_period("M")
    if periods.isna().any():
        raise ValueError("frame contains unparseable time values")
    if formation_period is None:
        if periods.empty:
            return pd.Series([], index=source.index, dtype="int64")
        formation_period = periods.max() + 1
    result = formation_period.ordinal - periods.map(lambda value: value.ordinal)
    if (result < 0).any():
        raise ValueError("frame contains rows after the formation month")
    return result.astype("int64")


def _relative_time_frame(
    frame: pd.DataFrame,
    *,
    formation_period: pd.Period | None = None,
) -> pd.DataFrame:
    """Return a copy whose model-facing time column is encoded as ``t-N``."""

    output = frame.copy()
    source = _time_source(output)
    offsets = _relative_offsets(source, formation_period=formation_period)
    output["time_key"] = offsets.map(lambda value: f"t-{int(value)}").astype("string")
    return output


def _contains_hidden_key(value: Any, path: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            normalized = _normalise_column(key_text)
            if (
                normalized in _HIDDEN_KEYS
                or normalized.startswith("ground_truth")
                or normalized.startswith("truth_")
                or (normalized.startswith("future_") and normalized not in {"future_labels_included"})
                or normalized in {"test", "test_rows", "test_returns", "test_labels"}
            ):
                found.append(f"{path}.{key_text}" if path else key_text)
            found.extend(_contains_hidden_key(child, f"{path}.{key_text}" if path else key_text))
    elif isinstance(value, (list, tuple)):
        # Embedded payloads must be checked in full; otherwise a hidden field
        # could be placed after an arbitrary row index. Large production
        # panels use artifact descriptors and do not pay this traversal cost.
        for index, child in enumerate(value):
            found.extend(_contains_hidden_key(child, f"{path}[{index}]"))
    return found


def validate_public_payload(payload: Any) -> list[str]:
    """Validate the public Task C payload and explicitly protect test inputs."""

    if not isinstance(payload, Mapping):
        return ["payload must be a JSON object"]
    errors: list[str] = []
    legacy_split_fields = {"time_key_range", "label_name", "label_definition"}
    legacy_version_fields = {
        "schema_version",
        "prompt_semantic_version",
        "semantic_card_schema_version",
    }
    if "semantic_card_schema_version" in payload:
        errors.append("payload.semantic_card_schema_version is a legacy payload field")
    errors.extend(f"hidden/evaluation field is not allowed: {path}" for path in _contains_hidden_key(payload))
    # ``future_labels_included`` is a harmless, explicit *false* metadata
    # invariant when it lives under ``metadata``.  At the payload root (or
    # with any other value) it is an evaluator routing field and must not be
    # allowed to smuggle a hidden-target declaration through the generic
    # top-level dictionary.
    if "future_labels_included" in payload:
        errors.append("future_labels_included is allowed only as metadata=false")
    for key in ("case_id", "task", "feature_columns", "train", "validation", "predict", "metadata"):
        if key not in payload:
            errors.append(f"missing payload field: {key}")
    if payload.get("task") not in {None, "stock_multifactor_model"}:
        errors.append("task must be stock_multifactor_model")
    raw_track = payload.get("track", "numeric_only")
    try:
        track = normalize_track(raw_track)
    except ValueError as exc:
        errors.append(str(exc))
        track = "numeric_only"
    raw_track_token = str(raw_track).strip().lower()
    if "track" in payload and raw_track_token not in {track, "semantic_numeric"}:
        errors.append("payload.track must use the canonical name numeric_only or semantic")
    horizon = payload.get("horizon_months")
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or int(horizon) != 1:
        errors.append("horizon_months must be the one-month value 1")
    case_id = payload.get("case_id")
    if not isinstance(case_id, str) or not case_id.strip():
        errors.append("case_id must be a non-empty string")
    else:
        if _CALENDAR_LITERAL_RE.search(case_id):
            errors.append("case_id must not expose a calendar date")
        if _unsafe_relative_path(case_id) or not re.fullmatch(r"[A-Za-z0-9_.-]+", case_id):
            errors.append("case_id must be a safe path-free identifier")
    features = payload.get("feature_columns")
    if not isinstance(features, list) or not features or any(
        not isinstance(x, str) or not x.strip() for x in features
    ):
        errors.append("feature_columns must be a non-empty string array")
        feature_names: list[str] = []
    else:
        feature_names = [str(feature) for feature in features]
        if len(set(feature_names)) != len(feature_names) or len({_normalise_column(feature) for feature in feature_names}) != len(feature_names):
            errors.append("feature_columns must not contain duplicates")
        forbidden_feature_names = [
            feature
            for feature in feature_names
            if _normalise_column(feature) in _LABEL_NAMES
            or _normalise_column(feature).startswith(("future_", "target_"))
            or _normalise_column(feature)
            in {"id", "time_key", "eom", "date", "month", "anchor_date", "formation_eom", "formation_month", "origin_month", "as_of_month"}
        ]
        if forbidden_feature_names:
            errors.append(f"feature_columns contain forbidden names: {forbidden_feature_names}")
    errors.extend(
        validate_semantic_cards(
            payload.get("factor_cards", []),
            feature_names,
            track=track,
        )
    )
    schema_marker = payload.get("semantic_card_schema_version")
    if track == "semantic" and schema_marker not in {None, SEMANTIC_CARD_SCHEMA_VERSION}:
        errors.append("semantic_card_schema_version is unsupported")
    if track == "numeric_only" and schema_marker is not None:
        errors.append("numeric_only payload must not declare semantic_card_schema_version")
    artifact_root = payload.get("artifact_root")
    if artifact_root is not None:
        if not isinstance(artifact_root, str) or not artifact_root.strip():
            errors.append("artifact_root must be a relative string")
        else:
            if _unsafe_relative_path(artifact_root):
                errors.append("artifact_root must stay relative to the case directory")
    split_offsets: dict[str, list[int]] = {}
    expected_history_columns = ["id", "time_key", *feature_names, "label"]
    for split_name in ("train", "validation"):
        split = payload.get(split_name)
        if not isinstance(split, Mapping):
            errors.append(f"{split_name} must be an object")
            continue
        for key in legacy_split_fields.intersection({str(item) for item in split}):
            errors.append(f"{split_name}.{key} is a legacy payload field; use rows and artifact only")
        rows = split.get("rows", [])
        if rows is None:
            rows = []
        if not isinstance(rows, list):
            errors.append(f"{split_name}.rows must be an array when provided")
            continue
        artifact = split.get("artifact")
        artifact_ok = isinstance(artifact, Mapping) and bool(str(artifact.get("path") or "").strip())
        if artifact_ok and rows:
            errors.append(f"{split_name} must use either embedded rows or an artifact, not both")
        if not rows and not artifact_ok:
            errors.append(f"{split_name}.rows must not be empty")
        if artifact_ok:
            artifact_path = str(artifact.get("path") or "").replace("\\", "/")
            if "\x00" in artifact_path:
                errors.append(f"{split_name}.artifact.path contains a NUL byte")
            if _unsafe_relative_path(artifact_path):
                errors.append(f"{split_name}.artifact.path must stay inside artifact_root")
            lowered_path = artifact_path.lower()
            if any(marker in lowered_path for marker in ("truth", "ground_truth", "id_mapping", "original_to_masked")):
                errors.append(f"{split_name}.artifact.path references a protected artifact")
            if _CALENDAR_LITERAL_RE.search(artifact_path):
                errors.append(f"{split_name}.artifact.path must not expose a calendar date")
            fmt = str(artifact.get("format") or "jsonl").lower()
            if fmt not in {"jsonl", "ndjson", "csv", "parquet", "pq"}:
                errors.append(f"{split_name}.artifact.format is unsupported: {fmt}")
            if "rows" in artifact:
                try:
                    if int(artifact["rows"]) < 1:
                        errors.append(f"{split_name}.artifact.rows must be positive")
                except (TypeError, ValueError):
                    errors.append(f"{split_name}.artifact.rows must be an integer")
            if "bytes" in artifact:
                try:
                    if int(artifact["bytes"]) < 0:
                        errors.append(f"{split_name}.artifact.bytes must be non-negative")
                except (TypeError, ValueError):
                    errors.append(f"{split_name}.artifact.bytes must be an integer")
            if "sha256" in artifact:
                digest = str(artifact.get("sha256") or "")
                if not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
                    errors.append(f"{split_name}.artifact.sha256 must be a 64-character hex digest")
            if artifact.get("columns") is not None and artifact.get("columns") != expected_history_columns:
                errors.append(
                    f"{split_name}.artifact.columns must be exactly {expected_history_columns}"
                )
        seen_keys: set[tuple[str, str]] = set()
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                errors.append(f"{split_name}.rows[{index}] must be an object")
                continue
            protected_row_columns = _find_protected_columns(row.keys())
            if protected_row_columns:
                errors.append(
                    f"{split_name}.rows[{index}] contains protected evaluator columns: "
                    f"{sorted(protected_row_columns)}"
                )
            calendar_keys = [
                str(key)
                for key in row
                if _normalise_column(key)
                in {"eom", "date", "month", "anchor_date", "formation_eom", "formation_month", "origin_month", "as_of_month"}
            ]
            if calendar_keys:
                errors.append(
                    f"{split_name}.rows[{index}] contains evaluator-only calendar columns: {calendar_keys}"
                )
            raw_id = row.get("id")
            if not _canonical_identifier(raw_id):
                errors.append(f"{split_name}.rows[{index}] has no id")
            if "time_key" not in row:
                errors.append(f"{split_name}.rows[{index}] has no time_key")
            if "time_key" in row and _relative_time_offset(row.get("time_key")) is None:
                errors.append(f"{split_name}.rows[{index}].time_key must use relative t-N keys")
            elif "time_key" in row:
                offset = _relative_time_offset(row.get("time_key"))
                if offset == 0:
                    errors.append(f"{split_name}.rows[{index}] cannot contain the formation-month label (t-0)")
                if offset is not None:
                    split_offsets.setdefault(split_name, []).append(int(offset))
            if "id" in row and "time_key" in row:
                raw_time_key = row.get("time_key")
                canonical_time_key = (
                    _relative_time_key(raw_time_key)
                    if _relative_time_offset(raw_time_key) is not None
                    else str(raw_time_key)
                )
                key = (_canonical_identifier(row.get("id")), canonical_time_key)
                if key in seen_keys:
                    errors.append(f"{split_name}.rows contains duplicate id/time_key: row {index}")
                seen_keys.add(key)
            if "label" not in row and not any(_normalise_column(k) in _LABEL_NAMES for k in row):
                errors.append(f"{split_name}.rows[{index}] has no historical label")
                break
            missing_features = [feature for feature in feature_names if feature not in row]
            if missing_features:
                errors.append(f"{split_name}.rows[{index}] is missing features: {missing_features}")
                break
            if list(row.keys()) != expected_history_columns:
                errors.append(
                    f"{split_name}.rows[{index}] columns must be exactly {expected_history_columns}"
                )
    train_offsets = split_offsets.get("train", [])
    valid_offsets = split_offsets.get("validation", [])
    for split_name, offsets in (("train", train_offsets), ("validation", valid_offsets)):
        if any(offset <= 0 for offset in offsets):
            errors.append(f"{split_name} historical rows must use offsets t-1 or older")
    if train_offsets and valid_offsets and min(train_offsets) <= max(valid_offsets):
        errors.append("train rows must be strictly earlier than validation rows")
    predict = payload.get("predict")
    if not isinstance(predict, Mapping):
        errors.append("predict must be an object")
    else:
        if predict.get("label_included") is not False:
            errors.append("predict.label_included must be false")
        if predict.get("rows_visible_to_llm") is not False:
            errors.append("predict.rows_visible_to_llm must be false")
        if predict.get("artifact") is not None:
            errors.append("predict.artifact must be omitted from the public payload")
        rows = predict.get("rows")
        if rows is not None:
            if not isinstance(rows, list):
                errors.append("predict.rows must be omitted or an array")
            else:
                for index, row in enumerate(rows):
                    if not isinstance(row, Mapping):
                        errors.append(f"predict.rows[{index}] must be an object")
                        break
                    protected_row_columns = _find_protected_columns(row.keys())
                    if protected_row_columns:
                        errors.append(
                            f"predict.rows[{index}] contains protected evaluator columns: "
                            f"{sorted(protected_row_columns)}"
                        )
                        break
                    calendar_keys = [
                        str(key)
                        for key in row
                        if _normalise_column(key)
                        in {"eom", "date", "month", "anchor_date", "formation_eom", "formation_month", "origin_month", "as_of_month"}
                    ]
                    if calendar_keys:
                        errors.append(
                            f"predict.rows[{index}] contains evaluator-only calendar columns: {calendar_keys}"
                        )
                        break
                    if _find_label_columns(row):
                        errors.append(f"predict.rows[{index}] contains a label/target column")
                        break
                    expected_predict_columns = ["id", "time_key", *feature_names]
                    if list(row.keys()) != expected_predict_columns:
                        errors.append(
                            f"predict.rows[{index}] columns must be exactly {expected_predict_columns}"
                        )
        schema = predict.get("schema")
        if isinstance(schema, list):
            expected_predict_columns = ["id", "time_key", *feature_names]
            if schema != expected_predict_columns:
                errors.append(f"predict.schema must be exactly {expected_predict_columns}")
            schema_names = {str(column) for column in schema}
            if len(schema_names) != len(schema):
                errors.append("predict.schema must not contain duplicate columns")
            required_predict_columns = {"id", "time_key", *feature_names}
            missing_predict_columns = sorted(required_predict_columns - schema_names)
            if missing_predict_columns:
                errors.append(f"predict.schema omits required columns: {missing_predict_columns}")
            calendar_columns = [
                str(column)
                for column in schema
                if _normalise_column(column)
                in {"eom", "date", "month", "anchor_date", "formation_eom", "formation_month", "origin_month", "as_of_month"}
            ]
            if calendar_columns:
                errors.append(f"predict.schema contains evaluator-only calendar columns: {calendar_columns}")
            if _find_label_columns(schema):
                errors.append("predict.schema contains a label/target column")
        else:
            errors.append("predict.schema must be an array")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        errors.append("metadata must be an object")
    elif metadata.get("future_labels_included") is True:
        errors.append("metadata.future_labels_included must be false")
    if isinstance(metadata, Mapping):
        for key in legacy_version_fields.intersection({str(item) for item in metadata}):
            errors.append(f"metadata.{key} is a legacy payload field")
        # Semantic cards have one authoritative location at the payload root.
        # A second list hidden in generic metadata could otherwise smuggle
        # cards into a numeric-only runtime or create a mismatched sidecar.
        for card_key in ("factor_cards", "semantic_cards"):
            if card_key in metadata and metadata.get(card_key) not in (None, [], {}):
                errors.append(f"metadata.{card_key} must not contain semantic cards")
        if metadata.get("id_column") not in {None, "id"}:
            errors.append("metadata.id_column must be id")
        if metadata.get("time_column") not in {None, "time_key"}:
            errors.append("metadata.time_column must be time_key")
        for key, value in metadata.items():
            normalized_key = _normalise_column(key)
            if normalized_key == "future_labels_included" and value is not False:
                errors.append("metadata.future_labels_included must be false")
            if any(token in normalized_key for token in ("path", "filename", "mapping", "secret", "token")):
                errors.append(f"metadata.{key} is evaluator-only")
            if isinstance(value, str) and _CALENDAR_LITERAL_RE.search(value):
                errors.append(f"metadata.{key} must not expose a calendar date")
    for key in (
        "formation_eom",
        "formation_month",
        "origin_month",
        "as_of_month",
        "anchor_date",
        "calendar_date",
    ):
        if isinstance(metadata, Mapping) and key in metadata:
            errors.append(f"metadata.{key} must not expose a calendar date")
    return list(dict.fromkeys(errors))


def _parse_month(value: Any) -> pd.Timestamp:
    if isinstance(value, pd.Period):
        return value.to_timestamp("M")
    if (
        isinstance(value, (int, np.integer))
        or isinstance(value, (float, np.floating)) and math.isfinite(float(value)) and float(value).is_integer()
    ) and 100001 <= int(value) <= 999912:
        parsed = pd.to_datetime(str(int(value)), format="%Y%m", errors="coerce")
    elif isinstance(value, str) and re.fullmatch(r"\d{6}", value.strip()):
        parsed = pd.to_datetime(value.strip(), format="%Y%m", errors="coerce")
    else:
        try:
            parsed = pd.to_datetime(value, format="mixed", errors="coerce", utc=True)
        except (TypeError, ValueError):  # pragma: no cover - old pandas
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(value, errors="coerce", utc=True)
        if getattr(parsed, "tzinfo", None) is not None:
            parsed = pd.Timestamp(parsed).tz_convert(None)
    if pd.isna(parsed):
        raise ValueError(f"could not parse month: {value!r}")
    return pd.Timestamp(parsed).to_period("M").to_timestamp("M")


def materialize_split(
    payload: Mapping[str, Any],
    split: str,
    *,
    include_label: bool | None = None,
    artifact_root: str | Path | None = None,
) -> pd.DataFrame:
    """Materialize a payload split while enforcing the label boundary."""

    if split not in {"train", "validation", "predict"}:
        raise ValueError("split must be train, validation, or predict")
    source = payload.get(split)
    if not isinstance(source, Mapping):
        raise ValueError(f"payload.{split} must be an object")
    rows = source.get("rows")
    frame: pd.DataFrame
    if isinstance(rows, list) and rows:
        invalid_rows = [index for index, row in enumerate(rows) if not isinstance(row, Mapping)]
        if invalid_rows:
            raise ValueError(f"payload.{split}.rows contains non-object rows: {invalid_rows[:5]}")
        frame = pd.DataFrame([dict(row) for row in rows])
    else:
        descriptor = source.get("artifact")
        if not isinstance(descriptor, Mapping) or not str(descriptor.get("path") or "").strip():
            raise ValueError(f"payload.{split}.rows is empty and no public artifact is declared")
        root = Path(artifact_root or payload.get("artifact_root") or ".").resolve()
        raw_artifact_path = str(descriptor["path"]).replace("\\", "/")
        if _unsafe_relative_path(raw_artifact_path):
            raise ValueError(f"payload.{split} artifact path must stay inside artifact_root")
        path = (root / raw_artifact_path).resolve()
        if root not in path.parents and path != root:
            raise ValueError(f"payload.{split} artifact escapes artifact_root")
        if not path.is_file():
            raise FileNotFoundError(f"public artifact does not exist: {path}")
        if "bytes" in descriptor:
            try:
                expected_bytes = int(descriptor["bytes"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"payload.{split} artifact.bytes must be an integer") from exc
            actual_bytes = int(path.stat().st_size)
            if actual_bytes != expected_bytes:
                raise ValueError(
                    f"payload.{split} artifact byte count mismatch: expected {expected_bytes}, got {actual_bytes}"
                )
        if descriptor.get("sha256"):
            expected_digest = str(descriptor["sha256"]).lower()
            if _file_sha256(path) != expected_digest:
                raise ValueError(f"payload.{split} artifact sha256 mismatch")
        fmt = str(descriptor.get("format") or path.suffix.lstrip(".") or "jsonl").lower()
        if fmt in {"jsonl", "ndjson"}:
            frame = pd.read_json(path, lines=True)
        elif fmt == "csv":
            frame = pd.read_csv(path)
        elif fmt in {"parquet", "pq"}:
            frame = pd.read_parquet(path)
        else:
            raise ValueError(f"unsupported public artifact format: {fmt}")
        if "rows" in descriptor:
            try:
                expected_rows = int(descriptor["rows"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"payload.{split} artifact.rows must be an integer") from exc
            if expected_rows != len(frame):
                raise ValueError(
                    f"payload.{split} artifact row count mismatch: expected {expected_rows}, got {len(frame)}"
                )
    if frame.empty:
        raise ValueError(f"payload.{split}.rows is empty")
    feature_names = [str(feature) for feature in payload.get("feature_columns", [])]
    expected_columns = ["id", "time_key", *feature_names]
    if split in {"train", "validation"}:
        expected_columns.append("label")
    if list(map(str, frame.columns)) != expected_columns:
        raise ValueError(
            f"payload.{split} columns must be exactly {expected_columns}; "
            f"got {list(map(str, frame.columns))}"
        )
    protected_columns = _find_protected_columns(frame.columns)
    if protected_columns:
        raise ValueError(
            f"payload.{split} contains protected evaluator columns: "
            f"{sorted(protected_columns)}"
        )
    if "id" not in frame.columns:
        raise ValueError(f"payload.{split} is missing id")
    frame["id"] = frame["id"].map(_canonical_identifier).astype("string")
    if frame["id"].isna().any() or frame["id"].str.strip().eq("").any():
        raise ValueError(f"payload.{split} contains null/empty ids")
    if "time_key" in frame.columns:
        frame["time_key"] = frame["time_key"].astype("string")
        if frame["time_key"].isna().any() or frame["time_key"].map(_relative_time_offset).isna().any():
            raise ValueError(f"payload.{split}.time_key must use relative t-N keys")
        # Collapse harmless case/format variations (``T-01`` vs ``t-1``)
        # before duplicate and chronological checks.
        frame["time_key"] = frame["time_key"].map(_relative_time_key).astype("string")
        if split in {"train", "validation"}:
            offsets = frame["time_key"].map(_relative_time_offset).astype("int64")
            if (offsets <= 0).any():
                raise ValueError(f"payload.{split} historical rows must use offsets t-1 or older")
        else:
            offsets = frame["time_key"].map(_relative_time_offset).astype("int64")
    elif split in {"train", "validation", "predict"}:
        raise ValueError(f"payload.{split} is missing time_key")
    # Calendar aliases are evaluator-only.  Keeping one in a mounted history
    # artifact would let a model infer the hidden origin even if the visible
    # key is relative, so reject them rather than silently dropping them.
    calendar_columns = [
        column
        for column in frame.columns
        if _normalise_column(column)
        in {"eom", "date", "month", "anchor_date", "formation_eom", "formation_month", "origin_month", "as_of_month"}
    ]
    if calendar_columns:
        raise ValueError(f"payload.{split} contains evaluator-only calendar columns: {calendar_columns}")
    feature_names = [str(feature) for feature in payload.get("feature_columns", [])] if isinstance(payload.get("feature_columns"), list) else []
    missing_features = [feature for feature in feature_names if feature not in frame.columns]
    if missing_features:
        raise ValueError(f"payload.{split} is missing requested features: {missing_features}")
    if include_label is None:
        include_label = split in {"train", "validation"}
    label_columns = _find_label_columns(frame.columns)
    if split == "predict" and label_columns:
        raise ValueError(f"predict split contains forbidden label columns: {sorted(label_columns)}")
    if include_label:
        # A historical frame may use one supported alias in place of ``label``;
        # multiple target-like columns are ambiguous and can hide a future
        # target behind a harmless-looking name.
        aliases = [column for column in frame.columns if _normalise_column(column) in _LABEL_NAMES]
        if "label" in frame.columns:
            extra_aliases = [column for column in aliases if column != "label"]
            if extra_aliases or any(
                _normalise_column(column).startswith(("future_", "target_")) for column in frame.columns
            ):
                raise ValueError(f"payload.{split} contains ambiguous/future label columns: {sorted(label_columns)}")
        elif len(aliases) > 1 or any(
            _normalise_column(column).startswith(("future_", "target_")) for column in frame.columns
        ):
            raise ValueError(f"payload.{split} contains ambiguous/future label columns: {sorted(label_columns)}")
    if include_label and "label" not in frame.columns:
        # Normalize one of the explicitly supported aliases.
        aliases = [column for column in frame.columns if _normalise_column(column) in _LABEL_NAMES]
        if aliases:
            frame = frame.rename(columns={aliases[0]: "label"})
    if include_label and "label" not in frame.columns:
        raise ValueError(f"payload.{split} has no label column")
    if not include_label:
        drop_columns = [column for column in frame.columns if column == "label" or column in _find_label_columns(frame.columns)]
        if drop_columns:
            frame = frame.drop(columns=drop_columns)
    if frame.duplicated(["id", "time_key"], keep=False).any():
        raise ValueError(f"payload.{split} contains duplicate id/time_key rows")
    return frame.reset_index(drop=True)


def _frame_rows(frame: pd.DataFrame, columns: Sequence[str]) -> list[dict[str, Any]]:
    existing = [column for column in columns if column in frame.columns]
    return [{str(k): _json_safe(v) for k, v in row.items()} for row in frame[existing].to_dict("records")]


def _public_artifact_descriptor(
    descriptor: Mapping[str, Any], columns: Sequence[str]
) -> dict[str, Any]:
    """Normalize the provider-facing descriptor without exposing old fields."""

    allowed = {"path", "format", "rows", "columns", "bytes", "sha256"}
    result = {
        str(key): _json_safe(value)
        for key, value in descriptor.items()
        if str(key) in allowed
    }
    result["columns"] = [str(column) for column in columns]
    return result


def build_code_generation_payload(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    case_id: str = "task_c_case",
    metadata: Mapping[str, Any] | None = None,
    include_rows: bool = True,
    train_artifact: Mapping[str, Any] | None = None,
    validation_artifact: Mapping[str, Any] | None = None,
    track: str = "numeric_only",
    factor_cards: Sequence[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a public code-generation payload from historical frames.

    ``validation_df`` is historical and must contain labels.  The predict
    section is intentionally schema-only; hidden test rows are injected by the
    local runner after code has been frozen.
    """

    track = normalize_track(track)
    # Artifact-only payloads should not duplicate a large historical frame in
    # memory. Rows are copied only when they are actually embedded in JSON.
    train = train_df.copy() if include_rows else train_df
    valid = validation_df.copy() if include_rows else validation_df
    for name, frame in (("train", train), ("validation", valid)):
        if "id" not in frame.columns:
            raise ValueError(f"{name}_df must contain id")
        if "label" not in frame.columns:
            raise ValueError(f"{name}_df must contain label")
    features = [str(value) for value in feature_columns]
    if (
        not features
        or any(not feature.strip() for feature in features)
        or len(set(features)) != len(features)
        or len({_normalise_column(feature) for feature in features}) != len(features)
    ):
        raise ValueError("feature_columns must be a non-empty unique sequence")
    forbidden_features = [
        feature
        for feature in features
        if _normalise_column(feature) in _LABEL_NAMES
        or _normalise_column(feature).startswith(("future_", "target_"))
        or _normalise_column(feature)
        in {"id", "time_key", "eom", "date", "month", "anchor_date", "formation_eom", "formation_month", "origin_month", "as_of_month"}
    ]
    if forbidden_features:
        raise ValueError(f"feature_columns contain label/future/calendar names: {forbidden_features}")
    missing_features = {
        name: [feature for feature in features if feature not in frame.columns]
        for name, frame in (("train", train), ("validation", valid))
    }
    missing_features = {name: values for name, values in missing_features.items() if values}
    if missing_features:
        raise ValueError(f"frames are missing requested features: {missing_features}")
    for name, frame in (("train", train), ("validation", valid)):
        if not any(column in frame.columns for column in ("time_key", "eom", "date")):
            raise ValueError(f"{name}_df must contain time_key/eom/date")
    metadata_input = dict(metadata or {})
    formation_period: pd.Period | None = None
    if metadata_input.get("formation_eom") is not None:
        formation_period = _parse_month(metadata_input["formation_eom"]).to_period("M")
    elif metadata_input.get("formation_period") is not None:
        formation_period = _parse_month(metadata_input["formation_period"]).to_period("M")
    # Infer one common origin when callers provide calendar dates but omit an
    # explicit formation month.  This is done even for artifact-only cases;
    # the large frame is not copied, only its vectorised month column is read.
    if formation_period is None:
        observed_periods: list[pd.Period] = []
        for frame in (train, valid):
            source = _time_source(frame)
            raw = source.astype("string")
            if not bool(raw.map(_relative_time_offset).notna().all()):
                parsed = _parse_calendar_series(source).dropna().dt.to_period("M")
                observed_periods.extend(parsed.tolist())
        if observed_periods:
            formation_period = max(observed_periods) + 1
    if include_rows:
        # Copy and rewrite embedded rows into the relative public namespace.
        train = _relative_time_frame(train, formation_period=formation_period)
        valid = _relative_time_frame(valid, formation_period=formation_period)
    elif train_artifact is None or validation_artifact is None:
        raise ValueError("include_rows=False requires both train_artifact and validation_artifact")
    else:
        # The descriptor points to the exact bytes that will be mounted later;
        # silently rewriting only the in-memory preview would create a payload
        # that validates but fails (or leaks dates) when the artifact is read.
        for name, frame in (("train", train), ("validation", valid)):
            calendar_columns = [
                column
                for column in frame.columns
                if _normalise_column(column)
                in {"eom", "date", "month", "anchor_date", "formation_eom", "formation_month", "origin_month", "as_of_month"}
            ]
            if calendar_columns:
                raise ValueError(f"artifact-only {name}_df must not contain calendar columns: {calendar_columns}")
            if "time_key" not in frame.columns or frame["time_key"].map(_relative_time_offset).isna().any():
                raise ValueError(f"artifact-only {name}_df must use relative t-N time_key values")
        for name, descriptor, frame in (
            ("train", train_artifact, train),
            ("validation", validation_artifact, valid),
        ):
            if not isinstance(descriptor, Mapping) or not str(descriptor.get("path") or "").strip():
                raise ValueError(f"{name}_artifact must declare a non-empty path")
            if isinstance(descriptor, Mapping) and descriptor.get("rows") is not None:
                try:
                    declared_rows = int(descriptor["rows"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{name}_artifact.rows must be an integer") from exc
                if declared_rows != len(frame):
                    raise ValueError(
                        f"{name}_artifact.rows does not match supplied frame: "
                        f"expected {len(frame)}, got {declared_rows}"
                    )
    # Normalize alternate date names to the stable string field exposed to the
    # model.  The real WRDS window builder already supplies ``time_key``; this
    # branch keeps the public constructor safe for hand-built fixtures.
    normalized_frames: list[pd.DataFrame] = []
    for frame in (train, valid):
        if "time_key" not in frame.columns:
            source_column = next(column for column in ("eom", "date") if column in frame.columns)
            frame = frame.copy()
            parsed = _parse_calendar_series(frame[source_column])
            frame["time_key"] = parsed.dt.to_period("M").astype("string")
        else:
            string_like = pd.api.types.is_string_dtype(frame["time_key"].dtype)
            if not string_like:
                frame = frame.copy()
                frame["time_key"] = frame["time_key"].astype("string")
            elif include_rows:
                frame = frame.copy()
                frame["time_key"] = frame["time_key"].astype("string")
        normalized_frames.append(frame)
    train, valid = normalized_frames
    common = ["id", "time_key", *features, "label"]
    common = list(dict.fromkeys(common))
    metadata_out = {
        "id_column": "id",
        "time_column": "time_key",
        "target_is_excess_return": True,
        "future_labels_included": False,
        "seed": 1,
    }
    if metadata:
        # Calendar dates, local paths, ID maps and test/truth routing fields are
        # evaluator-only. Preserve neutral experiment metadata but do not let a
        # caller accidentally re-introduce a private field through this generic
        # dictionary.
        blocked_metadata = {
            "formation_eom",
            "formation_period",
            "formation_month",
            "origin_month",
            "as_of_month",
            "anchor_date",
            "calendar_date",
            "formation_date",
            "as_of_date",
            "test_start",
            "test_end",
            "artifact_root",
            "public_artifact_files",
            "original_to_masked",
            "masked_to_original",
            "id_mapping",
            "truth",
            "ground_truth",
            "future_labels_included",
            "id_column",
            "time_column",
            "schema_version",
            "prompt_semantic_version",
            "semantic_card_schema_version",
            "track",
            "factor_cards",
            "semantic_cards",
        }
        for key, value in metadata.items():
            key_text = str(key)
            normalized_key = _normalise_column(key_text)
            if key_text in blocked_metadata or normalized_key in blocked_metadata:
                continue
            if any(
                token in normalized_key
                for token in ("path", "filename", "filepath", "secret", "api_key", "token")
            ):
                continue
            metadata_out[key_text] = _json_safe(value)
    metadata_out["time_key_encoding"] = "relative_months_from_formation"
    # Safety invariants must win over arbitrary caller metadata.
    metadata_out["future_labels_included"] = False
    train_section: dict[str, Any] = {
        "rows": _frame_rows(train, common) if include_rows else [],
    }
    valid_section: dict[str, Any] = {
        "rows": _frame_rows(valid, common) if include_rows else [],
    }
    if train_artifact is not None:
        train_section["artifact"] = _public_artifact_descriptor(train_artifact, common)
    if validation_artifact is not None:
        valid_section["artifact"] = _public_artifact_descriptor(validation_artifact, common)
    supplied_cards: list[dict[str, Any]] = []
    if factor_cards is not None:
        if isinstance(factor_cards, Mapping):
            # A mapping key is a valid card ID even when the nested object does
            # not repeat ``factor_id``.  This makes the public constructor's
            # advertised Mapping form deterministic and lets us order cards by
            # ``feature_columns`` instead of relying on insertion order.
            for key, value in factor_cards.items():
                if not isinstance(value, Mapping):
                    continue
                item = dict(value)
                # The mapping key is the public feature id; an enriched source
                # record may still carry its private/raw ``factor_id`` inside.
                # Override it rather than allowing that source id to escape.
                if str(item.get("factor_id", "")).strip() and str(item.get("factor_id")) != str(key):
                    item["_raw_factor_id"] = str(item["factor_id"])
                item["factor_id"] = str(key)
                supplied_cards.append(item)
        else:
            supplied_cards = [
                dict(value) for value in factor_cards if isinstance(value, Mapping)
            ]
    if track == "semantic":
        by_id: dict[str, dict[str, Any]] = {}
        for card in supplied_cards:
            factor_id = str(card.get("factor_id", "")).strip()
            if factor_id and factor_id not in by_id:
                by_id[factor_id] = card
        # The feature list is the canonical public order.  Missing/duplicate
        # cards are still rejected by ``validate_public_payload`` below.
        public_cards = [
            semantic_card_payload(by_id[feature], feature)
            for feature in features
            if feature in by_id
        ]
    else:
        # Preserve the fact that a caller supplied cards so the numeric-only
        # validator fails closed instead of silently discarding semantic data.
        public_cards = supplied_cards
    payload: dict[str, Any] = {
        "case_id": str(case_id),
        "task": "stock_multifactor_model",
        "horizon_months": 1,
        "track": track,
        "feature_columns": features,
        "factor_cards": public_cards,
        "train": train_section,
        "validation": valid_section,
        "predict": {
            "schema": ["id", "time_key", *features],
            "label_included": False,
            "rows_visible_to_llm": False,
        },
        "metadata": metadata_out,
    }
    errors = validate_public_payload(payload)
    if errors:
        raise ValueError("invalid code-generation payload: " + "; ".join(errors[:8]))
    return payload


def _normalise_panel_months(
    panel: pd.DataFrame,
    time_column: str | None = None,
    *,
    formation_period: pd.Period | None = None,
) -> pd.DataFrame:
    frame = panel.copy()
    column = time_column or next((c for c in ("eom", "date", "time_key", "month") if c in frame.columns), None)
    if column is None:
        raise ValueError("panel requires eom/date/time_key/month")
    raw_time = frame[column].astype("string")
    relative_offsets = raw_time.map(_relative_time_offset)
    if bool(relative_offsets.notna().any()):
        if not bool(relative_offsets.notna().all()):
            raise ValueError("panel time column mixes relative and calendar encodings")
        if formation_period is None:
            raise ValueError("relative panel time_key requires formation_eom")
        frame["_month"] = relative_offsets.map(
            lambda value: (formation_period - int(value)).to_timestamp("M")
        )
    elif isinstance(frame[column].dtype, pd.PeriodDtype):
        frame["_month"] = frame[column].dt.to_timestamp("M")
    else:
        parsed = _parse_calendar_series(frame[column])
        frame["_month"] = parsed.dt.to_period("M").dt.to_timestamp("M")
    frame = frame.dropna(subset=["_month"])
    if "id" not in frame.columns:
        raise ValueError("panel requires id")
    frame["id"] = frame["id"].map(_canonical_identifier).astype("string")
    if frame["id"].isna().any() or frame["id"].eq("").any():
        frame = frame.loc[frame["id"].notna() & frame["id"].ne("")].copy()
    duplicate = frame.duplicated(["id", "_month"], keep=False)
    if duplicate.any():
        sample = frame.loc[duplicate, ["id", "_month"]].head(5).to_dict("records")
        raise ValueError(f"panel contains duplicate id/month observations (sample={sample})")
    return frame


def build_rolling_windows(
    panel: pd.DataFrame,
    formation_eom: str | pd.Timestamp,
    *,
    feature_columns: Sequence[str],
    train_months: int = DEFAULT_TRAIN_MONTHS,
    valid_months: int = DEFAULT_VALID_MONTHS,
    label_column: str = "label",
    time_column: str | None = None,
    generate_lead_label_from: str | None = None,
    relative_time_keys: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Construct the leakage-safe ``train_t/valid_t/predict_t`` windows.

    For formation month ``t`` the default ranges are ``t-132..t-13`` for
    training, ``t-12..t-1`` for validation, and ``t`` for prediction.  The
    returned prediction frame never contains a label column.
    """

    if train_months < 1 or valid_months < 1:
        raise ValueError("train_months and valid_months must be positive")
    features = [str(c) for c in feature_columns]
    if (
        not features
        or any(not feature.strip() for feature in features)
        or len(set(features)) != len(features)
        or len({_normalise_column(feature) for feature in features}) != len(features)
    ):
        raise ValueError("feature_columns must be a non-empty unique sequence")
    forbidden_features = [
        column
        for column in features
        if _normalise_column(column) in _LABEL_NAMES
        or _normalise_column(column).startswith(("future_", "target_"))
        or _normalise_column(column)
        in {"id", "time_key", "eom", "date", "month", "anchor_date", "formation_eom", "formation_month", "origin_month", "as_of_month"}
    ]
    if forbidden_features:
        raise ValueError(f"feature_columns contain label/future/calendar columns: {forbidden_features}")
    if generate_lead_label_from is not None and (
        _normalise_column(generate_lead_label_from) in _LABEL_NAMES
        or _normalise_column(generate_lead_label_from).startswith(("future_", "target_"))
    ):
        raise ValueError("generate_lead_label_from must be an observed contemporaneous return column")
    t = _parse_month(formation_eom)
    period = t.to_period("M")
    frame = _normalise_panel_months(panel, time_column=time_column, formation_period=period)
    if generate_lead_label_from is not None and label_column not in frame.columns:
        if generate_lead_label_from not in frame.columns:
            raise ValueError(f"cannot generate label: missing {generate_lead_label_from}")
        frame = frame.sort_values(["id", "_month"])
        next_month = frame.groupby("id", sort=False)[generate_lead_label_from].shift(-1)
        next_date = frame.groupby("id", sort=False)["_month"].shift(-1)
        frame[label_column] = pd.to_numeric(next_month, errors="coerce").where(
            next_date.eq(frame["_month"] + pd.offsets.MonthEnd(1))
        )
    missing = [c for c in [*features, label_column] if c not in frame.columns]
    if missing:
        raise ValueError(f"panel missing required columns: {missing}")
    valid_start = period - valid_months
    valid_end = period - 1
    train_start = valid_start - train_months
    train_end = valid_start - 1
    train = frame.loc[frame["_month"].dt.to_period("M").between(train_start, train_end)].copy()
    valid = frame.loc[frame["_month"].dt.to_period("M").between(valid_start, valid_end)].copy()
    predict = frame.loc[frame["_month"].dt.to_period("M").eq(period)].copy()
    if train.empty or valid.empty or predict.empty:
        raise ValueError(
            f"insufficient rows for formation {t.strftime('%Y-%m')}: "
            f"train={len(train)}, validation={len(valid)}, predict={len(predict)}"
        )
    # Rows without a realized historical label cannot be used for fitting or
    # validation.  They are retained in diagnostics rather than imputed.
    train = train.dropna(subset=[label_column]).copy()
    valid = valid.dropna(subset=[label_column]).copy()
    if train.empty or valid.empty:
        raise ValueError("training/validation labels are all missing")
    columns = list(dict.fromkeys(["id", "_month", *features, label_column]))
    train = train[columns].rename(columns={"_month": "time_key"})
    valid = valid[columns].rename(columns={"_month": "time_key"})
    predict_columns = list(dict.fromkeys(["id", "_month", *features]))
    predict = predict[predict_columns].rename(columns={"_month": "time_key"})
    # Use relative month labels at the model boundary by default.  The
    # evaluator retains the actual formation date in its local audit row, but
    # the provider never receives calendar dates that could identify the test
    # period.  ``relative_time_keys=False`` remains available for diagnostics.
    def key_for(value: Any) -> str:
        stamp = pd.Timestamp(value).to_period("M")
        if relative_time_keys:
            offset = int(period.ordinal - stamp.ordinal)
            return f"t-{offset}"
        return stamp.strftime("%Y-%m")

    for subset in (train, valid, predict):
        subset["time_key"] = subset["time_key"].map(key_for)
    metadata = {
        "formation_key": "t-0" if relative_time_keys else t.strftime("%Y-%m-%d"),
        "train_month_range": [
            f"t-{period.ordinal - train_start.ordinal}",
            f"t-{period.ordinal - train_end.ordinal}",
        ] if relative_time_keys else [train_start.strftime("%Y-%m"), train_end.strftime("%Y-%m")],
        "validation_month_range": [
            f"t-{period.ordinal - valid_start.ordinal}",
            f"t-{period.ordinal - valid_end.ordinal}",
        ] if relative_time_keys else [valid_start.strftime("%Y-%m"), valid_end.strftime("%Y-%m")],
        "predict_month": "t-0" if relative_time_keys else period.strftime("%Y-%m"),
        "train_months": int(train_months),
        "validation_months": int(valid_months),
        "label_column": label_column,
        "feature_columns": features,
        "future_labels_included": False,
        "time_key_encoding": "relative_months_from_formation" if relative_time_keys else "calendar_month",
    }
    return train.reset_index(drop=True), valid.reset_index(drop=True), predict.reset_index(drop=True), metadata


def _rank_corr(x: pd.Series, y: pd.Series) -> float | None:
    frame = pd.concat([pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")], axis=1).dropna()
    if len(frame) < 2:
        return None
    a = frame.iloc[:, 0].rank(method="average").to_numpy(dtype=float)
    b = frame.iloc[:, 1].rank(method="average").to_numpy(dtype=float)
    if np.std(a) <= 0 or np.std(b) <= 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def validation_metrics(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    id_column: str = "id",
    time_column: str = "time_key",
    label_column: str = "label",
) -> dict[str, Any]:
    """Compute fixed, cross-sectional validation metrics outside model code."""

    required_pred = {id_column, "score"}
    required_label = {id_column, label_column}
    if not required_pred.issubset(predictions.columns):
        return {"rank_ic_mean": None, "rank_ic_std": None, "n_months": 0, "n_rows": 0, "error": "prediction columns missing"}
    if not required_label.issubset(labels.columns):
        return {"rank_ic_mean": None, "rank_ic_std": None, "n_months": 0, "n_rows": 0, "error": "label columns missing"}
    p = predictions.copy()
    y = labels.copy()
    p[id_column] = p[id_column].map(_canonical_identifier).astype("string")
    y[id_column] = y[id_column].map(_canonical_identifier).astype("string")
    keys = [id_column]
    if time_column in p.columns and time_column in y.columns:
        keys.append(time_column)
    merged = p.merge(y[keys + [label_column]], on=keys, how="inner", validate="one_to_one")
    merged["score"] = pd.to_numeric(merged["score"], errors="coerce")
    merged[label_column] = pd.to_numeric(merged[label_column], errors="coerce")
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna(subset=["score", label_column])
    if merged.empty:
        return {"rank_ic_mean": None, "rank_ic_std": None, "n_months": 0, "n_rows": 0, "mse": None, "directional_accuracy": None}
    if time_column in merged.columns:
        ics = [value for _, group in merged.groupby(time_column, sort=True) if (value := _rank_corr(group["score"], group[label_column])) is not None]
    else:
        value = _rank_corr(merged["score"], merged[label_column])
        ics = [] if value is None else [value]
    errors = merged["score"] - merged[label_column]
    return {
        "rank_ic_mean": float(np.mean(ics)) if ics else None,
        "rank_ic_std": float(np.std(ics, ddof=1)) if len(ics) > 1 else 0.0 if ics else None,
        "n_months": int(len(ics)),
        "n_rows": int(len(merged)),
        "mse": float(np.mean(np.square(errors.to_numpy(dtype=float)))),
        "directional_accuracy": float(np.mean(np.sign(merged["score"]) == np.sign(merged[label_column]))),
    }


def public_candidate_for_feedback(response: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Keep the previous generated source available to the next revision.

    Revision requests are stateless provider calls.  Sending only aggregate
    validation errors made a model regenerate the envelope from scratch and
    frequently lose fields that were already correct.  The previous candidate
    is public (it contains no evaluator truth), is bounded by the same
    source-size limit as the response contract, and is included only as repair
    context -- not as a test signal.  Mirrors the Task B helper so both tasks
    hand the model the same repair context.
    """

    if not isinstance(response, Mapping):
        return None
    candidate: dict[str, Any] = {}
    for key in (
        "revision",
        "status",
        "language",
        "entrypoint",
        "requirements",
        "input_contract",
        "output_contract",
        "explanation",
        "revision_summary",
    ):
        if key in response:
            candidate[key] = _json_safe(response.get(key))
    if "code" in response:
        candidate["code"] = str(response.get("code") or "")[:MAX_CODE_CHARS]
    return candidate


def validation_feedback(
    *,
    revision: int,
    validation: Mapping[str, Any] | None,
    io_checks: Mapping[str, Any] | None = None,
    runtime: Mapping[str, Any] | None = None,
    errors: Sequence[str] = (),
    allowed_next_revision: bool = True,
    previous_candidate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the bounded public envelope sent to the next LLM revision."""

    metrics = dict(validation or {})
    # Never copy row-level labels, IDs, dates, or raw trace text into feedback.
    public_metrics = {
        key: _json_safe(metrics.get(key))
        for key in (
            "rank_ic_mean",
            "rank_ic_std",
            "mse",
            "directional_accuracy",
            "net_return",
            "gross_return",
            "turnover",
            "coverage",
            "n_months",
            "n_rows",
        )
        if key in metrics
    }
    backtest_metrics = metrics.get("validation_backtest")
    if isinstance(backtest_metrics, Mapping):
        allowed_backtest = {
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
            "status",
        }
        public_metrics["validation_backtest"] = {
            str(key): _json_safe(value)
            for key, value in backtest_metrics.items()
            if str(key) in allowed_backtest
            and (
                isinstance(value, (bool, int, float, np.integer, np.floating, str))
            )
        }
    def redact_error(error: Any) -> str:
        # Validation errors are allowed to be useful for repair, but they must
        # not become an accidental side channel for row-level IDs, dates, or
        # local paths (especially on the explicit outer-holdout track).
        text = str(error)
        text = re.sub(r"sample\s*=\s*\[[^\]]*\]", "sample=[redacted]", text, flags=re.IGNORECASE)
        text = re.sub(r"\b(?:missing|unknown|extra|omits)\s+ids?\b[^;]*", "ID diagnostics [redacted]", text, flags=re.IGNORECASE)
        text = re.sub(r"(?:[A-Za-z]:)?[/\\][^\s;,)]+", "[redacted path]", text)
        text = re.sub(
            r"\b(?:(?:19|20)\d{2}-\d{2}(?:-\d{2})?(?:[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?)?|(?:19|20)\d{4})\b",
            "[redacted date]",
            text,
        )
        text = re.sub(r"\bI(?:[_-]?\d+)+\b|\bID[_-]?[A-Za-z0-9]+\b", "[redacted id]", text, flags=re.IGNORECASE)
        text = re.sub(r"\b[0-9a-f]{64}\b", "[redacted hash]", text, flags=re.IGNORECASE)
        return text[:500]

    public_errors = [redact_error(error) for error in list(errors)[:20]]

    def public_runtime_map(value: Mapping[str, Any] | None) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, raw in (value or {}).items():
            # Only scalar diagnostics are useful to a repair prompt.  Drop
            # arbitrary nested payloads/tracebacks so a caller cannot turn
            # this envelope into a row-level data side channel.
            if isinstance(raw, (bool, int, float, np.integer, np.floating, str)):
                safe = _json_safe(raw)
                if isinstance(safe, str):
                    safe = safe[:200]
                output[str(key)] = safe
        return output

    envelope = {
        "revision": int(revision),
        "candidate_status": "valid" if not public_errors else "invalid",
        "validation": public_metrics,
        "io_checks": public_runtime_map(io_checks),
        "runtime": public_runtime_map(runtime),
        "errors": public_errors,
        "allowed_next_revision": bool(allowed_next_revision),
    }
    prior = public_candidate_for_feedback(previous_candidate)
    if prior is not None:
        envelope["previous_candidate"] = prior
    return envelope


__all__ = [
    "PAYLOAD_SCHEMA_VERSION",
    "CODE_RESPONSE_SCHEMA_VERSION",
    "PROMPT_SEMANTIC_VERSION",
    "DEFAULT_TRAIN_MONTHS",
    "DEFAULT_VALID_MONTHS",
    "build_code_generation_payload",
    "build_rolling_windows",
    "code_response_hash",
    "materialize_split",
    "stable_hash",
    "validate_code_response",
    "validate_public_payload",
    "validation_feedback",
    "validation_metrics",
]

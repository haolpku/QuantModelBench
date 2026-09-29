"""Point-in-time and masking audits for Task C data preparation."""

from __future__ import annotations

import re
import warnings
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


def _canonical_identifier(value: Any) -> str:
    """Normalize numeric IDs consistently across CSV/Parquet boundaries."""

    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    text = str(value).strip()
    if text.lower() in {"", "nan", "<na>", "nat", "none"}:
        return ""
    # Normalize an unambiguous decimal/integer-like representation such as
    # ``123.0`` (CSV/Arrow may stringify an integer identifier this way), but
    # preserve explicit leading-zero identifiers such as ``00123``.  The
    # latter can be legitimate security codes rather than numeric casts.
    if re.fullmatch(r"[+-]?(?:0|[1-9]\d*)\.0+", text):
        try:
            return str(int(float(text)))
        except (OverflowError, ValueError):
            pass
    return text


def _formation_period(value: Any) -> pd.Period:
    if isinstance(value, pd.Period):
        return value.asfreq("M")
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
        raise ValueError(f"could not parse formation month: {value!r}")
    return pd.Timestamp(parsed).to_period("M")


def mask_identifiers(
    frames: Sequence[pd.DataFrame],
    *,
    id_column: str = "id",
    prefix: str = "I",
) -> tuple[list[pd.DataFrame], dict[str, str]]:
    """Return copies with deterministic masked IDs and the evaluator mapping."""

    if any(id_column not in frame.columns for frame in frames):
        raise ValueError(f"all frames must contain {id_column!r}")
    identifiers = sorted(
        {
            _canonical_identifier(value)
            for frame in frames
            for value in frame[id_column].dropna().tolist()
            if _canonical_identifier(value)
        }
    )
    mapping = {identifier: f"{prefix}{index:06d}" for index, identifier in enumerate(identifiers, start=1)}
    output: list[pd.DataFrame] = []
    for frame in frames:
        copied = frame.copy()
        copied[id_column] = copied[id_column].map(
            lambda value: mapping.get(_canonical_identifier(value)) if pd.notna(value) else pd.NA
        ).astype("string")
        output.append(copied)
    return output, mapping


def point_in_time_audit(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    *,
    formation_month: str | pd.Timestamp,
    time_column: str = "time_key",
    label_column: str = "label",
) -> dict[str, Any]:
    """Audit temporal boundaries and forbidden predict labels."""

    formation = _formation_period(formation_month)

    def month_values(frame: pd.DataFrame) -> pd.Series:
        if time_column in frame.columns:
            source = frame[time_column]
        elif "eom" in frame.columns:
            source = frame["eom"]
        else:
            return pd.Series(pd.NaT, index=frame.index, dtype="period[M]")
        raw = source.astype("string")
        offsets = raw.str.extract(r"^t-(\d+)$", expand=False, flags=re.IGNORECASE)
        result = pd.Series(pd.NaT, index=frame.index, dtype="period[M]")
        relative = offsets.notna()
        if bool(relative.any()):
            result.loc[relative] = offsets.loc[relative].astype(int).map(lambda value: formation - int(value))
        calendar = ~relative
        if bool(calendar.any()):
            raw_calendar = source.loc[calendar].astype("string").str.strip()
            compact = raw_calendar.str.fullmatch(r"\d{6}", na=False)
            parsed_dates = pd.Series(pd.NaT, index=raw_calendar.index, dtype="datetime64[ns]")
            if bool(compact.any()):
                parsed_dates.loc[compact] = pd.to_datetime(
                    raw_calendar.loc[compact], format="%Y%m", errors="coerce"
                )
            remaining = raw_calendar.notna() & ~compact
            if bool(remaining.any()):
                try:
                    converted = pd.to_datetime(
                        raw_calendar.loc[remaining], format="mixed", errors="coerce", utc=True
                    )
                except (TypeError, ValueError):  # pragma: no cover - old pandas
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        converted = pd.to_datetime(raw_calendar.loc[remaining], errors="coerce", utc=True)
                if getattr(converted.dt, "tz", None) is not None:
                    converted = converted.dt.tz_convert("UTC").dt.tz_localize(None)
                parsed_dates.loc[remaining] = converted
            parsed = parsed_dates.dt.to_period("M")
            result.loc[calendar] = parsed
        return result

    train_months = month_values(train_df)
    valid_months = month_values(validation_df)
    predict_months = month_values(predict_df)
    errors: list[str] = []
    for name, frame in (("train", train_df), ("validation", validation_df), ("predict", predict_df)):
        source = frame[time_column] if time_column in frame.columns else frame.get("eom")
        if source is None:
            continue
        raw = source.astype("string").str.strip()
        relative = raw.str.fullmatch(r"t-(\d+)", na=False, flags=re.IGNORECASE)
        calendar_values = raw.notna() & ~relative
        if bool(relative.any()) and bool(calendar_values.any()):
            errors.append(f"{name} mixes relative and calendar time encodings")
    for name, values in (("train", train_months), ("validation", valid_months), ("predict", predict_months)):
        if len(values) and values.isna().any():
            errors.append(f"{name} contains unparseable time values")
    if train_months.notna().any() and train_months.max() >= formation:
        errors.append("train contains formation-month or future rows")
    if valid_months.notna().any() and valid_months.max() >= formation:
        errors.append("validation contains formation-month or future rows")
    if predict_months.notna().any() and predict_months.min() != formation:
        errors.append("predict rows are not all in the formation month")
    def normalized(value: Any) -> str:
        return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")

    forbidden = [
        column
        for column in predict_df.columns
        if normalized(column) in {
            normalized(label_column),
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
        or normalized(column).startswith(("future_", "target_"))
    ]
    if forbidden:
        errors.append(f"predict contains forbidden label columns: {forbidden}")
    return {
        "point_in_time_valid": not errors,
        "errors": errors,
        "train_max_month": str(train_months.max()) if train_months.notna().any() else None,
        "validation_max_month": str(valid_months.max()) if valid_months.notna().any() else None,
        "predict_min_month": str(predict_months.min()) if predict_months.notna().any() else None,
        "formation_month": str(formation),
        "predict_label_absent": not forbidden,
    }


def assert_point_in_time(*args: Any, **kwargs: Any) -> dict[str, Any]:
    result = point_in_time_audit(*args, **kwargs)
    if not result["point_in_time_valid"]:
        raise ValueError("point-in-time audit failed: " + "; ".join(result["errors"]))
    return result


__all__ = ["assert_point_in_time", "mask_identifiers", "point_in_time_audit"]

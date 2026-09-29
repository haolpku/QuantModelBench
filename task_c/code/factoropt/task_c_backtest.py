"""Deterministic JKP-compatible stock portfolio formation and settlement.

The LLM never calls these functions. They consume only its validated score
table and evaluator-owned stock/daily-return data. Portfolio weights are formed
at month end, held through the following month, and settled one day at a time.
Missing returns and unavailable months are explicit in the returned tables;
they are never silently reallocated.
"""

from __future__ import annotations

import math
import re
import warnings
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


MAIN_SCREEN_COLUMNS = ("common", "exch_main", "primary_sec", "obs_main")
DEFAULT_MIN_LEG_STOCKS = 5
DEFAULT_MIN_BREAKPOINT_STOCKS = 10
# The WRDS monthly panel contains a small number of securities that do not
# have a matching daily observation in ``ctff_daily_ret`` (mostly the tiniest
# names).  A 95% weighted-coverage floor keeps ordinary months usable while
# still making materially incomplete months unavailable; 99% is available as
# a stricter pre-registered sensitivity.
DEFAULT_COVERAGE_THRESHOLD = 0.95
RETURN_POLICY = "raw_ret_exc_as_supplied_no_imputation"
DELISTING_RETURN_POLICY = "as_supplied_by_daily_source"


class BacktestDataError(ValueError):
    """Raised when a stock/daily table cannot satisfy the fixed contract."""


def _normalise_column(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    names = [_normalise_column(c) for c in out.columns]
    if len(set(names)) != len(names):
        duplicates = sorted({x for x in names if names.count(x) > 1})
        raise BacktestDataError(f"duplicate columns after normalization: {duplicates}")
    out.columns = names
    return out


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
    if pd.isna(parsed):
        raise BacktestDataError(f"could not parse month/date {value!r}")
    if getattr(parsed, "tzinfo", None) is not None:
        parsed = pd.Timestamp(parsed).tz_convert(None)
    return pd.Timestamp(parsed).to_period("M").to_timestamp("M")


def _parse_month_series(values: pd.Series) -> pd.Series:
    raw = values.astype("string").str.strip()
    compact = raw.str.fullmatch(r"\d{6}", na=False)
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    if bool(compact.any()):
        parsed.loc[compact] = pd.to_datetime(raw.loc[compact], format="%Y%m", errors="coerce")
    remaining = raw.notna() & ~compact
    if bool(remaining.any()):
        try:
            # ``utc=True`` makes mixed naive/aware exports deterministic and
            # avoids pandas' future warning about mixed time zones.
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
    return parsed.dt.to_period("M").dt.to_timestamp("M")


def _parse_daily_date_series(values: pd.Series) -> pd.Series:
    """Parse daily dates, including compact YYYYMMDD exports, consistently."""

    raw = values.astype("string").str.strip()
    compact = raw.str.fullmatch(r"\d{8}", na=False)
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    if bool(compact.any()):
        parsed.loc[compact] = pd.to_datetime(raw.loc[compact], format="%Y%m%d", errors="coerce")
    remaining = raw.notna() & ~compact
    if bool(remaining.any()):
        try:
            # ``utc=True`` makes mixed naive/aware exports deterministic and
            # avoids pandas' future warning about mixed time zones.
            converted = pd.to_datetime(
                raw.loc[remaining], format="mixed", errors="coerce", utc=True
            )
        except (TypeError, ValueError):  # pragma: no cover - old pandas
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                converted = pd.to_datetime(raw.loc[remaining], errors="coerce", utc=True)
        # A timezone-aware source cannot be assigned to the naive result
        # without an object fallback.  Convert it to UTC and drop the zone;
        # the backtest compares dates, not intraday timestamps.
        try:
            if getattr(converted.dt, "tz", None) is not None:
                converted = converted.dt.tz_convert("UTC").dt.tz_localize(None)
        except (AttributeError, TypeError):
            pass
        parsed.loc[remaining] = converted
    return parsed


def _parse_daily_date(value: Any) -> pd.Timestamp:
    parsed = _parse_daily_date_series(pd.Series([value])).iloc[0]
    if pd.isna(parsed):
        raise BacktestDataError(f"could not parse daily date {value!r}")
    return pd.Timestamp(parsed)


def _coerce_id(values: pd.Series) -> pd.Series:
    def one(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)) and float(value).is_integer():
            return str(int(value))
        text = str(value).strip()
        if text.lower() in {"", "nan", "<na>", "nat", "none"}:
            return ""
        # CSV/Arrow decimal columns sometimes arrive as ``Decimal('123.0')``
        # or a string with a zero fractional part while the panel side is an
        # integer. Normalize only an unambiguous integer-looking token; keep
        # genuine alphanumeric IDs untouched.
        # Convert an explicit integer-like decimal while preserving leading
        # zeros in string security identifiers (``00123`` is not necessarily
        # the same key as ``123``).
        if re.fullmatch(r"[+-]?(?:0|[1-9]\d*)\.0+", text):
            try:
                return str(int(float(text)))
            except (OverflowError, ValueError):
                pass
        return text

    return values.map(one).astype("string")


def _read_table(
    path: str | Path,
    *,
    columns: Sequence[str] | None = None,
    filters: Sequence[tuple[str, str, Any]] | None = None,
) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    try:
        if source.suffix.lower() in {".parquet", ".pq"}:
            kwargs: dict[str, Any] = {"columns": list(columns) if columns else None, "engine": "pyarrow"}
            if filters:
                kwargs["filters"] = list(filters)
            return _normalise_columns(pd.read_parquet(source, **kwargs))
        return _normalise_columns(pd.read_csv(source, usecols=list(columns) if columns else None))
    except ImportError as exc:
        raise BacktestDataError("Parquet input requires pyarrow (install repository requirements.txt)") from exc
    except Exception as exc:
        raise BacktestDataError(f"could not read {source}: {type(exc).__name__}: {exc}") from exc


def _source_columns(path: Path) -> dict[str, str] | None:
    """Inspect a Parquet/CSV schema without loading observations."""

    try:
        if path.suffix.lower() in {".parquet", ".pq"}:
            import pyarrow.parquet as parquet  # type: ignore

            names = parquet.ParquetFile(path).schema_arrow.names
        else:
            names = list(pd.read_csv(path, nrows=0).columns)
        return {_normalise_column(name): str(name) for name in names}
    except Exception:
        return None


def _projected_read(
    path: str | Path,
    desired: Sequence[str],
    *,
    filters: Sequence[tuple[str, str, Any]] | None = None,
) -> pd.DataFrame:
    source = Path(path)
    available = _source_columns(source)
    if available is None:
        return _read_table(source, filters=filters)
    selected = [
        available[name]
        for name in dict.fromkeys(_normalise_column(x) for x in desired)
        if name in available
    ]
    source_filters: Sequence[tuple[str, str, Any]] | None = None
    if filters:
        source_filters = [
            (available.get(_normalise_column(column), str(column)), operator, value)
            for column, operator, value in filters
        ]
    return _read_table(source, columns=selected or None, filters=source_filters)


def _flag(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(1, index=frame.index, dtype=float)
    values = frame[column]
    if pd.api.types.is_bool_dtype(values):
        return values.astype(int)
    return pd.to_numeric(values, errors="coerce")


def load_stock_panel(
    path: str | Path,
    *,
    feature_columns: Sequence[str] = (),
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    country: str = "USA",
    strict_screens: bool = True,
    allow_duplicates: bool = False,
) -> pd.DataFrame:
    """Load a projected WRDS stock panel and apply the four JKP screens."""

    desired = [
        "id", "permno", "gvkey", "iid", "eom", "date", "excntry", "me", "market_equity",
        "market_cap", "mktcap", "size_grp",
        "exchange", "exchange_name", "exchcd", "crsp_exchcd", *MAIN_SCREEN_COLUMNS,
        "ret_exc_lead1m", *[str(c) for c in feature_columns],
    ]
    available_schema = _source_columns(Path(path))
    has_eom = available_schema is None or "eom" in available_schema
    filters: list[tuple[str, str, Any]] = []
    # The supplied WRDS export contains all countries. Push the USA and main
    # JKP screens into the Parquet reader whenever the source schema supports
    # them; otherwise the identical filters are applied immediately below.
    if available_schema is None or "excntry" in available_schema:
        filters.append(("excntry", "==", str(country).upper()))
    for screen in MAIN_SCREEN_COLUMNS:
        if available_schema is None or screen in available_schema:
            filters.append((screen, "==", 1))
    if has_eom and start is not None:
        filters.append(("eom", ">=", _parse_month(start).date()))
    if has_eom and end is not None:
        filters.append(("eom", "<=", _parse_month(end).date()))
    frame = _projected_read(path, desired, filters=filters or None)
    missing_features = [
        str(column)
        for column in feature_columns
        if _normalise_column(column) not in frame.columns
    ]
    if missing_features:
        raise BacktestDataError(f"stock panel missing requested feature columns: {missing_features}")
    if "id" not in frame.columns:
        if "permno" in frame.columns:
            frame["id"] = frame["permno"]
        elif "gvkey" in frame.columns:
            suffix = frame["iid"].astype("string") if "iid" in frame.columns else ""
            frame["id"] = frame["gvkey"].astype("string") + ":" + suffix
        else:
            raise BacktestDataError("stock panel requires id, permno, or gvkey")
    elif "permno" in frame.columns:
        # Some WRDS exports leave the canonical id null for a CRSP-linked row;
        # use permno only for those missing entries and preserve the supplied
        # JKP id everywhere else.
        raw_id = frame["id"]
        missing_id = raw_id.isna() | raw_id.astype("string").str.strip().isin(["", "<NA>", "nan"])
        frame.loc[missing_id, "id"] = frame.loc[missing_id, "permno"]
    date_column = "eom" if "eom" in frame.columns else "date" if "date" in frame.columns else None
    if date_column is None:
        raise BacktestDataError("stock panel requires eom or date")
    # ``me`` is the canonical JKP name, but a few WRDS exports expose the
    # same point-in-time market equity as ``market_equity``/``market_cap``.
    # Normalize those aliases before applying screens rather than requiring a
    # caller to load the full panel and rename it manually.
    if "me" not in frame.columns:
        for alias in ("market_equity", "market_cap", "mktcap"):
            if alias in frame.columns:
                frame["me"] = frame[alias]
                break
    missing = sorted({"excntry", "me"} - set(frame.columns))
    if missing:
        raise BacktestDataError(f"stock panel missing required columns: {missing}")
    if strict_screens:
        missing_screens = [c for c in MAIN_SCREEN_COLUMNS if c not in frame.columns]
        if missing_screens:
            raise BacktestDataError(f"stock panel missing JKP screen columns: {missing_screens}")
    for column in MAIN_SCREEN_COLUMNS:
        if column in frame.columns:
            frame = frame.loc[_flag(frame, column).eq(1)].copy()
    frame["id"] = _coerce_id(frame["id"])
    frame["eom"] = _parse_month_series(frame[date_column])
    frame["excntry"] = frame["excntry"].astype("string").str.strip().str.upper()
    frame["me"] = pd.to_numeric(frame["me"], errors="coerce")
    for column in ("exchcd", "crsp_exchcd"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.loc[frame["id"].ne("") & frame["eom"].notna() & frame["me"].gt(0)].copy()
    if country:
        frame = frame.loc[frame["excntry"].eq(str(country).upper())].copy()
    if start is not None:
        frame = frame.loc[frame["eom"].ge(_parse_month(start))]
    if end is not None:
        frame = frame.loc[frame["eom"].le(_parse_month(end))]
    if frame.empty:
        raise BacktestDataError("no stock rows remain after filters")
    duplicate = frame.duplicated(["id", "eom"], keep=False)
    if duplicate.any() and not allow_duplicates:
        sample = frame.loc[duplicate, ["id", "eom"]].head(5).to_dict("records")
        raise BacktestDataError(f"duplicate id/eom observations (sample={sample})")
    if duplicate.any():
        frame = frame.sort_values(["id", "eom"]).drop_duplicates(["id", "eom"], keep="last")
    return frame.sort_values(["eom", "id"]).reset_index(drop=True)


def load_daily_returns(
    path: str | Path,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    return_column: str | None = None,
    allow_duplicates: bool = False,
) -> pd.DataFrame:
    """Load evaluator-owned daily ``ctff_daily_ret`` observations."""

    desired = ["id", "permno", "gvkey", "iid", "date", "eom", "ret_exc", "ret", "return", "ret_exc_daily"]
    if return_column:
        desired.append(str(return_column))
    available_schema = _source_columns(Path(path))
    has_date = available_schema is None or "date" in available_schema
    filters: list[tuple[str, str, Any]] = []
    if has_date and start is not None:
        filters.append(("date", ">=", _parse_daily_date(start).date()))
    if has_date and end is not None:
        filters.append(("date", "<=", _parse_daily_date(end).date()))
    frame = _projected_read(path, desired, filters=filters or None)
    if "id" not in frame.columns:
        if "permno" in frame.columns:
            frame["id"] = frame["permno"]
        elif "gvkey" in frame.columns:
            suffix = frame["iid"].astype("string") if "iid" in frame.columns else ""
            frame["id"] = frame["gvkey"].astype("string") + ":" + suffix
        else:
            raise BacktestDataError("daily returns require id, permno, or gvkey")
    elif "permno" in frame.columns:
        raw_id = frame["id"]
        missing_id = raw_id.isna() | raw_id.astype("string").str.strip().isin(["", "<NA>", "nan"])
        frame.loc[missing_id, "id"] = frame.loc[missing_id, "permno"]
    date_column = "date" if "date" in frame.columns else "eom" if "eom" in frame.columns else None
    if date_column is None:
        raise BacktestDataError("daily returns require date or eom")
    if return_column is None:
        return_column = next((c for c in ("ret_exc", "ret_exc_daily", "ret", "return") if c in frame.columns), None)
    else:
        return_column = _normalise_column(return_column)
    if return_column is None or return_column not in frame.columns:
        raise BacktestDataError("daily returns require ret_exc (or explicit return_column)")
    frame = frame.rename(columns={return_column: "ret_exc"})
    frame["id"] = _coerce_id(frame["id"])
    frame["date"] = _parse_daily_date_series(frame[date_column])
    frame["ret_exc"] = pd.to_numeric(frame["ret_exc"], errors="coerce")
    frame = frame.loc[frame["id"].ne("") & frame["date"].notna()].copy()
    if start is not None:
        frame = frame.loc[frame["date"].ge(_parse_daily_date(start))]
    if end is not None:
        frame = frame.loc[frame["date"].le(_parse_daily_date(end))]
    duplicate = frame.duplicated(["id", "date"], keep=False)
    if duplicate.any() and not allow_duplicates:
        sample = frame.loc[duplicate, ["id", "date"]].head(5).to_dict("records")
        raise BacktestDataError(f"duplicate id/date daily returns (sample={sample})")
    if duplicate.any():
        frame = frame.groupby(["id", "date"], as_index=False, sort=True)["ret_exc"].mean()
    return frame[["id", "date", "ret_exc"]].sort_values(["date", "id"]).reset_index(drop=True)


def load_nyse_cutoffs(path: str | Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    frame = _read_table(path)
    date_column = next((c for c in ("eom", "date", "month") if c in frame.columns), None)
    if date_column is None:
        raise BacktestDataError("cutoff file requires eom/date/month")
    frame["eom"] = _parse_month_series(frame[date_column])
    p20 = next((c for c in ("nyse_p20", "p20", "me_p20") if c in frame.columns), None)
    p80 = next((c for c in ("nyse_p80", "p80", "me_p80") if c in frame.columns), None)
    if p80 is None:
        raise BacktestDataError("cutoff file requires nyse_p80/p80/me_p80")
    rename = {p80: "nyse_p80"}
    if p20:
        rename[p20] = "nyse_p20"
    out = frame[["eom", p80] + ([p20] if p20 else [])].rename(columns=rename)
    for column in ("nyse_p20", "nyse_p80"):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out.dropna(subset=["eom", "nyse_p80"]).drop_duplicates("eom", keep="last")


def _discrete_quantile(values: Iterable[Any], probability: float) -> float | None:
    array = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna().sort_values().to_numpy(dtype=float)
    if array.size == 0:
        return None
    index = max(0, min(array.size - 1, int(math.ceil(float(probability) * array.size)) - 1))
    return float(array[index])


def _is_nyse(frame: pd.DataFrame) -> tuple[pd.Series, str]:
    for column in ("crsp_exchcd", "exchcd"):
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            if bool(values.notna().any()):
                return values.eq(1), column
    for column in ("exchange", "exchange_name"):
        if column in frame.columns:
            values = frame[column].astype("string").str.upper()
            if bool(values.notna().any()):
                return values.str.contains("NYSE", na=False), column
    # The supplied JKP global-factor export does not include an exchange-code
    # column.  Its ``size_grp`` labels do identify the non-micro breakpoint
    # universe (mega/large/small), which is the closest auditable proxy for the
    # NYSE reference sample available in this file.  Keep a conservative
    # fallback for synthetic/alternate exports with different labels.
    if "size_grp" in frame.columns:
        values = frame["size_grp"].astype("string").str.strip().str.lower()
        known = values.isin({"mega", "large", "small"})
        if bool(known.any()):
            return known, "size_grp_nonmicro_proxy"
    return pd.Series(True, index=frame.index), "eligible_proxy"


def _micro_mask(frame: pd.DataFrame, p20: float | None) -> tuple[pd.Series, str]:
    if "size_grp" in frame.columns:
        values = frame["size_grp"].astype("string").str.lower()
        known = (values.notna() & values.ne("")).fillna(False)
        if bool(known.any()):
            mask = values.str.contains("micro|nano", regex=True, na=False)
            # A partially populated size_grp must not silently classify its
            # unknown rows as ordinary stocks.  Use the official/sample NYSE
            # p20 only for those rows; labelled rows keep the source label.
            if p20 is not None and math.isfinite(float(p20)):
                me = pd.to_numeric(frame["me"], errors="coerce")
                mask = mask | (~known & me.lt(float(p20)))
                return mask.astype(bool), "size_grp_plus_nyse_p20"
            return mask.astype(bool), "size_grp_with_unknowns_unclassified"
    if p20 is not None and math.isfinite(p20):
        return pd.to_numeric(frame["me"], errors="coerce").lt(float(p20)), "nyse_p20_or_sample"
    return pd.Series(False, index=frame.index), "no_micro_cutoff"


def _assign_groups(values: pd.Series, n_groups: int) -> pd.Series:
    """Stable rank groups: 1 is low score, n_groups is high score."""

    ordered = pd.DataFrame({"score": values, "id": values.index.astype(str)}, index=values.index)
    ordered = ordered.sort_values(["score", "id"], kind="mergesort")
    ranks = np.arange(1, len(ordered) + 1, dtype=float)
    groups = np.ceil(ranks * int(n_groups) / max(1, len(ordered))).astype(int).clip(1, int(n_groups))
    output = pd.Series(groups, index=ordered.index, dtype="Int64")
    return output.reindex(values.index)


def form_jkp_weights(
    panel: pd.DataFrame,
    predictions: pd.DataFrame,
    formation_eom: str | pd.Timestamp,
    *,
    nyse_cutoffs: pd.DataFrame | None = None,
    n_groups: int = 3,
    min_breakpoint_stocks: int = DEFAULT_MIN_BREAKPOINT_STOCKS,
    min_leg_stocks: int = DEFAULT_MIN_LEG_STOCKS,
    strict_screens: bool = True,
    strict_cutoffs: bool = False,
    return_diagnostics: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    """Form capped-value, zero-net high-minus-low weights for one month."""

    if n_groups < 2 or min_leg_stocks < 1 or min_breakpoint_stocks < 1:
        raise ValueError("invalid grouping/minimum-stock parameters")
    frame = _normalise_columns(panel)
    pred = _normalise_columns(predictions)
    t = _parse_month(formation_eom)
    if "eom" not in frame.columns:
        date_column = next((c for c in ("date", "time_key", "month") if c in frame.columns), None)
        if date_column is None:
            raise BacktestDataError("panel requires eom/date/time_key")
        frame["eom"] = _parse_month_series(frame[date_column])
    else:
        frame["eom"] = _parse_month_series(frame["eom"])
    if "me" not in frame.columns:
        for alias in ("market_equity", "market_cap", "mktcap"):
            if alias in frame.columns:
                frame["me"] = frame[alias]
                break
    if "id" not in frame.columns or "me" not in frame.columns:
        raise BacktestDataError("panel requires id and me")
    if "excntry" in frame.columns:
        frame["excntry"] = frame["excntry"].astype("string").str.strip().str.upper()
        frame = frame.loc[frame["excntry"].eq("USA")].copy()
        if frame.empty:
            empty_columns = ["formation_eom", "holding_month", "id", "score", "leg", "weight", "me", "capped_me", "group"]
            diagnostics = {
                "formation_eom": t.strftime("%Y-%m-%d"),
                "holding_month": (t + pd.offsets.MonthEnd(1)).strftime("%Y-%m"),
                "tradable": False,
                "reason": "no_USA_rows",
            }
            empty = pd.DataFrame(columns=empty_columns)
            return (empty, diagnostics) if return_diagnostics else empty
    if "id" not in pred.columns or "score" not in pred.columns:
        raise BacktestDataError("predictions require id and score")
    frame["id"] = _coerce_id(frame["id"])
    pred["id"] = _coerce_id(pred["id"])
    pred["score"] = pd.to_numeric(pred["score"], errors="coerce")
    if "formation_eom" in pred.columns:
        pred["formation_eom"] = _parse_month_series(pred["formation_eom"])
        pred = pred.loc[pred["formation_eom"].eq(t)]
    if pred["id"].duplicated().any():
        raise BacktestDataError("predictions contain duplicate id for a formation month")
    frame = frame.loc[frame["eom"].eq(t)].copy()
    frame = frame.merge(pred[["id", "score"]], on="id", how="inner", validate="one_to_one")
    for column in MAIN_SCREEN_COLUMNS:
        if column not in frame.columns:
            if strict_screens:
                raise BacktestDataError(f"panel missing JKP screen column {column}")
        else:
            frame = frame.loc[_flag(frame, column).eq(1)].copy()
    frame["me"] = pd.to_numeric(frame["me"], errors="coerce")
    frame = frame.loc[frame["id"].ne("") & frame["me"].gt(0) & np.isfinite(frame["score"])].copy()
    nyse, nyse_source = _is_nyse(frame)
    supplied_row: Mapping[str, Any] | None = None
    if nyse_cutoffs is not None:
        cuts = _normalise_columns(nyse_cutoffs)
        if "eom" not in cuts.columns:
            date_column = next((c for c in ("date", "month") if c in cuts.columns), None)
            if date_column:
                cuts["eom"] = _parse_month_series(cuts[date_column])
        if "eom" in cuts.columns:
            rows = cuts.loc[cuts["eom"].eq(t)]
            if not rows.empty:
                supplied_row = rows.iloc[-1].to_dict()
    p20 = None
    p80 = None
    if supplied_row is not None:
        for column in ("nyse_p20", "p20", "me_p20"):
            if column in supplied_row and pd.notna(supplied_row[column]):
                p20 = float(supplied_row[column])
                break
        for column in ("nyse_p80", "p80", "me_p80"):
            if column in supplied_row and pd.notna(supplied_row[column]):
                p80 = float(supplied_row[column])
                break
    cutoff_source = "official_nyse" if p80 is not None else "sample_nyse"
    if strict_cutoffs and p80 is None:
        raise BacktestDataError(f"no official NYSE cutoff for {t.strftime('%Y-%m')}")
    micro, micro_source = _micro_mask(frame, p20)
    breakpoint_sample = frame.loc[nyse & ~micro].copy()
    if p20 is None:
        p20 = _discrete_quantile(breakpoint_sample["me"], 0.20)
        if p20 is not None:
            micro, micro_source = _micro_mask(frame, p20)
            breakpoint_sample = frame.loc[nyse & ~micro].copy()
    if p80 is None:
        p80 = _discrete_quantile(breakpoint_sample["me"], 0.80)
        cutoff_source = "sample_nyse_p80"
    if p80 is None or not math.isfinite(float(p80)) or p80 <= 0:
        p80 = float(frame["me"].quantile(0.80))
        cutoff_source = "sample_all_p80_fallback"
    diagnostics: dict[str, Any] = {
        "formation_eom": t.strftime("%Y-%m-%d"),
        "holding_month": (t + pd.offsets.MonthEnd(1)).strftime("%Y-%m"),
        "nyse_source": nyse_source,
        "micro_source": micro_source,
        "cutoff_source": cutoff_source,
        "nyse_p20": p20,
        "nyse_p80": p80,
        "breakpoint_n": int(len(breakpoint_sample)),
        "eligible_n": int(len(frame)),
        "tradable": False,
    }
    empty_columns = ["formation_eom", "holding_month", "id", "score", "leg", "weight", "me", "capped_me", "group"]
    if len(breakpoint_sample) < int(min_breakpoint_stocks):
        diagnostics["reason"] = "insufficient_breakpoint_stocks"
        empty = pd.DataFrame(columns=empty_columns)
        return (empty, diagnostics) if return_diagnostics else empty
    # Compute breakpoints only on non-micro NYSE stocks and apply them to all
    # eligible stocks. A stable id tie-break removes random portfolio churn.
    ordered_breakpoints = breakpoint_sample.sort_values(["score", "id"], kind="mergesort")
    boundary_rows = [
        ordered_breakpoints.iloc[
            min(len(ordered_breakpoints) - 1, max(0, int(math.ceil(i * len(ordered_breakpoints) / n_groups)) - 1))
        ]
        for i in range(1, n_groups)
    ]
    # Compare (score, masked-id) tuples rather than score alone. This gives a
    # deterministic tie-break when many stocks share a rounded signal while
    # retaining the NYSE reference breakpoints for micro stocks.
    boundaries = [(float(row["score"]), str(row["id"])) for row in boundary_rows]

    def classify(row: pd.Series) -> int:
        score = float(row["score"])
        identifier = str(row["id"])
        return int(1 + sum(
            score > boundary_score
            or (score == boundary_score and identifier > boundary_id)
            for boundary_score, boundary_id in boundaries
        ))

    frame["group"] = frame.apply(classify, axis=1).astype("Int64")
    low = frame.loc[frame["group"].eq(1)].copy()
    high = frame.loc[frame["group"].eq(n_groups)].copy()
    diagnostics["low_n"] = int(len(low))
    diagnostics["high_n"] = int(len(high))
    if len(low) < int(min_leg_stocks) or len(high) < int(min_leg_stocks):
        diagnostics["reason"] = "insufficient_leg_stocks"
        empty = pd.DataFrame(columns=empty_columns)
        return (empty, diagnostics) if return_diagnostics else empty

    def leg_rows(leg: pd.DataFrame, name: str, sign: float) -> pd.DataFrame:
        result = leg[["id", "score", "me", "group"]].copy()
        result["capped_me"] = np.minimum(result["me"].to_numpy(dtype=float), float(p80))
        denominator = float(result["capped_me"].sum())
        result["weight"] = sign * result["capped_me"] / denominator if denominator > 0 else np.nan
        result["leg"] = name
        return result

    output = pd.concat([leg_rows(high, "long", 1.0), leg_rows(low, "short", -1.0)], ignore_index=True)
    output["formation_eom"] = t
    output["holding_month"] = t + pd.offsets.MonthEnd(1)
    output = output[empty_columns].sort_values(["leg", "id"]).reset_index(drop=True)
    diagnostics.update(
        {
            "tradable": True,
            "long_weight_sum": float(output.loc[output["leg"].eq("long"), "weight"].sum()),
            "short_weight_sum": float(output.loc[output["leg"].eq("short"), "weight"].sum()),
            "gross_exposure": float(output["weight"].abs().sum()),
        }
    )
    output.attrs["diagnostics"] = diagnostics
    return (output, diagnostics) if return_diagnostics else output


def _normalise_daily(frame: pd.DataFrame) -> pd.DataFrame:
    daily = _normalise_columns(frame)
    if "id" not in daily.columns or "date" not in daily.columns:
        raise BacktestDataError("daily returns require id and date")
    ret_column = next((c for c in ("ret_exc", "ret_exc_daily", "ret", "return") if c in daily.columns), None)
    if ret_column is None:
        raise BacktestDataError("daily returns require ret_exc/ret")
    daily = daily.rename(columns={ret_column: "ret_exc"})[["id", "date", "ret_exc"]].copy()
    daily["id"] = _coerce_id(daily["id"])
    daily["date"] = _parse_daily_date_series(daily["date"])
    daily["ret_exc"] = pd.to_numeric(daily["ret_exc"], errors="coerce")
    return daily.loc[daily["id"].ne("") & daily["date"].notna()].copy()


def _holding_dates(holding_month: Any, daily: pd.DataFrame) -> pd.DatetimeIndex:
    month = _parse_month(holding_month).to_period("M")
    return pd.DatetimeIndex(sorted(daily.loc[daily["date"].dt.to_period("M").eq(month), "date"].unique()))


def settle_daily_returns(
    weights: pd.DataFrame,
    daily_returns: pd.DataFrame,
    holding_month: str | pd.Timestamp,
    *,
    turnover: float = 0.0,
    volume: float | None = None,
    cost_bps: float = 0.0,
    coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD,
    cash_when_empty: bool = False,
) -> dict[str, Any]:
    """Settle fixed signed weights over one calendar month.

    Missing return weight is reported and causes an unavailable month when
    coverage falls below ``coverage_threshold``. Observed weights are never
    renormalized; this prevents an accidental survivorship/zero-fill bias.
    """

    if (
        not math.isfinite(float(cost_bps))
        or cost_bps < 0
        or not math.isfinite(float(coverage_threshold))
        or coverage_threshold <= 0
        or coverage_threshold > 1
        or not math.isfinite(float(turnover))
        or turnover < 0
        or volume is not None and (not math.isfinite(float(volume)) or float(volume) < 0)
    ):
        raise ValueError("invalid cost_bps or coverage_threshold")
    required = {"id", "weight", "leg"}
    if not required.issubset(weights.columns):
        raise BacktestDataError(f"weights missing columns: {sorted(required - set(weights.columns))}")
    held = weights[["id", "weight", "leg"]].copy()
    held["id"] = _coerce_id(held["id"])
    held["weight"] = pd.to_numeric(held["weight"], errors="coerce")
    held = held.loc[held["id"].ne("") & np.isfinite(held["weight"])].copy()
    if held["id"].duplicated().any():
        raise BacktestDataError("weights contain duplicate ids")
    held["leg"] = held["leg"].astype("string").str.strip().str.lower()
    if not held.empty and (~held["leg"].isin(["long", "short"])).any():
        raise BacktestDataError("weights leg must be exactly long or short")
    if not held.empty and ((held.loc[held["leg"].eq("long"), "weight"] <= 0).any() or (
        held.loc[held["leg"].eq("short"), "weight"] >= 0
    ).any()):
        raise BacktestDataError("long weights must be positive and short weights must be negative")
    daily = _normalise_daily(daily_returns)
    dates = _holding_dates(holding_month, daily)
    volume_value = float(volume if volume is not None else 2.0 * float(turnover))
    cost_fraction = float(cost_bps) / 10_000.0 * max(0.0, volume_value)
    gross_capital = float(held["weight"].abs().sum())
    long_capital = float(held.loc[held["leg"].astype(str).str.lower().eq("long"), "weight"].abs().sum())
    short_capital = float(held.loc[held["leg"].astype(str).str.lower().eq("short"), "weight"].abs().sum())
    if held.empty and cash_when_empty and len(dates):
        # A valid netted portfolio can be entirely cash. Charge liquidation
        # once, with zero excess cash return, instead of treating it as failure.
        net = np.zeros(len(dates))
        net[0] = -cost_fraction
        days = pd.DataFrame({
            "holding_month": _parse_month(holding_month), "date": dates,
            "gross_return": 0.0, "net_return": net,
            "gross_excess_return": 0.0, "net_excess_return": net,
            "coverage": 1.0, "missing_weight": 0.0,
            "cost_fraction": -net,
        })
        return {"daily": days, "monthly": {
            "holding_month": _parse_month(holding_month).strftime("%Y-%m"),
            "status": "ok", "gross_return": 0.0, "net_return": -cost_fraction,
            "gross_excess_return": 0.0, "net_excess_return": -cost_fraction,
            "turnover": float(turnover), "volume": volume_value,
            "cost_fraction": cost_fraction, "coverage_min": 1.0, "coverage_mean": 1.0,
            "missing_weight_max": 0.0, "invalid_return_count": 0,
            "invalid_return_weight_max": 0.0, "nonpositive_wealth": False,
            "n_days": len(dates), "return_policy": RETURN_POLICY,
            "delisting_return_policy": DELISTING_RETURN_POLICY,
        }}
    if not held.empty and (long_capital <= 0 or short_capital <= 0):
        raise BacktestDataError("weights must contain non-empty long and short legs")
    rows: list[dict[str, Any]] = []
    if not len(dates) or gross_capital <= 0:
        monthly = {
            "holding_month": _parse_month(holding_month).strftime("%Y-%m"),
            "status": "unavailable_no_trading_days" if not len(dates) else "unavailable_empty_weights",
            "gross_return": None,
            "net_return": None,
            "long_return": None,
            "short_return": None,
            "short_signed_return": None,
            "gross_excess_return": None,
            "net_excess_return": None,
            "long_excess_return": None,
            "short_excess_return": None,
            "short_signed_excess_return": None,
            "turnover": float(turnover),
            "volume": volume_value,
            "cost_fraction": cost_fraction,
            "coverage_min": 0.0,
            "coverage_mean": 0.0,
            "missing_weight_max": 1.0,
            "invalid_return_count": 0,
            "invalid_return_weight_max": 0.0,
            "nonpositive_wealth": False,
            "n_days": int(len(dates)),
            "return_policy": RETURN_POLICY,
            "delisting_return_policy": DELISTING_RETURN_POLICY,
        }
        return {"daily": pd.DataFrame(rows), "monthly": monthly}
    daily = daily.loc[daily["date"].dt.to_period("M").eq(_parse_month(holding_month).to_period("M"))]
    duplicate = daily.duplicated(["id", "date"], keep=False)
    if duplicate.any():
        raise BacktestDataError("daily returns contain duplicate id/date rows")
    weight_index = held.set_index("id")
    for day_index, day in enumerate(dates):
        observed = daily.loc[daily["date"].eq(day), ["id", "ret_exc"]].set_index("id")
        joined = weight_index.join(observed, how="left")
        # Simple returns below -100% are not economically valid.  Treat them
        # as missing/invalid observations instead of allowing a negative
        # portfolio wealth path to silently compound.
        finite = (
            joined["ret_exc"].notna()
            & np.isfinite(joined["ret_exc"])
            & joined["ret_exc"].ge(-1.0)
        )
        invalid = joined["ret_exc"].notna() & ~finite
        observed_abs = float(joined.loc[finite, "weight"].abs().sum())
        coverage = observed_abs / gross_capital if gross_capital else 0.0
        missing_weight = max(0.0, gross_capital - observed_abs) / gross_capital if gross_capital else 1.0
        invalid_weight = float(joined.loc[invalid, "weight"].abs().sum()) / gross_capital if gross_capital else 0.0
        # No reallocation: missing positions contribute no observed return,
        # while the explicit coverage flag determines tradability.
        values = joined.loc[finite].copy()
        # Signed weights already have long sum +1 and short sum -1.  Their
        # signed sum is therefore the conventional high-minus-low return;
        # do not divide it by gross exposure (2), which would silently halve
        # every JKP spread.
        gross = float((values["weight"] * values["ret_exc"]).sum())
        long_values = values.loc[values["leg"].astype(str).str.lower().eq("long")]
        short_values = values.loc[values["leg"].astype(str).str.lower().eq("short")]
        long_ret = float((long_values["weight"] * long_values["ret_exc"]).sum() / long_capital) if long_capital else np.nan
        short_ret = float((short_values["weight"].abs() * short_values["ret_exc"]).sum() / short_capital) if short_capital else np.nan
        short_signed_ret = -short_ret if math.isfinite(short_ret) else np.nan
        net = (1.0 + gross) * (1.0 - cost_fraction) - 1.0 if day_index == 0 else gross
        rows.append(
            {
                "holding_month": _parse_month(holding_month),
                "date": pd.Timestamp(day),
                "gross_return": gross,
                "net_return": net,
                "long_return": long_ret,
                "short_return": short_ret,
                "short_signed_return": short_signed_ret,
                "gross_excess_return": gross,
                "net_excess_return": net,
                "long_excess_return": long_ret,
                "short_excess_return": short_ret,
                "short_signed_excess_return": short_signed_ret,
                "coverage": coverage,
                "missing_weight": missing_weight,
                "invalid_return_count": int(invalid.sum()),
                "invalid_return_weight": invalid_weight,
                "n_observed": int(finite.sum()),
                "n_held": int(len(held)),
                "cost_fraction": cost_fraction if day_index == 0 else 0.0,
            }
        )
    daily_frame = pd.DataFrame(rows)
    coverage_min = float(daily_frame["coverage"].min()) if not daily_frame.empty else 0.0
    coverage_mean = float(daily_frame["coverage"].mean()) if not daily_frame.empty else 0.0
    nonpositive_wealth = bool(
        not daily_frame.empty
        and (pd.to_numeric(daily_frame["gross_return"], errors="coerce") <= -1.0).any()
    )
    invalid_return_count = int(pd.to_numeric(daily_frame["invalid_return_count"], errors="coerce").fillna(0).sum())
    invalid_return_weight_max = float(
        pd.to_numeric(daily_frame["invalid_return_weight"], errors="coerce").fillna(0.0).max()
    )
    if invalid_return_count:
        status = "unavailable_invalid_return"
    elif nonpositive_wealth:
        status = "unavailable_nonpositive_wealth"
    elif coverage_min >= float(coverage_threshold):
        status = "ok"
    else:
        status = "unavailable_coverage"

    def compound(column: str) -> float | None:
        values = pd.to_numeric(daily_frame[column], errors="coerce")
        return float((1.0 + values).prod() - 1.0) if values.notna().all() else None

    gross_monthly = compound("gross_return") if status == "ok" else None
    net_monthly = compound("net_return") if status == "ok" else None
    long_monthly = compound("long_return") if status == "ok" and daily_frame["long_return"].notna().all() else None
    short_monthly = compound("short_return") if status == "ok" and daily_frame["short_return"].notna().all() else None
    short_signed_monthly = (
        compound("short_signed_return")
        if status == "ok" and daily_frame["short_signed_return"].notna().all()
        else None
    )
    monthly = {
        "holding_month": _parse_month(holding_month).strftime("%Y-%m"),
        "status": status,
        "gross_return": gross_monthly,
        "net_return": net_monthly,
        "long_return": long_monthly,
        "short_return": short_monthly,
        "short_signed_return": short_signed_monthly,
        "gross_excess_return": gross_monthly,
        "net_excess_return": net_monthly,
        "long_excess_return": long_monthly,
        "short_excess_return": short_monthly,
        "short_signed_excess_return": short_signed_monthly,
        "turnover": float(turnover),
        "volume": volume_value,
        "cost_fraction": cost_fraction,
        "coverage_min": coverage_min,
        "coverage_mean": coverage_mean,
        "missing_weight_max": float(daily_frame["missing_weight"].max()),
        "invalid_return_count": invalid_return_count,
        "invalid_return_weight_max": invalid_return_weight_max,
        "nonpositive_wealth": nonpositive_wealth,
        "n_days": int(len(daily_frame)),
        "return_policy": RETURN_POLICY,
        "delisting_return_policy": DELISTING_RETURN_POLICY,
    }
    return {"daily": daily_frame, "monthly": monthly}


def drift_weights(
    target_weights: pd.DataFrame,
    daily_returns: pd.DataFrame,
    holding_month: str | pd.Timestamp,
    *,
    preserve_leg_budget: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Mark target signed weights to the holding-month end for turnover.

    When any held capital lacks a finite daily return, the returned vector is
    diagnostic only and ``complete`` is false.  ``backtest_predictions`` then
    resets to cash instead of treating that missing observation as a zero
    return for a turnover calculation.
    """

    if not {"id", "weight"}.issubset(target_weights.columns):
        raise BacktestDataError("target weights require id and weight")
    held = target_weights[["id", "weight"]].copy()
    held["id"] = _coerce_id(held["id"])
    held["weight"] = pd.to_numeric(held["weight"], errors="coerce")
    held = held.loc[held["id"].ne("") & np.isfinite(held["weight"])].copy()
    if held["id"].duplicated().any():
        raise BacktestDataError("target weights contain duplicate ids")
    daily = _normalise_daily(daily_returns)
    month = _parse_month(holding_month).to_period("M")
    daily = daily.loc[daily["date"].dt.to_period("M").eq(month)]
    if daily.duplicated(["id", "date"], keep=False).any():
        raise BacktestDataError("daily returns contain duplicate id/date rows")
    gross_capital = float(held["weight"].abs().sum())
    if gross_capital <= 0:
        return held.assign(weight_pre=0.0), {"complete": False, "missing_weight": 1.0}
    if daily.empty:
        out = held[["id", "weight"]].copy()
        # With no observations the only honest pre-trade state is the prior
        # target itself.  Do not divide by gross exposure (that would turn a
        # +1/-1 long-short portfolio into +0.5/-0.5 and create artificial
        # turnover at the next rebalance).
        out["weight_pre"] = out["weight"]
        return out[["id", "weight_pre"]], {
            "complete": False,
            "missing_weight": 1.0,
            "gross_capital_denominator": gross_capital,
            "reason": "no_daily_returns",
        }
    # Mark each sleeve in its own capital account.  A long position's value is
    # multiplied by (1 + r); a short position's collateral/equity is
    # multiplied by (1 - r).  Applying (1 + r) to both signs would make a
    # losing short look more valuable and can materially understate turnover
    # when short-leg returns differ across names.
    growth: dict[str, float] = {str(identifier): 1.0 for identifier in held["id"]}
    value_error: str | None = None
    if "leg" in target_weights.columns:
        leg_values = target_weights[["id", "leg"]].copy()
        leg_values["id"] = _coerce_id(leg_values["id"])
        leg_map = {
            str(identifier): str(leg).strip().lower()
            for identifier, leg in zip(leg_values["id"], leg_values["leg"])
        }
    else:
        leg_map = {}
    missing_weight_max = 0.0
    for day in sorted(daily["date"].unique()):
        day_frame = daily.loc[daily["date"].eq(day)].set_index("id")
        day_missing_weight = 0.0
        for identifier, weight in zip(held["id"], held["weight"]):
            identifier_text = str(identifier)
            if identifier_text not in day_frame.index:
                day_missing_weight += abs(float(weight))
                continue
            value = day_frame.loc[identifier_text, "ret_exc"]
            try:
                finite = (
                    pd.notna(value)
                    and math.isfinite(float(value))
                    and float(value) >= -1.0
                )
            except (TypeError, ValueError):
                finite = False
            if not finite:
                day_missing_weight += abs(float(weight))
            else:
                leg = leg_map.get(identifier_text)
                if leg not in {"long", "short"}:
                    leg = "long" if float(weight) >= 0.0 else "short"
                signed_return = float(value) if leg == "long" else -float(value)
                multiplier = 1.0 + signed_return
                if not math.isfinite(multiplier) or multiplier < 0.0:
                    # A simple-return path with a negative capital value is
                    # not a valid mark-to-market state (for example a short
                    # position losing more than 100%).  Keep the failure
                    # explicit; the caller will flatten before turnover.
                    value_error = "nonpositive_mark_to_market_value"
                    continue
                growth[identifier_text] *= multiplier
        missing_weight_max = max(missing_weight_max, day_missing_weight)
    out = held.copy()
    # Keep the two sleeves on the same +1/-1 capital convention as the new
    # target.  A common gross denominator would halve both sleeves and produce
    # spurious turnover even when every holding had the same return.  This is
    # the usual high-minus-low portfolio convention: each leg is marked to
    # market, then re-expressed relative to that leg's capital budget.
    out["_marked_value"] = [
        float(weight) * growth[str(identifier)]
        for identifier, weight in zip(out["id"], out["weight"])
    ]
    if leg_map:
        out["_leg"] = out["id"].map(leg_map).where(
            out["id"].map(leg_map).isin(["long", "short"]),
            np.where(out["weight"].ge(0), "long", "short"),
        )
    else:
        out["_leg"] = np.where(out["weight"].ge(0), "long", "short")
    out["_leg_denominator"] = out.groupby("_leg", sort=False)["_marked_value"].transform(
        lambda values: float(values.abs().sum())
    )
    out["weight_pre"] = out["_marked_value"] / out["_leg_denominator"].replace(0.0, np.nan)
    out["weight_pre"] = out["weight_pre"].fillna(0.0)
    if preserve_leg_budget:
        out["weight_pre"] *= out.groupby("_leg")["weight"].transform(lambda x: x.abs().sum())
    diagnostics = {
        "complete": missing_weight_max <= 1e-12 and value_error is None and bool(
            (out.groupby("_leg", sort=False)["_leg_denominator"].first() > 0).all()
        ),
        "missing_weight": float(missing_weight_max / gross_capital),
        "gross_capital_denominator": gross_capital,
        "normalization": "preserve_target_leg_budgets" if preserve_leg_budget else "separate_long_short_legs",
        "short_mark_to_market": "signed_short_equity_1_minus_underlying_return",
    }
    if value_error is not None:
        diagnostics["reason"] = value_error
    if not diagnostics["complete"]:
        diagnostics["weight_pre_policy"] = "diagnostic_only_reset_before_turnover"
    return out[["id", "weight_pre"]], diagnostics


def performance_metrics(values: Iterable[Any]) -> dict[str, Any]:
    series = pd.to_numeric(pd.Series(list(values)), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    result: dict[str, Any] = {
        "months": int(len(series)),
        "mean_monthly_return": None,
        "annualized_return": None,
        "annualized_volatility": None,
        "sharpe": None,
        "cumulative_return": None,
        "max_drawdown": None,
        "positive_month_fraction": None,
    }
    if series.empty:
        return result
    wealth = (1.0 + series).cumprod()
    result["mean_monthly_return"] = float(series.mean())
    result["positive_month_fraction"] = float((series > 0).mean())
    result["cumulative_return"] = float(wealth.iloc[-1] - 1.0)
    if len(series) > 1:
        sample_std = float(series.std(ddof=1))
        result["annualized_volatility"] = sample_std * math.sqrt(12.0)
        result["sharpe"] = float(series.mean() / sample_std * math.sqrt(12.0)) if sample_std > 0 else None
    if float(wealth.iloc[-1]) > 0:
        result["annualized_return"] = float(wealth.iloc[-1] ** (12.0 / len(series)) - 1.0)
    result["max_drawdown"] = float((wealth / wealth.cummax() - 1.0).min())
    return result


def backtest_predictions(
    predictions: pd.DataFrame,
    panel: pd.DataFrame,
    daily_returns: pd.DataFrame,
    *,
    formation_months: Sequence[str | pd.Timestamp] | None = None,
    nyse_cutoffs: pd.DataFrame | None = None,
    n_groups: int = 3,
    min_breakpoint_stocks: int = DEFAULT_MIN_BREAKPOINT_STOCKS,
    min_leg_stocks: int = DEFAULT_MIN_LEG_STOCKS,
    strict_screens: bool = True,
    strict_cutoffs: bool = False,
    cost_bps: float = 10.0,
    coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD,
    _weight_builder: Callable[[pd.Timestamp], tuple[pd.DataFrame, dict[str, Any]]] | None = None,
    _preserve_leg_budget: bool = False,
) -> dict[str, Any]:
    """Run all formation months, including drift-based turnover and costs."""

    if (
        not math.isfinite(float(cost_bps))
        or cost_bps < 0
        or not math.isfinite(float(coverage_threshold))
        or coverage_threshold <= 0
        or coverage_threshold > 1
    ):
        raise ValueError("invalid cost_bps or coverage_threshold")

    pred = _normalise_columns(predictions)
    if pred.empty and formation_months is not None:
        # A rolling executor may legitimately produce no valid prediction
        # rows (for example every frozen-code month timed out).  With an
        # explicit formation schedule we can still emit one auditable
        # unavailable row per month instead of dropping the case.
        pred = pd.DataFrame(columns=["formation_eom", "id", "score"])
    elif "formation_eom" not in pred.columns:
        source = "time_key" if "time_key" in pred.columns else "eom" if "eom" in pred.columns else None
        if source is None:
            raise BacktestDataError("predictions require formation_eom/eom/time_key")
        pred["formation_eom"] = _parse_month_series(pred[source])
    else:
        pred["formation_eom"] = _parse_month_series(pred["formation_eom"])
    if formation_months is not None:
        requested_months = [_parse_month(x) for x in formation_months]
        if len(set(requested_months)) != len(requested_months):
            raise ValueError("formation_months must not contain duplicate months")
        # Turnover is path-dependent; always process an explicitly supplied
        # sequence chronologically even if a caller read the month list from
        # an unsorted file.
        months = sorted(requested_months)
    else:
        months = sorted(
            {
                _parse_month(value)
                for value in pred["formation_eom"].dropna().tolist()
            }
        )
    daily = _normalise_daily(daily_returns)
    previous_target: pd.DataFrame | None = None
    previous_holding_month: pd.Timestamp | None = None
    monthly_rows: list[dict[str, Any]] = []
    daily_frames: list[pd.DataFrame] = []
    weight_frames: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    for formation in months:
        current_pred = pred.loc[pred["formation_eom"].eq(formation)].copy()
        try:
            weights, formation_diag = _weight_builder(formation) if _weight_builder else form_jkp_weights(
                panel,
                current_pred,
                formation,
                nyse_cutoffs=nyse_cutoffs,
                n_groups=n_groups,
                min_breakpoint_stocks=min_breakpoint_stocks,
                min_leg_stocks=min_leg_stocks,
                strict_screens=strict_screens,
                strict_cutoffs=strict_cutoffs,
                return_diagnostics=True,
            )
        except Exception as exc:
            formation_diag = {
                "formation_eom": formation.strftime("%Y-%m-%d"),
                "tradable": False,
                "reason": f"formation_error: {type(exc).__name__}: {exc}",
            }
            weights = pd.DataFrame(columns=["formation_eom", "holding_month", "id", "score", "leg", "weight", "me", "group"])
        diagnostics.append(formation_diag)
        holding = formation + pd.offsets.MonthEnd(1)
        pre_trade: dict[str, float] = {}
        drift_diag: dict[str, Any] = {"complete": True, "missing_weight": 0.0}
        if previous_target is not None and previous_holding_month is not None and not previous_target.empty:
            try:
                drifted, drift_diag = drift_weights(
                    previous_target, daily, previous_holding_month,
                    preserve_leg_budget=_preserve_leg_budget,
                )
                if drift_diag.get("complete", False):
                    pre_trade = {str(row.id): float(row.weight_pre) for row in drifted.itertuples(index=False)}
                else:
                    # An incomplete mark-to-market state contains unknown
                    # capital.  Do not silently treat missing returns as zero
                    # and then compute turnover against a fabricated weight
                    # vector; flatten the prior state and make the reset part
                    # of the monthly audit instead.
                    pre_trade = {}
                    drift_diag["position_reset"] = "flat_after_incomplete_drift"
            except Exception as exc:
                drift_diag = {
                    "complete": False,
                    "missing_weight": 1.0,
                    "error": f"{type(exc).__name__}: {exc}",
                    "position_reset": "flat_after_drift_error",
                }
                diagnostics[-1]["drift_error"] = drift_diag["error"]
                pre_trade = {}
        target_map = {str(row.id): float(row.weight) for row in weights.itertuples(index=False)}
        all_ids = set(pre_trade) | set(target_map)
        volume = float(sum(abs(target_map.get(identifier, 0.0) - pre_trade.get(identifier, 0.0)) for identifier in all_ids))
        turnover = volume / 2.0
        entry_count = int(
            sum(
                abs(target_map.get(identifier, 0.0)) > 1e-12
                and abs(pre_trade.get(identifier, 0.0)) <= 1e-12
                for identifier in all_ids
            )
        )
        exit_count = int(
            sum(
                abs(pre_trade.get(identifier, 0.0)) > 1e-12
                and abs(target_map.get(identifier, 0.0)) <= 1e-12
                for identifier in all_ids
            )
        )
        side_switch_count = int(
            sum(
                target_map.get(identifier, 0.0) * pre_trade.get(identifier, 0.0) < -1e-12
                for identifier in all_ids
            )
        )
        long_volume = float(
            sum(
                abs(target_map.get(identifier, 0.0) - pre_trade.get(identifier, 0.0))
                for identifier in all_ids
                if target_map.get(identifier, 0.0) > 0.0 or pre_trade.get(identifier, 0.0) > 0.0
            )
        )
        short_volume = float(
            sum(
                abs(target_map.get(identifier, 0.0) - pre_trade.get(identifier, 0.0))
                for identifier in all_ids
                if target_map.get(identifier, 0.0) < 0.0 or pre_trade.get(identifier, 0.0) < 0.0
            )
        )
        cash_target = bool(_weight_builder and formation_diag.get("tradable") and weights.empty)
        if weights.empty and not cash_target:
            monthly = {
                "formation_eom": formation.strftime("%Y-%m-%d"),
                "holding_month": holding.strftime("%Y-%m"),
                # An unformable month is explicitly taken to cash.  Resetting
                # the prior target below prevents the same holding period from
                # being drifted twice on the next formation.
                "status": "unavailable_formation",
                "gross_return": None,
                "net_return": None,
                "long_return": None,
                "short_return": None,
                "short_signed_return": None,
                "gross_excess_return": None,
                "net_excess_return": None,
                "long_excess_return": None,
                "short_excess_return": None,
                "short_signed_excess_return": None,
                "turnover": turnover,
                "volume": volume,
                "one_way_turnover": turnover,
                "gross_turnover": volume,
                "long_volume": long_volume,
                "short_volume": short_volume,
                "entry_count": entry_count,
                "exit_count": exit_count,
                "side_switch_count": side_switch_count,
                "cost_fraction": cost_bps / 10_000.0 * volume,
                "coverage_min": None,
                "coverage_mean": None,
                "invalid_return_count": 0,
                "invalid_return_weight_max": None,
                "nonpositive_wealth": False,
                "n_days": 0,
                "n_long": 0,
                "n_short": 0,
                "drift_complete": drift_diag.get("complete", True),
                "drift_missing_weight": drift_diag.get("missing_weight", 0.0),
                "position_policy": "flat_after_unavailable_formation",
                "return_policy": RETURN_POLICY,
                "delisting_return_policy": DELISTING_RETURN_POLICY,
            }
            previous_target = None
            previous_holding_month = None
        else:
            try:
                settled = settle_daily_returns(
                    weights,
                    daily,
                    holding,
                    turnover=turnover,
                    volume=volume,
                    cost_bps=cost_bps,
                    coverage_threshold=coverage_threshold,
                    cash_when_empty=cash_target,
                )
            except Exception as exc:
                # Keep one malformed/misaligned holding month auditable and
                # continue the remaining origins. The unknown prior state is
                # flattened before the next formation.
                monthly = {
                    "formation_eom": formation.strftime("%Y-%m-%d"),
                    "holding_month": holding.strftime("%Y-%m"),
                    "status": "unavailable_settlement",
                    "gross_return": None,
                    "net_return": None,
                    "long_return": None,
                    "short_return": None,
                    "short_signed_return": None,
                    "gross_excess_return": None,
                    "net_excess_return": None,
                    "long_excess_return": None,
                    "short_excess_return": None,
                    "short_signed_excess_return": None,
                    "turnover": turnover,
                    "volume": volume,
                    "one_way_turnover": turnover,
                    "gross_turnover": volume,
                    "long_volume": long_volume,
                    "short_volume": short_volume,
                    "entry_count": entry_count,
                    "exit_count": exit_count,
                    "side_switch_count": side_switch_count,
                    "cost_fraction": cost_bps / 10_000.0 * volume,
                    "coverage_min": None,
                    "coverage_mean": None,
                    "invalid_return_count": 0,
                    "invalid_return_weight_max": None,
                    "nonpositive_wealth": False,
                    "n_days": 0,
                    "n_long": int((weights["leg"] == "long").sum()),
                    "n_short": int((weights["leg"] == "short").sum()),
                    "drift_complete": drift_diag.get("complete", True),
                    "drift_missing_weight": drift_diag.get("missing_weight", 0.0),
                    "position_policy": "flat_after_settlement_error",
                    "return_policy": RETURN_POLICY,
                    "delisting_return_policy": DELISTING_RETURN_POLICY,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                diagnostics[-1]["settlement_error"] = monthly["error"]
                weight_frames.append(weights.copy())
                weights = weights.iloc[0:0].copy()
                previous_target = None
                previous_holding_month = None
            else:
                monthly = dict(settled["monthly"])
                position_policy = (
                    "flat_after_incomplete_drift"
                    if not drift_diag.get("complete", True)
                    else "target_held_next_month"
                )
                monthly.update(
                    {
                        "formation_eom": formation.strftime("%Y-%m-%d"),
                        "n_long": int((weights["leg"] == "long").sum()),
                        "n_short": int((weights["leg"] == "short").sum()),
                        "drift_complete": drift_diag.get("complete", True),
                        "drift_missing_weight": drift_diag.get("missing_weight", 0.0),
                        "position_policy": position_policy,
                        "one_way_turnover": turnover,
                        "gross_turnover": volume,
                        "long_volume": long_volume,
                        "short_volume": short_volume,
                        "entry_count": entry_count,
                        "exit_count": exit_count,
                        "side_switch_count": side_switch_count,
                    }
                )
                day_frame = settled["daily"].copy()
                if not day_frame.empty:
                    day_frame.insert(0, "formation_eom", formation)
                    daily_frames.append(day_frame)
                weight_frames.append(weights.copy())
        monthly_rows.append(monthly)
        # A month whose daily path is unavailable is not a reliable mark-to-
        # market state for the next rebalance.  Flatten it explicitly so the
        # following turnover starts from cash rather than inventing a drifted
        # position from incomplete returns.
        if not weights.empty and monthly.get("status") == "ok":
            # Retain the sleeve label so mark-to-market applies the correct
            # long (1+r) versus short (1-r) equity convention.
            previous_target = weights[["id", "weight", "leg"]].copy()
            previous_holding_month = holding
        else:
            previous_target = None
            previous_holding_month = None
    monthly_frame = pd.DataFrame(monthly_rows)
    daily_frame = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    weights_frame = pd.concat(weight_frames, ignore_index=True) if weight_frames else pd.DataFrame()
    if not monthly_frame.empty and "status" in monthly_frame.columns:
        valid_rows = monthly_frame.loc[monthly_frame["status"].eq("ok")]
        gross_values = valid_rows["gross_return"].tolist()
        net_values = valid_rows["net_return"].tolist()
    else:
        gross_values, net_values = [], []
    def finite_mean(column: str) -> float | None:
        if monthly_frame.empty or column not in monthly_frame.columns:
            return None
        values = pd.to_numeric(monthly_frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        return float(values.mean()) if not values.empty else None

    return {
        "monthly": monthly_frame,
        "daily": daily_frame,
        "weights": weights_frame,
        "diagnostics": diagnostics,
        "summary": {
            "cost_bps": float(cost_bps),
            "coverage_threshold": float(coverage_threshold),
            "gross": performance_metrics(gross_values),
            "net": performance_metrics(net_values),
            "formation_months": int(len(months)),
            "tradable_months": int((monthly_frame["status"].eq("ok")).sum()) if not monthly_frame.empty and "status" in monthly_frame.columns else 0,
            "unavailable_months": int((~monthly_frame["status"].eq("ok")).sum()) if not monthly_frame.empty and "status" in monthly_frame.columns else 0,
            "average_one_way_turnover": finite_mean("one_way_turnover"),
            "average_coverage_min": finite_mean("coverage_min"),
            "return_policy": RETURN_POLICY,
            "delisting_return_policy": DELISTING_RETURN_POLICY,
        },
    }


def backtest_weights(
    weights: pd.DataFrame,
    daily_returns: pd.DataFrame,
    *,
    formation_months: Sequence[str | pd.Timestamp],
    cash_months: Sequence[str | pd.Timestamp] = (),
    cost_bps: float = 10.0,
    coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD,
) -> dict[str, Any]:
    """Settle preformed, netted stock weights without ranking or rescaling.

    Missing formations are unavailable; explicitly listed cash months are
    valid zero-excess-return targets. The existing Task C daily-return and
    incomplete-drift policies apply, with target leg budgets preserved.
    """
    required = {"formation_eom", "id", "weight"}
    if not required.issubset(weights):
        raise BacktestDataError(f"weights missing columns: {sorted(required - set(weights))}")
    frame = weights.copy()
    frame["formation_eom"] = _parse_month_series(frame["formation_eom"])
    if frame["formation_eom"].isna().any() or frame["id"].isna().any():
        raise BacktestDataError("weights contain null dates or IDs")
    frame["id"] = _coerce_id(frame["id"])
    frame["weight"] = pd.to_numeric(frame["weight"], errors="coerce")
    if frame["id"].eq("").any() or not np.isfinite(frame["weight"]).all():
        raise BacktestDataError("weights contain invalid IDs or nonfinite weights")
    if frame.duplicated(["formation_eom", "id"]).any():
        raise BacktestDataError("preformed weights must be netted to unique month/stock rows")
    months = [_parse_month(x) for x in formation_months]
    cash = {_parse_month(x) for x in cash_months}
    if not set(frame["formation_eom"]).issubset(months) or not cash.issubset(months):
        raise BacktestDataError("weights/cash months outside formation schedule")
    frame = frame.loc[frame["weight"].abs().gt(1e-12)].copy()
    if set(frame["formation_eom"]) & cash:
        raise BacktestDataError("cash month contains nonzero positions")
    frame["leg"] = np.where(frame["weight"].gt(0), "long", "short")
    frame["holding_month"] = frame["formation_eom"] + pd.offsets.MonthEnd(1)

    def build(formation: pd.Timestamp) -> tuple[pd.DataFrame, dict[str, Any]]:
        current = frame.loc[frame["formation_eom"].eq(formation)].copy()
        valid = not current.empty or formation in cash
        return current, {
            "formation_eom": formation.strftime("%Y-%m-%d"), "tradable": valid,
            "reason": "preformed_weights" if valid else "missing_preformed_weights",
            "gross_exposure": float(current["weight"].abs().sum()),
            "net_exposure": float(current["weight"].sum()),
            "cash_target": formation in cash,
        }

    return backtest_predictions(
        pd.DataFrame(), pd.DataFrame(), daily_returns, formation_months=months,
        cost_bps=cost_bps, coverage_threshold=coverage_threshold,
        _weight_builder=build, _preserve_leg_budget=True,
    )


__all__ = [
    "BacktestDataError",
    "DELISTING_RETURN_POLICY",
    "RETURN_POLICY",
    "backtest_predictions",
    "backtest_weights",
    "drift_weights",
    "form_jkp_weights",
    "load_daily_returns",
    "load_nyse_cutoffs",
    "load_stock_panel",
    "performance_metrics",
    "settle_daily_returns",
]

"""Task B (FactorRouterBench) utilities.

The implementation is deliberately small, deterministic, and provider-neutral.
It builds rolling-origin cases from monthly sleeve or raw-factor returns,
computes every feature from the history ending at the anchor month, validates
the structured router response, and settles predictions with one fixed
portfolio rule.

The module does not make model calls.  ``scripts/run_task_b.py`` is the HTTP
runner and ``scripts/evaluate_task_b.py`` is the local evaluator.  Keeping
those concerns separate makes it difficult for future returns to accidentally
enter an API payload.

The optional ``historical_panel`` helpers below build a separate public-input
variant for code-enabled experiments; they do not alter the canonical
one-row-per-factor payload contract.
"""

from __future__ import annotations

import inspect
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Public experiments have exactly two canonical tracks.  ``semantic_numeric``
# is retained as a read/CLI alias for old manifests, but new payloads always
# emit the shorter canonical name ``semantic``.
from .semantic import (  # noqa: E402
    CANONICAL_TRACKS,
    LEGACY_TRACK_ALIASES,
)

TRACKS = CANONICAL_TRACKS
LEGACY_TRACKS = tuple(LEGACY_TRACK_ALIASES)
HISTORICAL_PANEL_PROMPT_SUFFIX = "_historical_panel"

# These names are part of the local payload contract.  The aliases for ranks
# and z-scores are intentionally not emitted; keeping one canonical spelling
# makes the LLM input smaller and the trace validator unambiguous.
RETURN_WINDOWS = (1, 3, 6, 12, 36)
RISK_WINDOWS = (12, 36, 60)
MOMENT_WINDOWS = (12, 36)
MODEL_FEATURES = (
    "return_1m",
    "return_3m",
    "return_6m",
    "return_12m",
    "return_36m",
    "volatility_12m",
    "volatility_36m",
    "downside_volatility_12m",
    "downside_volatility_36m",
    "drawdown_12m",
    "drawdown_36m",
    "mean_return_12m",
    "mean_return_36m",
    "t_stat_12m",
    "t_stat_36m",
    "positive_ratio_12m",
    "positive_ratio_36m",
)
ROLLING_ML_METHODS = frozenset(
    {"logistic", "elastic_net", "tree", "mlp", "mlp_deep", "mlp_ensemble"}
)
ROLLING_ML_MIN_HISTORY_MONTHS = 12
ROLLING_ML_MIN_TRAINING_PERIODS = 3
ROLLING_ML_MIN_TRAINING_ROWS = 12
BASELINE_IMPLEMENTATION_VERSION = "task_b_baseline_v3_case_history_rolling_ml_dl"

# The regular Task B payload deliberately exposes one feature row per factor.
# ``HISTORICAL_PANEL_FEATURE_FIELDS`` is the smaller, non-pairwise schema used
# by the optional code-analysis payload.  It contains only statistics that can
# be computed from a prefix ending at the corresponding historical month.  In
# particular, it omits cross-factor correlations, ranks and z-scores: an
# agent may compute those itself without receiving a quadratic duplicated
# matrix in every historical snapshot.
HISTORICAL_PANEL_SCHEMA_VERSION = "task_b_historical_panel.v1"
HISTORICAL_PANEL_FEATURE_FIELDS = (
    # The same univariate signals used by the anchor feature contract.
    *MODEL_FEATURES,
    # Additional data-quality and risk fields make the panel useful for
    # missingness-aware ML while remaining linear in the number of factors.
    "volatility_60m",
    "downside_volatility_60m",
    "missing_rate_12m",
    "missing_rate_36m",
    "missing_rate_60m",
    "missing_12m",
    "missing_36m",
    "missing_60m",
    "n_obs_12m",
    "n_obs_36m",
    "n_obs_60m",
    "moment_n_obs_12m",
    "moment_n_obs_36m",
    "missing_current",
)
# Rolling ML uses the same point-in-time univariate signal and data-quality
# fields available to the benchmark.  Observation-count/missingness columns
# make partial 36/60-month windows explicit during the 12-month warm-up.
ROLLING_ML_FEATURES = HISTORICAL_PANEL_FEATURE_FIELDS


class RollingMLTrainingError(ValueError):
    """Raised when a rolling Task B ML baseline cannot be fitted honestly."""


def _normalise_column(value: Any) -> str:
    """Normalize a source column name without depending on Task A."""

    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _pick_column(columns: Iterable[Any], candidates: Sequence[str]) -> str | None:
    """Find a semantically named column in a long or wide input table."""

    normalised = {_normalise_column(column): str(column) for column in columns}
    for candidate in candidates:
        if candidate in normalised:
            return normalised[candidate]
    for candidate in candidates:
        for key, original in normalised.items():
            if key.endswith("_" + candidate) or key.startswith(candidate + "_"):
                return original
    return None


def _parse_month_end(values: pd.Series) -> pd.Series:
    """Parse YYYYMM or date-like values and normalize them to month end."""

    raw = values.astype("string").str.strip()
    compact = raw.str.fullmatch(r"\d{6}")
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    if compact.any():
        parsed.loc[compact] = pd.to_datetime(raw.loc[compact], format="%Y%m", errors="coerce")
    if (~compact).any():
        parsed.loc[~compact] = pd.to_datetime(raw.loc[~compact], errors="coerce")
    return parsed.dt.to_period("M").dt.to_timestamp("M")


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return _finite_float(value)
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        return _finite_float(value) if isinstance(value, float) else value
    if value is None or isinstance(value, (str, bool)):
        return value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _read_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _standardise_theme_frame(frame: pd.DataFrame, *, inferred_region: str = "WORLD") -> pd.DataFrame:
    """Normalise long or wide sleeve returns without guessing return units."""

    if frame.empty:
        raise ValueError("Returns table is empty")
    date_col = _pick_column(frame.columns, ("date", "eom", "month", "datetime", "time"))
    region_col = _pick_column(frame.columns, ("region", "country", "market", "excntry"))
    sleeve_col = _pick_column(
        frame.columns,
        ("theme_id", "sleeve_id", "factor_id", "theme", "sleeve", "factor", "portfolio"),
    )
    return_col = _pick_column(frame.columns, ("return", "ret", "factor_return", "theme_return", "value", "r"))
    if date_col is None:
        raise ValueError("Cannot find a date column in the sleeve returns table")

    if sleeve_col is not None and return_col is not None and sleeve_col != return_col:
        columns = [date_col, sleeve_col, return_col] + ([region_col] if region_col else [])
        out = frame[columns].copy()
        out.columns = ["date", "sleeve_id", "return"] + (["region"] if region_col else [])
    else:
        id_columns = [date_col] + ([region_col] if region_col else [])
        value_columns = [column for column in frame.columns if column not in id_columns]
        out = frame[id_columns + value_columns].melt(
            id_vars=id_columns,
            value_vars=value_columns,
            var_name="sleeve_id",
            value_name="return",
        )
        out = out.rename(columns={date_col: "date"})
        if region_col:
            out = out.rename(columns={region_col: "region"})

    if "region" not in out.columns:
        out["region"] = inferred_region.upper()
    out["date"] = _parse_month_end(out["date"])
    out["region"] = out["region"].astype("string").str.strip().replace({"": pd.NA})
    out["sleeve_id"] = out["sleeve_id"].astype("string").str.strip().replace({"": pd.NA})
    out["return"] = pd.to_numeric(out["return"], errors="coerce")
    out = out.dropna(subset=["date", "sleeve_id", "return"])
    out["region"] = out["region"].fillna("WORLD").astype(str).str.upper()
    out["sleeve_id"] = out["sleeve_id"].astype(str)
    out = (
        out.groupby(["date", "region", "sleeve_id"], as_index=False, sort=True)["return"]
        .mean()
        .sort_values(["region", "date", "sleeve_id"])
        .reset_index(drop=True)
    )
    if out.empty:
        raise ValueError("No usable sleeve returns remain after parsing")
    return out[["date", "region", "sleeve_id", "return"]]


def load_theme_returns(
    path: str | Path | pd.DataFrame,
    *,
    region: str | None = None,
    returns_in_percent: bool = False,
) -> pd.DataFrame:
    """Load JKP theme returns or an equivalent long/wide table.

    The unit conversion is explicit.  A magnitude heuristic is intentionally
    not used because it can silently alter a valid high-volatility series.
    """

    if isinstance(path, pd.DataFrame):
        out = _standardise_theme_frame(path)
    else:
        source = Path(path)
        if source.is_dir():
            paths = sorted(source.glob("*theme*.csv")) + sorted(source.glob("*theme*.parquet"))
            if not paths:
                paths = sorted(
                    item
                    for item in source.iterdir()
                    if item.suffix.lower() in {".csv", ".parquet", ".pq"}
                )
        else:
            paths = [source]
        if not paths:
            raise FileNotFoundError(f"No theme return files found under {source}")
        frames = [
            _standardise_theme_frame(_read_frame(item), inferred_region=item.stem.upper())
            for item in paths
        ]
        out = pd.concat(frames, ignore_index=True)
        out = (
            out.groupby(["date", "region", "sleeve_id"], as_index=False, sort=True)["return"]
            .mean()
            .sort_values(["region", "date", "sleeve_id"])
            .reset_index(drop=True)
        )
    if region is not None:
        out = out.loc[out["region"].str.upper() == str(region).upper()].copy()
    if returns_in_percent:
        out["return"] = out["return"] / 100.0
    if out.empty:
        raise ValueError("No sleeve returns remain after region filtering")
    return out


def _standardise_raw_factor_frame(
    frame: pd.DataFrame,
    *,
    inferred_region: str = "WORLD",
) -> pd.DataFrame:
    """Normalise a long/wide raw-factor return table.

    Raw-factor experiments must be explicit about the identifier column.  In
    particular, this helper deliberately does *not* accept ``theme_id`` or a
    generic ``theme`` column: accidentally pointing the rolling benchmark at
    the legacy 13-theme file should fail fast instead of silently changing the
    unit of observation.
    """

    if frame.empty:
        raise ValueError("Raw-factor returns table is empty")
    date_col = _pick_column(frame.columns, ("date", "eom", "month", "datetime", "time"))
    factor_col = _pick_column(
        frame.columns,
        ("factor_id", "factor", "characteristic", "signal", "variable", "portfolio"),
    )
    return_col = _pick_column(frame.columns, ("return", "ret", "factor_return", "value", "r"))
    region_col = _pick_column(frame.columns, ("region", "country", "market", "excntry"))
    if date_col is None:
        raise ValueError("Raw-factor source must contain a date/month column")
    if factor_col is None:
        raise ValueError(
            "Raw-factor source must contain factor_id (theme-only inputs are not accepted)"
        )
    if return_col is None:
        raise ValueError("Raw-factor source must contain a return column")
    # A file with an explicit ``theme_id`` but no factor_id is rejected above;
    # keep the check here as documentation and protection against ambiguous
    # column matching in future edits.
    if _normalise_column(factor_col) in {"theme", "theme_id", "sleeve", "sleeve_id"}:
        raise ValueError("Raw-factor loader received a theme/sleeve identifier, not factor_id")

    columns = [date_col, factor_col, return_col] + ([region_col] if region_col else [])
    out = frame[columns].copy()
    out.columns = ["date", "factor_id", "return"] + (["region"] if region_col else [])
    if "region" not in out.columns:
        out["region"] = str(inferred_region).upper()
    out["date"] = _parse_month_end(out["date"])
    out["factor_id"] = out["factor_id"].astype("string").str.strip().replace({"": pd.NA})
    out["region"] = out["region"].astype("string").str.strip().replace({"": pd.NA})
    out["return"] = pd.to_numeric(out["return"], errors="coerce")
    out = out.dropna(subset=["date", "factor_id", "return"])
    out["factor_id"] = out["factor_id"].astype(str)
    out["region"] = out["region"].fillna("WORLD").astype(str).str.upper()
    return (
        out.groupby(["date", "region", "factor_id"], as_index=False, sort=True)["return"]
        .mean()
        .sort_values(["region", "date", "factor_id"])
        .reset_index(drop=True)
    )[["date", "region", "factor_id", "return"]]


def load_raw_factor_returns(
    path: str | Path | pd.DataFrame,
    *,
    region: str | None = None,
    returns_in_percent: bool = False,
) -> pd.DataFrame:
    """Load monthly raw-factor returns without aggregating to themes.

    The returned schema is ``date, region, factor_id, return``.  A directory
    is restricted to files whose names contain ``factor`` when possible; a
    direct file is always validated for an explicit ``factor_id`` column.
    Units remain unchanged unless ``returns_in_percent=True`` is supplied.
    """

    if isinstance(path, pd.DataFrame):
        out = _standardise_raw_factor_frame(path)
    else:
        source = Path(path)
        if source.is_dir():
            paths = sorted(
                item
                for item in source.iterdir()
                if item.suffix.lower() in {".csv", ".parquet", ".pq"}
                and "factor" in item.name.lower()
            )
            if not paths:
                raise FileNotFoundError(
                    f"No raw-factor files (filename must contain 'factor') under {source}"
                )
        else:
            paths = [source]
        if not paths:
            raise FileNotFoundError(f"No raw-factor return files found under {source}")
        frames = [
            _standardise_raw_factor_frame(_read_frame(item), inferred_region="WORLD")
            for item in paths
        ]
        out = pd.concat(frames, ignore_index=True)
        out = (
            out.groupby(["date", "region", "factor_id"], as_index=False, sort=True)["return"]
            .mean()
            .sort_values(["region", "date", "factor_id"])
            .reset_index(drop=True)
        )
    if region is not None:
        out = out.loc[out["region"].str.upper() == str(region).upper()].copy()
    if returns_in_percent:
        out["return"] = out["return"] / 100.0
    if out.empty:
        raise ValueError("No raw-factor returns remain after region filtering")
    return out[["date", "region", "factor_id", "return"]]


# Explicit aliases make the source type visible at call sites while retaining
# the historical ``load_theme_returns``/``load_returns`` API for old dynamic
# and 13-theme experiments.
load_factor_returns = load_raw_factor_returns


# Task B's equivalent of Task A's ``load_returns``.  Keep the explicit name
# above for readability while offering the conventional task-local alias.
load_returns = load_theme_returns
load_router_returns = load_theme_returns


def load_sleeve_cards(
    path: str | Path | None,
    sleeve_ids: Iterable[str],
) -> dict[str, dict[str, Any]]:
    """Load semantic sleeve descriptions, defaulting to neutral cards."""

    ids = [str(value) for value in sleeve_ids]
    cards: dict[str, dict[str, Any]] = {
        sleeve_id: {
            "factor_id": sleeve_id,
            # Keep the private fallback empty.  The public serializer falls
            # back to the masked Sxx id; deriving a title from a raw JKP
            # mnemonic would defeat the anonymous/semantic boundary when a
            # metadata row is missing.
            "neutral_name": "",
            "neutral_description": "",
            "formula_or_definition": "",
            "input_fields": [],
            "source_paper": "",
        }
        for sleeve_id in ids
    }
    if path is None:
        return cards
    source = Path(path)
    if source.suffix.lower() == ".json":
        raw = json.loads(source.read_text(encoding="utf-8"))
        rows = list(raw.values()) if isinstance(raw, Mapping) else raw
        frame = pd.DataFrame(rows)
    elif source.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
        frame = pd.DataFrame(rows)
    elif source.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(source)
    else:
        frame = pd.read_csv(source)
    if frame.empty:
        return cards

    id_col = _pick_column(
        frame.columns,
        ("sleeve_id", "theme_id", "factor_id", "sleeve", "theme", "factor", "mnemonic", "abbreviation"),
    )
    if id_col is None:
        return cards
    name_col = _pick_column(frame.columns, ("neutral_name", "sleeve_name", "theme_name", "factor_name", "short_name", "name"))
    description_col = _pick_column(frame.columns, ("neutral_description", "description", "economic_description", "definition"))
    formula_col = _pick_column(frame.columns, ("formula_or_definition", "formula", "construction"))
    input_col = _pick_column(frame.columns, ("input_fields", "inputs", "fields"))
    paper_col = _pick_column(frame.columns, ("source_paper", "paper", "citation", "reference"))

    def text(row: pd.Series, column: str | None) -> str:
        if column is None:
            return ""
        value = row.get(column)
        if value is None:
            return ""
        if isinstance(value, (list, tuple)):
            return json.dumps(value, ensure_ascii=False)
        try:
            if pd.isna(value):
                return ""
        except (TypeError, ValueError):
            pass
        return str(value).strip()

    for _, row in frame.iterrows():
        sleeve_id = text(row, id_col)
        if sleeve_id not in cards:
            continue
        raw_inputs = text(row, input_col)
        inputs: list[str]
        if raw_inputs.startswith("["):
            try:
                parsed = json.loads(raw_inputs)
                inputs = [str(item) for item in parsed] if isinstance(parsed, list) else []
            except json.JSONDecodeError:
                inputs = []
        else:
            inputs = [item.strip() for item in re.split(r"[,;|]", raw_inputs) if item.strip()]
        cards[sleeve_id].update(
            {
                "neutral_name": text(row, name_col) or cards[sleeve_id]["neutral_name"],
                "neutral_description": text(row, description_col),
                "formula_or_definition": text(row, formula_col),
                "input_fields": inputs,
                # Kept in the private card cache for compatibility and
                # stratified selection.  ``semantic_card_payload`` applies
                # the public whitelist and therefore never emits this field.
                "source_paper": text(row, paper_col),
            }
        )
        # Formula-enriched JKP files carry these fields in addition to the
        # compact columns above.  Preserve only the semantic source material
        # here; the public serializer still performs a second whitelist pass.
        for field_name in (
            "formula_latex",
            "formula_latex_instantiated",
            "formula_inputs",
        ):
            if field_name not in row.index:
                continue
            value = row.get(field_name)
            if value is None:
                continue
            try:
                if pd.isna(value):
                    continue
            except (TypeError, ValueError):
                pass
            if isinstance(value, str):
                raw_value = value.strip()
                if field_name == "formula_inputs" and raw_value.startswith("["):
                    try:
                        raw_value = json.loads(raw_value)
                    except (TypeError, ValueError, json.JSONDecodeError):
                        pass
                cards[sleeve_id][field_name] = raw_value
            elif isinstance(value, (list, tuple, dict)):
                cards[sleeve_id][field_name] = value
    return cards


@dataclass
class RouterEpisode:
    """One month-end decision and its hidden forward-return target."""

    case_id: str
    region: str
    anchor_date: pd.Timestamp
    history: pd.DataFrame
    future: pd.DataFrame
    sleeve_cards: dict[str, dict[str, Any]]

    @property
    def sleeve_ids(self) -> list[str]:
        return [str(value) for value in self.history.columns]

    @property
    def factor_ids(self) -> list[str]:
        return self.sleeve_ids

    @property
    def horizon_months(self) -> int:
        return int(len(self.future.index))

    @property
    def target_definition(self) -> str:
        return (
            "single_month_return"
            if self.horizon_months == 1
            else "compounded_total_return_over_forward_window"
        )

    @property
    def target(self) -> pd.Series:
        if self.future.empty:
            return pd.Series(dtype=float)
        values = self.future.apply(pd.to_numeric, errors="coerce")
        if len(values.index) == 1:
            return values.iloc[0].copy()
        # A partial forward year is not treated as a complete annual label.
        # ``min_count`` preserves NaN for any sleeve missing at least one
        # target month instead of silently compounding a shorter interval.
        return (1.0 + values).prod(axis=0, min_count=len(values.index)) - 1.0


def _to_monthly_matrix(long_returns: pd.DataFrame, region: str) -> pd.DataFrame:
    subset = long_returns.loc[long_returns["region"].str.upper() == str(region).upper()].copy()
    if subset.empty:
        raise ValueError(f"No returns found for region={region}")
    matrix = (
        subset.pivot_table(index="date", columns="sleeve_id", values="return", aggfunc="mean")
        .sort_index()
    )
    if matrix.empty:
        raise ValueError(f"No usable sleeve matrix for region={region}")
    full_index = pd.date_range(matrix.index.min(), matrix.index.max(), freq="ME")
    matrix = matrix.reindex(full_index)
    matrix.index.name = "date"
    matrix.columns = [str(value) for value in matrix.columns]
    return matrix


def build_router_episodes(
    long_returns: str | Path | pd.DataFrame,
    sleeve_cards: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    regions: Sequence[str] | None = None,
    lookback: int = 120,
    horizon: int = 1,
    min_history: int | None = None,
    min_sleeves: int = 3,
    anchor_step: int = 1,
    sleeve_ids: Sequence[str] | None = None,
    max_cases: int | None = None,
) -> list[RouterEpisode]:
    """Build rolling-origin episodes using only data available at each anchor.

    The implementation is ID-agnostic: ``sleeve_ids`` may be theme sleeves or
    raw factor IDs.  The raw-factor window builder uses this same episode and
    payload contract while applying its own reproducible subset plan.
    """

    # Short-window benchmarks (2Y/3Y) are valid point-in-time episodes too.
    # Keep the generic builder's lower bound aligned with Task A's 12-month
    # minimum rather than hard-coding the legacy 5Y/60-month experiment.
    if lookback < 12 or horizon < 1 or min_sleeves < 1:
        raise ValueError("lookback must be at least 12 and horizon/min_sleeves must be positive")
    if anchor_step < 1:
        raise ValueError("anchor_step must be positive")
    # A sleeve/factor may have a small number of missing observations; those
    # are surfaced in the payload instead of being silently filled or
    # discarded by default.  Preserve the old 60-month minimum for legacy
    # windows while making the default feasible for 24- and 36-month windows.
    if min_history is not None:
        minimum = int(min_history)
    else:
        legacy_floor = 60 if lookback >= 60 else 0
        minimum = int(min(lookback, max(12, lookback - 12, legacy_floor)))
    if minimum > lookback:
        raise ValueError("min_history cannot exceed lookback")
    returns = (
        load_theme_returns(long_returns)
        if not isinstance(long_returns, pd.DataFrame)
        else _standardise_theme_frame(long_returns)
    )
    wanted_regions = [str(value).upper() for value in regions] if regions else sorted(returns["region"].unique())
    cards = {str(key): dict(value) for key, value in (sleeve_cards or {}).items()}
    episodes: list[RouterEpisode] = []

    for region in wanted_regions:
        matrix = _to_monthly_matrix(returns, region)
        if len(matrix.index) < lookback + horizon:
            continue
        requested = [str(value) for value in sleeve_ids] if sleeve_ids else list(matrix.columns)
        for end_index in range(lookback - 1, len(matrix.index) - horizon, anchor_step):
            # Sleeve availability is measured only in the history.  It is
            # deliberately not based on target returns or future statistics.
            selected = [
                sleeve
                for sleeve in requested
                if sleeve in matrix.columns
                and int(matrix.iloc[end_index - lookback + 1 : end_index + 1][sleeve].notna().sum()) >= minimum
            ]
            if len(selected) < min_sleeves:
                continue
            history = matrix.iloc[end_index - lookback + 1 : end_index + 1][selected].copy()
            future = matrix.iloc[end_index + 1 : end_index + horizon + 1][selected].copy()
            if future.empty or not future.notna().to_numpy().any():
                continue
            anchor = pd.Timestamp(matrix.index[end_index])
            case_id = f"{region}_{anchor.strftime('%Y%m%d')}"
            card_subset = {
                sleeve: dict(
                    cards.get(
                        sleeve,
                        {
                            "factor_id": sleeve,
                            "neutral_name": "",
                            "neutral_description": "",
                            "formula_or_definition": "",
                            "input_fields": [],
                            "source_paper": "",
                        },
                    )
                )
                for sleeve in selected
            }
            episodes.append(
                RouterEpisode(
                    case_id=case_id,
                    region=region,
                    anchor_date=anchor,
                    history=history,
                    future=future,
                    sleeve_cards=card_subset,
                )
            )
            if max_cases is not None and len(episodes) >= max_cases:
                return episodes
    return episodes


# A short alias mirrors Task A and makes small downstream scripts simpler.
build_episodes = build_router_episodes
# Explicit name for callers constructing a raw-factor Task B benchmark.  The
# returned ``RouterEpisode`` retains the historical ``sleeve_ids`` property
# for backward compatibility, but every ID is treated as one forecastable
# factor and no sleeve aggregation is performed.
build_raw_factor_episodes = build_router_episodes


def _window_values(series: pd.Series, window: int) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").tail(window).dropna()


def _cumulative_return(series: pd.Series, window: int) -> float | None:
    values = _window_values(series, window)
    if values.empty:
        return None
    return _finite_float(np.prod(1.0 + values.to_numpy(dtype=float)) - 1.0)


def _volatility(series: pd.Series, window: int) -> float | None:
    values = _window_values(series, window)
    if len(values) < 2:
        return None
    return _finite_float(values.std(ddof=1))


def _downside_volatility(series: pd.Series, window: int) -> float | None:
    values = _window_values(series, window).to_numpy(dtype=float)
    if len(values) == 0:
        return None
    downside = np.minimum(values, 0.0)
    return _finite_float(np.sqrt(np.mean(downside**2)))


def _maximum_drawdown(series: pd.Series, window: int) -> float | None:
    values = _window_values(series, window).to_numpy(dtype=float)
    if len(values) == 0:
        return None
    wealth = np.cumprod(1.0 + values)
    peaks = np.maximum.accumulate(wealth)
    return _finite_float(np.min(wealth / np.maximum(peaks, 1e-12) - 1.0))


def _mean_tstat_positive(series: pd.Series, window: int) -> tuple[float | None, float | None, float | None, int]:
    values = _window_values(series, window)
    n_obs = int(len(values))
    if n_obs == 0:
        return None, None, None, 0
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if n_obs >= 2 else float("nan")
    t_stat = mean / (std / math.sqrt(n_obs)) if math.isfinite(std) and std > 1e-12 else None
    return _finite_float(mean), _finite_float(t_stat), _finite_float((values > 0).mean()), n_obs


def _cross_sectional_values(rows: list[dict[str, Any]], field: str) -> tuple[dict[str, float | None], dict[str, float | None]]:
    numeric = {
        str(row["factor_id"]): _finite_float(row.get(field))
        for row in rows
    }
    valid = pd.Series({key: value for key, value in numeric.items() if value is not None}, dtype=float)
    ranks: dict[str, float | None] = {key: None for key in numeric}
    zscores: dict[str, float | None] = {key: None for key in numeric}
    if valid.empty:
        return ranks, zscores
    rank_values = valid.rank(method="average", pct=True)
    mean = float(valid.mean())
    std = float(valid.std(ddof=0))
    for key, value in valid.items():
        ranks[key] = _finite_float(rank_values[key])
        zscores[key] = _finite_float((float(value) - mean) / std) if std > 1e-12 else 0.0
    return ranks, zscores


def build_features(history: pd.DataFrame) -> pd.DataFrame:
    """Compute the point-in-time feature table for one anchor.

    ``history`` must end at the decision month.  Missing observations stay
    missing; no global imputation, PCA, z-score, or regime label is fitted.
    """

    if history.empty:
        raise ValueError("history must contain at least one month")
    frame = history.copy()
    frame.columns = [str(value) for value in frame.columns]
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.sort_index()
    ids = list(frame.columns)
    rows: list[dict[str, Any]] = []

    for sleeve_id in ids:
        series = frame[sleeve_id]
        row: dict[str, Any] = {"factor_id": sleeve_id}
        for window in RETURN_WINDOWS:
            row[f"return_{window}m"] = _cumulative_return(series, window)
        for window in RISK_WINDOWS:
            row[f"volatility_{window}m"] = _volatility(series, window)
            row[f"downside_volatility_{window}m"] = _downside_volatility(series, window)
            row[f"missing_rate_{window}m"] = _finite_float(1.0 - len(_window_values(series, window)) / max(window, 1))
            row[f"missing_{window}m"] = bool(len(_window_values(series, window)) < window)
            row[f"n_obs_{window}m"] = int(len(_window_values(series, window)))
        for window in MOMENT_WINDOWS:
            mean, t_stat, positive_ratio, n_obs = _mean_tstat_positive(series, window)
            row[f"mean_return_{window}m"] = mean
            row[f"t_stat_{window}m"] = t_stat
            row[f"positive_ratio_{window}m"] = positive_ratio
            row[f"moment_n_obs_{window}m"] = n_obs
            row[f"drawdown_{window}m"] = _maximum_drawdown(series, window)
        row["missing_current"] = bool(pd.isna(series.iloc[-1]))
        rows.append(row)

    # Pairwise correlations used to be computed by calling ``Series.corr``
    # once for every ordered pair.  That is quadratic Python overhead and is
    # prohibitively slow for the 153-factor raw-factor cases.  Compute each
    # rolling correlation matrix in one vectorized pandas operation instead;
    # the row-wise loop below only serializes the already-computed values into
    # the feature schema.
    correlation_matrices: dict[int, pd.DataFrame] = {}
    correlation_change_matrices: dict[int, pd.DataFrame] = {}
    for window in (12, 36):
        current = frame.tail(window).corr(min_periods=3)
        current = current.reindex(index=ids, columns=ids)
        correlation_matrices[window] = current
        if len(frame.index) >= 2 * window:
            previous = frame.iloc[:-window].tail(window).corr(min_periods=3)
            previous = previous.reindex(index=ids, columns=ids)
            correlation_change_matrices[window] = current - previous
        else:
            correlation_change_matrices[window] = pd.DataFrame(
                np.nan, index=ids, columns=ids, dtype=float
            )

    for left_id in ids:
        row = next(item for item in rows if item["factor_id"] == left_id)
        other_12: list[float] = []
        other_36: list[float] = []
        changes_12: list[float] = []
        changes_36: list[float] = []
        for right_id in ids:
            if right_id == left_id:
                continue
            corr_12 = _finite_float(correlation_matrices[12].at[left_id, right_id])
            corr_36 = _finite_float(correlation_matrices[36].at[left_id, right_id])
            change_12 = _finite_float(correlation_change_matrices[12].at[left_id, right_id])
            change_36 = _finite_float(correlation_change_matrices[36].at[left_id, right_id])
            row[f"correlation_{right_id}_12m"] = corr_12
            row[f"correlation_{right_id}_36m"] = corr_36
            row[f"correlation_change_{right_id}_12m"] = change_12
            row[f"correlation_change_{right_id}_36m"] = change_36
            if corr_12 is not None:
                other_12.append(corr_12)
            if corr_36 is not None:
                other_36.append(corr_36)
            if change_12 is not None:
                changes_12.append(change_12)
            if change_36 is not None:
                changes_36.append(change_36)
        row["mean_correlation_12m"] = _finite_float(np.mean(other_12)) if other_12 else None
        row["mean_correlation_36m"] = _finite_float(np.mean(other_36)) if other_36 else None
        row["mean_abs_correlation_12m"] = _finite_float(np.mean(np.abs(other_12))) if other_12 else None
        row["mean_abs_correlation_36m"] = _finite_float(np.mean(np.abs(other_36))) if other_36 else None
        row["mean_correlation_change_12m"] = _finite_float(np.mean(changes_12)) if changes_12 else None
        row["mean_correlation_change_36m"] = _finite_float(np.mean(changes_36)) if changes_36 else None

    cross_sectional_fields = list(MODEL_FEATURES)
    for field in cross_sectional_fields:
        ranks, zscores = _cross_sectional_values(rows, field)
        for row in rows:
            sleeve_id = str(row["factor_id"])
            row[f"rank_{field}"] = ranks[sleeve_id]
            row[f"zscore_{field}"] = zscores[sleeve_id]

    result = pd.DataFrame(rows).set_index("factor_id", drop=False)
    return result


def build_compact_features(history: pd.DataFrame) -> pd.DataFrame:
    """Build the linear-size univariate feature table for one history prefix.

    This is intentionally separate from :func:`build_features`.  The regular
    benchmark feature table also contains one correlation field for every
    ordered factor pair, which is useful at the anchor but would be needlessly
    expensive to repeat for every month in a code-analysis panel.  This helper
    emits the stable fields listed in ``HISTORICAL_PANEL_FEATURE_FIELDS`` and
    never looks beyond the last row of ``history``.
    """

    if history.empty:
        raise ValueError("history must contain at least one month")
    frame = history.copy()
    frame.columns = [str(value) for value in frame.columns]
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.sort_index()
    rows: list[dict[str, Any]] = []

    for sleeve_id in frame.columns:
        series = frame[sleeve_id]
        row: dict[str, Any] = {"factor_id": str(sleeve_id)}
        for window in RETURN_WINDOWS:
            row[f"return_{window}m"] = _cumulative_return(series, window)
        for window in RISK_WINDOWS:
            values = _window_values(series, window)
            row[f"volatility_{window}m"] = _volatility(series, window)
            row[f"downside_volatility_{window}m"] = _downside_volatility(series, window)
            row[f"missing_rate_{window}m"] = _finite_float(
                1.0 - len(values) / max(window, 1)
            )
            row[f"missing_{window}m"] = bool(len(values) < window)
            row[f"n_obs_{window}m"] = int(len(values))
        for window in MOMENT_WINDOWS:
            mean, t_stat, positive_ratio, n_obs = _mean_tstat_positive(series, window)
            row[f"mean_return_{window}m"] = mean
            row[f"t_stat_{window}m"] = t_stat
            row[f"positive_ratio_{window}m"] = positive_ratio
            row[f"moment_n_obs_{window}m"] = n_obs
            row[f"drawdown_{window}m"] = _maximum_drawdown(series, window)
        row["missing_current"] = bool(pd.isna(series.iloc[-1]))
        rows.append(row)

    result = pd.DataFrame(rows).set_index("factor_id", drop=False)
    # Keep the order explicit and stable even if a future edit adds an
    # intermediate calculation field above.
    return result[["factor_id", *HISTORICAL_PANEL_FEATURE_FIELDS]]


build_router_features = build_features


def _opaque_case_id(episode: RouterEpisode) -> str:
    # The digest is stable for resume/evaluation but does not expose a calendar
    # date or region to the model.  Preserve the original monthly digest while
    # namespacing forward-horizon cases so monthly and annual datasets cannot
    # collide when run logs are combined.
    import hashlib

    if episode.horizon_months == 1 and len(episode.history.index) == 120:
        identity = f"{episode.region}|{episode.case_id}"
    else:
        identity = (
            f"{episode.region}|{episode.case_id}|lookback={len(episode.history.index)}|"
            f"horizon={episode.horizon_months}|target={episode.target_definition}"
        )
    raw = identity.encode("utf-8")
    return f"case_{hashlib.sha256(raw).hexdigest()[:16]}"


def response_factor_ids(payload: Mapping[str, Any]) -> list[str]:
    features = payload.get("features") or []
    return [str(item["factor_id"]) for item in features if isinstance(item, Mapping) and "factor_id" in item]


def _cap_and_normalize(weights: pd.Series, max_weight: float = 0.40) -> pd.Series:
    values = pd.to_numeric(weights, errors="coerce").fillna(0.0).clip(lower=0.0).astype(float)
    if values.empty:
        return values
    if values.sum() <= 1e-12:
        values[:] = 1.0
    values = values / values.sum()
    # A 40% cap is feasible for the normal MVP universe (at least 3 sleeves).
    # For tiny synthetic universes, use the smallest feasible cap so weights
    # still sum to one and the function remains useful in unit tests.
    cap = max(float(max_weight), 1.0 / len(values))
    free = pd.Series(True, index=values.index)
    fixed = pd.Series(0.0, index=values.index)
    for _ in range(len(values) + 1):
        proposed = values.copy()
        proposed.loc[~free] = fixed.loc[~free]
        over = free & (proposed > cap + 1e-12)
        if not over.any():
            values = proposed
            break
        fixed.loc[over] = cap
        free.loc[over] = False
        remaining = max(0.0, 1.0 - float(fixed.loc[~free].sum()))
        if free.any():
            base = values.loc[free].clip(lower=0.0)
            if base.sum() <= 1e-12:
                base[:] = 1.0
            values.loc[free] = base / base.sum() * remaining
        else:
            values = fixed.copy()
            break
    total = float(values.sum())
    if total > 1e-12:
        values = values / total
    return values


def inverse_volatility_weights(
    history: pd.DataFrame,
    *,
    sleeve_ids: Sequence[str] | None = None,
    window: int = 36,
    max_weight: float = 0.40,
) -> pd.Series:
    """Long-only inverse-volatility/risk-parity approximation from history."""

    ids = [str(value) for value in (sleeve_ids or history.columns)]
    ids = [value for value in ids if value in history.columns]
    if not ids:
        return pd.Series(dtype=float)
    vol = history[ids].tail(window).std(ddof=1).replace([np.inf, -np.inf], np.nan)
    valid = vol[vol > 1e-12]
    fallback = float(valid.median()) if not valid.empty else 1.0
    vol = vol.fillna(fallback).clip(lower=1e-8)
    raw = 1.0 / vol
    return _cap_and_normalize(raw, max_weight=max_weight)


risk_parity_weights = inverse_volatility_weights


def turnover(current: Mapping[str, float] | pd.Series, previous: Mapping[str, float] | pd.Series | None = None) -> float:
    """Compute the README's single-sided turnover definition."""

    current_series = pd.Series(current, dtype=float)
    previous_series = pd.Series(previous, dtype=float) if previous is not None else pd.Series(dtype=float)
    index = current_series.index.union(previous_series.index)
    delta = current_series.reindex(index, fill_value=0.0) - previous_series.reindex(index, fill_value=0.0)
    return float(0.5 * delta.abs().sum())


def _prediction_map(
    predictions: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
    *,
    masked_to_original: Mapping[str, str] | None = None,
) -> dict[str, Mapping[str, Any]]:
    items = predictions.items() if isinstance(predictions, Mapping) else (
        (str(item.get("factor_id")), item) for item in predictions if isinstance(item, Mapping)
    )
    result: dict[str, Mapping[str, Any]] = {}
    for raw_id, item in items:
        original = masked_to_original.get(str(raw_id), str(raw_id)) if masked_to_original else str(raw_id)
        result[original] = item
    return result


def fixed_portfolio_weights(
    history: pd.DataFrame,
    predictions: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
    *,
    masked_to_original: Mapping[str, str] | None = None,
    top_k: int = 3,
    probability_threshold: float = 0.5,
    vol_window: int = 36,
    max_weight: float = 0.40,
) -> pd.Series:
    """Apply the frozen Top-3 inverse-volatility routing rule."""

    ids = [str(value) for value in history.columns]
    prediction_map = _prediction_map(predictions, masked_to_original=masked_to_original)
    rows = []
    for index, sleeve_id in enumerate(ids):
        item = prediction_map.get(sleeve_id, {})
        try:
            probability = float(item.get("probability_positive", 0.5))
        except (TypeError, ValueError):
            probability = 0.5
        try:
            rank = float(item.get("rank", index + 1))
        except (TypeError, ValueError):
            rank = float(index + 1)
        abstain = bool(item.get("abstain", False))
        rows.append((sleeve_id, probability, rank, abstain))
    positive = [row for row in rows if row[1] > probability_threshold and not row[3]]
    all_at_or_below = not any(row[1] > probability_threshold for row in rows)
    if all_at_or_below:
        return inverse_volatility_weights(history, sleeve_ids=ids, window=vol_window, max_weight=max_weight).reindex(ids, fill_value=0.0)
    ranked = sorted(rows, key=lambda row: (-row[1], row[2], row[0]))
    # The protocol says Top-3 whenever there is at least one non-fallback
    # signal.  Abstained rows are never selected unless needed to fill Top-3.
    selected = [row[0] for row in ranked if not row[3]][:top_k]
    if len(selected) < min(top_k, len(ids)):
        selected.extend(row[0] for row in ranked if row[0] not in selected and row[0] not in {x[0] for x in positive})
        selected = selected[:top_k]
    selected = selected or ids
    weights = inverse_volatility_weights(
        history,
        sleeve_ids=selected,
        window=vol_window,
        max_weight=max_weight,
    )
    return weights.reindex(ids, fill_value=0.0)


portfolio_weights = fixed_portfolio_weights


def _sigmoid(value: float) -> float:
    value = float(np.clip(value, -30.0, 30.0))
    return float(1.0 / (1.0 + math.exp(-value)))


def _ranked_predictions(ids: Sequence[str], scores: Mapping[str, float | None], *, method: str) -> dict[str, Any]:
    clean = {str(key): (_finite_float(value) if value is not None else None) for key, value in scores.items()}
    valid = [value for value in clean.values() if value is not None]
    mean = float(np.mean(valid)) if valid else 0.0
    std = float(np.std(valid)) if len(valid) > 1 else 0.0
    zscores = {
        key: ((float(value) - mean) / std if value is not None and std > 1e-12 else 0.0)
        for key, value in clean.items()
    }
    order = sorted(ids, key=lambda key: (-zscores.get(str(key), 0.0), str(key)))
    ranks = {sleeve_id: index + 1 for index, sleeve_id in enumerate(order)}
    predictions = []
    for sleeve_id in ids:
        z = zscores.get(str(sleeve_id), 0.0)
        missing = clean.get(str(sleeve_id)) is None
        predictions.append(
            {
                "factor_id": str(sleeve_id),
                "probability_positive": 0.5 if missing else round(_sigmoid(1.25 * z), 6),
                "expected_return_z": round(float(z), 6),
                "rank": int(ranks[str(sleeve_id)]),
                "confidence": round(0.25 if missing else min(0.99, 0.5 + 0.15 * abs(float(z))), 6),
                "abstain": bool(missing),
            }
        )
    return {"predictions": predictions, "market_view": f"Deterministic {method} baseline.", "data_quality_warning": []}


def _cached_features(
    episode: RouterEpisode,
    feature_tables: Mapping[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    if feature_tables is not None and episode.case_id in feature_tables:
        return feature_tables[episode.case_id]
    return build_features(episode.history)


def _model_feature_row(
    table: pd.DataFrame,
    factor_id: str,
    fields: Sequence[str],
) -> list[float]:
    """Return one numeric ML row while preserving missing values for imputation."""

    output: list[float] = []
    for field in fields:
        value = _finite_float(table.loc[factor_id].get(field))
        output.append(float("nan") if value is None else float(value))
    return output


def _rolling_training_target(history_window: pd.DataFrame) -> pd.Series:
    """Settle one fully observed historical horizon without reading episode.future."""

    values = history_window.apply(pd.to_numeric, errors="coerce")
    if len(values.index) == 1:
        return values.iloc[0].copy()
    return (1.0 + values).prod(axis=0, min_count=len(values.index)) - 1.0


def _rolling_model_training_rows(
    episode: RouterEpisode,
    fields: Sequence[str],
    *,
    min_history_months: int = ROLLING_ML_MIN_HISTORY_MONTHS,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Build point-in-time factor-month rows from the current case history.

    For a cutoff ``t`` the feature prefix ends at ``t`` and the label is the
    fully realised return over ``t+1 .. t+horizon``.  The final legal label
    therefore ends at the anchor; ``episode.future`` is never used for fit.
    """

    frame = episode.history.copy()
    frame.columns = [str(value) for value in frame.columns]
    frame = frame.apply(pd.to_numeric, errors="coerce").sort_index()
    horizon = max(int(episode.horizon_months), 1)
    x_rows: list[list[float]] = []
    y_rows: list[int] = []
    training_periods = 0
    first_cutoff = max(int(min_history_months) - 1, 0)
    # ``range`` stops before ``len(frame) - horizon``.  Consequently every
    # training horizon is fully contained in history and known at the anchor.
    for cutoff in range(first_cutoff, len(frame.index) - horizon):
        prefix = frame.iloc[: cutoff + 1]
        target_window = frame.iloc[cutoff + 1 : cutoff + 1 + horizon]
        if len(target_window.index) != horizon:
            continue
        table = build_compact_features(prefix)
        target = _rolling_training_target(target_window)
        rows_before = len(y_rows)
        for factor_id in frame.columns:
            value = _finite_float(target.get(factor_id))
            if value is None:
                continue
            x_rows.append(_model_feature_row(table, str(factor_id), fields))
            y_rows.append(int(value > 0.0))
        if len(y_rows) > rows_before:
            training_periods += 1
    if not x_rows:
        return (
            np.empty((0, len(fields)), dtype=float),
            np.empty((0,), dtype=int),
            training_periods,
        )
    return np.asarray(x_rows, dtype=float), np.asarray(y_rows, dtype=int), training_periods


def _model_predictions(
    episode: RouterEpisode,
    method: str,
    feature_tables: Mapping[str, pd.DataFrame] | None = None,
    training_cache: Mapping[str, tuple[np.ndarray, np.ndarray, int]] | None = None,
) -> dict[str, Any]:
    fields = list(ROLLING_ML_FEATURES)
    cached_training = training_cache.get(episode.case_id) if training_cache is not None else None
    if cached_training is None:
        cached_training = _rolling_model_training_rows(episode, fields)
    x_train, y_train, training_periods = cached_training
    classes = np.unique(y_train)
    if (
        training_periods < ROLLING_ML_MIN_TRAINING_PERIODS
        or len(y_train) < ROLLING_ML_MIN_TRAINING_ROWS
        or len(classes) < 2
    ):
        raise RollingMLTrainingError(
            f"rolling {method} baseline cannot fit without fallback: "
            f"history_months={len(episode.history.index)}, horizon_months={episode.horizon_months}, "
            f"training_periods={training_periods}, training_rows={len(y_train)}, "
            f"classes={classes.tolist()}"
        )
    table = _cached_features(episode, feature_tables)
    x_current = np.asarray(
        [_model_feature_row(table, sleeve_id, fields) for sleeve_id in episode.sleeve_ids],
        dtype=float,
    )
    if method == "elastic_net":
        elastic_net_options: dict[str, Any] = {
            "solver": "saga",
            "l1_ratio": 0.2,
            "C": 0.5,
            "max_iter": 500,
            "random_state": 0,
        }
        # scikit-learn 1.8 deprecated the explicit ``penalty`` parameter and
        # infers elastic net from l1_ratio.  Older supported releases still
        # require penalty="elasticnet".  Keep both branches warning-free.
        penalty_default = inspect.signature(LogisticRegression).parameters["penalty"].default
        if penalty_default != "deprecated":
            elastic_net_options["penalty"] = "elasticnet"
        estimator: Any = LogisticRegression(
            **elastic_net_options,
        )
    elif method == "tree":
        estimator = RandomForestClassifier(
            n_estimators=100,
            max_depth=4,
            min_samples_leaf=4,
            random_state=0,
            n_jobs=1,
        )
    elif method in {"mlp", "mlp_deep", "mlp_ensemble"}:
        hidden_layer_sizes = (32, 16) if method == "mlp" else (64, 32, 16)
        estimator = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size="auto",
            learning_rate_init=1e-3,
            max_iter=300,
            tol=1e-4,
            early_stopping=False,
            random_state=0,
        )
    else:
        estimator = LogisticRegression(C=0.5, max_iter=500, random_state=0)
    def make_model(seed: int = 0) -> Any:
        current = estimator
        if method == "mlp_ensemble":
            current = MLPClassifier(
                hidden_layer_sizes=(64, 32, 16),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                batch_size="auto",
                learning_rate_init=1e-3,
                max_iter=300,
                tol=1e-4,
                early_stopping=False,
                random_state=seed,
            )
        return make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            current,
        )
    try:
        models = [make_model(seed) for seed in (0, 1, 2)] if method == "mlp_ensemble" else [make_model()]
        probabilities = []
        scores_list = []
        for model in models:
            model.fit(x_train, y_train)
            probabilities.append(model.predict_proba(x_current)[:, 1])
            if hasattr(model[-1], "decision_function"):
                scores_list.append(model.decision_function(x_current))
            else:
                probability = probabilities[-1]
                scores_list.append(
                    np.log(np.clip(probability, 1e-6, 1 - 1e-6)
                           / np.clip(1 - probability, 1e-6, 1.0))
                )
        probability = np.mean(probabilities, axis=0)
        scores = np.mean(scores_list, axis=0)
        if method == "mlp_ensemble":
            model_label = "mlp_ensemble"
        elif method in {"mlp", "mlp_deep"}:
            model_label = method
        else:
            model_label = method
    except (ValueError, RuntimeError) as exc:
        raise RollingMLTrainingError(f"rolling {method} baseline fit failed: {exc}") from exc
    result = _ranked_predictions(
        episode.sleeve_ids,
        {sleeve_id: float(score) for sleeve_id, score in zip(episode.sleeve_ids, scores)},
        method=model_label,
    )
    for row, probability in zip(result["predictions"], probability):
        row["probability_positive"] = round(float(np.clip(probability, 0.0, 1.0)), 6)
    result["market_view"] = (
        f"Rolling-origin {method} baseline trained on {training_periods} historical cutoffs "
        f"and {len(y_train)} factor-horizon rows from this case only; "
        "no prediction-feedback features."
    )
    return result


def run_baseline(
    episode: RouterEpisode,
    method: str,
    *,
    training_episodes: Sequence[RouterEpisode] | None = None,
    feature_tables: Mapping[str, pd.DataFrame] | None = None,
    training_cache: Mapping[str, tuple[np.ndarray, np.ndarray, int]] | None = None,
) -> dict[str, Any]:
    """Return a schema-shaped prediction for a non-LLM baseline.

    Trainable methods use rolling factor-month rows derived only from the
    current case's history.  ``training_episodes`` remains accepted for API
    compatibility but is deliberately not used: pulling labels from other
    benchmark cases would give the ML baseline a different information set
    from the LLM.  No prediction-error or other feedback feature is added.
    """

    aliases = {"momentum": "momentum_12", "reversal": "reversal_1", "risk_parity": "inverse_vol"}
    method = aliases.get(method, method)
    _ = training_episodes
    if method in ROLLING_ML_METHODS:
        return _model_predictions(episode, method, feature_tables, training_cache)
    table = _cached_features(episode, feature_tables)
    if method == "equal_weight":
        return _ranked_predictions(episode.sleeve_ids, {key: 0.0 for key in episode.sleeve_ids}, method="equal_weight")
    if method == "inverse_vol":
        scores = {
            key: 1.0 / max(_finite_float(table.loc[key].get("volatility_36m")) or 1.0, 1e-8)
            for key in episode.sleeve_ids
        }
        return _ranked_predictions(episode.sleeve_ids, scores, method="inverse_vol")
    if method == "momentum_12":
        return _ranked_predictions(
            episode.sleeve_ids,
            {key: _finite_float(table.loc[key].get("return_12m")) for key in episode.sleeve_ids},
            method="momentum_12",
        )
    if method in {"t_stat_12", "t_stat_36"}:
        # Risk-adjusted (Sharpe-style) momentum: the t-statistic already scales
        # the mean return by its realized dispersion, so it is the signed
        # character that drives the ranking.
        field = "t_stat_12m" if method == "t_stat_12" else "t_stat_36m"
        return _ranked_predictions(
            episode.sleeve_ids,
            {key: (_finite_float(table.loc[key].get(field)) or 0.0) for key in episode.sleeve_ids},
            method=method,
        )
    if method == "downside_vol":
        # Inverse-downside-volatility long bias: prefer sleeves whose lower-tail
        # dispersion is smallest, mirroring the existing inverse_vol rule.
        return _ranked_predictions(
            episode.sleeve_ids,
            {
                key: 1.0 / max(_finite_float(table.loc[key].get("downside_volatility_12m")) or 1.0, 1e-8)
                for key in episode.sleeve_ids
            },
            method="downside_vol",
        )
    if method == "skip_month_momentum":
        # 12-1 momentum: compounded 12m return excluding the most recent month.
        # Pairs with reversal_1 to test whether short reversal and intermediate
        # momentum coexist in the cross-section of factor sleeves.
        return _ranked_predictions(
            episode.sleeve_ids,
            {
                key: (_finite_float(table.loc[key].get("return_12m")) or 0.0)
                - (_finite_float(table.loc[key].get("return_1m")) or 0.0)
                for key in episode.sleeve_ids
            },
            method="skip_month_momentum",
        )
    if method == "reversal_1":
        return _ranked_predictions(
            episode.sleeve_ids,
            {key: -(_finite_float(table.loc[key].get("return_1m")) or 0.0) for key in episode.sleeve_ids},
            method="reversal_1",
        )
    raise ValueError(f"Unknown baseline method: {method}")


def baseline_portfolio_weights(
    episode: RouterEpisode,
    method: str,
    *,
    training_episodes: Sequence[RouterEpisode] | None = None,
    feature_tables: Mapping[str, pd.DataFrame] | None = None,
    response: Mapping[str, Any] | None = None,
) -> pd.Series:
    """Return the portfolio for a named baseline under the frozen rules."""

    aliases = {"momentum": "momentum_12", "reversal": "reversal_1", "risk_parity": "inverse_vol"}
    method = aliases.get(method, method)
    if method == "equal_weight":
        return pd.Series(1.0 / len(episode.sleeve_ids), index=episode.sleeve_ids, dtype=float)
    if method == "inverse_vol":
        return inverse_volatility_weights(episode.history, sleeve_ids=episode.sleeve_ids)
    effective_response = response
    if effective_response is None:
        effective_response = run_baseline(
            episode,
            method,
            training_episodes=training_episodes,
            feature_tables=feature_tables,
        )
    return fixed_portfolio_weights(episode.history, effective_response["predictions"])


def _target_by_prediction_id(
    target: Mapping[str, Any] | pd.Series,
    predictions: Sequence[Mapping[str, Any]],
    masked_to_original: Mapping[str, str] | None = None,
) -> pd.Series:
    target_series = pd.Series(target, dtype=float)
    values: dict[str, float] = {}
    for item in predictions:
        prediction_id = str(item.get("factor_id"))
        original = masked_to_original.get(prediction_id, prediction_id) if masked_to_original else prediction_id
        raw_value = target_series.get(original)
        if raw_value is None and original != prediction_id:
            raw_value = target_series.get(prediction_id)
        value = _finite_float(raw_value)
        values[prediction_id] = value if value is not None else float("nan")
    return pd.Series(values, dtype=float)


def _ndcg(actual: np.ndarray, predicted_order: Sequence[int]) -> float | None:
    if len(actual) == 0:
        return None
    gains = (actual > 0.0).astype(float)
    order = np.asarray(predicted_order, dtype=int)
    dcg = float(sum(gains[index] / math.log2(position + 2.0) for position, index in enumerate(order)))
    ideal_order = np.argsort(-gains)
    ideal = float(sum(gains[index] / math.log2(position + 2.0) for position, index in enumerate(ideal_order)))
    return _finite_float(dcg / ideal) if ideal > 1e-12 else 0.0


def score_predictions(
    response: Mapping[str, Any],
    target: Mapping[str, Any] | pd.Series,
    *,
    masked_to_original: Mapping[str, str] | None = None,
    calibration_bins: int = 10,
) -> dict[str, Any]:
    """Calculate point-in-time direction/ranking/calibration metrics."""

    predictions = [item for item in response.get("predictions", []) if isinstance(item, Mapping)]
    if not predictions:
        return {"n": 0}
    actual_series = _target_by_prediction_id(target, predictions, masked_to_original)
    rows = []
    for item in predictions:
        prediction_id = str(item.get("factor_id"))
        actual = actual_series.get(prediction_id)
        if actual is None or not math.isfinite(float(actual)):
            continue
        probability = float(item.get("probability_positive", 0.5))
        confidence = float(item.get("confidence", 0.0))
        rows.append(
            {
                "factor_id": prediction_id,
                "actual": float(actual),
                "label": int(float(actual) > 0.0),
                "probability": float(np.clip(probability, 0.0, 1.0)),
                "confidence": float(np.clip(confidence, 0.0, 1.0)),
                "rank": int(item.get("rank", len(predictions))),
                "abstain": bool(item.get("abstain", False)),
            }
        )
    if not rows:
        return {"n": 0}
    labels = np.asarray([row["label"] for row in rows], dtype=int)
    probabilities = np.asarray([row["probability"] for row in rows], dtype=float)
    predicted = (probabilities >= 0.5).astype(int)
    accuracy = float(np.mean(predicted == labels))
    classes = np.unique(labels)
    if len(classes) == 2:
        balanced = float(
            0.5
            * (
                np.mean(predicted[labels == 0] == 0)
                + np.mean(predicted[labels == 1] == 1)
            )
        )
    else:
        balanced = accuracy
    clipped = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    brier = float(np.mean((probabilities - labels) ** 2))
    log_loss = float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)))
    ranked_rows = sorted(rows, key=lambda row: (row["rank"], row["factor_id"]))
    order = [rows.index(row) for row in ranked_rows]
    actual = np.asarray([row["actual"] for row in rows], dtype=float)
    rank_ic = None
    if len(rows) >= 2 and np.std(actual) > 1e-12 and np.std(np.asarray([row["rank"] for row in rows])) > 1e-12:
        rank_ic = _finite_float(
            pd.Series(actual).corr(pd.Series([-row["rank"] for row in rows]), method="spearman")
        )
    top = ranked_rows[: min(3, len(ranked_rows))]
    non_abstain = [row for row in rows if not row["abstain"]]
    calibration: list[dict[str, Any]] = []
    ece = 0.0
    for index in range(calibration_bins):
        low = index / calibration_bins
        high = (index + 1) / calibration_bins
        members = [row for row in rows if low <= row["confidence"] and (row["confidence"] < high or index == calibration_bins - 1)]
        if not members:
            continue
        observed = float(np.mean([row["label"] == (row["probability"] >= 0.5) for row in members]))
        mean_conf = float(np.mean([row["confidence"] for row in members]))
        ece += len(members) / len(rows) * abs(observed - mean_conf)
        calibration.append({"lower": low, "upper": high, "n": len(members), "confidence": mean_conf, "accuracy": observed})
    return {
        "n": len(rows),
        "accuracy": _finite_float(accuracy),
        "balanced_accuracy": _finite_float(balanced),
        "brier_score": _finite_float(brier),
        "log_loss": _finite_float(log_loss),
        "rank_ic": rank_ic,
        "top3_hit_rate": _finite_float(np.mean([row["label"] for row in top])) if top else None,
        "ndcg": _ndcg(actual, order),
        "top3_average_realized_return": _finite_float(np.mean([row["actual"] for row in top])) if top else None,
        "expected_calibration_error": _finite_float(ece),
        "calibration": calibration,
        "abstain_coverage": _finite_float(1.0 - len(non_abstain) / len(rows)),
        "accuracy_non_abstain": _finite_float(
            np.mean([row["label"] == (row["probability"] >= 0.5) for row in non_abstain])
        ) if non_abstain else None,
    }


def _annualized_return(values: pd.Series, periods_per_year: float) -> float | None:
    if values.empty:
        return None
    compounded = float(np.prod(1.0 + values.to_numpy(dtype=float)))
    return _finite_float(compounded ** (float(periods_per_year) / len(values)) - 1.0)


def _max_drawdown(values: pd.Series) -> float | None:
    if values.empty:
        return None
    wealth = np.cumprod(1.0 + values.to_numpy(dtype=float))
    peak = np.maximum.accumulate(wealth)
    return _finite_float(np.min(wealth / np.maximum(peak, 1e-12) - 1.0))


def evaluate_portfolio(
    gross_returns: Sequence[float] | pd.Series,
    turnovers: Sequence[float] | pd.Series,
    *,
    costs_bps: Sequence[float] = (0.0, 10.0, 25.0),
    periods_per_year: float = 12.0,
) -> dict[str, Any]:
    """Settle periodic portfolio returns at the requested turnover costs."""

    if not math.isfinite(float(periods_per_year)) or float(periods_per_year) <= 0.0:
        raise ValueError("periods_per_year must be positive and finite")
    gross = pd.to_numeric(pd.Series(gross_returns), errors="coerce").dropna().astype(float)
    turnover_series = pd.to_numeric(pd.Series(turnovers), errors="coerce").reindex(gross.index).fillna(0.0).astype(float)
    result: dict[str, Any] = {
        "periods": int(len(gross)),
        "periods_per_year": float(periods_per_year),
        "months": int(round(len(gross) * 12.0 / float(periods_per_year))),
        "average_turnover": _finite_float(turnover_series.mean()) if not turnover_series.empty else None,
        "cost_frontier": {},
    }
    for bps in costs_bps:
        cost_rate = float(bps) / 10000.0
        net = gross - turnover_series * cost_rate
        annual_return = _annualized_return(net, periods_per_year)
        annual_vol = _finite_float(net.std(ddof=1) * math.sqrt(periods_per_year)) if len(net) >= 2 else None
        sharpe = _finite_float(net.mean() / net.std(ddof=1) * math.sqrt(periods_per_year)) if len(net) >= 2 and net.std(ddof=1) > 1e-12 else None
        key = str(int(float(bps))) if float(bps).is_integer() else str(float(bps))
        result["cost_frontier"][key] = {
            "cost_bps": float(bps),
            "annualized_return": annual_return,
            "annualized_volatility": annual_vol,
            "sharpe": sharpe,
            "max_drawdown": _max_drawdown(net),
            "total_return": _finite_float(np.prod(1.0 + net) - 1.0) if not net.empty else None,
        }
    return result


def episode_to_truth_record(
    episode: RouterEpisode,
    payload: Mapping[str, Any],
    masked_to_original: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "case_id": str(payload.get("case_id", _opaque_case_id(episode))),
        "base_case_id": episode.case_id,
        "region": episode.region,
        "anchor_date": episode.anchor_date.strftime("%Y-%m-%d"),
        "horizon_months": episode.horizon_months,
        "target_definition": episode.target_definition,
        "target": {
            "columns": episode.sleeve_ids,
            "data": _jsonable(episode.target.to_list()),
        },
        "payload_factor_ids": response_factor_ids(payload),
        "masked_to_original": dict(masked_to_original),
        "history": {
            "index": [pd.Timestamp(value).strftime("%Y-%m-%d") for value in episode.history.index],
            "columns": episode.sleeve_ids,
            "data": _jsonable(episode.history.to_numpy().tolist()),
        },
        "future": {
            "index": [pd.Timestamp(value).strftime("%Y-%m-%d") for value in episode.future.index],
            "columns": episode.sleeve_ids,
            "data": _jsonable(episode.future.to_numpy().tolist()),
        },
        "sleeve_cards": _jsonable(episode.sleeve_cards),
    }


def episode_from_truth_record(record: Mapping[str, Any]) -> RouterEpisode:
    history_spec = record["history"]
    future_spec = record["future"]
    history = pd.DataFrame(
        history_spec["data"],
        index=pd.to_datetime(history_spec["index"]),
        columns=history_spec["columns"],
    )
    future = pd.DataFrame(
        future_spec["data"],
        index=pd.to_datetime(future_spec["index"]),
        columns=future_spec["columns"],
    )
    return RouterEpisode(
        case_id=str(record.get("base_case_id", record.get("case_id", "unknown"))),
        region=str(record.get("region", "WORLD")),
        anchor_date=pd.Timestamp(record["anchor_date"]),
        history=history,
        future=future,
        sleeve_cards={str(key): dict(value) for key, value in record.get("sleeve_cards", {}).items()},
    )


def make_smoke_fixture(*, months: int = 192, sleeves: int = 6) -> pd.DataFrame:
    """Create a deterministic monthly sleeve fixture; not a research dataset."""

    if months < 125 or sleeves < 3:
        raise ValueError("smoke fixture needs at least 125 months and 3 sleeves")
    dates = pd.date_range("2000-01-31", periods=months, freq="ME")
    rng = np.random.default_rng(19)
    latent = np.zeros((months, 3), dtype=float)
    shocks = rng.normal(0.0, 0.018, size=(months, 3))
    for index in range(1, months):
        latent[index] = 0.65 * latent[index - 1] + shocks[index]
    values: dict[str, np.ndarray] = {}
    for index in range(sleeves):
        loading = np.zeros(3)
        loading[index % 3] = 1.0
        if index % 2:
            loading[(index + 1) % 3] = 0.3
        values[f"theme_{index + 1:02d}"] = latent @ loading + rng.normal(0.0, 0.004, months)
    wide = pd.DataFrame(values, index=dates).rename_axis("date").reset_index()
    long = wide.melt(id_vars=["date"], var_name="theme_id", value_name="return")
    long["region"] = "WORLD"
    return long.loc[:, ["date", "region", "theme_id", "return"]]

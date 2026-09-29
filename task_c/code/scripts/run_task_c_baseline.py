#!/usr/bin/env python3
"""Run a deterministic, non-LLM Task C stock-factor baseline.

The primary reference is a JKP-direction signed equal-weight composite.  For
each formation month, every available characteristic is converted to its
cross-sectional percentile rank, multiplied by the frozen official JKP
direction, and averaged across characteristics.  It does not read labels and
therefore remains a useful no-learning reference for generated models.

Additional frozen references (selected with ``--method``):

- ``no_prior_signed_equal_weight``: same signed equal-weight composite but with
  every characteristic given sign +1, so no external JKP direction prior is
  used.  This isolates the extra information carried by the frozen prior.
- ``rolling_ridge`` / ``rolling_elastic_net``: a linear model fit on the
  labelled training window of each formation month, with characterisrics
  median-imputed and z-scored.  No hyperparameter search, no cross-validation.
- ``rolling_mlp`` / ``rolling_mlp_deep`` / ``rolling_mlp_ensemble``: standard
  Adam/ReLU multilayer perceptrons fit from scratch at every formation month;
  the ensemble averages three fixed-seed networks.
- ``rolling_ridge_tuned`` / ``rolling_elastic_net_tuned`` /
  ``rolling_mlp_tuned`` / ``rolling_mlp_deep_tuned`` /
  ``rolling_mlp_ensemble_tuned``: the same five models with their
  hyperparameters selected *inside every formation month*.  Selection scores a
  candidate grid on the labelled validation block (months ``t-12..t-1``) the
  contract supplies, then refits the winner on train + validation, which is the
  executor's declared ``select_on_valid_then_refit_train_plus_valid`` policy.
  Preprocessing is identical to the corresponding frozen baseline, so a frozen
  versus tuned pair isolates the hyperparameter choice.  These exist because a
  frozen baseline compared against a tuned candidate confounds "the model" with
  "the model was never tuned".

The baseline is deliberately expressed as a normal Task C ``fit_and_predict``
response and sent through the same rolling executor and JKP-compatible
backtester as an LLM response.  This keeps I/O, masking, point-in-time checks,
timeouts, portfolio formation, daily settlement, and costs identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from factoropt.task_c import (  # noqa: E402
    code_response_hash,
    materialize_split,
    validate_code_response,
    validate_public_payload,
)
from factoropt.semantic import normalize_track  # noqa: E402
from factoropt.task_c_backtest import (  # noqa: E402
    backtest_predictions,
    load_daily_returns,
    load_nyse_cutoffs,
    load_stock_panel,
)
from factoropt.task_c_iterative import (  # noqa: E402
    evaluate_candidate_on_validation,
    rolling_predict,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _case_metadata(case_dir: Path) -> dict[str, Any]:
    payload = _read_json(case_dir / "payload.json")
    payload_errors = validate_public_payload(payload)
    if payload_errors:
        raise ValueError("invalid public payload: " + "; ".join(payload_errors[:10]))
    feature_value = _read_json(case_dir / "evaluator_feature_map.json")
    id_value = _read_json(case_dir / "evaluator_id_map.json")
    raw_to_public = feature_value.get("original_to_masked")
    original_to_masked = id_value.get("original_to_masked")
    if not isinstance(raw_to_public, dict) or not isinstance(original_to_masked, dict):
        raise ValueError("evaluator maps must contain original_to_masked objects")
    raw_features = [str(key) for key in raw_to_public]
    public_features = [str(raw_to_public[key]) for key in raw_features]
    payload_features = [str(value) for value in payload.get("feature_columns", [])]
    if payload_features != public_features:
        raise ValueError("payload feature order does not match evaluator feature map")
    truth_path = case_dir / "rolling_truth.jsonl"
    formations: list[str] = []
    for line in truth_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            formations.append(str(row["formation_eom"])[:7])
    if not formations:
        raise ValueError(f"no rolling formation months in {truth_path}")
    manifest = _read_json(case_dir / "manifest.json")
    window = manifest.get("window") or {}
    train_months = int(window["train_months"])
    validation_months = int(window["validation_months"])
    return {
        "payload": payload,
        "track": normalize_track(payload.get("track", "numeric_only")),
        "factor_cards": [
            dict(card)
            for card in payload.get("factor_cards", [])
            if isinstance(card, dict)
        ],
        "raw_features": raw_features,
        "public_features": public_features,
        "feature_map": {str(k): str(v) for k, v in raw_to_public.items()},
        "id_map": {str(k): str(v) for k, v in original_to_masked.items()},
        "formations": formations,
        "train_months": train_months,
        "validation_months": validation_months,
    }


def _load_directions(path: Path, factor_ids: list[str]) -> dict[str, int]:
    """Return official JKP signs aligned to raw factor IDs."""

    frame = pd.read_csv(path)
    normalized = {str(column).strip().lower(): column for column in frame.columns}
    id_column = next(
        (normalized[name] for name in ("factor_id", "abr_jkp", "abbreviation", "mnemonic") if name in normalized),
        None,
    )
    direction_column = next(
        (normalized[name] for name in ("official_direction", "direction", "sign") if name in normalized),
        None,
    )
    if id_column is None or direction_column is None:
        raise ValueError(f"{path} lacks factor ID/direction columns")
    frame = frame[[id_column, direction_column]].copy()
    frame[id_column] = frame[id_column].astype("string").str.strip()
    frame[direction_column] = pd.to_numeric(frame[direction_column], errors="coerce")
    signs: dict[str, int] = {}
    for factor_id in factor_ids:
        values = frame.loc[frame[id_column].eq(factor_id), direction_column].dropna()
        if values.empty:
            raise ValueError(f"no official JKP direction for factor {factor_id!r}")
        value = float(values.iloc[0])
        if not np.isfinite(value) or value == 0:
            raise ValueError(f"invalid official JKP direction for factor {factor_id!r}: {value}")
        signs[factor_id] = 1 if value > 0 else -1
    return signs


def _ml_response_code(
    feature_columns: list[str], *, method: str, refit_on_validation: bool = False
) -> str:
    """Generate a frozen ``fit_and_predict`` source for a rolling ML baseline.

    The model is fit from scratch inside each per-month ``fit_and_predict``
    call, exactly as the Task C contract requires, and uses no labels outside
    the training window.  The generated code is static policy-checked and run
    in the same sandbox as an LLM response.

    ``refit_on_validation`` is the attribution control.  It keeps the frozen
    hyperparameters but raises the labelled window from train alone to
    train + validation, which is the window every tuned variant refits on.
    ``rolling_ridge`` vs ``rolling_ridge_refit`` therefore changes only how
    many months the model sees, and ``rolling_ridge_tuned`` vs
    ``rolling_ridge_refit`` isolates what the per-origin search itself is worth.
    """

    model_expr = {
        "rolling_ridge": 'Ridge(alpha=10.0)',
        "rolling_elastic_net": 'ElasticNet(alpha=0.005, l1_ratio=0.5, max_iter=2000)',
    }[method]
    features_repr = repr(feature_columns)
    if refit_on_validation:
        # Same model and hyperparameters as the frozen baseline; the only
        # difference is that validation months join the labelled window.
        return f'''import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler

_FEATURES = {features_repr}


def fit_and_predict(train_df, valid_df, predict_df, metadata):
    # Attribution control for the tuned baselines: identical model and
    # hyperparameters, refit on train + validation so the labelled window
    # matches the window the tuned variants refit on.
    columns = list(_FEATURES)
    x_train = train_df[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    y_train = train_df["label"].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    x_valid = valid_df[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    y_valid = valid_df["label"].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    x_predict = predict_df[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    x_train = np.vstack([x_train, x_valid])
    y_train = np.concatenate([y_train, y_valid])

    med = np.nanmedian(x_train, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    x_train = np.where(np.isfinite(x_train), x_train, med)
    x_predict = np.where(np.isfinite(x_predict), x_predict, med)
    x_train = np.nan_to_num(x_train, nan=0.0, posinf=0.0, neginf=0.0)
    x_predict = np.nan_to_num(x_predict, nan=0.0, posinf=0.0, neginf=0.0)
    y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_predict = scaler.transform(x_predict)

    model = {model_expr}
    model.fit(x_train, y_train)
    score = model.predict(x_predict)
    np.nan_to_num(score, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return pd.DataFrame({{"id": predict_df["id"].astype(str), "score": score.astype(float)}})
'''

    return f'''import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler

_FEATURES = {features_repr}


def fit_and_predict(train_df, valid_df, predict_df, metadata):
    # Rolling ridge / elastic-net reference: fit on the labelled training
    # window only, no cross-validation, no hyperparameter search, no label
    # data from the prediction month.  Anonymous characteristic columns are
    # imputed by training median, centred and scaled by training std.
    columns = list(_FEATURES)
    x_train = train_df[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    y_train = train_df["label"].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    x_predict = predict_df[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    # Impute missing features with the training medians (per column), falling
    # back to 0 for columns whose training median is undefined, then standardize.
    med = np.nanmedian(x_train, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    x_train = np.where(np.isfinite(x_train), x_train, med)
    x_predict = np.where(np.isfinite(x_predict), x_predict, med)
    x_train = np.nan_to_num(x_train, nan=0.0, posinf=0.0, neginf=0.0)
    x_predict = np.nan_to_num(x_predict, nan=0.0, posinf=0.0, neginf=0.0)
    y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_predict = scaler.transform(x_predict)

    model = {model_expr}
    model.fit(x_train, y_train)
    score = model.predict(x_predict)
    np.nan_to_num(score, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return pd.DataFrame({{"id": predict_df["id"].astype(str), "score": score.astype(float)}})
'''


def _mlp_response_code(
    feature_columns: list[str], *, method: str, refit_on_validation: bool = False
) -> str:
    """Generate deterministic rolling MLP source for the frozen runner.

    ``refit_on_validation`` is the MLP attribution control, matching the linear
    one: same architecture, same seed(s), same L2, but refit on train +
    validation so only the labelled window width differs from the frozen MLP.
    """

    hidden = "(64, 32)" if method == "rolling_mlp" else "(128, 64, 32)"
    seeds = "(0, 1, 2)" if method == "rolling_mlp_ensemble" else "(1,)"
    features_repr = repr(feature_columns)
    if refit_on_validation:
        return f'''import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

_FEATURES = {features_repr}


def fit_and_predict(train_df, valid_df, predict_df, metadata):
    # Attribution control for the tuned MLP baselines: identical architecture,
    # seeds, and L2, but refit on train + validation.  shuffle=False preserves
    # the chronological row order, as in the frozen baseline.
    columns = list(_FEATURES)
    x_train = train_df[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    y_train = train_df["label"].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    x_valid = valid_df[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    y_valid = valid_df["label"].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    x_predict = predict_df[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    x_train = np.vstack([x_train, x_valid])
    y_train = np.concatenate([y_train, y_valid])

    med = np.nanmedian(x_train, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    x_train = np.where(np.isfinite(x_train), x_train, med)
    x_predict = np.where(np.isfinite(x_predict), x_predict, med)
    x_train = np.nan_to_num(x_train, nan=0.0, posinf=0.0, neginf=0.0)
    x_predict = np.nan_to_num(x_predict, nan=0.0, posinf=0.0, neginf=0.0)
    y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
    x_train = (x_train - mean) / scale
    x_predict = (x_predict - mean) / scale
    base_seed = int(metadata.get("seed", 1))
    scores = []
    for offset in {seeds}:
        model = MLPRegressor(
            hidden_layer_sizes={hidden}, activation="relu", solver="adam",
            alpha=1e-4, batch_size=256, learning_rate_init=1e-3,
            max_iter=300, tol=1e-4, shuffle=False,
            early_stopping=False, random_state=base_seed + int(offset),
        )
        model.fit(x_train, y_train)
        scores.append(model.predict(x_predict))
    score = np.mean(np.asarray(scores, dtype=float), axis=0)
    np.nan_to_num(score, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return pd.DataFrame({{"id": predict_df["id"].astype(str), "score": score.astype(float)}})
'''

    return f'''import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

_FEATURES = {features_repr}


def fit_and_predict(train_df, valid_df, predict_df, metadata):
    # Fit from scratch using labels available before this formation month.
    # shuffle=False preserves the chronological row order in each origin.
    columns = list(_FEATURES)
    x_train = train_df[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    y_train = train_df["label"].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    x_predict = predict_df[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    med = np.nanmedian(x_train, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    x_train = np.where(np.isfinite(x_train), x_train, med)
    x_predict = np.where(np.isfinite(x_predict), x_predict, med)
    x_train = np.nan_to_num(x_train, nan=0.0, posinf=0.0, neginf=0.0)
    x_predict = np.nan_to_num(x_predict, nan=0.0, posinf=0.0, neginf=0.0)
    y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
    x_train = (x_train - mean) / scale
    x_predict = (x_predict - mean) / scale
    base_seed = int(metadata.get("seed", 1))
    scores = []
    for offset in {seeds}:
        model = MLPRegressor(
            hidden_layer_sizes={hidden}, activation="relu", solver="adam",
            alpha=1e-4, batch_size=256, learning_rate_init=1e-3,
            max_iter=300, tol=1e-4, shuffle=False,
            early_stopping=False, random_state=base_seed + int(offset),
        )
        model.fit(x_train, y_train)
        scores.append(model.predict(x_predict))
    score = np.mean(np.asarray(scores, dtype=float), axis=0)
    np.nan_to_num(score, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return pd.DataFrame({{"id": predict_df["id"].astype(str), "score": score.astype(float)}})
'''


def _tuned_response_code(feature_columns: list[str], *, method: str) -> str:
    """Generate a rolling baseline that tunes its hyperparameters per origin.

    The frozen ``rolling_ridge`` / ``rolling_elastic_net`` / ``rolling_mlp*``
    baselines hard-code their hyperparameters, so any comparison against a
    tuned LLM candidate confounds "the model" with "the model was never tuned".
    These variants implement the executor's declared
    ``select_on_valid_then_refit_train_plus_valid`` policy explicitly:

    1. score every candidate configuration on the labelled *validation* block
       that ``fit_and_predict`` receives (formation months ``t-12..t-1``),
       which is the last information set before the prediction month;
    2. refit the winning configuration on train + validation;
    3. predict formation month ``t``.

    Preprocessing is byte-for-byte the frozen baseline's (training median
    imputation, training mean/std standardisation) so the only difference
    between a frozen and a tuned run is the hyperparameter choice.

    Selection is deterministic and its grading is load-bearing: ties break
    toward the earlier entry in the grid, and each grid is ordered from the
    most to the least regularised configuration.  A validation block carrying
    no usable signal therefore lands on the most conservative configuration
    rather than the most flexible one, which is the failure mode that makes an
    unguarded in-loop search look better than it is.
    """

    if method in {"rolling_ridge_tuned", "rolling_elastic_net_tuned"}:
        return _tuned_linear_response_code(feature_columns, method=method)
    return _tuned_mlp_response_code(feature_columns, method=method)


# Shared helper emitted into every tuned variant.  Kept as one string so the
# linear and MLP generators cannot drift apart in how they score candidates.
_TUNED_HELPERS = '''

def _monthly_rank_ic(y_true, score, month):
    """Mean monthly Spearman IC between score and realized label.

    Monthly cross-sections are the unit of evaluation everywhere else in the
    benchmark, so a pooling-level correlation would be the wrong objective.
    Months with a degenerate (constant) score or label are skipped rather than
    scored as zero, and ``-inf`` signals "no usable validation month" so the
    caller can fall back to the grid's first entry.
    """
    frame = pd.DataFrame({"month": np.asarray(month), "y": y_true, "s": score})
    values = []
    for _, group in frame.groupby("month", sort=True):
        if len(group) < 30:
            continue
        ic = group["s"].corr(group["y"], method="spearman")
        if np.isfinite(ic):
            values.append(float(ic))
    if not values:
        return -np.inf
    return float(np.mean(values))


def _select(variants, x_fit, y_fit, month_fit, deadline):
    """Pick the configuration with the best validation IC, deterministically.

    ``variants`` is a sequence of ``(label, builder)`` pairs ordered from most
    to least conservative.  ``deadline`` is an absolute ``time.time()`` value:
    if fitting the remaining configurations would overrun it, selection stops
    and returns the best configuration scored so far, which guarantees the
    origin produces a prediction instead of being killed by the executor.
    The winning *builder* is returned so the caller refits a fresh estimator
    rather than reusing one already fitted on the validation block.
    """
    best_score, best_builder = -np.inf, None
    started = time.time()
    for position, (label, builder) in enumerate(variants):
        model = builder()
        model.fit(x_fit, y_fit)
        score = _monthly_rank_ic(y_fit, model.predict(x_fit), month_fit)
        if best_builder is None or score > best_score:
            best_score, best_builder = score, builder
        elapsed = time.time() - started
        remaining = len(variants) - position - 1
        if remaining and time.time() + elapsed * remaining > deadline:
            break
    return best_builder


def _preprocess(x_train, x_predict):
    """Frozen baseline preprocessing: training median, then train mean/std."""
    med = np.nanmedian(x_train, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    x_train = np.where(np.isfinite(x_train), x_train, med)
    x_predict = np.where(np.isfinite(x_predict), x_predict, med)
    x_train = np.nan_to_num(x_train, nan=0.0, posinf=0.0, neginf=0.0)
    x_predict = np.nan_to_num(x_predict, nan=0.0, posinf=0.0, neginf=0.0)
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
    return (x_train - mean) / scale, (x_predict - mean) / scale
'''


def _tuned_linear_response_code(feature_columns: list[str], *, method: str) -> str:
    """Tuned ridge / elastic-net source with a per-origin grid over alpha."""

    if method == "rolling_ridge_tuned":
        grid = "[(1e2, lambda: Ridge(alpha=1e2)), (1e1, lambda: Ridge(alpha=1e1)), " \
               "(1.0, lambda: Ridge(alpha=1.0)), (1e-1, lambda: Ridge(alpha=1e-1)), " \
               "(1e-2, lambda: Ridge(alpha=1e-2)), (1e-3, lambda: Ridge(alpha=1e-3))]"
        import_line = "from sklearn.linear_model import Ridge"
    else:
        grid = "[(1e-1, lambda: ElasticNet(alpha=1e-1, l1_ratio=0.5, max_iter=2000)), " \
               "(5e-2, lambda: ElasticNet(alpha=5e-2, l1_ratio=0.5, max_iter=2000)), " \
               "(1e-2, lambda: ElasticNet(alpha=1e-2, l1_ratio=0.5, max_iter=2000)), " \
               "(5e-3, lambda: ElasticNet(alpha=5e-3, l1_ratio=0.5, max_iter=2000)), " \
               "(1e-3, lambda: ElasticNet(alpha=1e-3, l1_ratio=0.5, max_iter=2000))]"
        import_line = "from sklearn.linear_model import ElasticNet"
    features_repr = repr(feature_columns)
    return f'''import time

import numpy as np
import pandas as pd
{import_line}

_FEATURES = {features_repr}

# Fraction of the executor's per-origin timeout this origin may spend on
# selection.  Leaves headroom for the final refit and prediction so an
# expensive grid degrades to fewer candidates instead of timing out.
_TUNE_BUDGET_FRACTION = 0.5
_DEFAULT_TIMEOUT_SECONDS = 1800.0
{_TUNED_HELPERS}

def fit_and_predict(train_df, valid_df, predict_df, metadata):
    columns = list(_FEATURES)
    n = len(columns)

    def matrix(frame):
        return frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    x_train_raw = matrix(train_df)
    x_valid_raw = matrix(valid_df)
    x_predict_raw = matrix(predict_df)
    y_train = np.nan_to_num(
        train_df["label"].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float),
        nan=0.0, posinf=0.0, neginf=0.0,
    )
    y_valid = np.nan_to_num(
        valid_df["label"].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float),
        nan=0.0, posinf=0.0, neginf=0.0,
    )
    # Impute and scale with training statistics only; the validation block is
    # transformed with the training fit so selection cannot see its own scale.
    x_train, x_valid = _preprocess(x_train_raw, x_valid_raw)
    _, x_predict = _preprocess(x_train_raw, x_predict_raw)

    timeout = float(metadata.get("timeout_seconds", _DEFAULT_TIMEOUT_SECONDS))
    deadline = time.time() + _TUNE_BUDGET_FRACTION * timeout

    variants = {grid}
    if len(x_valid) == 0 or not np.isfinite(y_valid).any():
        builder = variants[0][1]
        model = builder()
        model.fit(x_train, y_train)
    else:
        month_valid = valid_df["time_key"].to_numpy() if "time_key" in valid_df.columns else np.zeros(len(x_valid))
        # Selection scores each candidate on the validation block using the
        # *training* fit's imputation and scaling, so a candidate cannot tune
        # itself to statistics computed from the block it is scored on.
        _, x_valid_scored = _preprocess(x_train_raw, x_valid_raw)
        builder = _select(variants, x_valid_scored, y_valid, month_valid, deadline)
        # Refit the winner on train + validation, the executor's declared
        # policy, recomputing the preprocessing statistics over the union.
        x_refit, _ = _preprocess(np.vstack([x_train_raw, x_valid_raw]), x_train_raw)
        y_refit = np.concatenate([y_train, y_valid])
        model = builder()
        model.fit(x_refit, y_refit)

    score = model.predict(x_predict)
    score = np.nan_to_num(np.asarray(score, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    return pd.DataFrame({{"id": predict_df["id"].astype(str), "score": score}})
'''


def _tuned_mlp_response_code(feature_columns: list[str], *, method: str) -> str:
    """Tuned MLP source with a per-origin grid over width, lr, and L2."""

    if method == "rolling_mlp_ensemble_tuned":
        seeds = "(0, 1, 2)"
        grid = \
            "[(0, lambda: MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', " \
            "solver='adam', alpha=1e-4, batch_size=256, learning_rate_init=1e-3, max_iter=300, " \
            "tol=1e-4, shuffle=False, early_stopping=False, random_state=base_seed + 0)), " \
            "(1, lambda: MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', " \
            "solver='adam', alpha=1e-3, batch_size=256, learning_rate_init=1e-3, max_iter=300, " \
            "tol=1e-4, shuffle=False, early_stopping=False, random_state=base_seed + 1)), " \
            "(2, lambda: MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation='relu', " \
            "solver='adam', alpha=1e-4, batch_size=256, learning_rate_init=1e-3, max_iter=300, " \
            "tol=1e-4, shuffle=False, early_stopping=False, random_state=base_seed + 2))]"
    elif method == "rolling_mlp_deep_tuned":
        seeds = "(1,)"
        grid = \
            "[(0, lambda: MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation='relu', " \
            "solver='adam', alpha=1e-4, batch_size=256, learning_rate_init=1e-3, max_iter=300, " \
            "tol=1e-4, shuffle=False, early_stopping=False, random_state=base_seed + 0)), " \
            "(1, lambda: MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation='relu', " \
            "solver='adam', alpha=1e-3, batch_size=256, learning_rate_init=1e-3, max_iter=300, " \
            "tol=1e-4, shuffle=False, early_stopping=False, random_state=base_seed + 1)), " \
            "(2, lambda: MLPRegressor(hidden_layer_sizes=(256, 128, 64), activation='relu', " \
            "solver='adam', alpha=1e-4, batch_size=256, learning_rate_init=5e-4, max_iter=300, " \
            "tol=1e-4, shuffle=False, early_stopping=False, random_state=base_seed + 2))]"
    else:
        seeds = "(1,)"
        grid = \
            "[(0, lambda: MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', " \
            "solver='adam', alpha=1e-4, batch_size=256, learning_rate_init=1e-3, max_iter=300, " \
            "tol=1e-4, shuffle=False, early_stopping=False, random_state=base_seed + 0)), " \
            "(1, lambda: MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', " \
            "solver='adam', alpha=1e-3, batch_size=256, learning_rate_init=1e-3, max_iter=300, " \
            "tol=1e-4, shuffle=False, early_stopping=False, random_state=base_seed + 1)), " \
            "(2, lambda: MLPRegressor(hidden_layer_sizes=(32, 16), activation='relu', " \
            "solver='adam', alpha=1e-4, batch_size=256, learning_rate_init=1e-3, max_iter=300, " \
            "tol=1e-4, shuffle=False, early_stopping=False, random_state=base_seed + 2))]"
    features_repr = repr(feature_columns)
    return f'''import time

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

_FEATURES = {features_repr}

_TUNE_BUDGET_FRACTION = 0.5
_DEFAULT_TIMEOUT_SECONDS = 1800.0
{_TUNED_HELPERS}

def fit_and_predict(train_df, valid_df, predict_df, metadata):
    columns = list(_FEATURES)

    def matrix(frame):
        return frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    x_train_raw = matrix(train_df)
    x_valid_raw = matrix(valid_df)
    x_predict_raw = matrix(predict_df)
    y_train = np.nan_to_num(
        train_df["label"].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float),
        nan=0.0, posinf=0.0, neginf=0.0,
    )
    y_valid = np.nan_to_num(
        valid_df["label"].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float),
        nan=0.0, posinf=0.0, neginf=0.0,
    )
    x_train, x_valid = _preprocess(x_train_raw, x_valid_raw)
    _, x_predict = _preprocess(x_train_raw, x_predict_raw)

    base_seed = int(metadata.get("seed", 1))
    timeout = float(metadata.get("timeout_seconds", _DEFAULT_TIMEOUT_SECONDS))
    deadline = time.time() + _TUNE_BUDGET_FRACTION * timeout
    month_valid = valid_df["time_key"].to_numpy() if "time_key" in valid_df.columns else np.zeros(len(x_valid))

    _, x_valid_scored = _preprocess(x_train_raw, x_valid_raw)
    variants = {grid}
    best_score, best_builder = -np.inf, None
    started = time.time()
    for position, (label, builder) in enumerate(variants):
        # Score the configuration the way the frozen ensemble predicts: by
        # averaging the same fixed seeds, so selection ranks what will run.
        scores = []
        for offset in {seeds}:
            model = builder()
            model.set_params(random_state=base_seed + int(offset))
            model.fit(x_train, y_train)
            scores.append(model.predict(x_valid_scored))
        predictions = np.mean(np.asarray(scores, dtype=float), axis=0)
        score = _monthly_rank_ic(y_valid, predictions, month_valid)
        if best_builder is None or score > best_score:
            best_score, best_builder = score, builder
        elapsed = time.time() - started
        remaining = len(variants) - position - 1
        if remaining and time.time() + elapsed * remaining > deadline:
            break

    # Refit the winner on train + validation, the executor's declared policy.
    x_refit, _ = _preprocess(np.vstack([x_train_raw, x_valid_raw]), x_train_raw)
    y_refit = np.concatenate([y_train, y_valid])
    scores = []
    for offset in {seeds}:
        model = best_builder()
        model.set_params(random_state=base_seed + int(offset))
        model.fit(x_refit, y_refit)
        scores.append(model.predict(x_predict))
    score = np.mean(np.asarray(scores, dtype=float), axis=0)
    score = np.nan_to_num(np.asarray(score, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    return pd.DataFrame({{"id": predict_df["id"].astype(str), "score": score}})
'''


def _baseline_response(
    public_features: list[str],
    signs: dict[str, int] | None,
    *,
    method: str,
) -> dict[str, Any]:
    """Build a schema-valid frozen response with no label-dependent logic."""

    if method in {"rolling_ridge", "rolling_elastic_net"}:
        code = _ml_response_code(public_features, method=method)
        requirements = ["numpy", "pandas", "sklearn"]
        explanation = (
            "Rolling linear model reference: "
            + ("ridge (alpha=10)" if method == "rolling_ridge" else "elastic-net (alpha=0.005, l1_ratio=0.5)")
            + " fit on the labelled training window only, anonymous "
            "characteristic columns median-imputed and z-scored. No labels from the prediction month are used."
        )
    elif method in {"rolling_ridge_refit", "rolling_elastic_net_refit"}:
        base = "rolling_ridge" if method == "rolling_ridge_refit" else "rolling_elastic_net"
        code = _ml_response_code(public_features, method=base, refit_on_validation=True)
        requirements = ["numpy", "pandas", "sklearn"]
        explanation = (
            "Attribution control for the tuned linear baselines: the frozen "
            + ("ridge (alpha=10)" if base == "rolling_ridge" else "elastic-net (alpha=0.005, l1_ratio=0.5)")
            + " with unchanged hyperparameters, refit on train + validation so its "
            "labelled window matches the window the tuned variants refit on. "
            "Comparing this against the tuned variant isolates the value of the "
            "per-origin hyperparameter search; comparing it against the frozen "
            "baseline isolates the value of the extra twelve months. No label from "
            "the prediction month is used."
        )
    elif method in {"rolling_ridge_tuned", "rolling_elastic_net_tuned"}:
        code = _tuned_response_code(public_features, method=method)
        requirements = ["numpy", "pandas", "sklearn"]
        explanation = (
            "Tuned rolling linear model reference: "
            + ("ridge" if method == "rolling_ridge_tuned" else "elastic net")
            + " whose regularisation strength is selected inside every formation "
            "month on the labelled validation block the contract supplies, then "
            "refit on train + validation. Preprocessing matches the frozen linear "
            "baseline exactly, so only the hyperparameter choice differs. No label "
            "from the prediction month is used."
        )
    elif method in {"rolling_mlp", "rolling_mlp_deep", "rolling_mlp_ensemble"}:
        code = _mlp_response_code(public_features, method=method)
        requirements = ["numpy", "pandas", "sklearn"]
        descriptions = {
            "rolling_mlp": "two-hidden-layer MLP (64,32)",
            "rolling_mlp_deep": "three-hidden-layer MLP (128,64,32)",
            "rolling_mlp_ensemble": "three-seed ensemble of MLPs (128,64,32)",
        }
        explanation = (
            "Rolling neural-network reference: " + descriptions[method]
            + ", ReLU/Adam, fixed seed(s), no row shuffling, fit from scratch "
            "on the labelled training window after median imputation and z-scoring."
        )
    elif method in {"rolling_mlp_refit", "rolling_mlp_deep_refit", "rolling_mlp_ensemble_refit"}:
        base = method[: -len("_refit")]
        code = _mlp_response_code(public_features, method=base, refit_on_validation=True)
        requirements = ["numpy", "pandas", "sklearn"]
        descriptions = {
            "rolling_mlp_refit": "two-hidden-layer MLP (64,32)",
            "rolling_mlp_deep_refit": "three-hidden-layer MLP (128,64,32)",
            "rolling_mlp_ensemble_refit": "three-seed ensemble of MLPs (128,64,32)",
        }
        explanation = (
            "Attribution control for the tuned neural-network baselines: "
            + descriptions[method]
            + " with the frozen architecture, seeds, and L2 unchanged, refit on "
            "train + validation so its labelled window matches the window the tuned "
            "variants refit on. Comparing this against the tuned variant isolates the "
            "value of the per-origin search; comparing it against the frozen baseline "
            "isolates the value of the extra twelve months. No label from the "
            "prediction month is used."
        )
    elif method in {"rolling_mlp_tuned", "rolling_mlp_deep_tuned", "rolling_mlp_ensemble_tuned"}:
        code = _tuned_response_code(public_features, method=method)
        requirements = ["numpy", "pandas", "sklearn"]
        descriptions = {
            "rolling_mlp_tuned": "two-hidden-layer MLP (64,32) grid",
            "rolling_mlp_deep_tuned": "three-hidden-layer MLP (128,64,32) grid",
            "rolling_mlp_ensemble_tuned": "three-seed ensemble, architecture/L2 grid",
        }
        explanation = (
            "Tuned rolling neural-network reference: " + descriptions[method]
            + ", whose architecture width and L2 penalty are selected inside every "
            "formation month on the labelled validation block the contract supplies, "
            "then refit on train + validation. ReLU/Adam, fixed seed(s), no row "
            "shuffling, preprocessing identical to the frozen MLP baseline. No label "
            "from the prediction month is used."
        )
    else:
        sign_values = signs if signs is not None else {public: 1 for public in public_features}
        public_signs = {public: int(sign_values[raw]) for raw, public in zip(sign_values, public_features)}
        directions_literal = repr(public_signs)
        code = f'''import numpy as np
import pandas as pd

_DIRECTIONS = {directions_literal}


def fit_and_predict(train_df, valid_df, predict_df, metadata):
    # This reference intentionally ignores labels and fit history.  It is a
    # static signed equal-weight characteristic composite, evaluated monthly
    # through the same frozen-source interface as generated models.
    columns = list(_DIRECTIONS)
    values = predict_df[columns].apply(pd.to_numeric, errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan)
    ranks = values.rank(axis=0, method="average", pct=True)
    signed = ranks.mul(pd.Series(_DIRECTIONS), axis="columns")
    score = signed.mean(axis=1, skipna=True).fillna(0.0)
    return pd.DataFrame({{"id": predict_df["id"].astype(str), "score": score.astype(float)}})
'''
        requirements = ["numpy", "pandas"]
        if method == "jkp_signed_equal_weight":
            explanation = (
                "Static JKP-direction signed equal-weight composite: each available "
                "feature is percentile-ranked cross-sectionally and averaged after "
                "the frozen official JKP direction is applied. No labels are used."
            )
        else:
            explanation = (
                "Stock-level zero-prior signed equal-weight composite: every available "
                "anonymous characteristic enters with sign +1 for the long/short ranking. "
                "No labels, no external factor-prior direction, and no fit history are used."
            )

    features_with_labels = ["id", "time_key", *public_features, "label"]
    predict_features = ["id", "time_key", *public_features]
    return {
        "revision": 0,
        "status": "final",
        "language": "python",
        "code": code,
        "entrypoint": "fit_and_predict",
        "requirements": requirements,
        "input_contract": {
            "train_columns": features_with_labels,
            "validation_columns": features_with_labels,
            "predict_columns": predict_features,
        },
        "output_contract": {"columns": ["id", "score"], "one_row_per_id": True},
        "explanation": explanation,
        "revision_summary": f"Deterministic non-LLM reference baseline ({method}).",
    }


def _write_table(frame: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_parquet(path, index=False)
        return str(path)
    except (ImportError, ModuleNotFoundError):
        csv_path = path.with_suffix(".csv")
        frame.to_csv(csv_path, index=False)
        return str(csv_path)


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def run_case(
    *,
    case_dir: Path,
    panel_path: Path,
    daily_path: Path,
    cutoffs_path: Path,
    factor_details_path: Path,
    output_dir: Path,
    cost_bps: float,
    coverage_threshold: float,
    timeout_seconds: float,
    strict_cutoffs: bool,
    method: str,
) -> dict[str, Any]:
    info = _case_metadata(case_dir)
    if method == "no_prior_signed_equal_weight":
        # Zero-prior variant: every anonymous characteristic is signed +1, so
        # no external JKP direction file is consulted.
        signs = None
    else:
        signs = _load_directions(factor_details_path, info["raw_features"])
    metadata = {
        "seed": 1,
        "source": "wrds",
        "baseline_method": method,
        "track": info["track"],
        # The tuned variants size their per-origin selection budget from this,
        # so it must match the timeout actually enforced by the executor.
        "timeout_seconds": float(timeout_seconds),
    }
    if info["track"] == "semantic":
        metadata["factor_cards"] = info["factor_cards"]
    response = _baseline_response(info["public_features"], signs, method=method)
    response_errors = validate_code_response(response, payload=info["payload"])
    if response_errors:
        raise ValueError("generated baseline response is invalid: " + "; ".join(response_errors))

    output_dir.mkdir(parents=True, exist_ok=True)
    response_path = output_dir / "baseline_response.json"
    response_path.write_text(json.dumps(response, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Code evaluation uses the public train/validation artifacts and keeps
    # labels outside the candidate process exactly as the normal evaluator.
    payload = info["payload"]
    train = materialize_split(payload, "train", include_label=True, artifact_root=case_dir)
    valid = materialize_split(payload, "validation", include_label=True, artifact_root=case_dir)
    validation_metrics, validation_predictions, validation_exec = evaluate_candidate_on_validation(
        response,
        train_df=train,
        validation_df=valid,
        # Keep the same semantic sidecar available during local validation as
        # during the rolling evaluator. ``metadata`` already contains the
        # canonical track and cards for semantic cases; passing only the
        # payload's inner metadata would silently drop that context.
        metadata=metadata,
        timeout_seconds=timeout_seconds,
        scratch_root=output_dir / "validation_scratch",
    )
    validation_result = {
        "metrics": validation_metrics,
        "prediction_rows": int(len(validation_predictions)),
        "executions": [item.audit_dict() for item in validation_exec],
        "schema_valid": True,
        "code_sha256": code_response_hash(response),
    }
    (output_dir / "validation.json").write_text(
        json.dumps(validation_result, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    first = pd.Period(info["formations"][0], freq="M")
    last = pd.Period(info["formations"][-1], freq="M")
    data_start = (first - info["train_months"] - info["validation_months"]).to_timestamp("M")
    data_end = last.to_timestamp("M")
    panel = load_stock_panel(
        panel_path,
        feature_columns=info["raw_features"],
        start=data_start,
        end=data_end,
    )
    if "label" not in panel.columns:
        panel = panel.rename(columns={"ret_exc_lead1m": "label"})
    panel = panel.rename(columns=info["feature_map"])
    formation_months = [period.to_timestamp("M") for period in pd.period_range(first, last, freq="M")]
    predictions, rolling_audit = rolling_predict(
        response,
        panel,
        formation_months,
        feature_columns=info["public_features"],
        train_months=info["train_months"],
        valid_months=info["validation_months"],
        timeout_seconds=timeout_seconds,
        metadata=dict(metadata),
        id_mapping=info["id_map"],
        scratch_root=output_dir / "rolling_scratch",
    )
    daily_start = (first + 1).to_timestamp("M").replace(day=1)
    daily_end = (last + 1).to_timestamp("M")
    daily = load_daily_returns(daily_path, start=daily_start, end=daily_end)
    cutoffs = load_nyse_cutoffs(cutoffs_path)
    result = backtest_predictions(
        predictions,
        panel,
        daily,
        formation_months=formation_months,
        nyse_cutoffs=cutoffs,
        strict_cutoffs=strict_cutoffs,
        cost_bps=cost_bps,
        coverage_threshold=coverage_threshold,
    )

    artifacts = {
        "response": str(response_path),
        "validation": str(output_dir / "validation.json"),
        "predictions": _write_table(predictions, output_dir / "predictions.parquet"),
        "rolling_audit": _write_table(rolling_audit, output_dir / "rolling_audit.parquet"),
        "monthly": _write_table(result["monthly"], output_dir / "backtest_monthly.parquet"),
        "daily": _write_table(result["daily"], output_dir / "backtest_daily.parquet"),
        "weights": _write_table(result["weights"], output_dir / "weights.parquet"),
    }
    # Keep the same per-formation weight partitions as the normal rolling
    # runner, so one rebalance can be audited without loading the combined
    # table.
    weight_partitions: dict[str, str] = {}
    if not result["weights"].empty and "formation_eom" in result["weights"].columns:
        for formation, subset in result["weights"].groupby("formation_eom", sort=True):
            stamp = (
                formation.to_timestamp("M")
                if isinstance(formation, pd.Period)
                else pd.Timestamp(formation)
            ).strftime("%Y%m")
            weight_partitions[stamp] = _write_table(
                subset,
                output_dir / f"weights_{stamp}.parquet",
            )
    artifacts["weight_partitions"] = weight_partitions
    summary = {
        "case_id": str(payload.get("case_id", case_dir.name)),
        "method": method,
        "track": info["track"],
        "semantic_card_count": (
            len(info["factor_cards"]) if info["track"] == "semantic" else 0
        ),
        "factor_count": len(info["public_features"]),
        "train_months": info["train_months"],
        "validation_months": info["validation_months"],
        "formation_months": len(formation_months),
        "cost_bps": float(cost_bps),
        "coverage_threshold": float(coverage_threshold),
        "code": {
            "schema_valid": True,
            "validation_execution_valid": bool(validation_metrics.get("execution_valid", False)),
            "validation_rank_ic_mean": validation_metrics.get("rank_ic_mean"),
            "validation_rank_ic_std": validation_metrics.get("rank_ic_std"),
            "validation_months_scored": validation_metrics.get("n_months"),
            "rolling_execution_valid": int(rolling_audit.get("execution_valid", pd.Series(dtype=bool)).sum())
            if "execution_valid" in rolling_audit
            else 0,
            "rolling_origins": int(len(rolling_audit)),
            "rolling_llm_calls": int(rolling_audit.get("llm_calls", pd.Series(dtype=int)).sum())
            if "llm_calls" in rolling_audit
            else 0,
        },
        "backtest": result["summary"],
        "artifacts": artifacts,
        "directions_sha256": _sha256_json(signs) if signs is not None else None,
        "response_sha256": _sha256_json(response),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", required=True, type=Path)
    parser.add_argument("--panel", required=True, type=Path)
    parser.add_argument("--daily", required=True, type=Path)
    parser.add_argument("--nyse-cutoffs", required=True, type=Path)
    parser.add_argument("--factor-details", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--method",
        default="jkp_signed_equal_weight",
        choices=[
            "jkp_signed_equal_weight",
            "no_prior_signed_equal_weight",
            "rolling_ridge",
            "rolling_elastic_net",
            "rolling_mlp",
            "rolling_mlp_deep",
            "rolling_mlp_ensemble",
            "rolling_ridge_refit",
            "rolling_elastic_net_refit",
            "rolling_mlp_refit",
            "rolling_mlp_deep_refit",
            "rolling_mlp_ensemble_refit",
            "rolling_ridge_tuned",
            "rolling_elastic_net_tuned",
            "rolling_mlp_tuned",
            "rolling_mlp_deep_tuned",
            "rolling_mlp_ensemble_tuned",
        ],
        help="Baseline method to run (frozen deterministic reference).",
    )
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--coverage-threshold", type=float, default=0.95)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--strict-cutoffs", action="store_true")
    args = parser.parse_args(argv)
    summary = run_case(
        case_dir=args.case_dir.resolve(),
        panel_path=args.panel.resolve(),
        daily_path=args.daily.resolve(),
        cutoffs_path=args.nyse_cutoffs.resolve(),
        factor_details_path=args.factor_details.resolve(),
        output_dir=args.output_dir.resolve(),
        cost_bps=float(args.cost_bps),
        coverage_threshold=float(args.coverage_threshold),
        timeout_seconds=float(args.timeout_seconds),
        strict_cutoffs=bool(args.strict_cutoffs),
        method=args.method,
    )
    print(json.dumps(summary, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

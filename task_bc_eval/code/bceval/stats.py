"""Inference and null-distribution machinery.

Two jobs:

1. Uncertainty that respects the design -- cases within a `family` share a data
   subset and anchor, so they are resampled as blocks, never as independent
   draws (rubric 1.2).
2. Noise floors -- `tau_noise` comes from an explicit no-skill null per metric,
   not from a copied constant.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .metrics import MONTHS_PER_YEAR, annualized_return, iqm, sharpe

DEFAULT_DRAWS = 2000
DEFAULT_SEED = 20260923


# --------------------------------------------------------------------------
# Block bootstrap
# --------------------------------------------------------------------------


def family_block_bootstrap_ci(
    values: Sequence[float],
    families: Sequence,
    n_boot: int = DEFAULT_DRAWS,
    alpha: float = 0.05,
    seed: int = DEFAULT_SEED,
    statistic=np.mean,
) -> dict:
    """Percentile CI from resampling whole families with replacement."""
    frame = pd.DataFrame({"value": np.asarray(values, dtype=float), "family": list(families)})
    frame = frame[np.isfinite(frame["value"])]
    if frame.empty:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n_cases": 0, "n_families": 0, "n_boot": 0}

    groups = [g["value"].to_numpy() for _, g in frame.groupby("family", sort=True)]
    rng = np.random.default_rng(seed)
    point = float(statistic(frame["value"].to_numpy()))

    draws = np.empty(n_boot)
    n_groups = len(groups)
    for i in range(n_boot):
        picks = rng.integers(0, n_groups, size=n_groups)
        sample = np.concatenate([groups[j] for j in picks])
        draws[i] = statistic(sample)

    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "point": point,
        "lo": float(lo),
        "hi": float(hi),
        "n_cases": int(len(frame)),
        "n_families": n_groups,
        "n_boot": n_boot,
    }


# --------------------------------------------------------------------------
# Paired comparisons
# --------------------------------------------------------------------------


def paired_delta(
    model: pd.DataFrame,
    baseline: pd.DataFrame,
    metric: str,
    sign: int = 1,
    key: str | Sequence[str] = "case_id",
) -> pd.DataFrame:
    """Per-case direction-unified difference, inner-joined on `key`.

    Cases missing on either side are dropped, never imputed; the caller reports
    how many pairs survived.

    `key` is a single column for the usual model-vs-baseline pairing, where the
    two sides are different methods by construction.  Pass a composite key when
    BOTH sides span more than one method -- the semantic-vs-numeric treatment
    effect across two models is the case that matters: joining on `case_id`
    alone silently emits cross-model pairs (model A's semantic minus model B's
    numeric), which doubles `n_pairs` with contrasts nobody asked for and
    corrupts the bootstrap CI.
    """
    keys = [key] if isinstance(key, str) else list(key)
    left = model[keys + [metric]].rename(columns={metric: "model"})
    right = baseline[keys + [metric]].rename(columns={metric: "baseline"})
    merged = left.merge(right, on=keys, how="inner")
    merged["delta"] = sign * (merged["model"] - merged["baseline"])
    return merged.dropna(subset=["delta"])


def improved_rate(deltas: Sequence[float], epsilon: float) -> dict:
    """P(delta > practical equivalence margin) -- 'how often does it win?'."""
    arr = np.asarray([d for d in np.asarray(deltas, dtype=float) if np.isfinite(d)])
    if arr.size == 0:
        return {"improved_rate": float("nan"), "n": 0, "n_improved": 0, "epsilon": epsilon}
    wins = int((arr > epsilon).sum())
    return {
        "improved_rate": wins / arr.size,
        "n": int(arr.size),
        "n_improved": wins,
        "n_worse": int((arr < -epsilon).sum()),
        "n_tied": int((np.abs(arr) <= epsilon).sum()),
        "epsilon": float(epsilon),
    }


def uplift_magnitude(deltas: Sequence[float], scale: float) -> float:
    """IQM of `clip(delta / a_m, -1, 1)` -- 'when it wins, by how much?'.

    Returned on the native [-1, 1] scale; `outcome` maps it to [0, 1].
    """
    if not np.isfinite(scale) or scale <= 0:
        return float("nan")
    arr = np.asarray(deltas, dtype=float)
    return iqm(np.clip(arr / scale, -1.0, 1.0))


def active_ir(model_monthly: pd.DataFrame, baseline_monthly: pd.DataFrame,
              value: str = "net_return", key: str = "holding_month") -> dict:
    """Annualised Information Ratio of the active series (Goodwin 1998).

    Requires both sides to expose a monthly series; a missing series yields NaN
    rather than a zero-filled active return.
    """
    if model_monthly is None or baseline_monthly is None or not len(model_monthly) or not len(baseline_monthly):
        return {"active_ir": float("nan"), "months": 0, "mean_active": float("nan")}
    merged = model_monthly[[key, value]].merge(
        baseline_monthly[[key, value]], on=key, how="inner", suffixes=("_m", "_b")
    )
    active = (merged[f"{value}_m"] - merged[f"{value}_b"]).to_numpy(dtype=float)
    active = active[np.isfinite(active)]
    if active.size < 2 or active.std(ddof=1) == 0:
        return {"active_ir": float("nan"), "months": int(active.size),
                "mean_active": float(active.mean()) if active.size else float("nan")}
    return {
        "active_ir": float(np.sqrt(MONTHS_PER_YEAR) * active.mean() / active.std(ddof=1)),
        "months": int(active.size),
        "mean_active": float(active.mean()),
    }


# --------------------------------------------------------------------------
# No-skill nulls for tau_noise
# --------------------------------------------------------------------------


def null_rank_ic(n_per_origin: Sequence[int], n_draws: int = DEFAULT_DRAWS,
                 seed: int = DEFAULT_SEED) -> np.ndarray:
    """Null of the case-level mean Rank IC under random ranking.

    Spearman's null is distribution-free given n, so the labels are never
    touched: each origin contributes the correlation between a random
    permutation and a fixed ranking.
    """
    rng = np.random.default_rng(seed)
    sizes = [int(n) for n in n_per_origin if n and n >= 3]
    if not sizes:
        return np.array([])
    out = np.empty(n_draws)
    for d in range(n_draws):
        ics = []
        for n in sizes:
            perm = rng.permutation(n)
            base = np.arange(n)
            ics.append(np.corrcoef(perm, base)[0, 1])
        out[d] = float(np.mean(ics))
    return out


def null_accuracy(n_per_origin: Sequence[int], p: float = 0.5,
                  n_draws: int = DEFAULT_DRAWS, seed: int = DEFAULT_SEED) -> np.ndarray:
    """Null of mean directional accuracy under coin-flip predictions."""
    rng = np.random.default_rng(seed)
    sizes = np.array([int(n) for n in n_per_origin if n and n > 0])
    if sizes.size == 0:
        return np.array([])
    draws = rng.binomial(sizes[None, :], p, size=(n_draws, sizes.size)) / sizes[None, :]
    return draws.mean(axis=1)


def null_top3_hit(positive_fraction_per_origin: Sequence[float], n_per_origin: Sequence[int],
                  k: int = 3, n_draws: int = DEFAULT_DRAWS, seed: int = DEFAULT_SEED) -> np.ndarray:
    """Null of Top-k hit rate under random selection.

    `top3_hit_rate` is the share of the k picked factors with a positive
    realised return, so the random control is a hypergeometric draw against
    that origin's actual positive-return base rate -- the 'random selection
    control' the rubric asks for in 3.1.2.
    """
    rng = np.random.default_rng(seed)
    pairs = [(float(f), int(n)) for f, n in zip(positive_fraction_per_origin, n_per_origin)
             if np.isfinite(f) and n and n >= k]
    if not pairs:
        return np.array([])
    out = np.empty(n_draws)
    for d in range(n_draws):
        hits = []
        for frac, n in pairs:
            n_pos = int(round(frac * n))
            pool = np.array([1] * n_pos + [0] * (n - n_pos))
            hits.append(rng.choice(pool, size=k, replace=False).mean())
        out[d] = float(np.mean(hits))
    return out


def null_sign_flip(monthly: Sequence[float], n_draws: int = DEFAULT_DRAWS,
                   seed: int = DEFAULT_SEED) -> dict[str, np.ndarray]:
    """Null of annualised return / Sharpe by randomly flipping monthly signs.

    Sign flipping preserves the volatility and the fat tails of the realised
    series while destroying the sign of the mean -- the relevant 'no edge' null
    for a long-short book.
    """
    arr = np.asarray([v for v in np.asarray(monthly, dtype=float) if np.isfinite(v)])
    if arr.size < 2:
        return {"annual_return": np.array([]), "sharpe": np.array([])}
    rng = np.random.default_rng(seed)
    rets = np.empty(n_draws)
    shp = np.empty(n_draws)
    for d in range(n_draws):
        flipped = arr * rng.choice([-1.0, 1.0], size=arr.size)
        rets[d] = annualized_return(flipped)
        shp[d] = sharpe(flipped)
    return {"annual_return": rets, "sharpe": shp}


def null_task_b_random_top3(truth_case: pd.DataFrame, k: int = 3, cost_bps: float = 10.0,
                            n_draws: int = 500, seed: int = DEFAULT_SEED) -> dict[str, np.ndarray]:
    """Task B economic null: pick k factors at random each month, equal-weight.

    This mirrors the evaluator's Top-k structure (it equal-weights rather than
    inverse-vol-weights), so the resulting return / Sharpe floor is the payoff
    of the selection protocol with the signal removed.
    """
    months = sorted(truth_case["holding_month"].unique())
    by_month = {m: truth_case[truth_case["holding_month"] == m]["ret"].to_numpy(dtype=float)
                for m in months}
    factors = {m: truth_case[truth_case["holding_month"] == m]["factor_id"].to_numpy() for m in months}
    rng = np.random.default_rng(seed)
    rets, shp = np.empty(n_draws), np.empty(n_draws)
    for d in range(n_draws):
        series, prev = [], None
        for m in months:
            n = len(by_month[m])
            if n < k:
                continue
            pick = rng.choice(n, size=k, replace=False)
            weights = {factors[m][i]: 1.0 / k for i in pick}
            gross = float(np.nanmean(by_month[m][pick]))
            keys = set(weights) | set(prev or {})
            turn = (sum(abs(weights.get(x, 0.0) - (prev or {}).get(x, 0.0)) for x in keys) / 2.0
                    if prev is not None else 1.0)
            series.append(gross - turn * cost_bps / 1e4)
            prev = weights
        rets[d] = annualized_return(series)
        shp[d] = sharpe(series)
    return {"annual_return": rets, "sharpe": shp}


def quantile(values: Sequence[float], q: float) -> float:
    arr = np.asarray([v for v in np.asarray(values, dtype=float) if np.isfinite(v)])
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q))

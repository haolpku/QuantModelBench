"""Metric primitives shared by the Outcome and Process axes.

Nothing here touches the filesystem; everything takes arrays / frames and
returns numbers, so each function is unit-testable in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12


# --------------------------------------------------------------------------
# Rank correlation
# --------------------------------------------------------------------------


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation with pairwise NaN removal.

    Returns NaN when fewer than 3 usable pairs remain or either side is
    constant -- a constant side makes the correlation undefined, and filling it
    with 0.0 would silently reward a degenerate predictor.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    ra = pd.Series(a[ok]).rank().to_numpy()
    rb = pd.Series(b[ok]).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def rank_ic_by_group(
    frame: pd.DataFrame, group: str, score: str, target: str
) -> pd.DataFrame:
    """Per-group Spearman IC plus the pair count that produced it."""
    rows = []
    for key, sub in frame.groupby(group, sort=True):
        ok = sub[[score, target]].notna().all(axis=1)
        rows.append(
            {
                group: key,
                "rank_ic": spearman(sub.loc[ok, score].to_numpy(), sub.loc[ok, target].to_numpy()),
                "n_pairs": int(ok.sum()),
                "n_rows": int(len(sub)),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Robust aggregation
# --------------------------------------------------------------------------


def iqm(values) -> float:
    """Interquartile mean: mean of the middle 50% (Agarwal et al. 2021)."""
    arr = np.asarray([v for v in np.asarray(values, dtype=float) if np.isfinite(v)])
    if arr.size == 0:
        return float("nan")
    if arr.size < 4:
        return float(arr.mean())
    lo, hi = np.percentile(arr, [25, 75])
    mid = arr[(arr >= lo) & (arr <= hi)]
    return float(mid.mean()) if mid.size else float(arr.mean())


def nan_iqr(values) -> float:
    arr = np.asarray([v for v in np.asarray(values, dtype=float) if np.isfinite(v)])
    if arr.size < 2:
        return float("nan")
    q1, q3 = np.percentile(arr, [25, 75])
    return float(q3 - q1)


def case_macro(frame: pd.DataFrame, value: str, case_col: str = "case_id") -> float:
    """Case-macro mean: aggregate within case first, then average across cases."""
    per_case = frame.groupby(case_col, sort=False)[value].mean()
    return float(per_case.mean(skipna=True))


# --------------------------------------------------------------------------
# Portfolio arithmetic
# --------------------------------------------------------------------------


def net_from_gross(gross: np.ndarray, turnover: np.ndarray, cost_bps: float) -> np.ndarray:
    """Task B cost convention, verified against the persisted cost_frontier."""
    return np.asarray(gross, dtype=float) - np.asarray(turnover, dtype=float) * cost_bps / 1e4


def annualized_return(monthly: np.ndarray) -> float:
    """Geometric annualisation of a monthly simple-return series."""
    arr = np.asarray([v for v in np.asarray(monthly, dtype=float) if np.isfinite(v)])
    if arr.size == 0:
        return float("nan")
    return float(np.prod(1.0 + arr) ** (MONTHS_PER_YEAR / arr.size) - 1.0)


def annualized_volatility(monthly: np.ndarray) -> float:
    arr = np.asarray([v for v in np.asarray(monthly, dtype=float) if np.isfinite(v)])
    if arr.size < 2:
        return float("nan")
    return float(arr.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR))


def sharpe(monthly: np.ndarray) -> float:
    """Arithmetic-mean Sharpe: `12 * mean(r) / (sd(r, ddof=1) * sqrt(12))`.

    Verified against the persisted Task B `cost_frontier` and Task C
    `summary.net`: both evaluators annualise the *return* geometrically but the
    Sharpe numerator arithmetically, so the two must not be mixed.
    """
    arr = np.asarray([v for v in np.asarray(monthly, dtype=float) if np.isfinite(v)])
    if arr.size < 2:
        return float("nan")
    vol = annualized_volatility(arr)
    if not np.isfinite(vol) or vol == 0:
        return float("nan")
    return float(arr.mean() * MONTHS_PER_YEAR / vol)


def max_drawdown(monthly: np.ndarray) -> float:
    """Worst peak-to-trough on the compounded wealth path (negative number)."""
    arr = np.asarray([v for v in np.asarray(monthly, dtype=float) if np.isfinite(v)])
    if arr.size == 0:
        return float("nan")
    wealth = np.cumprod(1.0 + arr)
    peak = np.maximum.accumulate(wealth)
    return float((wealth / peak - 1.0).min())


def portfolio_stats(monthly: np.ndarray) -> dict[str, float]:
    return {
        "annualized_return": annualized_return(monthly),
        "annualized_volatility": annualized_volatility(monthly),
        "sharpe": sharpe(monthly),
        "max_drawdown": max_drawdown(monthly),
        "months": int(np.isfinite(np.asarray(monthly, dtype=float)).sum()),
    }


# --------------------------------------------------------------------------
# Score composition with explicit missing-component handling
# --------------------------------------------------------------------------


@dataclass
class WeightedScore:
    """A weighted average that renormalises instead of zero-filling.

    A missing sub-component has its weight removed from the denominator and its
    name recorded in `dropped`.  Treating it as 0.0 would hand the lowest
    possible score to whichever track merely lacks the measurement -- exactly
    the failure mode the rubric forbids for runtime failures.
    """

    value: float
    weight_used: float
    dropped: tuple[str, ...]
    components: dict[str, float]

    @property
    def complete(self) -> bool:
        return not self.dropped

    def to_dict(self) -> dict:
        return {
            "value": None if not np.isfinite(self.value) else round(self.value, 6),
            "weight_used": round(self.weight_used, 6),
            "dropped": list(self.dropped),
            "components": {
                k: (None if v is None or not np.isfinite(v) else round(float(v), 6))
                for k, v in self.components.items()
            },
        }


def weighted_score(components: dict[str, float | None], weights: dict[str, float]) -> WeightedScore:
    """Renormalising weighted mean over the components that are present."""
    used: dict[str, float] = {}
    dropped: list[str] = []
    for name, weight in weights.items():
        val = components.get(name)
        if val is None or not np.isfinite(float(val)):
            dropped.append(name)
        else:
            used[name] = float(val) * weight
    total_weight = sum(weights[name] for name in used)
    value = sum(used.values()) / total_weight if total_weight > 0 else float("nan")
    return WeightedScore(
        value=value,
        weight_used=total_weight,
        dropped=tuple(dropped),
        components={k: components.get(k) for k in weights},
    )


def clip01(x: float) -> float:
    if x is None or not np.isfinite(x):
        return float("nan")
    return float(min(1.0, max(0.0, x)))


def three_threshold_score(value: float, tau_noise: float, tau_target: float) -> float:
    """`clip((M - tau_noise) / (tau_target - tau_noise), 0, 1)` (rubric 3.1.1).

    Callers must pass direction-unified values: higher is always better.
    """
    if value is None or not np.isfinite(value):
        return float("nan")
    if not np.isfinite(tau_noise) or not np.isfinite(tau_target):
        return float("nan")
    span = tau_target - tau_noise
    if span <= 0:
        return float("nan")
    return clip01((value - tau_noise) / span)

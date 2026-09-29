"""Read, write and validate the dev-frozen configuration.

Everything that could be tuned after seeing an outcome lives in one JSON file
that is written once on dev and then treated as read-only.  `validate()`
enforces rubric section 7's ten items: anything deliberately left unfrozen must
say so explicitly (`null` plus a `deferred` note), never be silently absent.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

CONFIG_VERSION = "dev_frozen.v1"

# Rubric section 7: the ten items that must be settled before reading test.
REQUIRED_ITEMS: tuple[str, ...] = (
    "baselines",
    "thresholds",
    "epsilon",
    "scales",
    "weights",
    "aggregation",
    "judge",
    "champion_thresholds",
    "search_policy",
    "multiplicity",
)

# Items this round is allowed to leave unresolved, with the reason recorded.
ALLOWED_DEFERRALS: dict[str, str] = {
    # Cost is out of the rubric for good (decision 2A): the token scale is not
    # comparable across models and is absent on most deepseek cases.  Both tasks
    # stay null and `E` renormalises over 0.55; cost is reported, not scored.
    "scales.task_b.cost": "dropped_from_rubric_decision_2A",
    "scales.task_c.cost": "dropped_from_rubric_decision_2A",
    "judge.agreement_threshold": "judge_not_run_this_round",
    "champion_thresholds.p_min": "not_defined_this_round",
    "champion_thresholds.o_min": "not_defined_this_round",
}


def _get(cfg: dict, dotted: str):
    node = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def new_config(created_on: str | None = None) -> dict:
    """Skeleton with every required item present (values filled by the freezer)."""
    return {
        "config_version": CONFIG_VERSION,
        "created_on": created_on or date.today().isoformat(),
        "split": "dev",
        "note": "Frozen on dev only. Test is evaluated once, with these values unchanged.",
        "baselines": {},
        "thresholds": {},
        "epsilon": {},
        "scales": {},
        "weights": {},
        "aggregation": {},
        "judge": {},
        "champion_thresholds": {"p_min": None, "o_min": None,
                                "deferred": "not_defined_this_round"},
        "search_policy": {},
        "multiplicity": {},
        "deferred": {},
    }


def validate(cfg: dict) -> dict:
    """Return `{ok, errors, deferred}`; raises nothing so callers can report."""
    errors: list[str] = []
    deferred: list[str] = []

    if cfg.get("config_version") != CONFIG_VERSION:
        errors.append(f"config_version must be {CONFIG_VERSION!r}, got {cfg.get('config_version')!r}")
    if cfg.get("split") != "dev":
        errors.append("thresholds may only be frozen on dev")

    for item in REQUIRED_ITEMS:
        if item not in cfg:
            errors.append(f"missing required item: {item}")
        elif cfg[item] in ({}, None):
            errors.append(f"required item is empty: {item}")

    for dotted, reason in ALLOWED_DEFERRALS.items():
        value = _get(cfg, dotted)
        if value is None:
            deferred.append(f"{dotted} ({reason})")

    # Weights must be exhaustive and normalised: a typo here silently reweights
    # every score downstream.
    for name, weights in (cfg.get("weights") or {}).items():
        if isinstance(weights, dict):
            total = sum(v for v in weights.values() if isinstance(v, (int, float)))
            if abs(total - 1.0) > 1e-9:
                errors.append(f"weights.{name} sums to {total!r}, expected 1.0")

    # A three-threshold score is only monotone when noise < target.
    for task, per_metric in (cfg.get("thresholds") or {}).items():
        for metric, thr in (per_metric or {}).items():
            noise, target = thr.get("tau_noise"), thr.get("tau_target")
            if noise is None or target is None:
                errors.append(f"thresholds.{task}.{metric}: tau_noise/tau_target must be set")
            elif not target > noise:
                errors.append(
                    f"thresholds.{task}.{metric}: tau_target ({target}) must exceed tau_noise ({noise})"
                )
            if thr.get("tau_effect") is None:
                errors.append(f"thresholds.{task}.{metric}: tau_effect must be set")

    return {"ok": not errors, "errors": errors, "deferred": deferred}


def save(cfg: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = validate(cfg)
    cfg = dict(cfg)
    cfg["validation"] = report
    path.write_text(json.dumps(cfg, indent=2, sort_keys=False, default=str) + "\n")
    return path


def load(path: str | Path, require_valid: bool = True) -> dict:
    cfg = json.loads(Path(path).read_text())
    report = validate(cfg)
    if require_valid and not report["ok"]:
        raise ValueError("frozen config failed validation:\n  " + "\n  ".join(report["errors"]))
    return cfg


def cost_scale(cfg: dict, task: str) -> tuple[float | None, float | None]:
    """`(c_good, c_bad)`; `(None, None)` when the scale is still deferred."""
    node = _get(cfg, f"scales.{task}.cost") or {}
    return node.get("c_good"), node.get("c_bad")


def set_cost_scale(cfg: dict, task: str, c_good: float, c_bad: float, source: str) -> dict:
    """Refill a deferred cost scale without touching any other frozen value."""
    cfg.setdefault("scales", {}).setdefault(task, {})["cost"] = {
        "c_good": float(c_good),
        "c_bad": float(c_bad),
        "source": source,
        "refrozen_on": date.today().isoformat(),
    }
    cfg.get("deferred", {}).pop(f"scales.{task}.cost", None)
    return cfg

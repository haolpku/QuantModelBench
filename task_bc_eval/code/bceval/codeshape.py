"""Heuristic classification of the frozen code into model families.

This is a keyword + AST-import scan, not a human annotation.  It is good
enough to answer "did semantic prompting change *what kind of model* the LLM
wrote?", and every report that uses it must say it is heuristic.
"""

from __future__ import annotations

import ast
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

# Ordered: the first family whose pattern matches wins when a file mentions
# several, because later entries are the more specific estimators.
FAMILY_PATTERNS: tuple[tuple[str, str], ...] = (
    ("gbdt_lightgbm", r"\blightgbm\b|\bLGBMRegressor\b|\bLGBMClassifier\b"),
    ("gbdt_xgboost", r"\bxgboost\b|\bXGBRegressor\b|\bXGBClassifier\b"),
    ("gbdt_catboost", r"\bcatboost\b|\bCatBoost"),
    ("gbdt_sklearn", r"HistGradientBoosting|GradientBoosting"),
    ("random_forest", r"RandomForest|ExtraTrees"),
    ("mlp", r"\bMLPRegressor\b|\bMLPClassifier\b|torch\.nn|\bkeras\b"),
    ("elastic_net", r"\bElasticNet"),
    ("lasso", r"\bLasso\b"),
    ("ridge", r"\bRidge\b|ridge_solve|closed_form_ridge"),
    ("logistic", r"LogisticRegression"),
    ("pls_pca", r"\bPLSRegression\b|\bPCA\b|TruncatedSVD"),
    ("ols", r"LinearRegression|np\.linalg\.lstsq|scipy\.linalg\.lstsq"),
    ("rank_composite", r"rank\(pct=True\)|rankdata|cross_sectional_rank|zscore"),
)

ENSEMBLE_HINT = r"\baverage\b|\bblend\b|\bensemble\b|\bstack"


def classify_code(code: str) -> dict:
    """Family label plus the evidence that produced it."""
    if not code:
        return {"family": "unknown", "matched": [], "is_ensemble": False,
                "imports": [], "code_chars": 0, "n_functions": 0}

    matched = [name for name, pattern in FAMILY_PATTERNS if re.search(pattern, code)]
    estimators = [m for m in matched if m != "rank_composite"]
    if not estimators:
        family = "rank_composite" if "rank_composite" in matched else "unknown"
    elif len(estimators) == 1:
        family = estimators[0]
    else:
        family = "hybrid:" + "+".join(sorted(estimators)[:3])

    imports: list[str] = []
    n_functions = 0
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module.split(".")[0])
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                n_functions += 1
    except SyntaxError:
        imports = ["<unparsable>"]

    return {
        "family": family,
        "matched": matched,
        "is_ensemble": bool(re.search(ENSEMBLE_HINT, code)) or len(estimators) > 1,
        "imports": sorted(set(imports)),
        "code_chars": len(code),
        "n_functions": n_functions,
    }


def read_frozen_code(task: str, case_dir: Path) -> str | None:
    """Task B stores the selected response beside the case; Task C nests it."""
    if task == "task_b":
        payload = case_dir / "selected_response.json"
        if payload.exists():
            return json.loads(payload.read_text()).get("code")
        audit = case_dir / "revision_audit.json"
        if audit.exists():
            return (json.loads(audit.read_text()).get("selected_response") or {}).get("code")
        return None
    audit = case_dir / "revision_audit.json"
    if audit.exists():
        return (json.loads(audit.read_text()).get("selected_response") or {}).get("code")
    return None


def classify_run(task: str, case_dirs: list[Path], run_key: str) -> pd.DataFrame:
    rows = []
    for case_dir in case_dirs:
        code = read_frozen_code(task, case_dir)
        info = classify_code(code or "")
        rows.append(
            {
                "run_key": run_key,
                "case_id": case_dir.name,
                "family": info["family"],
                "is_ensemble": info["is_ensemble"],
                "code_chars": info["code_chars"],
                "n_functions": info["n_functions"],
                "matched": ",".join(info["matched"]),
                "imports": ",".join(info["imports"]),
            }
        )
    return pd.DataFrame(rows)


def family_distribution(shapes: pd.DataFrame) -> dict:
    counts = Counter(shapes["family"])
    total = sum(counts.values())
    return {
        "counts": dict(counts.most_common()),
        "n_distinct": len(counts),
        "share": {k: round(v / total, 4) for k, v in counts.most_common()} if total else {},
        "ensemble_rate": float(shapes["is_ensemble"].mean()) if len(shapes) else float("nan"),
        "median_code_chars": float(shapes["code_chars"].median()) if len(shapes) else float("nan"),
    }


def spec_case_dirs(spec) -> list[Path]:
    """Case directories that hold the frozen code for a run spec.

    Task B keeps `selected_response.json` beside the rolling result; Task C
    keeps it in a separate `codegen` tree, so the rolling root is not usable.
    """
    if spec.task == "task_b":
        return sorted(p for p in spec.root.glob("block_*/case_*") if p.is_dir())
    root = spec.codegen_root or spec.root
    return sorted(p for p in root.glob("block_*/c_case_*") if p.is_dir())


def classify_spec(spec) -> pd.DataFrame:
    return classify_run(spec.task, spec_case_dirs(spec), spec.key)

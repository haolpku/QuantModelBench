"""Policy-check and execute one generated Task C model in a fresh process.

This is a containment layer for benchmark code, not a hardened operating-
system sandbox.  It combines an explicit AST/import policy, restricted
builtins, a minimal child environment, a scratch working directory, resource
limits where the platform supports them, and a wall-clock timeout.  Production
runs should additionally use an OS/container sandbox with networking disabled.
"""

from __future__ import annotations

import ast
import hashlib
import os
import pickle
import json
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .semantic import normalize_track, semantic_card_payload


MAX_CODE_CHARS = 120_000
MAX_STDIO_CHARS = 12_000
# Child-side traceback file, read back by the parent outside the stdio budget.
_DIAGNOSTIC_NAME = "child_diagnostic.txt"
DEFAULT_TIMEOUT_SECONDS = 1_800.0
# Upper bound on the child process address space.  Deep-learning candidates
# (torch / gradient boosting) need a real budget, but an unbounded fit can take
# the whole machine down with it, so the limit is requested per execution.
# ``RLIMIT_AS`` is enforced by the kernel on Linux; on macOS this process
# already sits above the settable maximum, so the cap is skipped there and the
# wall-clock timeout remains the only bound.
DEFAULT_MEMORY_LIMIT_BYTES = 8 * 1024 ** 3
# Small, fixed thread budget: enough for a torch matmul to use the machine,
# small enough that per-thread memory stays inside DEFAULT_MEMORY_LIMIT_BYTES.
_THREAD_LIMIT = "4"
ALLOWED_IMPORT_ROOTS = frozenset(
    {
        # Core numerical/scientific stack.
        "numpy",
        "pandas",
        "sklearn",
        "scipy",
        # Gradient boosting / statistical model families.  The benchmark scores
        # every model family on the same footing, so the whitelist must not
        # silently exclude them.
        "xgboost",
        "lightgbm",
        "catboost",
        "statsmodels",
        "optuna",
        # Neural networks.
        "torch",
        # Standard-library helpers commonly used in modelling code.
        "abc",
        "bisect",
        "copy",
        "csv",
        "json",
        "math",
        "numbers",
        "re",
        "statistics",
        "collections",
        "typing",
        "functools",
        "itertools",
        "dataclasses",
        "warnings",
        "random",
        "time",
        "__future__",
        # Declared runtime dependencies of the model packages above (torch's
        # ``typing_extensions``/``sympy``/``filelock``/``fsspec``/``networkx``/
        # ``jinja2`` chain, and the boosters' shared helpers).  A candidate that
        # imports a whitelisted package must not fail because that package
        # imports one of these on its own behalf.
        "typing_extensions",
        "setuptools",
        "sympy",
        "networkx",
        "filelock",
        "fsspec",
        "jinja2",
        "markupsafe",
        "mpmath",
        "packaging",
        "yaml",
        "tqdm",
    }
)
FORBIDDEN_IMPORT_PREFIXES = (
    "pandas.io",
    "numpy.lib.npyio",
    "scipy.io",
    "sklearn.datasets",
)
FORBIDDEN_CALL_NAMES = frozenset(
    {
        "open",
        "eval",
        "exec",
        "compile",
        "input",
        "help",
        "breakpoint",
        "__import__",
        "globals",
        "locals",
        "vars",
        "dir",
        "getattr",
        "hasattr",
        "setattr",
    }
)
FORBIDDEN_ATTRIBUTES = frozenset(
    {
        # Reaching further into a live object graph is what turns a
        # ``__class__``-style dunder into a sandbox escape, so the interception
        # dunders stay blocked even though ordinary class definition does not.
        "__globals__",
        "__subclasses__",
        "__bases__",
        "__mro__",
        "__code__",
        "__closure__",
        "__func__",
        "__self__",
        "__dict__",
        "__getattribute__",
        "__reduce__",
        "__reduce_ex__",
        "__setstate__",
        "dump",
        "dumps",
        "fromfile",
        "genfromtxt",
        "load",
        "loads",
        "memmap",
        # Reads are neither write nor escape primitives, and the agentic
        # ``python_exec`` preamble materializes exactly the evaluator-declared
        # public train/validation files into the workspace.  Both prompts point
        # candidates at ``pd.read_parquet``/``pd.read_json`` for row-level
        # inspection, so a blanket read ban only broke documented instructions.
        # ``read_*`` names are therefore allowed; the write and process
        # primitives below stay blocked.
        "save",
        "savetxt",
        "savez",
        "savez_compressed",
        "tofile",
        "to_csv",
        "to_excel",
        "to_feather",
        "to_hdf",
        "to_json",
        "to_parquet",
        "to_pickle",
        "system",
        "popen",
        "spawn",
        "fork",
        "remove",
        "unlink",
        "rmdir",
        "mkdir",
        "makedirs",
        "rename",
        # ``DataFrame.replace`` is a normal, in-memory missing-value
        # operation used by many generated models.  File/network modules are
        # blocked separately (and ``os``/``pathlib`` cannot be imported), so
        # treating the generic method name as a write primitive made valid
        # modeling code fail before it could run.
        "connect",
        "request",
        "urlopen",
    }
)


@dataclass
class CodeExecutionResult:
    """Structured result from syntax/policy/runtime/output validation."""

    valid: bool
    status: str
    errors: list[str] = field(default_factory=list)
    prediction: pd.DataFrame | None = field(default=None, repr=False)
    code_sha256: str | None = None
    runtime_seconds: float | None = None
    timed_out: bool = False
    return_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    checks: dict[str, Any] = field(default_factory=dict)

    def audit_dict(self, *, include_prediction: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "valid": bool(self.valid),
            "status": self.status,
            "errors": list(self.errors),
            "code_sha256": self.code_sha256,
            "runtime_seconds": self.runtime_seconds,
            "timed_out": bool(self.timed_out),
            "return_code": self.return_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "checks": dict(self.checks),
        }
        if include_prediction and self.prediction is not None:
            result["prediction"] = self.prediction.to_dict("records")
        elif self.prediction is not None:
            result["prediction_rows"] = int(len(self.prediction))
            result["prediction_columns"] = [str(c) for c in self.prediction.columns]
        return result


class _PolicyVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.entrypoints: list[ast.FunctionDef] = []

    def _error(self, node: ast.AST, message: str) -> None:
        self.errors.append(f"line {getattr(node, 'lineno', '?')}: {message}")

    def visit_Import(self, node: ast.Import) -> Any:
        for alias in node.names:
            root = alias.name.split(".", 1)[0]
            blocked = any(
                alias.name == prefix or alias.name.startswith(prefix + ".")
                for prefix in FORBIDDEN_IMPORT_PREFIXES
            )
            if root not in ALLOWED_IMPORT_ROOTS or blocked:
                self._error(node, f"import {alias.name!r} is not allowed")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        if node.level:
            self._error(node, "relative imports are not allowed")
        module = str(node.module or "")
        root = module.split(".", 1)[0]
        if root not in ALLOWED_IMPORT_ROOTS or any(
            module == prefix or module.startswith(prefix + ".")
            for prefix in FORBIDDEN_IMPORT_PREFIXES
        ):
            self._error(node, f"import from {module!r} is not allowed")
        for alias in node.names:
            imported_name = str(alias.name).split(".")[-1]
            json_serialization = root == "json" and imported_name in {"dump", "dumps", "load", "loads"}
            if imported_name in FORBIDDEN_ATTRIBUTES and not json_serialization:
                self._error(node, f"imported attribute {imported_name!r} is not allowed")
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        if node.name == "fit_and_predict":
            self.entrypoints.append(node)
            positional = len(node.args.posonlyargs) + len(node.args.args)
            if positional < 4 and node.args.vararg is None:
                self._error(node, "fit_and_predict must accept train_df, valid_df, predict_df, metadata")
            if node.decorator_list:
                self._error(node, "fit_and_predict decorators are not allowed")
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        if node.name == "fit_and_predict":
            self._error(node, "fit_and_predict must be synchronous")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id in {"__builtins__", "__loader__", "__spec__"}:
            self._error(node, f"access to {node.id} is not allowed")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        if node.attr in FORBIDDEN_ATTRIBUTES:
            self._error(node, f"attribute {node.attr!r} is not allowed")
        # JSON's in-memory serializers/parsers are safe and useful for
        # deterministic diagnostics.  Keep numpy/pandas file loaders blocked;
        # the exception is deliberately limited to a direct ``json.<method>``
        # receiver rather than allowing arbitrary objects named ``loads``.
        json_serialization = (
            isinstance(node.value, ast.Name)
            and node.value.id == "json"
            and node.attr in {"dump", "dumps", "load", "loads"}
        )
        if node.attr in FORBIDDEN_ATTRIBUTES and not json_serialization:
            self._error(node, f"attribute {node.attr!r} is not allowed")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> Any:
        if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALL_NAMES:
            self._error(node, f"call to {node.func.id!r} is not allowed")
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> Any:
        self._error(node, "global declarations are not allowed")

    def visit_Nonlocal(self, node: ast.Nonlocal) -> Any:
        self._error(node, "nonlocal declarations are not allowed")

    def visit_Constant(self, node: ast.Constant) -> Any:
        if isinstance(node.value, str):
            lowered = node.value.lower().replace("\\", "/")
            if any(marker in lowered for marker in ("truth.jsonl", "ground_truth.json", "/truth/", "/ground_truth/")):
                self._error(node, "references to hidden truth artifacts are not allowed")
        self.generic_visit(node)


def check_code_policy(code: Any) -> list[str]:
    """Return syntax and static-policy errors without executing source."""

    if not isinstance(code, str):
        return ["code must be a string"]
    if len(code) > MAX_CODE_CHARS:
        return [f"code exceeds {MAX_CODE_CHARS} characters"]
    if not code.strip():
        return ["code is empty"]
    try:
        tree = ast.parse(code, filename="<task_c_candidate>", mode="exec")
    except SyntaxError as exc:
        return [f"syntax error at line {exc.lineno}: {exc.msg}"]
    visitor = _PolicyVisitor()
    visitor.visit(tree)
    if len(visitor.entrypoints) != 1:
        visitor.errors.append(
            f"source must define exactly one synchronous fit_and_predict function; found {len(visitor.entrypoints)}"
        )
    return list(dict.fromkeys(visitor.errors))


def _imports_valid(policy_errors: Sequence[str]) -> bool:
    """Return whether static import checks passed independently of other policy rules."""

    return not any(
        "import " in str(error).lower()
        or "imported attribute" in str(error).lower()
        for error in policy_errors
    )


_CHILD_RUNNER = (
    r'''from __future__ import annotations
import builtins
import inspect
import pickle
import random
import sys
import traceback
import warnings

# Generated from the module-level whitelist so the in-process policy check and
# the child's runtime import gate cannot drift apart.
ALLOWED_IMPORT_ROOTS = '''
    + repr(sorted(ALLOWED_IMPORT_ROOTS))
    + r'''
_real_import = builtins.__import__
_STDIO_LIMIT = '''
    + str(MAX_STDIO_CHARS)
    + r'''

class _BoundedStream:
    # Raw, never-wrapped stderr.  Diagnostics written through this handle do
    # not consume the candidate's output budget and cannot be affected by it.
    UNLIMITED = sys.stderr

    def __init__(self, target, label="stdio"):
        self.target = target
        self.label = label
        self.count = 0
        self.truncated = False

    def write(self, value):
        text = str(value)
        if self.truncated:
            return len(text)
        remaining = _STDIO_LIMIT - self.count
        if len(text) >= remaining:
            # Truncate and drop the rest.  Raising here would turn "the
            # candidate printed too much" into a process crash, and the crash
            # would land while the traceback was being written -- destroying
            # the one diagnostic that explains the failure.
            self.target.write(text[:remaining])
            self.target.flush()
            self.count += remaining
            self.truncated = True
            _unlimited(
                "\n[task-c] " + self.label
                + " exceeded " + str(_STDIO_LIMIT)
                + " chars; further candidate output dropped\n"
            )
            return len(text)
        self.target.write(text)
        self.target.flush()
        self.count += len(text)
        return len(text)

    def flush(self):
        try:
            self.target.flush()
        except Exception:
            pass

    @property
    def closed(self):
        return False

    def isatty(self):
        return False

    def fileno(self):
        return self.target.fileno()

    def writable(self):
        return True


def _unlimited(text):
    """Write a diagnostic to raw stderr, outside the candidate budget."""
    try:
        _BoundedStream.UNLIMITED.write(str(text))
        _BoundedStream.UNLIMITED.flush()
    except Exception:
        pass


def _install_stdio_isolation():
    """Silence warning floods and keep diagnostics off the candidate budget."""

    # Library deprecation warnings (pandas FutureWarning above all) arrive on
    # stderr and, under the shared budget, crowd out the real traceback.
    try:
        warnings.filterwarnings("ignore")
    except Exception:
        pass
    _unlimited("[task-c] child stdio isolation active (warnings suppressed)\n")

def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level:
        raise ImportError("relative imports are disabled")
    root = str(name).split(".", 1)[0]
    if root not in ALLOWED_IMPORT_ROOTS:
        raise ImportError("import is not allowed: " + str(name))
    return _real_import(name, globals, locals, fromlist, level)

SAFE_BUILTINS = {
    name: getattr(builtins, name)
    for name in (
        "__build_class__", "abs", "all", "any", "bool", "bytes", "callable",
        "dict", "divmod", "enumerate", "Exception", "filter", "float", "format",
        "frozenset", "hash", "int", "isinstance", "issubclass",
        "iter", "len", "list", "map", "max", "memoryview", "min", "next", "object",
        "ord", "pow", "print", "property", "range", "repr", "reversed", "round",
        "RuntimeError", "set", "slice", "sorted", "staticmethod", "StopIteration",
        "str", "sum", "super", "tuple", "type", "TypeError", "ValueError",
        "KeyError", "IndexError", "LookupError", "ZeroDivisionError",
        "NotImplementedError", "ArithmeticError", "AssertionError", "zip",
        "MemoryError", "OverflowError", "FloatingPointError", "RecursionError",
    )
}
SAFE_BUILTINS["__import__"] = _safe_import

def _limits(memory_limit_bytes=None):
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024 * 1024, 16 * 1024 * 1024))
    except Exception:
        pass
    if memory_limit_bytes:
        try:
            # ``RLIMIT_AS`` is enforced by the kernel on Linux; on platforms
            # where the limit cannot be expressed it is silently skipped.
            import resource
            limit = int(memory_limit_bytes)
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        except Exception:
            pass

def main():
    input_path, source_path, output_path = sys.argv[1:4]
    memory_limit = None
    if len(sys.argv) > 4:
        try:
            memory_limit = int(sys.argv[4])
        except (TypeError, ValueError):
            memory_limit = None
    _limits(memory_limit)
    _install_stdio_isolation()
    sys.stdout = _BoundedStream(sys.stdout, "candidate stdout")
    sys.stderr = _BoundedStream(sys.stderr, "candidate stderr")
    with open(input_path, "rb") as handle:
        payload = pickle.load(handle)
    try:
        seed = int(payload.get("metadata", {}).get("seed", 1))
    except (TypeError, ValueError):
        seed = 1
    random.seed(seed)
    try:
        import numpy as _np
        _np.random.seed(seed)
    except Exception:
        pass
    with open(source_path, "r", encoding="utf-8") as handle:
        source = handle.read()
    namespace = {
        "__builtins__": SAFE_BUILTINS,
        "__name__": "task_c_generated_model",
        "__package__": None,
    }
    compiled = compile(source, "<task_c_candidate>", "exec")
    exec(compiled, namespace, namespace)
    function = namespace.get("fit_and_predict")
    if not callable(function):
        raise TypeError("fit_and_predict is not callable")
    inspect.signature(function).bind(
        payload["train_df"], payload["valid_df"], payload["predict_df"], payload["metadata"]
    )
    result = function(
        payload["train_df"].copy(deep=True),
        payload["valid_df"].copy(deep=True),
        payload["predict_df"].copy(deep=True),
        dict(payload["metadata"]),
    )
    # JSON is intentionally used for the child -> parent boundary.  Loading a
    # pickle produced by untrusted candidate code in the evaluator would allow
    # a crafted ``__reduce__`` object to execute outside the sandbox.  The
    # public output contract is a tabular DataFrame, so records JSON is enough
    # and rejects arbitrary Python objects before they cross the boundary.
    if not hasattr(result, "to_json"):
        raise TypeError("fit_and_predict must return pandas.DataFrame")
    text = result.to_json(orient="records", date_format="iso", force_ascii=False)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(text)

def _record_failure():
    """Persist the active traceback where the parent is guaranteed to find it.

    Writing it to stderr alone is not enough: the parent keeps only the first
    ``MAX_STDIO_CHARS`` of stderr, so candidate output printed earlier can push
    the traceback past the cut.  A dedicated file has no such competition.
    """
    import os
    text = traceback.format_exc()
    destination = os.environ.get("TASK_C_DIAGNOSTIC_PATH")
    if destination:
        try:
            with open(destination, "w", encoding="utf-8", errors="replace") as handle:
                handle.write(text)
                handle.flush()
        except Exception:
            pass
    _unlimited(text)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        _record_failure()
        raise
'''
)


def _bounded(value: str, limit: int = MAX_STDIO_CHARS) -> str:
    text = str(value or "")
    return text if len(text) <= limit else text[:limit] + f"\n...[truncated at {limit} chars]"


def _with_diagnostic(scratch: Path, stderr: str) -> str:
    """Put the child's own failure report at the head of `stderr`.

    Ordering matters more than the content.  `_bounded` truncates from the tail,
    so anything the candidate library printed ahead of the traceback is exactly
    what pushes it out of the captured window.  Leading with the diagnostic file
    makes the reason for the failure survive regardless of how much noise the
    candidate emitted first.
    """
    diagnostic = ""
    try:
        candidate = scratch / _DIAGNOSTIC_NAME
        if candidate.is_file():
            diagnostic = candidate.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        diagnostic = ""
    if not diagnostic:
        return stderr
    return f"[child traceback]\n{diagnostic}\n[child stderr]\n{stderr}"


def _child_environment(scratch: Path, seed: int) -> dict[str, str]:
    """Keep only variables needed to import the frozen scientific runtime."""

    env: dict[str, str] = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONHASHSEED": str(int(seed)),
        # Thread counts are pinned so an unbounded library default (torch and
        # the boosters all read these) cannot multiply the memory footprint
        # measured against DEFAULT_MEMORY_LIMIT_BYTES.
        "OMP_NUM_THREADS": _THREAD_LIMIT,
        "OPENBLAS_NUM_THREADS": _THREAD_LIMIT,
        "MKL_NUM_THREADS": _THREAD_LIMIT,
        "NUMEXPR_NUM_THREADS": _THREAD_LIMIT,
        "VECLIB_MAXIMUM_THREADS": _THREAD_LIMIT,
        "MPLCONFIGDIR": str(scratch / "mpl"),
        "TMPDIR": str(scratch),
        # The child writes its traceback here.  stderr alone cannot carry it:
        # the parent keeps only the head of the captured stderr, so a candidate
        # that prints first and crashes second loses the one line that explains
        # the crash.  A separate file has no budget to compete for.
        "TASK_C_DIAGNOSTIC_PATH": str(scratch / _DIAGNOSTIC_NAME),
    }
    # A virtual environment may rely on these search roots.  They contain no
    # benchmark truth and are recorded by the outer runtime audit.
    for key in ("VIRTUAL_ENV", "CONDA_PREFIX", "SYSTEMROOT", "WINDIR"):
        if os.environ.get(key):
            env[key] = os.environ[key]
    return env


def _canonical_identifier(value: Any) -> str:
    """Normalize the common numeric/string ID representations at the runner boundary.

    WRDS Parquet columns often arrive as floating-point values (``123.0``),
    while a generated model may stringify them as ``123``.  Treat those two
    representations as the same key, but preserve leading-zero string IDs,
    which can be meaningful security identifiers.
    """

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


def _forbidden_calendar_columns(columns: Iterable[Any]) -> list[str]:
    """Return evaluator-only calendar columns that must not cross the model boundary."""

    calendar_names = {
        "eom",
        "date",
        "month",
        "anchor_date",
        "formation_eom",
        "formation_date",
        "formation_month",
        "origin_month",
        "as_of_month",
        "calendar_date",
    }
    return [
        str(column)
        for column in columns
        if re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_") in calendar_names
    ]


def _forbidden_historical_columns(columns: Iterable[Any]) -> list[str]:
    """Find hidden targets/calendar fields in train or validation inputs."""

    forbidden: list[str] = []
    label_names = {
        "labels",
        "target",
        "y",
        "next_return",
        "next_month_return",
        "ret_exc_lead1m",
        "ret_exc_lead_1m",
        "lead_return",
        "return_next_month",
        "truth",
        "ground_truth",
        "real_id",
        "real_ids",
        "id_mapping",
        "masked_to_original",
        "original_to_masked",
    }
    for column in columns:
        normalized = re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_")
        # ``label`` itself is the sole historical target explicitly allowed by
        # the Task C contract.  Aliases and future/target-prefixed fields are
        # rejected rather than silently handed to generated code.
        if normalized in label_names or normalized.startswith(
            ("future_", "target_", "truth_", "ground_truth")
        ):
            forbidden.append(str(column))
    forbidden.extend(_forbidden_calendar_columns(columns))
    return list(dict.fromkeys(forbidden))


def _declared_input_errors(
    code: str | Mapping[str, Any],
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    predict_df: pd.DataFrame,
) -> list[str]:
    """Check the response's declared input contract against actual frames."""

    if not isinstance(code, Mapping):
        return []
    contract = code.get("input_contract")
    if not isinstance(contract, Mapping):
        return []
    errors: list[str] = []
    for key, frame in (
        ("train_columns", train_df),
        ("validation_columns", valid_df),
        ("predict_columns", predict_df),
    ):
        declared = contract.get(key)
        if not isinstance(declared, list):
            continue
        missing = [str(column) for column in declared if str(column) not in frame.columns]
        if missing:
            errors.append(f"{key} declares columns not supplied at runtime: {missing[:20]}")
    return errors


_RUNTIME_DATE_LITERAL_RE = re.compile(
    r"(?<!\d)(?:(?:19|20)\d{2}[-_/]\d{1,2}(?:[-_/]\d{1,2})?|(?:19|20)\d{4})(?!\d)"
)


def _safe_runtime_metadata(value: Mapping[str, Any] | None) -> dict[str, Any]:
    """Remove evaluator-only metadata before it enters generated code.

    The normal rolling path already constructs a sanitized metadata mapping,
    but ``run_candidate`` is also a public low-level API.  Applying the same
    boundary here prevents a direct caller from accidentally handing a
    formation date, real-ID map, local path, or truth object to a candidate.
    Neutral scalar/list settings (seed, feature names, protocol labels) are
    retained so the model remains reproducible.
    """

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

    # ``factor_cards`` is a public semantic sidecar, but this low-level runner
    # is also callable directly by scripts.  Do not trust a direct caller to
    # have applied the card whitelist (or to keep cards out of numeric-only
    # metadata); normalize it again below after the generic metadata pass.
    source_value = value or {}
    try:
        runtime_track = normalize_track(source_value.get("track", "numeric_only"))
    except ValueError:
        runtime_track = "numeric_only"
    raw_cards = source_value.get("factor_cards")
    declared_features = source_value.get("feature_columns")
    declared_public_ids = (
        {str(item) for item in declared_features if str(item).strip()}
        if isinstance(declared_features, (list, tuple))
        else set()
    )

    def clean(key: Any, item: Any) -> tuple[bool, Any]:
        normalized = re.sub(r"[^a-z0-9]+", "_", str(key).strip().lower()).strip("_")
        if normalized in {"factor_cards", "semantic_cards"}:
            return False, None
        if normalized in blocked_exact or any(
            marker in normalized for marker in ("path", "filename", "mapping", "secret", "token")
        ):
            return False, None
        if isinstance(item, (pd.Timestamp, pd.Period, np.datetime64, datetime, date)):
            return False, None
        if isinstance(item, str) and _RUNTIME_DATE_LITERAL_RE.search(item):
            return False, None
        if isinstance(item, Mapping):
            output: dict[str, Any] = {}
            for child_key, child_value in item.items():
                keep, cleaned = clean(child_key, child_value)
                if keep:
                    output[str(child_key)] = cleaned
            return True, output
        if isinstance(item, (list, tuple)):
            output_list: list[Any] = []
            for index, child_value in enumerate(item):
                keep, cleaned = clean(f"item_{index}", child_value)
                if keep:
                    output_list.append(cleaned)
            return True, output_list
        return True, item

    output: dict[str, Any] = {}
    for key, item in source_value.items():
        keep, cleaned = clean(key, item)
        if keep:
            output[str(key)] = cleaned
    if runtime_track == "semantic" and isinstance(raw_cards, list):
        public_cards: list[dict[str, Any]] = []
        for card in raw_cards:
            if not isinstance(card, Mapping):
                continue
            card_id = str(card.get("factor_id") or "").strip()
            # Normal payloads use Sxx/fxxx identifiers.  If a low-level caller
            # supplies a raw mnemonic instead, dropping the card is safer than
            # allowing it to cross the model boundary without an evaluator map.
            if declared_public_ids and card_id not in declared_public_ids:
                continue
            if not declared_public_ids and not re.fullmatch(r"(?:S|f)\d+", card_id):
                continue
            sanitized = semantic_card_payload(card, card_id)
            keep, cleaned = clean("card", sanitized)
            if keep and isinstance(cleaned, Mapping):
                public_cards.append(dict(cleaned))
        if public_cards:
            output["factor_cards"] = public_cards
    # ``semantic_cards`` is a legacy/private alias and is intentionally never
    # forwarded to generated code.
    output.pop("semantic_cards", None)
    return output


def _validate_prediction(
    value: Any,
    predict_df: pd.DataFrame,
    *,
    id_column: str,
) -> tuple[pd.DataFrame | None, list[str], dict[str, Any]]:
    errors: list[str] = []
    checks: dict[str, Any] = {
        "is_dataframe": isinstance(value, pd.DataFrame),
        "columns_ok": False,
        "one_row_per_id": False,
        "coverage": 0.0,
        "finite_score": False,
        "rows_ok": False,
    }
    if not isinstance(value, pd.DataFrame):
        return None, ["fit_and_predict must return pandas.DataFrame"], checks
    frame = value.copy()
    if frame.columns.duplicated().any():
        duplicate_columns = [str(column) for column in frame.columns[frame.columns.duplicated()].tolist()]
        return None, [f"prediction contains duplicate columns: {duplicate_columns}"], checks
    missing_columns = [c for c in (id_column, "score") if c not in frame.columns]
    if missing_columns:
        return None, [f"prediction missing columns: {missing_columns}"], checks
    checks["columns_ok"] = True
    output_columns = [column for column in frame.columns if column not in {id_column, "score", "time_key"}]
    forbidden_output = _forbidden_predict_columns(output_columns)
    forbidden_calendar = _forbidden_calendar_columns(output_columns)
    if forbidden_output:
        return None, [f"prediction contains forbidden label columns: {forbidden_output}"], checks
    if forbidden_calendar:
        return None, [f"prediction contains evaluator-only calendar columns: {forbidden_calendar}"], checks
    if id_column not in predict_df.columns:
        return None, [f"predict_df missing id column {id_column!r}"], checks
    expected = predict_df[id_column].map(_canonical_identifier).astype("string")
    if expected.isna().any() or expected.eq("").any() or expected.duplicated().any():
        return None, ["predict_df must contain one non-null row per id"], checks
    frame[id_column] = frame[id_column].map(_canonical_identifier).astype("string")
    if frame[id_column].isna().any() or frame[id_column].eq("").any() or frame[id_column].duplicated().any():
        errors.append("prediction ids must be non-null and unique")
    else:
        checks["one_row_per_id"] = True
    expected_set = set(expected.tolist())
    actual_set = set(frame[id_column].tolist())
    missing = sorted(expected_set - actual_set)
    extra = sorted(actual_set - expected_set)
    checks["coverage"] = float(len(expected_set & actual_set) / max(1, len(expected_set)))
    if missing:
        errors.append(f"prediction omits {len(missing)} ids (sample={missing[:5]})")
    if extra:
        errors.append(f"prediction contains {len(extra)} unknown ids (sample={extra[:5]})")
    scores = pd.to_numeric(frame["score"], errors="coerce")
    finite = np.isfinite(scores.to_numpy(dtype=float, na_value=np.nan))
    if not bool(finite.all()):
        errors.append(f"prediction contains {int((~finite).sum())} non-finite scores")
    else:
        checks["finite_score"] = True
    checks["rows_ok"] = len(frame) == len(expected) and not missing and not extra
    if errors:
        return None, errors, checks
    frame["score"] = scores.astype(float)
    order = pd.DataFrame({id_column: expected, "_order": np.arange(len(expected), dtype=int)})
    frame = order.merge(frame, on=id_column, how="left", validate="one_to_one").sort_values("_order")
    frame = frame.drop(columns=["_order"]).reset_index(drop=True)
    if "time_key" in predict_df.columns:
        time_values = predict_df[[id_column, "time_key"]].copy()
        time_values[id_column] = time_values[id_column].map(_canonical_identifier).astype("string")
        time_values = time_values.rename(columns={"time_key": "_predict_time_key"})
        frame = frame.merge(time_values, on=id_column, how="left", validate="one_to_one")
        frame["time_key"] = frame["_predict_time_key"]
        frame = frame.drop(columns=["_predict_time_key"])
    return frame, [], checks


def _forbidden_predict_columns(columns: Iterable[Any]) -> list[str]:
    """Find label/future-return columns using the same aliases as payload checks."""

    forbidden: list[str] = []
    for column in columns:
        normalized = "".join(ch if ch.isalnum() else "_" for ch in str(column).strip().lower()).strip("_")
        if (
            normalized
            in {
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
            or normalized in {
                "truth",
                "ground_truth",
                "real_id",
                "real_ids",
                "id_mapping",
                "masked_to_original",
                "original_to_masked",
            }
            or normalized.startswith("future_")
            or normalized.startswith("target_")
            or normalized.startswith("truth_")
            or normalized.startswith("ground_truth")
        ):
            forbidden.append(str(column))
    return forbidden


def run_candidate(
    code: str | Mapping[str, Any],
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    metadata: Mapping[str, Any] | None = None,
    *,
    entrypoint: str = "fit_and_predict",
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    memory_limit_bytes: int | None = DEFAULT_MEMORY_LIMIT_BYTES,
    scratch_root: str | Path | None = None,
    id_column: str = "id",
) -> CodeExecutionResult:
    """Execute frozen model source and validate the complete output contract."""

    source = str(code.get("code") or "") if isinstance(code, Mapping) else str(code)
    if isinstance(code, Mapping):
        entrypoint = str(code.get("entrypoint") or entrypoint)
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if entrypoint != "fit_and_predict":
        return CodeExecutionResult(
            valid=False,
            status="entrypoint_error",
            errors=["entrypoint must be fit_and_predict"],
            code_sha256=digest,
        )
    policy_errors = check_code_policy(source)
    if policy_errors:
        return CodeExecutionResult(
            valid=False,
            status="policy_error",
            errors=policy_errors,
            code_sha256=digest,
            checks={
                "syntax_valid": not any("syntax error" in e for e in policy_errors),
                "imports_valid": _imports_valid(policy_errors),
                "policy_valid": False,
            },
        )
    if timeout_seconds <= 0 or not np.isfinite(timeout_seconds):
        raise ValueError("timeout_seconds must be positive and finite")
    if not all(isinstance(frame, pd.DataFrame) for frame in (train_df, valid_df, predict_df)):
        raise TypeError("train_df, valid_df, and predict_df must be pandas DataFrames")
    for frame_name, frame in (("train_df", train_df), ("valid_df", valid_df), ("predict_df", predict_df)):
        if frame.columns.duplicated().any():
            duplicate_columns = [str(column) for column in frame.columns[frame.columns.duplicated()].tolist()]
            return CodeExecutionResult(
                valid=False,
                status="input_error",
                errors=[f"{frame_name} contains duplicate columns: {duplicate_columns}"],
                code_sha256=digest,
                checks={"syntax_valid": True, "imports_valid": True, "policy_valid": True, "future_labels_absent": True},
            )
    for frame_name, frame in (("train_df", train_df), ("valid_df", valid_df)):
        forbidden_historical = _forbidden_historical_columns(frame.columns)
        if forbidden_historical:
            return CodeExecutionResult(
                valid=False,
                status="input_leakage",
                errors=[
                    f"{frame_name} contains forbidden future/alias/calendar columns: "
                    f"{forbidden_historical}"
                ],
                code_sha256=digest,
                checks={
                    "syntax_valid": True,
                    "policy_valid": True,
                    "future_labels_absent": False,
                },
            )
    declared_errors = _declared_input_errors(code, train_df, valid_df, predict_df)
    if declared_errors:
        return CodeExecutionResult(
            valid=False,
            status="input_error",
            errors=declared_errors,
            code_sha256=digest,
            checks={"syntax_valid": True, "imports_valid": True, "policy_valid": True, "future_labels_absent": True},
        )
    forbidden_predict = _forbidden_predict_columns(predict_df.columns)
    if forbidden_predict:
        return CodeExecutionResult(
            valid=False,
            status="input_leakage",
            errors=[f"predict_df contains label/future-return columns: {forbidden_predict}"],
            code_sha256=digest,
            checks={"syntax_valid": True, "imports_valid": True, "policy_valid": True, "future_labels_absent": False},
        )
    forbidden_predict_calendar = _forbidden_calendar_columns(predict_df.columns)
    if forbidden_predict_calendar:
        return CodeExecutionResult(
            valid=False,
            status="input_leakage",
            errors=[
                "predict_df contains evaluator-only calendar columns: "
                f"{forbidden_predict_calendar}"
            ],
            code_sha256=digest,
            checks={"syntax_valid": True, "imports_valid": True, "policy_valid": True, "future_labels_absent": False},
        )
    root = Path(scratch_root).resolve() if scratch_root is not None else None
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    temporary = tempfile.TemporaryDirectory(prefix="task_c_exec_", dir=str(root) if root else None)
    try:
        scratch = Path(temporary.name)
        input_path = scratch / "input.pkl"
        source_path = scratch / "candidate.py"
        runner_path = scratch / "runner.py"
        output_path = scratch / "output.json"
        source_path.write_text(source, encoding="utf-8")
        runner_path.write_text(_CHILD_RUNNER, encoding="utf-8")
        safe_metadata = _safe_runtime_metadata(metadata)
        payload = {
            "train_df": train_df.copy(deep=True),
            "valid_df": valid_df.copy(deep=True),
            "predict_df": predict_df.copy(deep=True),
            "metadata": safe_metadata,
        }
        with input_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        try:
            seed = int(safe_metadata.get("seed", 1))
        except (TypeError, ValueError):
            seed = 1
        try:
            # `-W ignore` is the parent-side half of warning suppression: the child
            # also installs `warnings.filterwarnings("ignore")`, but candidate code
            # may call `resetwarnings()`, and a deprecation flood is pure noise that
            # would otherwise eat the stdio budget it shares with the real traceback.
            command = [
                sys.executable,
                "-W",
                "ignore",
                "-I",
                str(runner_path),
                str(input_path),
                str(source_path),
                str(output_path),
            ]
            if memory_limit_bytes:
                command.append(str(int(memory_limit_bytes)))
            completed = subprocess.run(
                command,
                cwd=scratch,
                env=_child_environment(scratch, seed),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                errors="replace",
                timeout=float(timeout_seconds),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return CodeExecutionResult(
                valid=False,
                status="timeout",
                errors=[f"candidate exceeded {float(timeout_seconds):g} seconds"],
                code_sha256=digest,
                runtime_seconds=round(time.perf_counter() - started, 6),
                timed_out=True,
                stdout=_bounded(exc.stdout or ""),
                stderr=_bounded(_with_diagnostic(scratch, exc.stderr or "")),
                checks={"syntax_valid": True, "imports_valid": True, "policy_valid": True, "future_labels_absent": True},
            )
        runtime = round(time.perf_counter() - started, 6)
        stdout = _bounded(completed.stdout)
        stderr = _bounded(_with_diagnostic(scratch, completed.stderr))
        if completed.returncode != 0 or not output_path.exists():
            message = f"candidate subprocess exited with code {completed.returncode}"
            return CodeExecutionResult(
                valid=False,
                status="runtime_error",
                errors=[message],
                code_sha256=digest,
                runtime_seconds=runtime,
                return_code=int(completed.returncode),
                stdout=stdout,
                stderr=stderr,
                checks={"syntax_valid": True, "imports_valid": True, "policy_valid": True, "future_labels_absent": True},
            )
        try:
            value = json.loads(output_path.read_text(encoding="utf-8"))
            if not isinstance(value, list):
                raise TypeError("candidate output JSON must be an array")
            value = pd.DataFrame(value)
        except Exception as exc:
            return CodeExecutionResult(
                valid=False,
                status="serialization_error",
                errors=[f"could not read candidate output: {type(exc).__name__}: {exc}"],
                code_sha256=digest,
                runtime_seconds=runtime,
                return_code=int(completed.returncode),
                stdout=stdout,
                stderr=stderr,
                checks={"syntax_valid": True, "imports_valid": True, "policy_valid": True, "future_labels_absent": True},
            )
        if isinstance(code, Mapping):
            declared_output = code.get("output_contract")
            declared_columns = declared_output.get("columns") if isinstance(declared_output, Mapping) else None
            if isinstance(declared_columns, list):
                missing_declared = [
                    str(column) for column in declared_columns if str(column) not in value.columns
                ]
                if missing_declared:
                    return CodeExecutionResult(
                        valid=False,
                        status="output_error",
                        errors=[
                            "prediction omits columns declared in output_contract: "
                            f"{missing_declared[:20]}"
                        ],
                        code_sha256=digest,
                        runtime_seconds=runtime,
                        return_code=int(completed.returncode),
                        stdout=stdout,
                        stderr=stderr,
                        checks={
                            "syntax_valid": True,
                            "imports_valid": True,
                            "policy_valid": True,
                            "future_labels_absent": True,
                        },
                    )
        prediction, output_errors, checks = _validate_prediction(value, predict_df, id_column=id_column)
        checks.update({"syntax_valid": True, "imports_valid": True, "policy_valid": True, "future_labels_absent": True})
        if output_errors:
            return CodeExecutionResult(
                valid=False,
                status="output_error",
                errors=output_errors,
                code_sha256=digest,
                runtime_seconds=runtime,
                return_code=int(completed.returncode),
                stdout=stdout,
                stderr=stderr,
                checks=checks,
            )
        return CodeExecutionResult(
            valid=True,
            status="success",
            prediction=prediction,
            code_sha256=digest,
            runtime_seconds=runtime,
            return_code=int(completed.returncode),
            stdout=stdout,
            stderr=stderr,
            checks=checks,
        )
    finally:
        temporary.cleanup()


__all__ = [
    "ALLOWED_IMPORT_ROOTS",
    "CodeExecutionResult",
    "FORBIDDEN_IMPORT_PREFIXES",
    "MAX_CODE_CHARS",
    "check_code_policy",
    "run_candidate",
]

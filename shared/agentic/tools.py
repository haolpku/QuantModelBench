"""Generic, bounded tools for the code-assisted benchmark.

The model is given only generic tools.  ``python_exec`` receives a copy of
the public payload as the in-process variable ``payload`` and runs in a
separate subprocess.  ``read_artifact`` and ``write_artifact`` are scoped to
the case scratch directory.  These checks are a useful research guardrail,
not a substitute for a container/VM when executing untrusted code at scale.
"""

from __future__ import annotations

import ast
import builtins
import hashlib
import importlib
from importlib import metadata as importlib_metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping


MAX_CODE_CHARS = 20_000
MAX_PURPOSE_CHARS = 160
MAX_TOOL_OUTPUT_CHARS = 12_000
MAX_READ_CHARS = 20_000
MAX_ARTIFACT_BYTES = 2_000_000
MAX_ARTIFACT_COUNT = 64
MAX_ARTIFACT_TOTAL_BYTES = 8_000_000
MAX_TIMEOUT_SECONDS = 30


def runtime_metadata() -> dict[str, str | None]:
    """Return the small reproducibility fingerprint for tool subprocesses."""

    versions: dict[str, str | None] = {"python": platform.python_version()}
    for package, key in (
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scipy", "scipy"),
        ("scikit-learn", "sklearn"),
    ):
        try:
            versions[key] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[key] = None
    return versions

# These are intentionally broad, task-independent numerical/scientific
# packages.  Task-specific evaluators are never exposed as tools.
# The list must stay aligned with the generated-code whitelist in
# ``task_c_code_runner.ALLOWED_IMPORT_ROOTS``: otherwise a model that inspects
# the data with ``import xgboost`` inside ``python_exec`` sees a failure that
# says nothing about the code it finally submits.
ALLOWED_IMPORT_ROOTS = {
    "abc",
    "bisect",
    "catboost",
    "collections",
    "copy",
    "csv",
    "dataclasses",
    "functools",
    "itertools",
    "json",
    "lightgbm",
    "math",
    "numpy",
    "numbers",
    "optuna",
    "pandas",
    "random",
    "re",
    "scipy",
    "sklearn",
    "statistics",
    "statsmodels",
    "time",
    "torch",
    "typing",
    "warnings",
    "xgboost",
    # Declared runtime dependencies of the model packages above (torch's
    # typing_extensions/sympy/filelock/fsspec/networkx/jinja2 chain and the
    # boosters' shared helpers) so a whitelisted import cannot fail on its own
    # transitive imports.
    "typing_extensions",
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

BLOCKED_IMPORT_ROOTS = {
    "builtins",
    "ctypes",
    "glob",
    "importlib",
    "multiprocessing",
    "os",
    "pathlib",
    "pickle",
    "requests",
    "shutil",
    "socket",
    "subprocess",
    "sys",
    "tempfile",
    "urllib",
    "yaml",
}

BLOCKED_NAMES = {
    "__builtins__",
    "__import__",
    "breakpoint",
    "compile",
    "dir",
    "eval",
    "exec",
    "exit",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "open",
    "quit",
    "setattr",
    "vars",
}

# Methods that can turn an otherwise harmless numerical object into a file or
# network side effect.  ``json.loads`` and ordinary model methods remain fine.
# ``read_*`` names are deliberately absent: reading is neither a write nor an
# escape primitive, the evaluator copies exactly the declared public
# train/validation files into the case workspace, and both task prompts point
# candidates at ``pd.read_parquet``/``pd.read_json`` for row-level inspection.
BLOCKED_ATTRIBUTE_NAMES = {
    "dump",
    "dumps",
    "fromfile",
    "genfromtxt",
    "load",
    "loads",
    "loadtxt",
    "memmap",
    "save",
    "savetxt",
    "savez",
    "savez_compressed",
    "tofile",
    "to_csv",
    "to_excel",
    "to_feather",
    "to_json",
    "to_parquet",
    "to_pickle",
    "to_sql",
    "unlink",
    "write_text",
    "write_bytes",
}


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "python_exec",
            "description": (
                "Run deterministic numerical Python on the public payload only. "
                "The variables payload, numpy as np, pandas as pd, and a fixed seed are available."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "code": {
                        "type": "string",
                        "maxLength": MAX_CODE_CHARS,
                        "description": "Python source; print a compact result rather than a large table.",
                    },
                    "timeout_seconds": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_TIMEOUT_SECONDS,
                    },
                    "purpose": {
                        "type": "string",
                        "maxLength": MAX_PURPOSE_CHARS,
                    },
                },
                "required": ["code", "timeout_seconds", "purpose"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_artifact",
            "description": "Read a small text/JSON/CSV artifact produced in this case scratch directory.",
            "strict": True,
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "path": {"type": "string", "maxLength": 240},
                    "max_chars": {"type": "integer", "minimum": 1, "maximum": MAX_READ_CHARS},
                },
                "required": ["path", "max_chars"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_artifact",
            "description": "Write a compact diagnostic/table/plot description to the case artifact directory.",
            "strict": True,
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "path": {"type": "string", "maxLength": 240},
                    "content": {"type": "string", "maxLength": MAX_ARTIFACT_BYTES},
                    "format": {"type": "string", "enum": ["text", "json", "csv"]},
                },
                "required": ["path", "content", "format"],
            },
        },
    },
]


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _bounded_text(value: Any, limit: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    if value is None:
        text = ""
    elif isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated at {limit} characters]"


def _safe_relative_path(raw_path: Any) -> Path:
    """Validate a relative artifact path without resolving it against cwd."""

    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("path must be a non-empty relative string")
    candidate = raw_path.strip().replace("\\", "/")
    path = Path(candidate)
    # ``Path.is_absolute`` does not catch a Windows drive on all platforms.
    if path.is_absolute() or path.drive or candidate.startswith("/"):
        raise ValueError("absolute artifact paths are not allowed")
    parts = [part for part in path.parts if part not in ("", ".")]
    if not parts or any(part == ".." for part in parts):
        raise ValueError("artifact path must stay inside the case directory")
    if any("\x00" in part for part in parts):
        raise ValueError("artifact path contains a NUL byte")
    return Path(*parts)


def _validate_python_code(code: Any) -> str:
    if not isinstance(code, str):
        raise ValueError("code must be a string")
    if not code.strip():
        raise ValueError("code must not be empty")
    if len(code) > MAX_CODE_CHARS:
        raise ValueError(f"code exceeds {MAX_CODE_CHARS} characters")
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise ValueError(f"Python syntax error: {exc}") from exc
    if sum(1 for _ in ast.walk(tree)) > 20_000:
        raise ValueError("code has too many syntax-tree nodes")

    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            name = str(node.id)
            if name in BLOCKED_NAMES or name.startswith("__"):
                raise ValueError(f"name is not allowed: {name}")
        elif isinstance(node, ast.Attribute):
            attribute = str(node.attr)
            json_serialization = (
                isinstance(node.value, ast.Name)
                and node.value.id == "json"
                and attribute in {"dump", "dumps", "load", "loads"}
            )
            if attribute.startswith("__") or (
                attribute in BLOCKED_ATTRIBUTE_NAMES and not json_serialization
            ):
                raise ValueError(f"attribute is not allowed: {attribute}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                root = str(alias.name).split(".", 1)[0]
                if root not in ALLOWED_IMPORT_ROOTS or root in BLOCKED_IMPORT_ROOTS:
                    raise ValueError(f"import is not allowed: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise ValueError("relative imports are not allowed")
            root = str(node.module or "").split(".", 1)[0]
            if root not in ALLOWED_IMPORT_ROOTS or root in BLOCKED_IMPORT_ROOTS:
                raise ValueError(f"import is not allowed: {node.module}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            # Catch common attempts to point a reader at hidden benchmark files
            # before the code reaches the subprocess.
            lowered = node.value.lower().replace("\\", "/")
            hidden_path_markers = (
                "truth.jsonl",
                "ground_truth.json",
                "/truth/",
                "/ground_truth/",
            )
            if any(marker in lowered for marker in hidden_path_markers):
                raise ValueError("references to hidden truth artifacts are not allowed")
    return code


def _safe_import(name: str, globals_: Any = None, locals_: Any = None, fromlist: tuple[str, ...] = (), level: int = 0) -> Any:
    if level:
        raise ImportError("relative imports are disabled")
    root = str(name).split(".", 1)[0]
    if root not in ALLOWED_IMPORT_ROOTS or root in BLOCKED_IMPORT_ROOTS:
        raise ImportError(f"import is not allowed: {name}")
    return importlib.import_module(name)


def _safe_builtins() -> dict[str, Any]:
    names = (
        "abs",
        "all",
        "any",
        "BaseException",
        "bool",
        "bytes",
        "bytearray",
        "complex",
        "dict",
        "enumerate",
        "Exception",
        "IndexError",
        "float",
        "frozenset",
        "int",
        "isinstance",
        "issubclass",
        "iter",
        "KeyError",
        "len",
        "LookupError",
        "list",
        "map",
        "max",
        "min",
        "next",
        "object",
        "print",
        "range",
        "reversed",
        "round",
        "RuntimeError",
        "set",
        "slice",
        "sorted",
        "str",
        "sum",
        "type",
        "tuple",
        "TypeError",
        "ValueError",
        "zip",
    )
    result = {name: getattr(builtins, name) for name in names}
    # Class definitions need this interpreter hook implicitly; user code
    # cannot name it directly because dunder names remain blocked by the AST
    # policy below.
    result["__build_class__"] = builtins.__build_class__
    result["__import__"] = _safe_import
    return result


def _minimal_child_environment(scratch: Path) -> dict[str, str]:
    """Keep dynamic code away from project secrets and unrelated settings."""

    result: dict[str, str] = {}
    for key in ("PATH", "SystemRoot", "WINDIR", "LD_LIBRARY_PATH"):
        value = os.environ.get(key)
        if value:
            result[key] = value
    result.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONIOENCODING": "utf-8",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "TMP": str(scratch),
            "TEMP": str(scratch),
        }
    )
    return result


class ScratchWorkspace:
    """One isolated scratch/artifact directory for one benchmark case."""

    def __init__(
        self,
        root: Path,
        payload: Mapping[str, Any],
        *,
        keep_artifacts: bool = True,
        seed: int | None = 0,
        public_artifact_files: Mapping[str, str] | None = None,
    ) -> None:
        self.root = root.resolve()
        self.keep_artifacts = bool(keep_artifacts)
        self.seed = seed
        self.runtime = runtime_metadata()
        self.root.mkdir(parents=True, exist_ok=True)
        self.artifact_root = self.root / "artifacts"
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        # Task C may provide large, evaluator-owned historical artifacts.  They
        # live outside the writable artifact directory and are copied into a
        # read-only-by-convention ``public`` subtree for each case.  Keeping
        # this optional key here preserves the A/B API while allowing the same
        # python_exec tool to materialize Task C train/validation rows.
        payload_copy = dict(payload)
        # ``public_artifact_files`` is an evaluator-side mount specification.
        # It is deliberately separate from the payload sent to the provider so
        # local absolute paths never enter the prompt or payload hash.  The
        # private ``_public_artifact_files`` key remains accepted for backwards
        # compatibility with callers that construct a ScratchWorkspace
        # directly.
        public_specs = public_artifact_files
        private_specs = payload_copy.get("_public_artifact_files")
        if public_specs is None and isinstance(private_specs, Mapping):
            public_specs = private_specs
        if isinstance(public_specs, Mapping):
            staged: dict[str, str] = {}
            manifest: list[dict[str, Any]] = []
            public_root = self.root / "public"
            for split, raw_path in public_specs.items():
                if str(split) not in {"train", "validation"}:
                    raise ValueError("only train and validation public artifacts may be mounted")
                if not isinstance(raw_path, str) or not raw_path.strip():
                    raise ValueError(f"public artifact path for {split!r} must be a non-empty string")
                source = Path(raw_path).expanduser().resolve()
                if not source.is_file():
                    raise FileNotFoundError(f"public artifact does not exist: {source}")
                target = public_root / str(split) / source.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                staged[str(split)] = target.relative_to(self.root).as_posix()
                digest = hashlib.sha256()
                with source.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                digest_text = digest.hexdigest()
                descriptor = payload_copy.get(str(split), {})
                descriptor = descriptor.get("artifact", {}) if isinstance(descriptor, Mapping) else {}
                if isinstance(descriptor, Mapping) and descriptor.get("bytes") is not None:
                    if int(descriptor["bytes"]) != int(source.stat().st_size):
                        raise ValueError(f"public {split} artifact byte count does not match its descriptor")
                if isinstance(descriptor, Mapping) and descriptor.get("sha256"):
                    if str(descriptor["sha256"]).lower() != digest_text:
                        raise ValueError(f"public {split} artifact sha256 does not match its descriptor")
                manifest.append(
                    {
                        "split": str(split),
                        "source_name": source.name,
                        "staged_path": staged[str(split)],
                        "bytes": int(source.stat().st_size),
                        "sha256": digest_text,
                    }
                )
            # This key is written only to the evaluator-local payload file that
            # python_exec reads. It is not present in the provider request.
            payload_copy["_public_artifact_files"] = staged
            self.public_artifact_manifest = manifest
        else:
            self.public_artifact_manifest = []
        self.payload_path = self.root / "payload.json"
        self.payload_path.write_text(
            json.dumps(payload_copy, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        self._execution_index = 0

    def _artifact_path(self, raw_path: Any) -> tuple[Path, Path]:
        relative = _safe_relative_path(raw_path)
        target = (self.artifact_root / relative).resolve()
        artifact_root = self.artifact_root.resolve()
        if target != artifact_root and artifact_root not in target.parents:
            raise ValueError("artifact path escapes the case artifact directory")
        return relative, target

    def write_artifact(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        relative, target = self._artifact_path(arguments.get("path"))
        content = arguments.get("content")
        if not isinstance(content, str):
            raise ValueError("content must be a string")
        format_name = str(arguments.get("format") or "text").lower()
        if format_name not in {"text", "json", "csv"}:
            raise ValueError("format must be text, json, or csv")
        encoded = content.encode("utf-8")
        if len(encoded) > MAX_ARTIFACT_BYTES:
            raise ValueError(f"artifact exceeds {MAX_ARTIFACT_BYTES} bytes")
        existing_files = [path for path in self.artifact_root.rglob("*") if path.is_file() and not path.is_symlink()]
        existing_bytes = sum(path.stat().st_size for path in existing_files)
        replacing_bytes = target.stat().st_size if target.exists() and target.is_file() else 0
        if target not in existing_files and len(existing_files) >= MAX_ARTIFACT_COUNT:
            raise ValueError(f"case artifact count exceeds {MAX_ARTIFACT_COUNT}")
        if existing_bytes - replacing_bytes + len(encoded) > MAX_ARTIFACT_TOTAL_BYTES:
            raise ValueError(f"case artifacts exceed {MAX_ARTIFACT_TOTAL_BYTES} bytes")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        temporary.write_bytes(encoded)
        os.replace(temporary, target)
        return {
            "ok": True,
            "path": relative.as_posix(),
            "bytes": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "format": format_name,
        }

    def read_artifact(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        relative, target = self._artifact_path(arguments.get("path"))
        if not target.exists() or not target.is_file():
            raise FileNotFoundError(f"artifact does not exist: {relative.as_posix()}")
        max_chars = int(arguments.get("max_chars", MAX_READ_CHARS))
        max_chars = max(1, min(max_chars, MAX_READ_CHARS))
        size = target.stat().st_size
        if size > MAX_ARTIFACT_BYTES:
            raise ValueError("artifact is too large to read")
        raw = target.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return {
                "ok": True,
                "path": relative.as_posix(),
                "binary": True,
                "bytes": size,
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        return {
            "ok": True,
            "path": relative.as_posix(),
            "binary": False,
            "bytes": size,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "truncated": len(text) > max_chars,
            "content": _bounded_text(text, max_chars),
        }

    def _list_artifacts(self) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        if not self.artifact_root.exists():
            return output
        for path in sorted(p for p in self.artifact_root.rglob("*") if p.is_file()):
            if path.is_symlink():
                # Do not hash or promote a link that could point outside the
                # case workspace.
                continue
            raw = path.read_bytes()
            output.append(
                {
                    "path": path.relative_to(self.artifact_root).as_posix(),
                    "bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
        return output

    def python_exec(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        code = _validate_python_code(arguments.get("code"))
        purpose = str(arguments.get("purpose") or "")[:MAX_PURPOSE_CHARS]
        try:
            timeout = int(arguments.get("timeout_seconds", 30))
        except (TypeError, ValueError) as exc:
            raise ValueError("timeout_seconds must be an integer") from exc
        timeout = max(1, min(timeout, MAX_TIMEOUT_SECONDS))
        self._execution_index += 1
        stem = f"_python_exec_{self._execution_index:02d}"
        code_path = self.root / f"{stem}.py"
        runner_path = self.root / f"{stem}_runner.py"
        code_path.write_text(code, encoding="utf-8")
        # The wrapper imports scientific packages before replacing builtins for
        # user code.  The user namespace gets only the public payload and
        # numerical helpers; no truth path or project root is supplied.
        allowed_builtin_names = [
            name for name in _safe_builtins()
            if name != "__import__"
        ]
        allowed_imports = sorted(ALLOWED_IMPORT_ROOTS)
        runner = f'''\
import builtins as _builtins
import importlib as _importlib
import json as _json
import math as _math
import pathlib as _pathlib
import sys as _sys
import traceback as _traceback
import numpy as _np
import pandas as _pd

_OUTPUT_LIMIT = {MAX_TOOL_OUTPUT_CHARS!r}

class _BoundedStream:
    def __init__(self, target):
        self.target = target
        self.count = 0

    def write(self, value):
        text = str(value)
        remaining = _OUTPUT_LIMIT - self.count
        if remaining <= 0:
            raise RuntimeError("tool output exceeded the configured limit")
        if len(text) > remaining:
            self.target.write(text[:remaining])
            self.target.flush()
            self.count += remaining
            raise RuntimeError("tool output exceeded the configured limit")
        self.target.write(text)
        self.target.flush()
        self.count += len(text)
        return len(text)

    def flush(self):
        self.target.flush()

    @property
    def encoding(self):
        return getattr(self.target, "encoding", "utf-8")

_sys.stdout = _BoundedStream(_sys.__stdout__)
_sys.stderr = _BoundedStream(_sys.__stderr__)
_seed = {self.seed!r}
if _seed is not None:
    _np.random.seed(_seed)

_payload = _json.loads(_pathlib.Path({str(self.payload_path)!r}).read_text(encoding="utf-8"))

# A Task C artifact-only payload keeps the chat message compact.  Materialize
# only the evaluator-declared public train/validation files into the in-memory
# payload visible to this one python_exec call.  The hidden test panel has no
# descriptor and therefore cannot cross this boundary.
_public_files = _payload.get("_public_artifact_files", {{}})
if isinstance(_public_files, dict):
    for _split, _relative in list(_public_files.items()):
        if not isinstance(_relative, str):
            continue
        _source = _pathlib.Path({str(self.root)!r}) / _relative
        if not _source.is_file() or _split not in _payload or not isinstance(_payload[_split], dict):
            continue
        _format = str(_payload[_split].get("artifact", {{}}).get("format", _source.suffix.lstrip("."))).lower()
        if _format in {{"jsonl", "ndjson"}}:
            _rows_frame = _pd.read_json(_source, lines=True)
        elif _format == "csv":
            _rows_frame = _pd.read_csv(_source)
        elif _format in {{"parquet", "pq"}}:
            _rows_frame = _pd.read_parquet(_source)
        else:
            raise ValueError("unsupported public Task C artifact format: " + _format)
        _descriptor = _payload[_split].get("artifact", {{}})
        if _payload.get("task") == "stock_multifactor_model":
            # Artifact-only Task C payloads are not traversed by the outer
            # JSON validator. Audit the materialized header before exposing
            # rows to python_exec, otherwise a hidden target/calendar/mapping
            # column could bypass the public-payload boundary merely by being
            # moved from embedded JSON into a large file.
            def _normalise_public_column(_value):
                return "".join(
                    _char if _char.isalnum() else "_"
                    for _char in str(_value).strip().lower()
                ).strip("_")
            _protected = {{
                "truth", "ground_truth", "future", "test", "test_rows", "test_labels",
                "test_returns", "real_id", "real_ids", "id_mapping",
                "masked_to_original", "original_to_masked",
                "future_labels_included", "public_artifact_files",
            }}
            _bad_columns = [
                str(_column) for _column in _rows_frame.columns
                if (
                    _normalise_public_column(_column) in _protected
                    or _normalise_public_column(_column).startswith(
                        ("truth_", "ground_truth", "future_", "target_")
                    )
                    or _normalise_public_column(_column) in {{
                        "eom", "date", "month", "anchor_date", "formation_eom",
                        "formation_month", "origin_month", "as_of_month",
                    }}
                )
            ]
            if _bad_columns:
                raise ValueError(
                    "public Task C artifact contains protected columns: "
                    + str(_bad_columns[:20])
                )
            if "id" not in _rows_frame.columns or "time_key" not in _rows_frame.columns:
                raise ValueError("public Task C artifact requires id and time_key")
            _label_aliases = [
                _column for _column in _rows_frame.columns
                if _normalise_public_column(_column) in {{
                    "label", "labels", "target", "y", "next_return",
                    "next_month_return", "ret_exc_lead1m", "ret_exc_lead_1m",
                    "lead_return", "return_next_month",
                }}
            ]
            if len(_label_aliases) != 1:
                raise ValueError("historical Task C artifact requires exactly one label column")
            if _label_aliases[0] != "label":
                _rows_frame = _rows_frame.rename(columns={{_label_aliases[0]: "label"}})
        if isinstance(_descriptor, dict) and _descriptor.get("rows") is not None:
            try:
                _expected_rows = int(_descriptor["rows"])
            except (TypeError, ValueError) as _exc:
                raise ValueError("public artifact rows descriptor must be an integer") from _exc
            if _expected_rows != len(_rows_frame):
                raise ValueError(
                    "public artifact row count mismatch: expected "
                    + str(_expected_rows) + ", got " + str(len(_rows_frame))
                )
        _payload[_split]["rows"] = _rows_frame.to_dict("records")

def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level:
        raise ImportError("relative imports are disabled")
    root = str(name).split(".", 1)[0]
    if root not in {allowed_imports!r}:
        raise ImportError("import is not allowed: " + str(name))
    return _importlib.import_module(name)

_allowed = {{
    name: getattr(_builtins, name)
    for name in {allowed_builtin_names!r}
}}
_allowed["__import__"] = _safe_import

_namespace = {{
    "__name__": "__agent_code__",
    "__builtins__": _allowed,
    "payload": _payload,
    "np": _np,
    "pd": _pd,
    "math": _math,
    "json": _json,
    "seed": _seed,
}}
try:
    _source = _pathlib.Path({str(code_path)!r}).read_text(encoding="utf-8")
    exec(compile(_source, {str(code_path)!r}, "exec"), _namespace, _namespace)
except BaseException as _exc:
    _traceback.print_exc(file=_sys.__stderr__)
    raise
'''
        runner_path.write_text(runner, encoding="utf-8")
        command = [sys.executable, str(runner_path)]
        timed_out = False
        try:
            completed = subprocess.run(
                command,
                cwd=str(self.root),
                env=_minimal_child_environment(self.root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=False,
            )
            exit_code = int(completed.returncode)
            stdout = completed.stdout.decode("utf-8", errors="replace")
            stderr = completed.stderr.decode("utf-8", errors="replace")
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            exit_code = 124
            stdout = (exc.stdout or b"").decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else str(exc.stdout or "")
            stderr = (exc.stderr or b"").decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else str(exc.stderr or "")
            stderr = (stderr + "\nprocess timed out").strip()
        finally:
            # The payload and generated source are never promoted to the
            # persistent artifact directory.  They are removed after each call.
            code_path.unlink(missing_ok=True)
            runner_path.unlink(missing_ok=True)
        # Tracebacks from the evaluator subprocess contain the generated
        # scratch path.  It is an implementation detail, not useful evidence
        # for the provider, and can reveal local directory layout through a
        # tool result.  Replace it before the bounded result crosses the
        # agentic boundary.
        for _private_path in (str(self.root), str(self.payload_path)):
            if _private_path:
                stdout = stdout.replace(_private_path, "[scratch]")
                stderr = stderr.replace(_private_path, "[scratch]")
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
        return {
            "ok": exit_code == 0 and not timed_out,
            "exit_code": exit_code,
            "stdout": _bounded_text(stdout),
            "stderr": _bounded_text(stderr),
            "artifacts": self._list_artifacts(),
            "elapsed_ms": elapsed_ms,
            "code_sha256": _sha256_text(code),
            "purpose": purpose,
            "timed_out": timed_out,
            "runtime": dict(self.runtime),
        }

    def dispatch(self, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        if name == "python_exec":
            return self.python_exec(arguments)
        if name == "read_artifact":
            return self.read_artifact(arguments)
        if name == "write_artifact":
            return self.write_artifact(arguments)
        raise ValueError(f"unknown tool: {name}")

    def finalize(self, destination: Path | None) -> dict[str, Any]:
        """Promote explicitly written artifacts and remove temporary inputs."""

        artifacts = self._list_artifacts()
        promoted = False
        if destination is not None and self.keep_artifacts and artifacts:
            destination = destination.resolve()
            destination.mkdir(parents=True, exist_ok=True)
            for source in self.artifact_root.rglob("*"):
                if not source.is_file() or source.is_symlink():
                    continue
                relative = source.relative_to(self.artifact_root)
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            promoted = True
        shutil.rmtree(self.root, ignore_errors=True)
        return {"artifacts": artifacts, "promoted": promoted}


class ToolDispatcher:
    """Dispatch tool arguments and turn failures into model-readable results."""

    _ARGUMENT_KEYS = {
        "python_exec": {"code", "timeout_seconds", "purpose"},
        "read_artifact": {"path", "max_chars"},
        "write_artifact": {"path", "content", "format"},
    }

    def __init__(self, workspace: ScratchWorkspace) -> None:
        self.workspace = workspace

    def dispatch(self, name: str, arguments: Any) -> dict[str, Any]:
        if not isinstance(arguments, Mapping):
            return {"ok": False, "error_type": "invalid_arguments", "error": "arguments must be an object"}
        expected = self._ARGUMENT_KEYS.get(name)
        if expected is None:
            return {"ok": False, "error_type": "unknown_tool", "error": f"unknown tool: {name}"}
        unknown = sorted(str(key) for key in arguments.keys() if str(key) not in expected)
        if unknown:
            return {
                "ok": False,
                "error_type": "invalid_arguments",
                "error": "unexpected argument(s): " + ", ".join(unknown),
            }
        missing = sorted(key for key in expected if key not in arguments)
        if missing:
            return {
                "ok": False,
                "error_type": "invalid_arguments",
                "error": "missing argument(s): " + ", ".join(missing),
            }
        try:
            result = self.workspace.dispatch(name, arguments)
            return result if isinstance(result, dict) else {"ok": True, "result": result}
        except Exception as exc:  # noqa: BLE001 - tool errors are returned to the model
            return {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": _bounded_text(str(exc), 2_000),
            }

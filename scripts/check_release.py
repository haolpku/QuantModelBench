#!/usr/bin/env python3
"""Check tracked public-release files without printing possible secret values."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "credential-shaped token": re.compile(r"\b(?:hf_|gh[pousr]_|github_pat_|sk-)[A-Za-z0-9_-]{20,}"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "machine-specific path": re.compile(r"/(?:Users|home|vepfs[^/]*)/[^\s\"']+"),
    "embedded SSH host": re.compile(r"\b(?:root|ubuntu)@(?:\d{1,3}\.){3}\d{1,3}"),
}


def main() -> int:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT, check=True, capture_output=True,
    )
    paths = sorted(set(p for p in result.stdout.decode().split("\0") if p))
    errors = []
    for name in paths:
        path = ROOT / name
        if not path.is_file():
            errors.append(f"{name}: missing file")
            continue
        if (name.startswith("data/") and name != "data/README.md") or any(
            part in {"runs", "outputs", "outputs_test", ".venv"} for part in path.relative_to(ROOT).parts
        ):
            errors.append(f"{name}: private/generated path in public release")
        if path.suffix in {".parquet", ".pickle", ".pkl", ".env", ".pem", ".key"} or path.name == ".env":
            errors.append(f"{name}: data/credential file in public release")
        if path.stat().st_size > 5_000_000:
            errors.append(f"{name}: unexpected large file")
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            errors.append(f"{name}: unexpected binary file")
            continue
        for number, line in enumerate(content.splitlines(), 1):
            for kind, pattern in PATTERNS.items():
                if pattern.search(line):
                    errors.append(f"{name}:{number}: {kind}")
        try:
            if path.suffix == ".py":
                ast.parse(content, filename=name)
            elif path.suffix == ".json":
                json.loads(content)
        except (SyntaxError, ValueError) as exc:
            errors.append(f"{name}: invalid {path.suffix} ({type(exc).__name__})")
    if errors:
        print("\n".join(errors))
        return 1
    print(f"Release check passed: {len(paths)} source/configuration/documentation files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

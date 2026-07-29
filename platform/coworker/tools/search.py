"""Fast code search (`grep`) — ripgrep when available, a Python walk otherwise.

ripgrep respects `.gitignore`, but only inside a git repository, so both engines are
additionally given the same `_IGNORE_DIRS` list of heavy dirs — the same query must not
return different results depending on whether rg is installed. Read-only, workspace-scoped.
Returns file:line:text.
"""

from __future__ import annotations

import fnmatch
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional

import aisuite as ai

_IGNORE_DIRS = {
    ".git",
    "node_modules",
    "target",
    "dist",
    "build",
    ".venv",
    "venv",
    "__pycache__",
    ".next",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".idea",
}

# Same exclusions for ripgrep. `.gitignore` only applies inside a git repository, and
# even there it won't list a dir the project happens not to ignore, so relying on it
# alone made results differ between the two engines.
_RG_IGNORE_GLOBS = [f"!**/{d}/**" for d in sorted(_IGNORE_DIRS)]

_SCHEMA = {
    "type": "function",
    "function": {
        "name": "grep",
        "description": (
            "Search the workspace for a regular-expression pattern and return matching lines as "
            "file:line:text. Fast and .gitignore-aware (skips node_modules, build dirs, etc.). "
            "Prefer this over reading files blindly to locate code. Read-only."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regular expression to search for.",
                },
                "path": {
                    "type": "string",
                    "description": "Subdirectory to search (default: whole workspace).",
                },
                "glob": {
                    "type": "string",
                    "description": "Optional filename glob filter, e.g. '*.py'.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max matches (default 100, max 1000).",
                },
            },
            "required": ["pattern"],
        },
    },
}


def search_tools(workspace: str) -> list:
    root = Path(workspace).resolve()

    def grep(
        pattern: str,
        path: str = ".",
        glob: Optional[str] = None,
        max_results: int = 100,
    ) -> dict[str, Any]:
        n = max_results if isinstance(max_results, int) and max_results > 0 else 100
        n = min(n, 1000)
        base = (root / (path or ".")).resolve()
        try:
            base.relative_to(root)  # keep searches inside the workspace
        except ValueError:
            return {"error": "path escapes the workspace"}

        rg = shutil.which("rg")
        if rg:
            cmd = [
                rg,
                "--line-number",
                "--no-heading",
                "--color=never",
                # NUL between the path and the line number: `base` is absolute, so on
                # Windows every match line starts with "D:\..." and a plain ":" split
                # would break on the drive letter.
                "--null",
                "--max-count",
                str(n),
                "-e",
                pattern,
            ]
            if glob:
                cmd += ["--glob", glob]
            # After the caller's glob: later globs win in ripgrep, so the heavy dirs
            # stay excluded even when `glob` would otherwise match inside them.
            for ignore in _RG_IGNORE_GLOBS:
                cmd += ["--glob", ignore]
            cmd.append(str(base))
            try:
                # Decode explicitly: `text=True` alone uses the platform's preferred
                # encoding, a legacy codepage on a stock Windows install, and any
                # matched line whose bytes that codec rejects makes subprocess drop
                # `stdout` to None -- the whole search is lost over one line.
                out = subprocess.run(
                    cmd,
                    capture_output=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=30,
                )
            except Exception as exc:
                return {"error": f"grep failed: {exc}"}
            if out.returncode not in (0, 1):  # 1 = no matches
                return {"error": (out.stderr or "ripgrep error").strip()[:300]}
            return {"engine": "ripgrep", **_parse_rg(out.stdout, root, n)}

        return {"engine": "python", **_py_grep(root, base, pattern, glob, n)}

    grep.__name__ = "grep"
    grep.__doc__ = _SCHEMA["function"]["description"]
    grep.__aisuite_tool_metadata__ = ai.ToolMetadata(
        name="grep",
        category="search",
        risk_level="low",
        capabilities=["search"],
        requires_approval=False,
    )
    grep.__coworker_schema__ = _SCHEMA
    return [grep]


def _rel(path: str, root: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(root))
    except (ValueError, OSError):
        return path


def _parse_rg(stdout: str, root: Path, n: int) -> dict[str, Any]:
    """Parse `rg --null --line-number --no-heading` output: PATH \\0 LINE ':' TEXT."""
    matches: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        # The NUL ends the path, so the path may contain ':' (a Windows drive
        # letter) without confusing the split. Only the line number is separated
        # by ':', and the text keeps every ':' it contained.
        f, sep, rest = line.partition("\0")
        if not sep:
            continue
        ln, sep, txt = rest.partition(":")
        if not sep:
            continue
        matches.append(
            {
                "file": _rel(f, root),
                "line": int(ln) if ln.isdigit() else 0,
                "text": txt[:300],
            }
        )
        if len(matches) >= n:
            break
    return {"count": len(matches), "matches": matches}


def _py_grep(
    root: Path, base: Path, pattern: str, glob: Optional[str], n: int
) -> dict[str, Any]:
    try:
        rx = re.compile(pattern)
    except re.error as exc:
        return {"error": f"invalid regex: {exc}", "count": 0, "matches": []}
    matches: list[dict[str, Any]] = []
    for dirpath, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in _IGNORE_DIRS]
        for fn in files:
            if glob and not fnmatch.fnmatch(fn, glob):
                continue
            fp = Path(dirpath) / fn
            try:
                with open(fp, "r", encoding="utf-8", errors="ignore") as fh:
                    for i, line in enumerate(fh, 1):
                        if rx.search(line):
                            matches.append(
                                {
                                    "file": _rel(str(fp), root),
                                    "line": i,
                                    "text": line.rstrip()[:300],
                                }
                            )
                            if len(matches) >= n:
                                return {"count": len(matches), "matches": matches}
            except OSError:
                continue
    return {"count": len(matches), "matches": matches}

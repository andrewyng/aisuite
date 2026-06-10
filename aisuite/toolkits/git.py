from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..agents import ToolMetadata, tool


def git(*, root: str | Path, max_output_chars: int = 20000) -> list:
    """Return root-scoped read-only git tools."""
    toolkit = GitToolkit(root=root, max_output_chars=max_output_chars)

    def git_status() -> dict[str, object]:
        """Return `git status --short --branch` for the configured root."""
        return toolkit.status()

    def git_diff(path: Optional[str] = None, staged: bool = False) -> dict[str, object]:
        """Return a read-only git diff for the configured root."""
        return toolkit.diff(path=path, staged=staged)

    return [
        tool(
            git_status,
            metadata=ToolMetadata(
                category="git",
                risk_level="low",
                capabilities=["git_status"],
            ),
        ),
        tool(
            git_diff,
            metadata=ToolMetadata(
                category="git",
                risk_level="low",
                capabilities=["git_diff"],
            ),
        ),
    ]


@dataclass
class GitToolkit:
    root: str | Path
    max_output_chars: int = 20000

    def __post_init__(self) -> None:
        self.root = Path(self.root).expanduser().resolve()

    def status(self) -> dict[str, object]:
        return self._run(["git", "status", "--short", "--branch"])

    def diff(
        self, path: Optional[str] = None, staged: bool = False
    ) -> dict[str, object]:
        args = ["git", "diff"]
        if staged:
            args.append("--staged")
        if path:
            self._resolve_path(path)
            args.extend(["--", path])
        return self._run(args)

    def _run(self, args: list[str]) -> dict[str, object]:
        completed = subprocess.run(
            args,
            cwd=self.root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        stdout = self._clip(completed.stdout)
        stderr = self._clip(completed.stderr)
        return {
            "command": " ".join(args),
            "exit_code": completed.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "truncated": len(completed.stdout) > len(stdout)
            or len(completed.stderr) > len(stderr),
        }

    def _resolve_path(self, path: str) -> Path:
        candidate = (self.root / path).expanduser().resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise PermissionError(f"Path escapes git toolkit root: {path}") from exc
        return candidate

    def _clip(self, value: str) -> str:
        if len(value) <= self.max_output_chars:
            return value
        return value[: self.max_output_chars - 3] + "..."

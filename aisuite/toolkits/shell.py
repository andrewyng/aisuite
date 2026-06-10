from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Optional

from ..agents import ToolMetadata, tool
from ..agents.context import get_active_run_context

UNSUPPORTED_SHELL_TOKENS = {"|", "||", "&&", ";", ">", ">>", "<", "<<", "2>", "2>>"}


def shell(
    *,
    cwd: str | Path,
    allowed_commands: Optional[list[str]] = None,
    allow_all: bool = False,
    allow_shell: bool = False,
    default_timeout_seconds: int = 30,
    max_output_chars: int = 20_000,
) -> list:
    """Return shell command tools scoped to a working directory."""
    toolkit = ShellToolkit(
        cwd=cwd,
        allowed_commands=allowed_commands,
        allow_all=allow_all,
        allow_shell=allow_shell,
        default_timeout_seconds=default_timeout_seconds,
        max_output_chars=max_output_chars,
    )

    def run_shell(command: str, timeout_seconds: Optional[int] = None) -> dict:
        """Run an allowed shell command in the configured working directory."""
        return toolkit.run_shell(
            command=command,
            timeout_seconds=timeout_seconds,
        )

    return [
        tool(
            run_shell,
            metadata=ToolMetadata(
                category="shell",
                risk_level="high",
                capabilities=["run_command"],
                requires_approval=True,
            ),
        )
    ]


class ShellToolkit:
    def __init__(
        self,
        *,
        cwd: str | Path,
        allowed_commands: Optional[list[str]],
        allow_all: bool,
        allow_shell: bool,
        default_timeout_seconds: int,
        max_output_chars: int,
    ):
        if not allow_all and not allowed_commands:
            raise ValueError(
                "Shell toolkit requires allowed_commands or allow_all=True."
            )

        self.cwd = Path(cwd).expanduser().resolve()
        if not self.cwd.exists():
            raise ValueError(f"cwd does not exist: {cwd}")
        if not self.cwd.is_dir():
            raise ValueError(f"cwd is not a directory: {cwd}")

        self.allowed_commands = allowed_commands or []
        self.allow_all = allow_all
        self.allow_shell = allow_shell
        self.default_timeout_seconds = default_timeout_seconds
        self.max_output_chars = max_output_chars

    def run_shell(self, command: str, timeout_seconds: Optional[int] = None) -> dict:
        """Run an allowed shell command in the configured working directory."""
        timeout = timeout_seconds or self.default_timeout_seconds
        self._validate_command(command)
        try:
            completed = subprocess.run(
                command if self.allow_shell else shlex.split(command),
                cwd=self.cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=self.allow_shell,
                check=False,
            )
            return {
                "command": command,
                "cwd": self.cwd.as_posix(),
                "exit_code": completed.returncode,
                "stdout": self._output_value(completed.stdout),
                "stderr": self._output_value(completed.stderr),
                "timed_out": False,
            }
        except subprocess.TimeoutExpired as exc:
            return {
                "command": command,
                "cwd": self.cwd.as_posix(),
                "exit_code": None,
                "stdout": self._output_value(exc.stdout or ""),
                "stderr": self._output_value(exc.stderr or ""),
                "timed_out": True,
            }

    def _validate_command(self, command: str) -> None:
        if not self.allow_shell:
            self._validate_no_shell_syntax(command)
        if self.allow_all:
            return
        for allowed in self.allowed_commands:
            if command == allowed or command.startswith(f"{allowed} "):
                return
        raise PermissionError(f"Command is not allowed: {command}")

    def _validate_no_shell_syntax(self, command: str) -> None:
        if "\n" in command:
            raise ValueError(
                "Shell redirection, heredocs, and multi-line syntax are disabled "
                "for this tool. Use file tools for multi-line file writes, or "
                "configure allow_shell=True."
            )
        try:
            tokens = shlex.split(command)
        except ValueError as exc:
            raise ValueError(f"Invalid shell command: {exc}") from exc
        if any(self._looks_like_shell_operator(token) for token in tokens):
            raise ValueError(
                "Shell redirection, pipes, heredocs, and command chaining are "
                "disabled for this tool. Use file tools such as write_file or "
                "apply_unified_diff for file edits, or configure allow_shell=True."
            )

    def _looks_like_shell_operator(self, token: str) -> bool:
        if token in UNSUPPORTED_SHELL_TOKENS:
            return True
        if token.startswith((">", ">>", "2>", "2>>")):
            return True
        return any(operator in token for operator in ("&&", "||", "<<", ">>"))

    def _output_value(self, value: str) -> str:
        value = self._decode_output(value)
        context = get_active_run_context()
        if context and context.artifact_store is not None:
            return value
        return self._truncate(value)

    def _truncate(self, value: str) -> str:
        value = self._decode_output(value)
        if len(value) <= self.max_output_chars:
            return value
        return value[: self.max_output_chars]

    def _decode_output(self, value: str) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        if value is None:
            return ""
        return value

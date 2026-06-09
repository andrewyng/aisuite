"""Persistent shell behind an `Executor` boundary.

`LocalExecutor` keeps one long-lived shell process, so `cd`, `export`, activated venvs,
etc. persist across `run_shell` calls (unlike a per-call `subprocess.run`). The `Executor`
interface is the hedge for a future `ContainerExecutor`/`VMExecutor` (sandboxing) without
touching the engine.

Safety here is permission-gating (high-risk tool → approval) + per-command timeout +
best-effort non-interactive enforcement. A timed-out command is interrupted (SIGINT to
the process group); the shell survives so session state is preserved.
"""

from __future__ import annotations

import os
import queue
import signal
import subprocess
import threading
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import aisuite as ai

# Env defaults that discourage commands from blocking on a prompt.
_NONINTERACTIVE_ENV = {
    "GIT_TERMINAL_PROMPT": "0",
    "DEBIAN_FRONTEND": "noninteractive",
    "PYTHONUNBUFFERED": "1",
    "PIP_NO_INPUT": "1",
}


class Executor(ABC):
    @abstractmethod
    def run(self, command: str, timeout: Optional[float] = None) -> dict[str, Any]:
        ...

    def interrupt(self) -> None:  # pragma: no cover - default no-op
        pass

    def close(self) -> None:  # pragma: no cover - default no-op
        pass


class LocalExecutor(Executor):
    def __init__(
        self,
        *,
        cwd: str | Path,
        env: Optional[dict[str, str]] = None,
        shell_path: str = "/bin/bash",
        default_timeout: float = 30.0,
        max_output_chars: int = 20_000,
    ) -> None:
        self.cwd = str(Path(cwd).expanduser().resolve())
        self.default_timeout = default_timeout
        self.max_output_chars = max_output_chars
        self._marker = f"__COWORKER_DONE_{uuid.uuid4().hex}__"

        full_env = {**os.environ, **_NONINTERACTIVE_ENV, **(env or {})}
        self._proc = subprocess.Popen(
            [shell_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=self.cwd,
            text=True,
            bufsize=1,
            env=full_env,
            start_new_session=True,
        )
        self._queue: "queue.Queue[Optional[str]]" = queue.Queue()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _read_loop(self) -> None:
        try:
            assert self._proc.stdout is not None
            for line in self._proc.stdout:
                self._queue.put(line)
        finally:
            self._queue.put(None)  # EOF sentinel

    def run(self, command: str, timeout: Optional[float] = None) -> dict[str, Any]:
        if self._proc.poll() is not None or self._proc.stdin is None:
            return self._result(command, None, "", timed_out=False, error="shell not running")

        timeout = timeout or self.default_timeout
        # Run the command, then emit a marker line with exit code + cwd.
        self._proc.stdin.write(command + "\n")
        self._proc.stdin.write(
            f'printf "\\n%s %s %s\\n" "{self._marker}" "$?" "$PWD"\n'
        )
        self._proc.stdin.flush()

        deadline = time.monotonic() + timeout
        interrupted = False
        timed_out = False
        exit_code: Optional[int] = None
        lines: list[str] = []

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                if not interrupted:
                    # First deadline: interrupt the running command and keep reading
                    # until ITS marker arrives, so the stream stays in sync for the
                    # next command. SIGINT makes the command exit and the trailer
                    # printf emit the marker.
                    interrupted = True
                    timed_out = True
                    self._interrupt()
                    deadline = time.monotonic() + 3.0  # grace to resync on the marker
                    continue
                # Grace expired and still no marker: the shell is wedged. Hard-kill
                # so future commands don't desync (session state is lost).
                self.close()
                break
            try:
                item = self._queue.get(timeout=min(remaining, 0.5))
            except queue.Empty:
                continue
            if item is None:
                break  # shell died
            if self._marker in item:
                exit_code = _parse_exit_code(item, self._marker)
                cwd = _parse_cwd(item, self._marker)
                if cwd:
                    self.cwd = cwd
                break
            lines.append(item)

        output = "".join(lines)
        truncated = len(output) > self.max_output_chars
        if truncated:
            output = output[: self.max_output_chars]
        return self._result(
            command, exit_code, output, timed_out=timed_out, truncated=truncated
        )

    def _interrupt(self) -> None:
        # Interrupt the shell's foreground child(ren), not the shell itself, so the
        # session survives. SIGINT makes the command exit; the queued trailer printf
        # then emits the marker and the stream resyncs.
        try:
            found = subprocess.run(
                ["pgrep", "-P", str(self._proc.pid)],
                capture_output=True,
                text=True,
            )
            for pid in found.stdout.split():
                try:
                    os.kill(int(pid), signal.SIGINT)
                except (ProcessLookupError, ValueError, OSError):
                    pass
        except (FileNotFoundError, OSError):
            pass

    def interrupt(self) -> None:
        self._interrupt()

    def close(self) -> None:
        try:
            self._proc.terminate()
        except (ProcessLookupError, OSError):
            pass

    def _result(self, command, exit_code, output, *, timed_out, truncated=False, error=None):
        result = {
            "command": command,
            "cwd": self.cwd,
            "exit_code": exit_code,
            "output": output,
            "timed_out": timed_out,
            "truncated": truncated,
        }
        if error:
            result["error"] = error
        return result


def _parse_exit_code(line: str, marker: str) -> Optional[int]:
    parts = line.strip().split()
    try:
        return int(parts[parts.index(marker) + 1])
    except (ValueError, IndexError):
        return None


def _parse_cwd(line: str, marker: str) -> Optional[str]:
    parts = line.strip().split()
    try:
        return " ".join(parts[parts.index(marker) + 2 :]) or None
    except (ValueError, IndexError):
        return None


def shell_tools(executor: Executor) -> list:
    """Return the `run_shell` tool bound to a persistent executor."""

    def run_shell(command: str, timeout_seconds: Optional[int] = None) -> dict:
        """Run a shell command in the persistent session (cwd and env persist)."""
        return executor.run(command, timeout=timeout_seconds)

    return [
        ai.tool(
            run_shell,
            metadata=ai.ToolMetadata(
                category="shell",
                risk_level="high",
                capabilities=["run_command"],
                requires_approval=True,
            ),
        )
    ]

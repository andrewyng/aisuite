"""Permission engine — decides allow / deny / ask-user for each proposed tool call.

Modes: Plan (read-only) · Interactive (auto reads, ask on writes/commands) · Auto
(allow, still path-scoped). Refined by argument patterns (path-under-root, command
prefixes) and a session allowlist. The engine only *decides*; the turn engine routes
`needs_user` decisions to a surface for approval and records the outcome.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class Mode(str, Enum):
    PLAN = "plan"  # read-only
    INTERACTIVE = "interactive"  # ask for approval (default)
    AUTO = "auto"  # full access
    CUSTOM = "custom"  # interactive + auto-allow the config's `auto_allow` tools


@dataclass
class Decision:
    allowed: bool
    reason: str = ""
    needs_user: bool = False  # True → surface should prompt the user for approval


# Tools that mutate the workspace (used for path-scoping + plan-mode blocking).
WRITE_TOOLS = {"write_file", "replace_in_file", "apply_patch", "apply_unified_diff"}
SHELL_TOOL = "run_shell"


@dataclass
class PermissionEngine:
    workspace_root: Path
    mode: Mode = Mode.INTERACTIVE
    allowed_commands: list[str] = field(default_factory=list)
    auto_allow_tools: set[str] = field(default_factory=set)
    session_allow_tools: set[str] = field(default_factory=set)
    session_allow_commands: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.workspace_root = Path(self.workspace_root).expanduser().resolve()
        self.auto_allow_tools = set(self.auto_allow_tools)

    def evaluate(
        self, tool_name: str, arguments: dict[str, Any], metadata: Any = None
    ) -> Decision:
        arguments = arguments or {}
        requires_approval = bool(getattr(metadata, "requires_approval", False))
        is_connector = getattr(metadata, "category", "") == "connector"
        is_write = tool_name in WRITE_TOOLS
        is_shell = tool_name == SHELL_TOOL
        consequential = is_write or is_shell or requires_approval

        # Plan mode: read-only.
        if self.mode is Mode.PLAN and consequential:
            return Decision(False, "plan mode is read-only", needs_user=False)

        # Path scoping for writes that name a path (all modes).
        if is_write:
            path = arguments.get("path")
            if path is not None and not self._under_root(path):
                return Decision(False, f"path escapes workspace: {path}")

        # Non-consequential tools always run.
        if not consequential:
            return Decision(True, "low risk")

        # Full access.
        if self.mode is Mode.AUTO:
            return Decision(True, "full access")

        # interactive / custom: allowlists.
        if is_shell:
            command = str(arguments.get("command", ""))
            if self._command_allowed(command):
                return Decision(True, "command on allowlist")
            if command and command in self.session_allow_commands:
                return Decision(True, "command allowed for session")
        if tool_name in self.session_allow_tools and not is_connector:
            return Decision(True, "tool allowed for session")

        # Custom mode auto-approves the configured tools.
        if self.mode is Mode.CUSTOM and tool_name in self.auto_allow_tools:
            return Decision(True, "auto-allowed by config")

        # Otherwise: ask the user.
        return Decision(False, "requires approval", needs_user=True)

    # -- session memory ---------------------------------------------------------
    def allow_tool_for_session(self, tool_name: str) -> None:
        self.session_allow_tools.add(tool_name)

    def allow_command_for_session(self, command: str) -> None:
        if command:
            self.session_allow_commands.add(command)

    # -- helpers ----------------------------------------------------------------
    def _under_root(self, path: str) -> bool:
        try:
            (self.workspace_root / path).expanduser().resolve().relative_to(
                self.workspace_root
            )
            return True
        except ValueError:
            return False

    def _command_allowed(self, command: str) -> bool:
        for allowed in self.allowed_commands:
            if command == allowed or command.startswith(f"{allowed} "):
                return True
        return False

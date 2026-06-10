"""Configuration — layered TOML: built-in defaults < global < per-workspace.

Global:    <state-dir>/config.toml   (see `secrets.state_dir`; platform-native)
Workspace: <workspace>/.coworker/config.toml   (overrides global)
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .secrets import state_dir

DEFAULT_ALLOWED_COMMANDS = [
    "ls",
    "cat",
    "pwd",
    "echo",
    "head",
    "tail",
    "grep",
    "find",
    "wc",
    "git status",
    "git diff",
    "git log",
    "git show",
    "python3",
    "python",
    "pytest",
    "node",
    "npm",
    "npx",
]


@dataclass
class Config:
    model: str = "gpt-5.5"
    mode: str = "interactive"
    max_iterations: int = 150
    allowed_commands: list[str] = field(
        default_factory=lambda: list(DEFAULT_ALLOWED_COMMANDS)
    )
    # In "custom" permission mode, these tools are auto-approved (e.g. file edits)
    # while everything else still asks.
    auto_allow: list[str] = field(default_factory=list)
    host: str = "127.0.0.1"
    port: int = 8765
    # Always-on super-agent (inbound messaging): its workspace (default: <state>/superagent).
    superagent_workspace: Optional[str] = None
    # Web search provider: "duckduckgo" (keyless default) | "tavily" | "brave" (need a key).
    web_search_provider: str = "duckduckgo"


_FIELDS = {
    "model",
    "mode",
    "max_iterations",
    "allowed_commands",
    "auto_allow",
    "host",
    "port",
    "superagent_workspace",
    "web_search_provider",
}


def global_config_path() -> Path:
    return state_dir() / "config.toml"


def _read(path: Path) -> dict[str, Any]:
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def load_config(
    workspace: Optional[str | Path] = None, *, global_path: Optional[Path] = None
) -> Config:
    cfg = Config()
    data: dict[str, Any] = {}

    g = Path(global_path) if global_path is not None else global_config_path()
    if g.is_file():
        data.update(_read(g))
    if workspace:
        w = Path(workspace).expanduser() / ".coworker" / "config.toml"
        if w.is_file():
            data.update(_read(w))

    for key, value in data.items():
        if key in _FIELDS:
            setattr(cfg, key, value)
    return cfg

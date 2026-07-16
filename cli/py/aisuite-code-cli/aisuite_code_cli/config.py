from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

DEFAULT_ALLOWED_COMMANDS = [
    "npm",
    "npx",
    "node",
    "python",
    "python3",
    "pytest",
    "git status",
    "git diff",
    "git log",
    "mkdir",
    "ls",
    "pwd",
]


@dataclass
class CliConfig:
    model: str
    cwd: Path
    trace_file: Path
    artifact_root: Path = field(default_factory=lambda: Path(".aisuite/artifacts"))
    trace_http: Optional[str] = None
    allow_write: bool = True
    allow_shell_all: bool = False
    allowed_commands: list[str] = field(default_factory=list)
    max_turns: int = 8
    start_viewer: bool = False
    enable_reviewer: bool = True

    def __post_init__(self) -> None:
        self.cwd = self.cwd.expanduser().resolve()
        if not self.trace_file.is_absolute():
            self.trace_file = self.cwd / self.trace_file
        if not self.artifact_root.is_absolute():
            self.artifact_root = self.cwd / self.artifact_root


def parse_args(argv: Optional[list[str]] = None) -> CliConfig:
    parser = argparse.ArgumentParser(description="Run the aisuite coding agent CLI.")
    parser.add_argument("--model", default="openai:gpt-4o-mini")
    parser.add_argument("--cwd", default=".")
    parser.add_argument("--trace-file", default=".aisuite/code.jsonl")
    parser.add_argument("--artifact-root", default=".aisuite/artifacts")
    parser.add_argument(
        "--trace-http",
        default=None,
        help="Optional viewer /api/events endpoint for live trace streaming.",
    )
    parser.add_argument(
        "--read-only",
        action="store_true",
        help="Disable file write tools.",
    )
    parser.add_argument("--allow-shell-all", action="store_true")
    parser.add_argument(
        "--allow-command",
        action="append",
        dest="allowed_commands",
        default=[],
        help="Command prefix to allow. Can be supplied multiple times.",
    )
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Start the local viewer when the CLI launches.",
    )
    parser.add_argument(
        "--no-reviewer",
        action="store_true",
        help="Disable the default read-only reviewer subagent tool.",
    )
    args = parser.parse_args(argv)
    cwd = Path(args.cwd).expanduser().resolve()
    trace_file = Path(args.trace_file)
    if not trace_file.is_absolute():
        trace_file = cwd / trace_file
    artifact_root = Path(args.artifact_root)
    if not artifact_root.is_absolute():
        artifact_root = cwd / artifact_root
    return CliConfig(
        model=args.model,
        cwd=cwd,
        trace_file=trace_file,
        artifact_root=artifact_root,
        trace_http=args.trace_http,
        allow_write=not args.read_only,
        allow_shell_all=args.allow_shell_all,
        allowed_commands=args.allowed_commands or list(DEFAULT_ALLOWED_COMMANDS),
        max_turns=args.max_turns,
        start_viewer=args.viewer,
        enable_reviewer=not args.no_reviewer,
    )

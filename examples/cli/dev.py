"""
Minimal aisuite developer CLI.

Run from the repository root:

    python examples/cli/dev.py --model openai:gpt-4o-mini

Useful commands inside the CLI:

    /help
    /viewer
    /status
    /clear
    /exit
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, TextIO

import aisuite as ai

DEFAULT_ALLOWED_COMMANDS = [
    "python",
    "python3",
    "pytest",
    "git status",
    "git diff",
    "ls",
    "pwd",
]


@dataclass
class CliConfig:
    model: str
    cwd: Path
    trace_file: Path
    trace_http: Optional[str] = None
    allow_write: bool = False
    allow_shell_all: bool = False
    allowed_commands: list[str] = field(default_factory=list)
    max_turns: int = 5


class ApprovalController:
    def __init__(
        self,
        *,
        input_stream: TextIO = sys.stdin,
        output_stream: TextIO = sys.stdout,
    ):
        self.input_stream = input_stream
        self.output_stream = output_stream
        self.always_allow_tools: set[str] = set()

    def evaluate(self, context: ai.ToolPolicyContext) -> ai.ToolPolicyDecision:
        metadata = context.tool_metadata
        if context.tool_name in self.always_allow_tools:
            return ai.ToolPolicyDecision(
                allowed=True,
                reason="allowed for session",
            )
        if metadata is None or not metadata.requires_approval:
            return ai.ToolPolicyDecision(allowed=True, reason="low risk")

        self._print_approval_request(context)
        choice = self._read_choice()
        if choice == "a":
            self.always_allow_tools.add(context.tool_name)
            return ai.ToolPolicyDecision(
                allowed=True,
                reason="allowed for session",
            )
        if choice == "y":
            return ai.ToolPolicyDecision(allowed=True, reason="approved by user")
        return ai.ToolPolicyDecision(allowed=False, reason="denied by user")

    def _print_approval_request(self, context: ai.ToolPolicyContext) -> None:
        metadata = context.tool_metadata
        print("\nPermission required", file=self.output_stream)
        print(f"  tool: {context.tool_name}", file=self.output_stream)
        if metadata:
            print(
                f"  risk: {metadata.risk_level}"
                f" · category: {metadata.category or '-'}",
                file=self.output_stream,
            )
        print("  arguments:", file=self.output_stream)
        for key, value in context.arguments.items():
            print(f"    {key}: {value}", file=self.output_stream)
        print(
            "  allow? [y] yes  [n] no  [a] always for this session",
            file=self.output_stream,
        )

    def _read_choice(self) -> str:
        print("> ", end="", file=self.output_stream, flush=True)
        choice = self.input_stream.readline().strip().lower()
        if choice in {"y", "yes"}:
            return "y"
        if choice in {"a", "always"}:
            return "a"
        return "n"


class DevCli:
    def __init__(
        self,
        config: CliConfig,
        *,
        input_stream: TextIO = sys.stdin,
        output_stream: TextIO = sys.stdout,
    ):
        self.config = config
        self.input_stream = input_stream
        self.output_stream = output_stream
        self.approvals = ApprovalController(
            input_stream=input_stream,
            output_stream=output_stream,
        )
        self.trace_sinks = [ai.tracing.LocalTraceSink(config.trace_file)]
        if config.trace_http:
            self.trace_sinks.append(ai.tracing.HttpTraceSink(config.trace_http))
        self.agent = self._build_agent()
        self.result: Optional[ai.RunResult] = None
        self.viewer: Optional[ai.tracing.ViewerServer] = None

    def run(self) -> None:
        self._print_header()
        try:
            while True:
                print("\nYou > ", end="", file=self.output_stream, flush=True)
                user_input = self.input_stream.readline()
                if not user_input:
                    print("", file=self.output_stream)
                    break
                user_input = user_input.strip()
                if not user_input:
                    continue
                if user_input.startswith("/"):
                    if self._handle_command(user_input):
                        break
                    continue
                self._run_agent(user_input)
        finally:
            if self.viewer is not None:
                self.viewer.stop()

    def _build_agent(self) -> ai.Agent:
        tools = [
            *ai.toolkits.files(
                root=self.config.cwd,
                allow_write=self.config.allow_write,
            ),
            *ai.toolkits.shell(
                cwd=self.config.cwd,
                allowed_commands=self.config.allowed_commands,
                allow_all=self.config.allow_shell_all,
            ),
        ]
        return ai.Agent(
            name="aisuite_dev",
            model=self.config.model,
            instructions=(
                "You are a concise coding assistant. Use file tools to inspect "
                "the project and shell tools to run focused commands when useful. "
                "Explain what changed and mention command results."
            ),
            tools=tools,
            model_settings={"temperature": 0.2},
            tags=["cli", "dev"],
            metadata={"app": "aisuite_dev_cli"},
        )

    def _run_agent(self, user_input: str) -> None:
        try:
            if self.result is None:
                self.result = ai.Runner.run_sync(
                    self.agent,
                    user_input,
                    max_turns=self.config.max_turns,
                    run_name="cli_turn",
                    group_id="aisuite-dev-cli",
                    trace_sinks=self.trace_sinks,
                    tool_policy=self.approvals,
                )
            else:
                self.result = ai.Runner.continue_sync(
                    self.result,
                    user_input,
                    max_turns=self.config.max_turns,
                    trace_sinks=self.trace_sinks,
                    tool_policy=self.approvals,
                )
        except Exception as exc:
            print(f"\nError: {exc}", file=self.output_stream)
            return

        self._print_steps(self.result)
        print("\nAssistant", file=self.output_stream)
        print(f"  {self.result.final_output or ''}", file=self.output_stream)

    def _print_steps(self, result: ai.RunResult) -> None:
        tool_steps = [
            step for step in result.steps if step.type in {"tool_call", "tool_result"}
        ]
        for step in tool_steps[-8:]:
            data = step.data
            if step.type == "tool_call":
                status = "allowed" if data.get("allowed") else "denied"
                print(
                    f"\nTool request · {step.name} · {status}",
                    file=self.output_stream,
                )
                self._print_mapping("arguments", data.get("arguments"))
                if data.get("reason"):
                    print(f"  reason: {data['reason']}", file=self.output_stream)
            elif step.type == "tool_result":
                print(f"\nTool result · {step.name}", file=self.output_stream)
                if data.get("status"):
                    print(f"  status: {data['status']}", file=self.output_stream)
                if data.get("result_preview"):
                    self._print_result_preview(data["result_preview"])

    def _handle_command(self, command: str) -> bool:
        if command in {"/exit", "/quit"}:
            return True
        if command == "/help":
            self._print_help()
            return False
        if command == "/viewer":
            print(
                "\nRun viewer:",
                file=self.output_stream,
            )
            print(
                f"  python -m aisuite.agents.viewer --trace-file {self.config.trace_file}",
                file=self.output_stream,
            )
            print(
                "  live endpoint: "
                f"{self.config.trace_http or '(pass --trace-http to stream events)'}",
                file=self.output_stream,
            )
            return False
        if command == "/viewer start":
            if self.viewer is None:
                self.viewer = ai.tracing.start_viewer(
                    self.config.trace_file,
                    port=0,
                )
            print(f"\nViewer: {self.viewer.url}", file=self.output_stream)
            return False
        if command == "/status":
            print(f"\nmodel: {self.config.model}", file=self.output_stream)
            print(f"cwd: {self.config.cwd}", file=self.output_stream)
            print(f"trace_file: {self.config.trace_file}", file=self.output_stream)
            print(
                f"trace_http: {self.config.trace_http or '-'}", file=self.output_stream
            )
            print(
                f"allowed_commands: {', '.join(self.config.allowed_commands) or 'all'}",
                file=self.output_stream,
            )
            return False
        if command == "/clear":
            self.result = None
            print("\nConversation state cleared.", file=self.output_stream)
            return False
        print(f"\nUnknown command: {command}", file=self.output_stream)
        self._print_help()
        return False

    def _print_mapping(self, label: str, value: object) -> None:
        if not value:
            return
        print(f"  {label}:", file=self.output_stream)
        if isinstance(value, dict):
            for key, item in value.items():
                print(
                    f"    {key}: {self._compact_value(item)}", file=self.output_stream
                )
            return
        print(f"    {self._compact_value(value)}", file=self.output_stream)

    def _print_result_preview(self, preview: str) -> None:
        try:
            value = json.loads(preview)
        except json.JSONDecodeError:
            print(f"  result: {self._compact_value(preview)}", file=self.output_stream)
            return

        if isinstance(value, dict) and {"stdout", "stderr", "exit_code"} & set(value):
            for key in ("exit_code", "timed_out"):
                if key in value:
                    print(f"  {key}: {value[key]}", file=self.output_stream)
            for key in ("stdout", "stderr"):
                if value.get(key):
                    print(
                        f"  {key}: {self._compact_value(value[key])}",
                        file=self.output_stream,
                    )
            return
        print(f"  result: {self._compact_value(value)}", file=self.output_stream)

    def _compact_value(self, value: object, max_chars: int = 500) -> str:
        if isinstance(value, str):
            rendered = value
        else:
            rendered = json.dumps(value, sort_keys=True)
        rendered = rendered.replace("\n", "\\n")
        if len(rendered) <= max_chars:
            return rendered
        return rendered[: max_chars - 3] + "..."

    def _print_header(self) -> None:
        print("aisuite dev", file=self.output_stream)
        print("-" * 48, file=self.output_stream)
        print(f"model: {self.config.model}", file=self.output_stream)
        print(f"cwd: {self.config.cwd}", file=self.output_stream)
        print(f"traces: {self.config.trace_file}", file=self.output_stream)
        print("type /help for commands", file=self.output_stream)

    def _print_help(self) -> None:
        print("\nCommands", file=self.output_stream)
        print("  /viewer        show local viewer command", file=self.output_stream)
        print("  /viewer start  start local viewer server", file=self.output_stream)
        print("  /status  show current configuration", file=self.output_stream)
        print("  /clear   clear conversation state", file=self.output_stream)
        print("  /exit    quit", file=self.output_stream)


def parse_args(argv: Optional[list[str]] = None) -> CliConfig:
    parser = argparse.ArgumentParser(description="Run the aisuite developer CLI.")
    parser.add_argument("--model", default="openai:gpt-4o-mini")
    parser.add_argument("--cwd", default=".")
    parser.add_argument("--trace-file", default=".aisuite/events.jsonl")
    parser.add_argument(
        "--trace-http",
        default=None,
        help="Optional viewer /api/events endpoint for live trace streaming.",
    )
    parser.add_argument("--allow-write", action="store_true")
    parser.add_argument("--allow-shell-all", action="store_true")
    parser.add_argument(
        "--allow-command",
        action="append",
        dest="allowed_commands",
        default=[],
        help="Command prefix to allow. Can be supplied multiple times.",
    )
    parser.add_argument("--max-turns", type=int, default=5)
    args = parser.parse_args(argv)
    allowed_commands = args.allowed_commands or list(DEFAULT_ALLOWED_COMMANDS)
    return CliConfig(
        model=args.model,
        cwd=Path(args.cwd).expanduser().resolve(),
        trace_file=Path(args.trace_file),
        trace_http=args.trace_http,
        allow_write=args.allow_write,
        allow_shell_all=args.allow_shell_all,
        allowed_commands=allowed_commands,
        max_turns=args.max_turns,
    )


def main(argv: Optional[list[str]] = None) -> None:
    config = parse_args(argv)
    DevCli(config).run()


if __name__ == "__main__":
    main()

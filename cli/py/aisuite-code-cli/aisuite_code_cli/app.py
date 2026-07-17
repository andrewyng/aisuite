from __future__ import annotations

import json
import sys
from typing import Optional, TextIO

import aisuite as ai

from .agent import build_agent
from .approval import ApprovalController
from .config import CliConfig
from .rendering import print_steps


class CodeCli:
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
        self.artifact_store = ai.FileArtifactStore(config.artifact_root)
        if config.trace_http:
            self.trace_sinks.append(ai.tracing.HttpTraceSink(config.trace_http))
        self.agent = build_agent(config)
        self.result: Optional[ai.RunResult] = None
        self.viewer: Optional[ai.tracing.ViewerServer] = None

    def run(self) -> None:
        self._print_header()
        if self.config.start_viewer:
            self._start_viewer()
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

    def _run_agent(self, user_input: str) -> None:
        print("\nWorking...", file=self.output_stream)
        try:
            if self.result is None:
                self.result = ai.Runner.run_sync(
                    self.agent,
                    user_input,
                    max_turns=self.config.max_turns,
                    run_name="code_cli_turn",
                    group_id="aisuite-code-cli",
                    trace_sinks=self.trace_sinks,
                    tool_policy=self.approvals,
                    artifact_store=self.artifact_store,
                )
            else:
                self.result = ai.Runner.continue_sync(
                    self.result,
                    user_input,
                    max_turns=self.config.max_turns,
                    trace_sinks=self.trace_sinks,
                    tool_policy=self.approvals,
                    artifact_store=self.artifact_store,
                )
        except Exception as exc:
            self._print_error(exc)
            return

        print_steps(self.result, self.output_stream)
        print("\nAssistant", file=self.output_stream)
        final_output = str(self.result.final_output or "").strip()
        if final_output:
            for line in final_output.splitlines():
                print(f"  {line}", file=self.output_stream)
        else:
            print("  (no final output)", file=self.output_stream)
        self._print_trace_hint()

    def _handle_command(self, command: str) -> bool:
        if command in {"/exit", "/quit"}:
            return True
        if command == "/help":
            self._print_help()
            return False
        if command == "/viewer":
            print("\nViewer", file=self.output_stream)
            if self.viewer is not None:
                print(f"  open: {self.viewer.url}", file=self.output_stream)
            print("  start here: /viewer start", file=self.output_stream)
            print(
                "  run separately: python -m aisuite.tracing.viewer "
                f"--trace-file {self.config.trace_file} "
                f"--artifact-root {self.config.artifact_root}",
                file=self.output_stream,
            )
            print(
                "  live endpoint: "
                f"{self.config.trace_http or '(pass --trace-http to stream events)'}",
                file=self.output_stream,
            )
            return False
        if command == "/viewer start":
            self._start_viewer()
            return False
        if command == "/status":
            self._print_status()
            return False
        if command == "/examples":
            self._print_examples()
            return False
        if command == "/last":
            self._print_last_turn()
            return False
        if command == "/clear":
            self.result = None
            print("\nConversation state cleared.", file=self.output_stream)
            return False
        print(f"\nUnknown command: {command}", file=self.output_stream)
        self._print_help()
        return False

    def _start_viewer(self) -> None:
        if self.viewer is None:
            self.viewer = ai.tracing.start_viewer(
                self.config.trace_file,
                port=0,
                artifact_root=self.config.artifact_root,
            )
        print(f"\nViewer: {self.viewer.url}", file=self.output_stream)

    def _print_status(self) -> None:
        print(f"\nmodel: {self.config.model}", file=self.output_stream)
        print(f"cwd: {self.config.cwd}", file=self.output_stream)
        print(f"trace_file: {self.config.trace_file}", file=self.output_stream)
        print(f"artifact_root: {self.config.artifact_root}", file=self.output_stream)
        print(f"trace_http: {self.config.trace_http or '-'}", file=self.output_stream)
        print(f"write_tools: {self.config.allow_write}", file=self.output_stream)
        print(f"reviewer: {self.config.enable_reviewer}", file=self.output_stream)
        print(
            f"allowed_commands: {', '.join(self.config.allowed_commands) or 'all'}",
            file=self.output_stream,
        )

    def _print_header(self) -> None:
        print("aisuite-code", file=self.output_stream)
        print("-" * 56, file=self.output_stream)
        print("Session", file=self.output_stream)
        print(f"  model:    {self.config.model}", file=self.output_stream)
        print(f"  cwd:      {self.config.cwd}", file=self.output_stream)
        print(
            f"  tools:    writes {'on' if self.config.allow_write else 'off'} · "
            f"shell {'all' if self.config.allow_shell_all else 'limited'} · "
            f"reviewer {'on' if self.config.enable_reviewer else 'off'}",
            file=self.output_stream,
        )
        print(f"  traces:   {self.config.trace_file}", file=self.output_stream)
        print(f"  artifacts:{self.config.artifact_root}", file=self.output_stream)
        if self.config.trace_http:
            print(f"  stream:   {self.config.trace_http}", file=self.output_stream)
        print(
            "\nType /help for commands. Use /viewer start for local traces.",
            file=self.output_stream,
        )
        self._print_starter_prompts()

    def _print_help(self) -> None:
        print("\nCommands", file=self.output_stream)
        print(
            "  /viewer        show viewer status and command", file=self.output_stream
        )
        print("  /viewer start  start local viewer server", file=self.output_stream)
        print("  /status        show current configuration", file=self.output_stream)
        print("  /examples      show good starter prompts", file=self.output_stream)
        print("  /last          show last turn details", file=self.output_stream)
        print("  /clear         clear conversation state", file=self.output_stream)
        print("  /exit          quit", file=self.output_stream)
        self._print_examples()

    def _print_trace_hint(self) -> None:
        if self.result is None:
            return
        print(f"\nTrace: {self.result.trace_id}", file=self.output_stream)
        if self.viewer is not None:
            focused_url = f"{self.viewer.url}?embed=1&trace_id={self.result.trace_id}"
            print(f"  focused viewer: {focused_url}", file=self.output_stream)
        else:
            print("  start viewer: /viewer start", file=self.output_stream)

    def _print_starter_prompts(self) -> None:
        print("\nTry", file=self.output_stream)
        for prompt in (
            "List files in this directory and tell me what you see.",
            "Read README.md and summarize the project in 5 bullets.",
            "Create app.py with add(a, b), then run it.",
        ):
            print(f"  {prompt}", file=self.output_stream)

    def _print_examples(self) -> None:
        self._print_starter_prompts()
        print(
            "  Run the focused tests for the files you changed.",
            file=self.output_stream,
        )
        print(
            "  Ask the reviewer subagent to review the current changes.",
            file=self.output_stream,
        )

    def _print_last_turn(self) -> None:
        if self.result is None:
            print("\nNo turns yet.", file=self.output_stream)
            return
        print("\nLast turn", file=self.output_stream)
        print(f"  trace: {self.result.trace_id}", file=self.output_stream)
        print(f"  status: {self.result.status}", file=self.output_stream)
        print(
            f"  input: {self._compact(self.result.input, 260)}", file=self.output_stream
        )
        print("  output:", file=self.output_stream)
        output = str(self.result.final_output or "").strip() or "(no final output)"
        for line in output.splitlines():
            print(f"    {line}", file=self.output_stream)
        if not self.result.steps:
            return
        print("  steps:", file=self.output_stream)
        for step in self.result.steps:
            print(
                f"    - {step.type}: {step.name or '-'} ({step.id})",
                file=self.output_stream,
            )
            if step.data:
                print(f"      {self._compact(step.data, 700)}", file=self.output_stream)

    def _compact(self, value: object, max_chars: int = 500) -> str:
        if isinstance(value, str):
            rendered = value
        else:
            rendered = json.dumps(value, sort_keys=True, default=str)
        rendered = rendered.replace("\n", "\\n")
        if len(rendered) <= max_chars:
            return rendered
        return rendered[: max_chars - 3] + "..."

    def _print_error(self, exc: Exception) -> None:
        message = str(exc)
        print(f"\nError: {message}", file=self.output_stream)
        if "No module named 'openai'" in message:
            print(
                "  Install CLI dependencies from cli/py/aisuite-code-cli with "
                "python3 -m poetry install.",
                file=self.output_stream,
            )
        if "OPENAI_API_KEY" in message or "API key" in message:
            print(
                "  Set OPENAI_API_KEY in your shell, or source the repo .env file.",
                file=self.output_stream,
            )
        print(
            "  Use /status to inspect the active model, cwd, and trace files.",
            file=self.output_stream,
        )

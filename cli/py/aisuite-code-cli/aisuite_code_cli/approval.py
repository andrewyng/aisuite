from __future__ import annotations

import json
import sys
from typing import Any, TextIO

import aisuite as ai


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
        self.always_allow_commands: set[str] = set()

    def evaluate(self, context: ai.ToolPolicyContext) -> ai.ToolPolicyDecision:
        metadata = context.tool_metadata
        command = _exact_command(context)
        if command and command in self.always_allow_commands:
            return ai.ToolPolicyDecision(
                allowed=True,
                reason="command allowed for session",
            )
        if context.tool_name in self.always_allow_tools:
            return ai.ToolPolicyDecision(
                allowed=True,
                reason="tool allowed for session",
            )
        if metadata is None or not metadata.requires_approval:
            return ai.ToolPolicyDecision(allowed=True, reason="low risk")

        self._print_approval_request(context)
        choice = self._read_choice()
        if choice == "c" and command:
            self.always_allow_commands.add(command)
            return ai.ToolPolicyDecision(
                allowed=True,
                reason="command allowed for session",
            )
        if choice == "a":
            self.always_allow_tools.add(context.tool_name)
            return ai.ToolPolicyDecision(
                allowed=True,
                reason="tool allowed for session",
            )
        if choice == "y":
            return ai.ToolPolicyDecision(allowed=True, reason="approved by user")
        return ai.ToolPolicyDecision(allowed=False, reason="denied by user")

    def _print_approval_request(self, context: ai.ToolPolicyContext) -> None:
        metadata = context.tool_metadata
        action = _approval_action(context)
        effect = _approval_effect(context)
        preview = _approval_preview(context)
        risk = metadata.risk_level if metadata else "unknown"
        category = metadata.category if metadata and metadata.category else "-"

        print("\nPermission required", file=self.output_stream)
        print("  Action", file=self.output_stream)
        print(f"    {action}", file=self.output_stream)
        print("  Risk", file=self.output_stream)
        print(f"    {risk} · {category}", file=self.output_stream)
        print("  Effect", file=self.output_stream)
        print(f"    {effect}", file=self.output_stream)
        print("  Preview", file=self.output_stream)
        for line in preview:
            print(f"    {line}", file=self.output_stream)
        prompt = "  Allow? [y] once  [n] deny  [a] always this tool"
        if _exact_command(context):
            prompt += "  [c] always this command"
        print(prompt, file=self.output_stream)

    def _read_choice(self) -> str:
        print("> ", end="", file=self.output_stream, flush=True)
        choice = self.input_stream.readline().strip().lower()
        if choice in {"y", "yes"}:
            return "y"
        if choice in {"a", "always"}:
            return "a"
        if choice in {"c", "command"}:
            return "c"
        return "n"


def _approval_action(context: ai.ToolPolicyContext) -> str:
    args = context.arguments or {}
    if context.tool_name == "run_shell":
        command = args.get("command") or "-"
        return f"run shell command: {_single_line(str(command), 140)}"
    if context.tool_name == "write_file":
        return f"write file: {args.get('path', '-')}"
    if context.tool_name == "replace_in_file":
        return f"edit file: {args.get('path', '-')}"
    if context.tool_name == "apply_patch":
        return "apply Codex-style patch"
    if context.tool_name == "apply_unified_diff":
        return "apply unified diff"
    if context.tool_name == "review_changes":
        return "invoke reviewer subagent"
    return f"run tool: {context.tool_name}"


def _approval_effect(context: ai.ToolPolicyContext) -> str:
    args = context.arguments or {}
    if context.tool_name == "run_shell":
        return "Executes a command in the configured workspace."
    if context.tool_name == "write_file":
        overwrite = args.get("overwrite", True)
        mode = "create or overwrite" if overwrite else "create if missing"
        return f"May {mode} a file under the configured workspace."
    if context.tool_name == "replace_in_file":
        expected = args.get("expected_replacements", 1)
        return f"Replaces exact text in one file; expected replacements: {expected}."
    if context.tool_name == "apply_patch":
        return "Applies one or more targeted file edits from a Codex-style patch."
    if context.tool_name == "apply_unified_diff":
        return "Applies one or more file edits from a unified diff."
    if context.tool_name == "review_changes":
        return "Runs a read-only reviewer subagent and records a child trace."
    return "Runs the tool with the summarized arguments below."


def _approval_preview(context: ai.ToolPolicyContext) -> list[str]:
    args = context.arguments or {}
    if context.tool_name == "run_shell":
        return [f"command: {_summarize_argument(args.get('command'), max_chars=260)}"]
    if context.tool_name == "write_file":
        return [
            f"path: {args.get('path', '-')}",
            f"content: {_text_stats(args.get('content'))}",
        ]
    if context.tool_name == "replace_in_file":
        return [
            f"path: {args.get('path', '-')}",
            f"old: {_text_stats(args.get('old'))}",
            f"new: {_text_stats(args.get('new'))}",
        ]
    if context.tool_name == "apply_patch":
        return [f"patch: {_text_stats(args.get('patch'))}"]
    if context.tool_name == "apply_unified_diff":
        return [f"diff: {_text_stats(args.get('diff'))}"]
    if context.tool_name == "review_changes":
        return [f"input: {_summarize_argument(args.get('input'), max_chars=260)}"]
    if not args:
        return ["no arguments"]
    return [f"{key}: {_summarize_argument(value)}" for key, value in args.items()]


def _summarize_argument(value: Any, max_chars: int = 700) -> str:
    if _is_artifact_ref(value):
        ref = value.get("artifact_ref") or {}
        preview = value.get("preview")
        pieces = [
            f"artifact {ref.get('artifact_id')}",
            (
                f"{ref.get('size_bytes')} bytes"
                if ref.get("size_bytes") is not None
                else None
            ),
        ]
        if preview:
            pieces.append(f"preview={_single_line(str(preview), 180)}")
        return " · ".join(piece for piece in pieces if piece)
    if isinstance(value, str):
        lines = len(value.splitlines())
        if len(value) <= max_chars:
            return value
        return (
            f"{len(value)} chars · {lines} lines · "
            f"preview={_single_line(value, 360)}"
        )
    try:
        rendered = json.dumps(value, sort_keys=True)
    except TypeError:
        rendered = str(value)
    if len(rendered) <= max_chars:
        return rendered
    return f"{len(rendered)} chars · preview={_single_line(rendered, 360)}"


def _is_artifact_ref(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and value.get("type") == "artifact_ref"
        and isinstance(value.get("artifact_ref"), dict)
    )


def _single_line(value: str, max_chars: int) -> str:
    rendered = value.replace("\n", "\\n")
    if len(rendered) <= max_chars:
        return rendered
    return rendered[: max_chars - 3] + "..."


def _exact_command(context: ai.ToolPolicyContext) -> str | None:
    if context.tool_name != "run_shell":
        return None
    command = context.arguments.get("command") if context.arguments else None
    return command if isinstance(command, str) and command else None


def _text_stats(value: Any) -> str:
    if not isinstance(value, str):
        return "-"
    return f"{len(value)} chars · {len(value.splitlines())} lines"

from __future__ import annotations

import json
from typing import TextIO

import aisuite as ai

WRITE_TOOL_NAMES = {
    "write_file",
    "replace_in_file",
    "apply_patch",
    "apply_unified_diff",
}


def print_steps(result: ai.RunResult, output_stream: TextIO) -> None:
    tool_steps = [
        step for step in result.steps if step.type in {"tool_call", "tool_result"}
    ]
    if not tool_steps:
        return
    print("\nActivity", file=output_stream)
    for step in tool_steps[-10:]:
        data = step.data
        if step.type == "tool_call":
            status = "allowed" if data.get("allowed") else "denied"
            summary = summarize_tool_arguments(step.name, data.get("arguments"))
            print(
                f"  tool request: {step.name} · {status}{summary}",
                file=output_stream,
            )
            if data.get("reason"):
                print(f"    reason: {data['reason']}", file=output_stream)
        elif step.type == "tool_result":
            result_summary = summarize_result_preview(
                step.name, data.get("result_preview")
            )
            status = data.get("status") or "completed"
            print(
                f"  tool result: {step.name} · {status}{result_summary}",
                file=output_stream,
            )


def summarize_tool_arguments(tool_name: str, value: object) -> str:
    if not isinstance(value, dict) or not value:
        return ""
    if tool_name == "run_shell":
        command = value.get("command")
        return f" · {compact_value(command, 180)}" if command else ""
    if tool_name in WRITE_TOOL_NAMES:
        path = value.get("path")
        if path:
            if tool_name == "write_file" and "content" in value:
                return f" · {path} · {text_stats(value.get('content'))}"
            if tool_name == "replace_in_file":
                return f" · {path} · replace {text_stats(value.get('old'))}"
            return f" · {path}"
        if "patch" in value:
            return f" · patch · {text_stats(value.get('patch'))}"
        if "diff" in value:
            return f" · diff · {text_stats(value.get('diff'))}"
    if tool_name == "review_changes":
        return " · reviewer subagent"
    return " · " + compact_value(value, 220)


def summarize_result_preview(tool_name: str, preview: object) -> str:
    if not preview:
        return ""
    try:
        value = json.loads(preview) if isinstance(preview, str) else preview
    except json.JSONDecodeError:
        return " · " + compact_value(preview, 220)

    if isinstance(value, dict) and {"stdout", "stderr", "exit_code"} & set(value):
        exit_code = value.get("exit_code")
        stdout = value.get("stdout") or ""
        stderr = value.get("stderr") or ""
        parts = [f"exit {exit_code}" if exit_code is not None else None]
        if stdout:
            parts.append(f"stdout {text_stats(stdout)}")
        if stderr:
            parts.append(f"stderr {text_stats(stderr)}")
        return " · " + " · ".join(part for part in parts if part)
    if isinstance(value, dict):
        if "path" in value and "content" in value:
            return f" · {value['path']} · {text_stats(value.get('content'))}"
        changed = value.get("changed_files") or []
        if changed:
            files = ", ".join(changed[:3])
            more = "..." if len(changed) > 3 else ""
            return f" · {len(changed)} file(s): {files}{more}"
    return " · " + compact_value(value, 220)


def print_mapping(label: str, value: object, output_stream: TextIO) -> None:
    if not value:
        return
    print(f"  {label}:", file=output_stream)
    if isinstance(value, dict):
        for key, item in value.items():
            print(f"    {key}: {compact_value(item)}", file=output_stream)
        return
    print(f"    {compact_value(value)}", file=output_stream)


def print_result_preview(preview: str, output_stream: TextIO) -> None:
    try:
        value = json.loads(preview)
    except json.JSONDecodeError:
        print(f"  result: {compact_value(preview)}", file=output_stream)
        return

    if isinstance(value, dict) and {"stdout", "stderr", "exit_code"} & set(value):
        for key in ("exit_code", "timed_out"):
            if key in value:
                print(f"  {key}: {value[key]}", file=output_stream)
        for key in ("stdout", "stderr"):
            if value.get(key):
                print(
                    f"  {key}: {compact_value(value[key])}",
                    file=output_stream,
                )
        return
    print(f"  result: {compact_value(value)}", file=output_stream)


def text_stats(value: object) -> str:
    if not isinstance(value, str):
        return "-"
    line_count = len(value.splitlines())
    return f"{len(value)} chars, {line_count} lines"


def compact_value(value: object, max_chars: int = 500) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        rendered = value
    else:
        rendered = json.dumps(value, sort_keys=True)
    rendered = rendered.replace("\n", "\\n")
    if len(rendered) <= max_chars:
        return rendered
    return rendered[: max_chars - 3] + "..."

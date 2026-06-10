"""
Create a local agent trace for the aisuite runs viewer.

Run from the repository root:

    python examples/cli/create_demo_trace.py --trace-file .aisuite/demo.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import aisuite as ai
from aisuite.framework.message import ChatCompletionMessageToolCall, Function, Message


def tool_call(name: str, arguments: str, call_id: str) -> ChatCompletionMessageToolCall:
    return ChatCompletionMessageToolCall(
        id=call_id,
        type="function",
        function=Function(name=name, arguments=arguments),
    )


def response(content: str | None = "done", tool_calls=None):
    message = Message(role="assistant", content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class ScriptedProvider:
    def __init__(self, responses):
        self._responses = list(responses)

    def chat_completions_create(self, *_args, **_kwargs):
        if not self._responses:
            raise RuntimeError("No scripted responses left for demo trace.")
        return self._responses.pop(0)


LARGE_FILE_CONTENT = "demo artifact content\n" * 1300
LARGE_STDOUT_COMMAND = "python3 -c 'print(\"demo artifact stdout\\n\" * 1300)'"


def build_client() -> ai.Client:
    client = ai.Client()
    client.providers["openai"] = ScriptedProvider(
        [
            response(
                None,
                [
                    tool_call(
                        "list_files",
                        '{"path": "tests/toolkits"}',
                        "call_demo_list",
                    )
                ],
            ),
            response("Found `test_files.py` and `test_shell.py` in `tests/toolkits`."),
            response(
                None,
                [
                    tool_call(
                        "read_file_lines",
                        (
                            '{"path": "tests/toolkits/test_shell.py", '
                            '"start_line": 1, "max_lines": 10}'
                        ),
                        "call_demo_lines",
                    )
                ],
            ),
            response(
                "Read the first 10 lines of `tests/toolkits/test_shell.py` "
                "using the line-range file tool."
            ),
            response(
                None,
                [
                    tool_call(
                        "run_shell",
                        '{"command": "python3 -m pytest tests/toolkits -q"}',
                        "call_demo_tests",
                    )
                ],
            ),
            response("The focused toolkit test suite passed."),
            response(
                None,
                [
                    tool_call(
                        "write_file",
                        json.dumps(
                            {
                                "path": ".aisuite/demo-work/large-artifact.txt",
                                "content": LARGE_FILE_CONTENT,
                            }
                        ),
                        "call_demo_write_large",
                    )
                ],
            ),
            response(
                "Wrote a large demo file so the viewer can show artifactized arguments."
            ),
            response(
                None,
                [
                    tool_call(
                        "run_shell",
                        json.dumps({"command": LARGE_STDOUT_COMMAND}),
                        "call_demo_large_stdout",
                    )
                ],
            ),
            response(
                "Captured large shell output so the viewer can show artifactized results."
            ),
            response(
                None,
                [
                    tool_call(
                        "run_shell",
                        json.dumps({"command": "python3 -c 'print(\"blocked\")'"}),
                        "call_demo_denied_shell",
                    )
                ],
            ),
            response("The demo policy denied the blocked shell command."),
            response(
                None,
                [
                    tool_call(
                        "review_changes",
                        json.dumps(
                            {"input": "Review the demo trace changes and risks."}
                        ),
                        "call_demo_review",
                    )
                ],
            ),
            response(
                "Reviewer found no material issues. Residual risk: this is scripted demo data."
            ),
            response("Reviewer subagent completed and the demo trace is ready."),
        ]
    )
    return client


def approve_demo_tool(context: ai.ToolPolicyContext) -> ai.ToolPolicyDecision:
    if context.tool_name == "run_shell":
        command = str(context.arguments.get("command", ""))
        if "blocked" in command:
            return ai.ToolPolicyDecision(allowed=False, reason="blocked by demo policy")
        return ai.ToolPolicyDecision(allowed=True, reason="approved by demo")
    return ai.ToolPolicyDecision(allowed=True, reason="low risk")


def create_demo_trace(
    trace_file: Path,
    cwd: Path,
    append: bool = False,
    artifact_root: Path | None = None,
) -> None:
    if trace_file.exists() and not append:
        trace_file.unlink()
    trace_file.parent.mkdir(parents=True, exist_ok=True)
    sink = ai.tracing.LocalTraceSink(trace_file)
    artifact_store = ai.FileArtifactStore(
        artifact_root or trace_file.parent / "artifacts"
    )
    reviewer = ai.Agent(
        name="reviewer",
        model="openai:gpt-4o-mini",
        instructions=(
            "Review the demo trace work. Do not edit files. Keep the response concise."
        ),
        tools=ai.toolkits.files(root=cwd, allow_write=False),
        tags=["demo", "dev", "reviewer"],
        metadata={"app": "aisuite_demo_trace", "role": "reviewer"},
    )
    agent = ai.Agent(
        name="aisuite_demo_dev",
        model="openai:gpt-4o-mini",
        instructions=(
            "You are a concise coding assistant. Use tools to inspect the "
            "project and run focused checks."
        ),
        tools=[
            *ai.toolkits.files(root=cwd, allow_write=True),
            *ai.toolkits.git(root=cwd),
            *ai.toolkits.shell(cwd=cwd, allowed_commands=["python3"]),
            ai.agent_tool(
                reviewer,
                name="review_changes",
                description="Ask a read-only reviewer subagent to inspect the demo work.",
            ),
        ],
        tags=["demo", "dev"],
        metadata={"app": "aisuite_demo_trace"},
    )

    client = build_client()
    result = ai.Runner.run_sync(
        agent,
        "List files in tests/toolkits.",
        client=client,
        run_name="demo_list_files",
        group_id="aisuite-demo",
        trace_sinks=[sink],
        tool_policy=approve_demo_tool,
        artifact_store=artifact_store,
    )
    result = ai.Runner.continue_sync(
        result,
        "Read the first 10 lines of tests/toolkits/test_shell.py.",
        max_turns=5,
        trace_sinks=[sink],
        tool_policy=approve_demo_tool,
        artifact_store=artifact_store,
    )
    result = ai.Runner.continue_sync(
        result,
        "Run the focused toolkit tests.",
        max_turns=5,
        trace_sinks=[sink],
        tool_policy=approve_demo_tool,
        artifact_store=artifact_store,
    )
    result = ai.Runner.continue_sync(
        result,
        "Write a large demo file for artifact preview testing.",
        max_turns=5,
        trace_sinks=[sink],
        tool_policy=approve_demo_tool,
        artifact_store=artifact_store,
    )
    result = ai.Runner.continue_sync(
        result,
        "Run a command with large stdout for artifact preview testing.",
        max_turns=5,
        trace_sinks=[sink],
        tool_policy=approve_demo_tool,
        artifact_store=artifact_store,
    )
    result = ai.Runner.continue_sync(
        result,
        "Attempt a blocked shell command so the viewer shows a denied event.",
        max_turns=5,
        trace_sinks=[sink],
        tool_policy=approve_demo_tool,
        artifact_store=artifact_store,
    )
    ai.Runner.continue_sync(
        result,
        "Ask the reviewer subagent to review the demo work.",
        max_turns=5,
        trace_sinks=[sink],
        tool_policy=approve_demo_tool,
        artifact_store=artifact_store,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a sample aisuite trace.")
    parser.add_argument("--trace-file", default=".aisuite/demo.jsonl")
    parser.add_argument("--cwd", default=".")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--artifact-root", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_demo_trace(
        trace_file=Path(args.trace_file),
        cwd=Path(args.cwd).expanduser().resolve(),
        append=args.append,
        artifact_root=Path(args.artifact_root) if args.artifact_root else None,
    )
    print(f"Wrote demo trace to {args.trace_file}")


if __name__ == "__main__":
    main()

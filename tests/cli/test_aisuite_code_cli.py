from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path
from unittest.mock import Mock

import pytest

try:
    import tomllib  # stdlib from Python 3.11
except ModuleNotFoundError:  # Python 3.10
    tomllib = None

import aisuite as ai

CLI_PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "cli/py/aisuite-code-cli"
sys.path.insert(0, str(CLI_PACKAGE_ROOT))

from aisuite_code_cli import DEFAULT_ALLOWED_COMMANDS, CliConfig, CodeCli  # noqa: E402
from aisuite_code_cli.agent import build_agent, build_reviewer_agent  # noqa: E402
from aisuite_code_cli.approval import ApprovalController  # noqa: E402
from aisuite_code_cli.config import parse_args  # noqa: E402
from aisuite_code_cli.rendering import (  # noqa: E402
    print_steps,
    summarize_result_preview,
    summarize_tool_arguments,
)
from aisuite.framework.message import (
    ChatCompletionMessageToolCall,
    Function,
    Message,
)  # noqa: E402
from tests.agents.helpers import chat_response  # noqa: E402


def make_context(tool_name="run_shell", metadata=None):
    return ai.ToolPolicyContext(
        agent_name="assistant",
        tool_name=tool_name,
        arguments={"command": "npm run build"},
        run_name="code_cli_turn",
        trace_id="trace_1",
        group_id="group_1",
        tags=[],
        metadata={},
        messages=[],
        tool_metadata=metadata,
    )


def tool_call(name, arguments, call_id="call_1"):
    return ChatCompletionMessageToolCall(
        id=call_id,
        type="function",
        function=Function(name=name, arguments=arguments),
    )


@pytest.mark.skipif(tomllib is None, reason="tomllib requires Python 3.11+")
def test_cli_package_declares_openai_and_root_shim_exists():
    package_root = CLI_PACKAGE_ROOT
    pyproject = tomllib.loads((package_root / "pyproject.toml").read_text())

    assert "openai" in pyproject["tool"]["poetry"]["dependencies"]
    assert pyproject["tool"]["poetry"]["scripts"]["aisuite-code"] == (
        "aisuite_code_cli.main:main"
    )
    shim = package_root.parents[2] / "scripts" / "aisuite-code"
    assert shim.exists()
    assert "poetry run aisuite-code" in shim.read_text()


def test_parse_args_uses_code_defaults(tmp_path):
    config = parse_args(
        [
            "--model",
            "openai:gpt-4o-mini",
            "--cwd",
            str(tmp_path),
            "--trace-http",
            "http://127.0.0.1:8780/api/events",
        ]
    )

    assert config.model == "openai:gpt-4o-mini"
    assert config.cwd == tmp_path.resolve()
    assert config.trace_file == tmp_path / ".aisuite/code.jsonl"
    assert config.artifact_root == tmp_path / ".aisuite/artifacts"
    assert config.allowed_commands == DEFAULT_ALLOWED_COMMANDS
    assert config.allow_write is True
    assert config.allow_shell_all is False
    assert config.trace_http == "http://127.0.0.1:8780/api/events"
    assert config.enable_reviewer is True


def test_parse_args_can_disable_writes_and_override_commands(tmp_path):
    config = parse_args(
        [
            "--cwd",
            str(tmp_path),
            "--read-only",
            "--allow-command",
            "python3",
            "--allow-command",
            "pytest",
            "--viewer",
            "--no-reviewer",
        ]
    )

    assert config.allow_write is False
    assert config.allowed_commands == ["python3", "pytest"]
    assert config.start_viewer is True
    assert config.enable_reviewer is False


def test_build_agent_includes_code_metadata(tmp_path):
    agent = build_agent(
        CliConfig(
            model="openai:gpt-4o-mini",
            cwd=tmp_path,
            trace_file=tmp_path / "events.jsonl",
            allowed_commands=["python3"],
        )
    )

    assert agent.name == "aisuite_code"
    assert agent.metadata["app"] == "aisuite_code_cli"
    assert "code" in agent.tags
    assert any(getattr(tool, "__name__", "") == "write_file" for tool in agent.tools)
    assert any(
        getattr(tool, "__name__", "") == "replace_in_file" for tool in agent.tools
    )
    assert any(getattr(tool, "__name__", "") == "git_status" for tool in agent.tools)
    assert any(getattr(tool, "__name__", "") == "git_diff" for tool in agent.tools)
    assert any(getattr(tool, "__name__", "") == "apply_patch" for tool in agent.tools)
    assert any(
        getattr(tool, "__name__", "") == "review_changes" for tool in agent.tools
    )
    assert "replace_in_file for exact" in agent.instructions
    assert "apply_patch accepts only this Codex-style envelope" in agent.instructions
    assert "*** Begin Patch" in agent.instructions
    assert "apply_unified_diff is different" in agent.instructions


def test_build_agent_can_disable_reviewer_subagent_tool(tmp_path):
    agent = build_agent(
        CliConfig(
            model="openai:gpt-4o-mini",
            cwd=tmp_path,
            trace_file=tmp_path / "events.jsonl",
            allowed_commands=["python3"],
            enable_reviewer=False,
        )
    )

    assert all(
        getattr(tool, "__name__", "") != "review_changes" for tool in agent.tools
    )


def test_build_agent_can_include_reviewer_subagent_tool(tmp_path):
    agent = build_agent(
        CliConfig(
            model="openai:gpt-4o-mini",
            cwd=tmp_path,
            trace_file=tmp_path / "events.jsonl",
            allowed_commands=["python3"],
            enable_reviewer=True,
        )
    )
    reviewer = build_reviewer_agent(
        CliConfig(
            model="openai:gpt-4o-mini",
            cwd=tmp_path,
            trace_file=tmp_path / "events.jsonl",
            allowed_commands=["python3"],
            enable_reviewer=True,
        )
    )

    review_tool = next(
        tool for tool in agent.tools if tool.__name__ == "review_changes"
    )

    assert "call review_changes" in agent.instructions
    assert "Do not edit files" in reviewer.instructions
    assert reviewer.metadata["role"] == "reviewer"
    assert "write_file" not in {tool.__name__ for tool in reviewer.tools}
    assert "reviewer subagent" in (review_tool.__doc__ or "")


def test_reviewer_subagent_tool_emits_child_trace(tmp_path):
    client = ai.Client()
    provider = Mock()
    first_response = chat_response(None)
    first_response.choices[0].message = Message(
        role="assistant",
        tool_calls=[
            tool_call(
                "review_changes",
                '{"input": "Review the current project."}',
            )
        ],
    )
    provider.chat_completions_create.side_effect = [
        first_response,
        chat_response("No material issues found."),
        chat_response("Reviewer found no material issues."),
    ]
    client.providers["openai"] = provider
    sink = ai.tracing.InMemoryTraceSink()
    agent = build_agent(
        CliConfig(
            model="openai:gpt-4o-mini",
            cwd=tmp_path,
            trace_file=tmp_path / "events.jsonl",
            allowed_commands=["python3"],
            enable_reviewer=True,
        )
    )

    result = ai.Runner.run_sync(
        agent,
        "Use the reviewer subagent.",
        client=client,
        group_id="cli-review",
        trace_sinks=[sink],
    )

    assert result.final_output == "Reviewer found no material issues."
    completed_runs = [
        event.data["run"]
        for event in sink.events
        if event.event_type == "run.completed"
    ]
    child = next(run for run in completed_runs if run["agent_name"] == "reviewer")
    assert child["parent_run_id"] == result.trace_id
    assert child["group_id"] == "cli-review"


def test_approval_controller_allows_low_risk_tool_without_prompt():
    output = StringIO()
    controller = ApprovalController(input_stream=StringIO(""), output_stream=output)

    decision = controller.evaluate(make_context(tool_name="read_file"))

    assert decision.allowed is True
    assert decision.reason == "low risk"
    assert output.getvalue() == ""


def test_approval_controller_denies_high_risk_tool_by_default():
    output = StringIO()
    metadata = ai.ToolMetadata(
        category="shell",
        risk_level="high",
        requires_approval=True,
    )
    controller = ApprovalController(input_stream=StringIO("\n"), output_stream=output)

    decision = controller.evaluate(make_context(metadata=metadata))

    assert decision.allowed is False
    assert decision.reason == "denied by user"
    rendered = output.getvalue()
    assert "Permission required" in rendered
    assert "Action" in rendered
    assert "run shell command: npm run build" in rendered
    assert "Risk" in rendered
    assert "high · shell" in rendered
    assert "Effect" in rendered
    assert "Executes a command in the configured workspace." in rendered
    assert "Preview" in rendered
    assert "Allow? [y] once" in rendered


def test_approval_controller_can_allow_tool_for_session():
    output = StringIO()
    metadata = ai.ToolMetadata(
        category="shell",
        risk_level="high",
        requires_approval=True,
    )
    controller = ApprovalController(input_stream=StringIO("a\n"), output_stream=output)

    first = controller.evaluate(make_context(metadata=metadata))
    second = controller.evaluate(make_context(metadata=metadata))

    assert first.allowed is True
    assert first.reason == "tool allowed for session"
    assert second.allowed is True
    assert second.reason == "tool allowed for session"
    assert output.getvalue().count("Permission required") == 1


def test_approval_controller_can_allow_exact_shell_command_for_session():
    output = StringIO()
    metadata = ai.ToolMetadata(
        category="shell",
        risk_level="high",
        requires_approval=True,
    )
    controller = ApprovalController(input_stream=StringIO("c\n"), output_stream=output)

    first = controller.evaluate(make_context(metadata=metadata))
    second = controller.evaluate(make_context(metadata=metadata))

    assert first.allowed is True
    assert first.reason == "command allowed for session"
    assert second.allowed is True
    assert second.reason == "command allowed for session"
    rendered = output.getvalue()
    assert "always this command" in rendered
    assert "run shell command: npm run build" in rendered
    assert rendered.count("Permission required") == 1


def test_rendering_summarizes_tool_activity():
    assert (
        summarize_tool_arguments(
            "write_file",
            {"path": "app.py", "content": "one\ntwo\n"},
        )
        == " · app.py · 8 chars, 2 lines"
    )
    assert (
        summarize_tool_arguments(
            "run_shell",
            {"command": "python3 app.py"},
        )
        == " · python3 app.py"
    )
    assert (
        summarize_result_preview(
            "run_shell",
            '{"exit_code": 0, "stdout": "ok\\n", "stderr": "", "timed_out": false}',
        )
        == " · exit 0 · stdout 3 chars, 1 lines"
    )


def test_approval_controller_summarizes_large_arguments():
    output = StringIO()
    metadata = ai.ToolMetadata(
        category="filesystem",
        risk_level="medium",
        requires_approval=True,
    )
    controller = ApprovalController(input_stream=StringIO("n\n"), output_stream=output)
    context = make_context(tool_name="write_file", metadata=metadata)
    context.arguments = {
        "path": "large.txt",
        "content": "0123456789" * 200,
    }

    decision = controller.evaluate(context)

    rendered = output.getvalue()
    assert decision.allowed is False
    assert "write file: large.txt" in rendered
    assert "May create or overwrite a file" in rendered
    assert "content: 2000 chars" in rendered
    assert "0123456789" * 80 not in rendered
    assert "always this command" not in rendered


def test_approval_controller_summarizes_patch_without_dumping_content():
    output = StringIO()
    metadata = ai.ToolMetadata(
        category="filesystem",
        risk_level="medium",
        requires_approval=True,
    )
    controller = ApprovalController(input_stream=StringIO("n\n"), output_stream=output)
    context = make_context(tool_name="apply_patch", metadata=metadata)
    context.arguments = {
        "patch": "*** Begin Patch\n" + ("+0123456789\n" * 250) + "*** End Patch",
    }

    decision = controller.evaluate(context)

    rendered = output.getvalue()
    assert decision.allowed is False
    assert "apply Codex-style patch" in rendered
    assert "Applies one or more targeted file edits" in rendered
    assert "patch:" in rendered
    assert "+0123456789" * 40 not in rendered
    assert "always this command" not in rendered


def test_approval_controller_renders_reviewer_subagent_action():
    output = StringIO()
    metadata = ai.ToolMetadata(
        category="subagent",
        risk_level="medium",
        requires_approval=True,
    )
    controller = ApprovalController(input_stream=StringIO("y\n"), output_stream=output)
    context = make_context(tool_name="review_changes", metadata=metadata)
    context.arguments = {"input": "Review the current changes."}

    decision = controller.evaluate(context)

    rendered = output.getvalue()
    assert decision.allowed is True
    assert "invoke reviewer subagent" in rendered
    assert "Runs a read-only reviewer subagent" in rendered
    assert "input: Review the current changes." in rendered


def test_print_steps_renders_shell_result(tmp_path):
    output = StringIO()
    agent = build_agent(
        CliConfig(
            model="openai:gpt-4o-mini",
            cwd=tmp_path,
            trace_file=tmp_path / "events.jsonl",
            allowed_commands=["python3"],
        )
    )
    result = ai.RunResult(
        final_output="done",
        status="completed",
        agent=agent,
        last_agent=agent,
        input="run tests",
        messages=[],
        new_items=[],
        raw_responses=[],
        run_name="code_cli_turn",
        trace_id="trace_1",
        parent_run_id=None,
        group_id="group",
        tags=[],
        metadata={},
        steps=[
            ai.RunStep(
                id="step_1",
                type="tool_call",
                name="run_shell",
                trace_id="trace_1",
                data={
                    "allowed": True,
                    "arguments": {"command": "python3 -m pytest tests/toolkits"},
                    "reason": "approved by user",
                },
            ),
            ai.RunStep(
                id="step_2",
                type="tool_result",
                name="run_shell",
                trace_id="trace_1",
                data={
                    "status": "success",
                    "result_preview": (
                        '{"exit_code": 0, "stdout": "2 passed\\n", '
                        '"stderr": "", "timed_out": false}'
                    ),
                },
            ),
        ],
        max_turns=5,
    )

    print_steps(result, output)

    rendered = output.getvalue()
    assert "Activity" in rendered
    assert (
        "tool request: run_shell · allowed · python3 -m pytest tests/toolkits"
        in rendered
    )
    assert "tool result: run_shell · success · exit 0" in rendered
    assert "stdout 9 chars, 1 lines" in rendered


def test_code_cli_header_help_and_error_guidance(tmp_path):
    output = StringIO()
    cli = CodeCli(
        CliConfig(
            model="openai:gpt-4o-mini",
            cwd=tmp_path,
            trace_file=tmp_path / "events.jsonl",
            allowed_commands=DEFAULT_ALLOWED_COMMANDS,
        ),
        input_stream=StringIO(""),
        output_stream=output,
    )

    cli._print_header()
    cli._print_help()
    cli._print_error(ModuleNotFoundError("No module named 'openai'"))

    rendered = output.getvalue()
    assert "Session" in rendered
    assert "tools:    writes on" in rendered
    assert "Type /help for commands. Use /viewer start for local traces." in rendered
    assert "Create app.py with add(a, b), then run it." in rendered
    assert "Read README.md and summarize the project in 5 bullets." in rendered
    assert "python3 -m poetry install" in rendered
    assert "Use /status" in rendered
    assert "/last          show last turn details" in rendered


def test_code_cli_examples_command_prints_starter_prompts(tmp_path):
    output = StringIO()
    cli = CodeCli(
        CliConfig(
            model="openai:gpt-4o-mini",
            cwd=tmp_path,
            trace_file=tmp_path / "events.jsonl",
            allowed_commands=DEFAULT_ALLOWED_COMMANDS,
        ),
        input_stream=StringIO(""),
        output_stream=output,
    )

    assert cli._handle_command("/examples") is False

    rendered = output.getvalue()
    assert "Try" in rendered
    assert "Run the focused tests" in rendered


def make_run_result(cli, *, final_output="done", steps=None):
    return ai.RunResult(
        final_output=final_output,
        status="completed",
        agent=cli.agent,
        last_agent=cli.agent,
        input="hello",
        messages=[],
        new_items=[],
        raw_responses=[],
        run_name="code_cli_turn",
        trace_id="trace_123",
        parent_run_id=None,
        group_id="group",
        tags=[],
        metadata={},
        steps=steps or [],
        max_turns=5,
    )


def test_code_cli_prints_focused_viewer_trace_hint(tmp_path):
    output = StringIO()
    cli = CodeCli(
        CliConfig(
            model="openai:gpt-4o-mini",
            cwd=tmp_path,
            trace_file=tmp_path / "events.jsonl",
            allowed_commands=DEFAULT_ALLOWED_COMMANDS,
        ),
        input_stream=StringIO(""),
        output_stream=output,
    )
    cli.viewer = type("Viewer", (), {"url": "http://127.0.0.1:1234"})()
    cli.result = make_run_result(cli)

    cli._print_trace_hint()

    rendered = output.getvalue()
    assert "Trace: trace_123" in rendered
    assert "?embed=1&trace_id=trace_123" in rendered


def test_code_cli_run_agent_prints_working_and_answer(tmp_path, monkeypatch):
    output = StringIO()
    cli = CodeCli(
        CliConfig(
            model="openai:gpt-4o-mini",
            cwd=tmp_path,
            trace_file=tmp_path / "events.jsonl",
            allowed_commands=DEFAULT_ALLOWED_COMMANDS,
        ),
        input_stream=StringIO(""),
        output_stream=output,
    )

    monkeypatch.setattr(
        ai.Runner, "run_sync", lambda *args, **kwargs: make_run_result(cli)
    )

    cli._run_agent("hello")

    rendered = output.getvalue()
    assert "Working..." in rendered
    assert "Assistant" in rendered
    assert "done" in rendered
    assert "Trace: trace_123" in rendered


def test_code_cli_last_command_prints_last_turn_details(tmp_path):
    output = StringIO()
    cli = CodeCli(
        CliConfig(
            model="openai:gpt-4o-mini",
            cwd=tmp_path,
            trace_file=tmp_path / "events.jsonl",
            allowed_commands=DEFAULT_ALLOWED_COMMANDS,
        ),
        input_stream=StringIO(""),
        output_stream=output,
    )
    cli.result = make_run_result(
        cli,
        steps=[
            ai.RunStep(
                id="step_1",
                type="tool_call",
                name="read_file",
                trace_id="trace_123",
                data={"arguments": {"path": "README.md"}},
            )
        ],
    )

    assert cli._handle_command("/last") is False

    rendered = output.getvalue()
    assert "Last turn" in rendered
    assert "trace: trace_123" in rendered
    assert "tool_call: read_file" in rendered


def test_code_cli_last_command_handles_empty_session(tmp_path):
    output = StringIO()
    cli = CodeCli(
        CliConfig(
            model="openai:gpt-4o-mini",
            cwd=tmp_path,
            trace_file=tmp_path / "events.jsonl",
            allowed_commands=DEFAULT_ALLOWED_COMMANDS,
        ),
        input_stream=StringIO(""),
        output_stream=output,
    )

    assert cli._handle_command("/last") is False

    assert "No turns yet." in output.getvalue()


def test_code_cli_viewer_command_prints_hint(tmp_path):
    output = StringIO()
    cli = CodeCli(
        CliConfig(
            model="openai:gpt-4o-mini",
            cwd=tmp_path,
            trace_file=tmp_path / "events.jsonl",
            allowed_commands=DEFAULT_ALLOWED_COMMANDS,
        ),
        input_stream=StringIO(""),
        output_stream=output,
    )

    assert cli._handle_command("/viewer") is False

    rendered = output.getvalue()
    assert "python -m aisuite.tracing.viewer" in rendered
    assert "--artifact-root" in rendered
    assert "pass --trace-http" in rendered


def test_code_cli_viewer_start_launches_viewer(tmp_path, monkeypatch):
    output = StringIO()
    cli = CodeCli(
        CliConfig(
            model="openai:gpt-4o-mini",
            cwd=tmp_path,
            trace_file=tmp_path / "events.jsonl",
            allowed_commands=DEFAULT_ALLOWED_COMMANDS,
        ),
        input_stream=StringIO(""),
        output_stream=output,
    )
    viewer = type(
        "Viewer",
        (),
        {"url": "http://127.0.0.1:1234", "stop": lambda self: None},
    )()
    calls = []

    def start_viewer(trace_file, port, artifact_root=None):
        calls.append((trace_file, port, artifact_root))
        return viewer

    monkeypatch.setattr(ai.tracing, "start_viewer", start_viewer)

    assert cli._handle_command("/viewer start") is False

    assert calls == [(tmp_path / "events.jsonl", 0, tmp_path / ".aisuite/artifacts")]
    assert "http://127.0.0.1:1234" in output.getvalue()

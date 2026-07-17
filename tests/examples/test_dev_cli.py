from io import StringIO
from pathlib import Path

import aisuite as ai
from examples.cli.dev import (
    DEFAULT_ALLOWED_COMMANDS,
    ApprovalController,
    CliConfig,
    DevCli,
    parse_args,
)
from examples.cli.create_demo_trace import create_demo_trace


def make_context(tool_name="run_shell", metadata=None):
    return ai.ToolPolicyContext(
        agent_name="assistant",
        tool_name=tool_name,
        arguments={"command": "python3 -m pytest"},
        run_name="cli_turn",
        trace_id="trace_1",
        group_id="group_1",
        tags=[],
        metadata={},
        messages=[],
        tool_metadata=metadata,
    )


def test_parse_args_uses_default_allowed_commands():
    config = parse_args(
        [
            "--model",
            "openai:gpt-4o-mini",
            "--cwd",
            ".",
            "--trace-http",
            "http://127.0.0.1:8780/api/events",
        ]
    )

    assert config.model == "openai:gpt-4o-mini"
    assert config.allowed_commands == DEFAULT_ALLOWED_COMMANDS
    assert config.allow_write is False
    assert config.allow_shell_all is False
    assert config.trace_http == "http://127.0.0.1:8780/api/events"


def test_parse_args_accepts_repeated_allowed_commands():
    config = parse_args(
        [
            "--allow-command",
            "python3",
            "--allow-command",
            "pytest",
        ]
    )

    assert config.allowed_commands == ["python3", "pytest"]


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
    assert "Permission required" in output.getvalue()
    assert "run_shell" in output.getvalue()


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
    assert first.reason == "allowed for session"
    assert second.allowed is True
    assert second.reason == "allowed for session"
    assert output.getvalue().count("Permission required") == 1


def test_dev_cli_prints_tool_arguments_and_result_preview(tmp_path):
    output = StringIO()
    cli = DevCli(
        CliConfig(
            model="openai:gpt-4o-mini",
            cwd=tmp_path,
            trace_file=tmp_path / "events.jsonl",
            allowed_commands=DEFAULT_ALLOWED_COMMANDS,
        ),
        input_stream=StringIO(""),
        output_stream=output,
    )
    result = ai.RunResult(
        final_output="done",
        status="completed",
        agent=cli.agent,
        last_agent=cli.agent,
        input="run tests",
        messages=[],
        new_items=[],
        raw_responses=[],
        run_name="cli_turn",
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

    cli._print_steps(result)

    rendered = output.getvalue()
    assert "arguments:" in rendered
    assert "command: python3 -m pytest tests/toolkits" in rendered
    assert "exit_code: 0" in rendered
    assert "stdout: 2 passed\\n" in rendered


def test_dev_cli_viewer_command_prints_hint(tmp_path):
    output = StringIO()
    cli = DevCli(
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
    assert "python -m aisuite.agents.viewer" in rendered
    assert "pass --trace-http" in rendered


def test_dev_cli_viewer_start_launches_viewer(tmp_path, monkeypatch):
    output = StringIO()
    cli = DevCli(
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

    def start_viewer(trace_file, port):
        calls.append((trace_file, port))
        return viewer

    monkeypatch.setattr(ai.tracing, "start_viewer", start_viewer)

    assert cli._handle_command("/viewer start") is False

    assert calls == [(tmp_path / "events.jsonl", 0)]
    assert "http://127.0.0.1:1234" in output.getvalue()


def test_create_demo_trace_writes_sample_runs(tmp_path):
    trace_file = tmp_path / "demo.jsonl"

    create_demo_trace(trace_file=trace_file, cwd=Path.cwd())

    runs = ai.tracing.read_trace_file(trace_file)
    assert len(runs) == 8
    assert {run["group_id"] for run in runs} == {"aisuite-demo"}
    assert {run["metadata"]["app"] for run in runs} == {"aisuite_demo_trace"}
    assert any(
        step["name"] == "read_file_lines"
        for run in runs
        for step in run["steps"]
        if step["type"] == "tool_call"
    )
    assert any(
        step["name"] == "run_shell"
        for run in runs
        for step in run["steps"]
        if step["type"] == "tool_result"
    )
    assert any(
        step["name"] == "write_file" and step["data"].get("argument_artifacts")
        for run in runs
        for step in run["steps"]
        if step["type"] == "tool_call"
    )
    assert any(
        step["name"] == "run_shell" and step["data"].get("result_artifacts")
        for run in runs
        for step in run["steps"]
        if step["type"] == "tool_result"
    )


def test_create_demo_trace_includes_subagent_and_denied_event(tmp_path):
    trace_file = tmp_path / "demo.jsonl"

    create_demo_trace(trace_file=trace_file, cwd=Path.cwd())

    runs = ai.tracing.read_trace_file(trace_file)
    assert any(run["agent_name"] == "reviewer" for run in runs)
    parent = next(
        run
        for run in runs
        if run["agent_name"] == "aisuite_demo_dev" and run["display"]["child_count"]
    )
    reviewer = next(run for run in runs if run["agent_name"] == "reviewer")
    assert reviewer["parent_run_id"] == parent["trace_id"]
    assert any(
        event["event_type"] == "tool.denied"
        and event["data"].get("reason") == "blocked by demo policy"
        for run in runs
        for event in run["events"]
    )
    assert any(
        activity.get("tool_name") == "review_changes"
        and activity.get("child_run", {}).get("trace_id") == reviewer["trace_id"]
        for run in runs
        for activity in run["display"]["activities"]
    )

import json
from unittest.mock import Mock

import pytest

import aisuite as ai
from aisuite.framework.message import ChatCompletionMessageToolCall, Function, Message
from tests.agents.helpers import chat_response


def tool_call(name, arguments, call_id="call_1"):
    return ChatCompletionMessageToolCall(
        id=call_id,
        type="function",
        function=Function(name=name, arguments=arguments),
    )


def shell_tool(**kwargs):
    return {fn.__name__: fn for fn in ai.toolkits.shell(**kwargs)}["run_shell"]


def test_shell_toolkit_requires_allowlist_or_allow_all(tmp_path):
    with pytest.raises(ValueError):
        ai.toolkits.shell(cwd=tmp_path)


def test_shell_toolkit_runs_allowed_command(tmp_path):
    run_shell = shell_tool(cwd=tmp_path, allowed_commands=["python3"])

    result = run_shell("python3 -c 'print(\"hello\")'")

    assert result["exit_code"] == 0
    assert result["stdout"] == "hello\n"
    assert result["stderr"] == ""
    assert result["timed_out"] is False


def test_shell_toolkit_denies_unlisted_command(tmp_path):
    run_shell = shell_tool(cwd=tmp_path, allowed_commands=["python3"])

    with pytest.raises(PermissionError):
        run_shell("git status")


def test_shell_toolkit_rejects_shell_syntax_when_shell_disabled(tmp_path):
    run_shell = shell_tool(cwd=tmp_path, allowed_commands=["python3", "cat", "echo"])

    with pytest.raises(ValueError, match="Shell redirection"):
        run_shell("echo hello > out.txt")
    with pytest.raises(ValueError, match="Shell redirection"):
        run_shell("cat <<'EOF'\nhello\nEOF")
    with pytest.raises(ValueError, match="Shell redirection"):
        run_shell("python3 -m pytest && npm test")


def test_shell_toolkit_allows_shell_syntax_when_shell_enabled(tmp_path):
    run_shell = shell_tool(
        cwd=tmp_path,
        allowed_commands=["echo"],
        allow_shell=True,
    )

    result = run_shell("echo hello > out.txt")

    assert result["exit_code"] == 0
    assert (tmp_path / "out.txt").read_text(encoding="utf-8") == "hello\n"


def test_shell_toolkit_captures_nonzero_exit_code(tmp_path):
    run_shell = shell_tool(cwd=tmp_path, allowed_commands=["python3"])

    result = run_shell("python3 -c 'import sys; sys.exit(7)'")

    assert result["exit_code"] == 7
    assert result["timed_out"] is False


def test_shell_toolkit_reports_timeout(tmp_path):
    run_shell = shell_tool(cwd=tmp_path, allowed_commands=["python3"])

    result = run_shell("python3 -c 'import time; time.sleep(2)'", timeout_seconds=1)

    assert result["exit_code"] is None
    assert result["timed_out"] is True


def test_shell_toolkit_timeout_output_is_text(tmp_path):
    run_shell = shell_tool(cwd=tmp_path, allowed_commands=["python3"])

    result = run_shell(
        "python3 -c 'import time; print(\"before\", flush=True); time.sleep(2)'",
        timeout_seconds=1,
    )

    assert result["timed_out"] is True
    assert isinstance(result["stdout"], str)
    assert "before" in result["stdout"]


def test_shell_toolkit_attaches_high_risk_metadata(tmp_path):
    run_shell = shell_tool(cwd=tmp_path, allowed_commands=["python3"])

    metadata = run_shell.__aisuite_tool_metadata__

    assert metadata.category == "shell"
    assert metadata.risk_level == "high"
    assert metadata.requires_approval is True
    assert metadata.capabilities == ["run_command"]


def test_shell_toolkit_can_be_used_by_agent_tool_loop(tmp_path):
    client = ai.Client()
    provider = Mock()
    first_response = chat_response(None)
    first_response.choices[0].message = Message(
        role="assistant",
        tool_calls=[
            tool_call(
                "run_shell",
                '{"command": "python3 -c \\"print(42)\\""}',
            )
        ],
    )
    provider.chat_completions_create.side_effect = [
        first_response,
        chat_response("Command printed 42."),
    ]
    client.providers["openai"] = provider

    result = ai.Runner.run_sync(
        ai.Agent(
            name="runner",
            model="openai:gpt-4o",
            tools=ai.toolkits.shell(cwd=tmp_path, allowed_commands=["python3"]),
        ),
        "Run the command",
        client=client,
    )

    assert result.final_output == "Command printed 42."
    assert result.steps[-2].data["tool_metadata"]["category"] == "shell"
    tool_message = provider.chat_completions_create.call_args_list[1].args[1][-1]
    assert '"stdout": "42\\n"' in tool_message["content"]


def test_shell_tool_stores_large_output_artifact_without_hiding_model_context(tmp_path):
    client = ai.Client()
    provider = Mock()
    first_response = chat_response(None)
    first_response.choices[0].message = Message(
        role="assistant",
        tool_calls=[
            tool_call(
                "run_shell",
                json.dumps({"command": "python3 -c 'print(\"x\" * 21001)'"}),
            )
        ],
    )
    provider.chat_completions_create.side_effect = [
        first_response,
        chat_response("saw full output"),
    ]
    client.providers["openai"] = provider
    artifact_store = ai.InMemoryArtifactStore()

    result = ai.Runner.run_sync(
        ai.Agent(
            name="runner",
            model="openai:gpt-4o",
            tools=ai.toolkits.shell(cwd=tmp_path, allowed_commands=["python3"]),
        ),
        "Run the command",
        client=client,
        artifact_store=artifact_store,
    )

    tool_message = provider.chat_completions_create.call_args_list[1].args[1][-1]
    assert "x" * 21001 in tool_message["content"]
    tool_result_steps = [step for step in result.steps if step.type == "tool_result"]
    assert (
        tool_result_steps[0].data["result_artifacts"][0]["artifact_ref"]["metadata"][
            "field"
        ]
        == "stdout"
    )
    ref = ai.ArtifactRef.from_dict(
        tool_result_steps[0].data["result_artifacts"][0]["artifact_ref"]
    )
    assert artifact_store.get(ref).text().startswith("x" * 21001)

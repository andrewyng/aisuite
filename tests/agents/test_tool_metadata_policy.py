from unittest.mock import Mock

import aisuite as ai
from aisuite.framework.message import ChatCompletionMessageToolCall, Function, Message
from tests.agents.helpers import chat_response


def tool_call(name, arguments, call_id="call_1"):
    return ChatCompletionMessageToolCall(
        id=call_id,
        type="function",
        function=Function(name=name, arguments=arguments),
    )


def test_plain_custom_tool_still_runs_without_metadata_or_policy():
    client = ai.Client()
    provider = Mock()
    first_response = chat_response(None)
    first_response.choices[0].message = Message(
        role="assistant",
        tool_calls=[tool_call("lookup", '{"city": "Paris"}')],
    )
    provider.chat_completions_create.side_effect = [
        first_response,
        chat_response("done"),
    ]
    client.providers["openai"] = provider
    calls = []

    def lookup(city: str) -> str:
        calls.append(city)
        return f"{city} found"

    result = ai.Runner.run_sync(
        ai.Agent(name="assistant", model="openai:gpt-4o", tools=[lookup]),
        "Find Paris",
        client=client,
    )

    assert result.final_output == "done"
    assert calls == ["Paris"]
    assert result.steps[-2].data["allowed"] is True
    assert "tool_metadata" not in result.steps[-2].data


def test_tool_metadata_is_available_to_policy_and_traces():
    client = ai.Client()
    provider = Mock()
    first_response = chat_response(None)
    first_response.choices[0].message = Message(
        role="assistant",
        tool_calls=[tool_call("write_note", '{"path": "note.txt"}')],
    )
    provider.chat_completions_create.side_effect = [
        first_response,
        chat_response("done"),
    ]
    client.providers["openai"] = provider
    seen_metadata = []

    def write_note(path: str) -> str:
        return f"wrote {path}"

    wrapped_tool = ai.tool(
        write_note,
        metadata=ai.ToolMetadata(
            category="filesystem",
            risk_level="medium",
            capabilities=["write_file"],
            requires_approval=True,
        ),
    )

    def policy(context):
        seen_metadata.append(context.tool_metadata)
        return ai.ToolPolicyDecision(allowed=True, reason="approved")

    result = ai.Runner.run_sync(
        ai.Agent(name="assistant", model="openai:gpt-4o", tools=[wrapped_tool]),
        "Write a note",
        client=client,
        tool_policy=policy,
    )

    assert seen_metadata[0].category == "filesystem"
    assert seen_metadata[0].risk_level == "medium"
    assert result.steps[-2].data["tool_metadata"]["category"] == "filesystem"
    assert result.steps[-2].data["tool_metadata"]["requires_approval"] is True


def test_require_approval_policy_can_deny_each_invocation():
    client = ai.Client()
    provider = Mock()
    first_response = chat_response(None)
    first_response.choices[0].message = Message(
        role="assistant",
        tool_calls=[tool_call("run_shell", '{"command": "rm -rf /tmp/x"}')],
    )
    provider.chat_completions_create.side_effect = [
        first_response,
        chat_response("denied"),
    ]
    client.providers["openai"] = provider
    approvals = []

    def run_shell(command: str) -> str:
        return command

    shell_tool = ai.tool(
        run_shell,
        metadata=ai.ToolMetadata(category="shell", risk_level="high"),
    )

    def approval_callback(context):
        approvals.append((context.tool_name, context.arguments["command"]))
        return ai.ToolPolicyDecision(allowed=False, reason="user denied")

    result = ai.Runner.run_sync(
        ai.Agent(name="assistant", model="openai:gpt-4o", tools=[shell_tool]),
        "Run cleanup",
        client=client,
        tool_policy=ai.RequireApprovalPolicy(approval_callback),
    )

    assert result.final_output == "denied"
    assert approvals == [("run_shell", "rm -rf /tmp/x")]
    assert result.steps[-1].data["allowed"] is False
    assert result.steps[-1].data["reason"] == "user denied"
    assert result.steps[-1].data["tool_metadata"]["risk_level"] == "high"


def test_allow_tools_policy_allows_named_tools_only():
    client = ai.Client()
    provider = Mock()
    first_response = chat_response(None)
    first_response.choices[0].message = Message(
        role="assistant",
        tool_calls=[tool_call("lookup", '{"city": "Paris"}')],
    )
    provider.chat_completions_create.side_effect = [
        first_response,
        chat_response("blocked"),
    ]
    client.providers["openai"] = provider
    calls = []

    def lookup(city: str) -> str:
        calls.append(city)
        return city

    result = ai.Runner.run_sync(
        ai.Agent(name="assistant", model="openai:gpt-4o", tools=[lookup]),
        "Find Paris",
        client=client,
        tool_policy=ai.AllowToolsPolicy(["read_file"]),
    )

    assert result.final_output == "blocked"
    assert calls == []
    assert result.steps[-1].data["allowed"] is False
    assert result.steps[-1].data["reason"] == "tool not in allowlist"

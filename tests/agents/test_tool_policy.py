from unittest.mock import Mock

from aisuite import Agent, Client, RunState, Runner, ToolPolicyDecision
from aisuite.framework.message import ChatCompletionMessageToolCall, Function, Message
from tests.agents.helpers import chat_response


def tool_call(name, arguments, call_id="call_1"):
    return ChatCompletionMessageToolCall(
        id=call_id,
        type="function",
        function=Function(name=name, arguments=arguments),
    )


def test_allowed_tool_policy_executes_tool():
    client = Client()
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
        """Lookup a city."""
        calls.append(city)
        return f"{city} found"

    def policy(context):
        assert context.agent_name == "assistant"
        assert context.tool_name == "lookup"
        assert context.arguments == {"city": "Paris"}
        assert context.metadata == {"request_id": "req_1"}
        return True

    agent = Agent(name="assistant", model="openai:gpt-4o", tools=[lookup])

    result = Runner.run_sync(
        agent,
        "Find Paris",
        client=client,
        metadata={"request_id": "req_1"},
        tool_policy=policy,
    )

    assert result.final_output == "done"
    assert calls == ["Paris"]
    assert [step.type for step in result.steps[-2:]] == ["tool_call", "tool_result"]
    assert result.steps[-2].name == "lookup"
    assert result.steps[-2].data == {
        "type": "tool_call",
        "tool_name": "lookup",
        "tool_call_id": "call_1",
        "allowed": True,
        "reason": None,
        "metadata": {},
    }
    assert result.steps[-1].data == {
        "type": "tool_result",
        "tool_name": "lookup",
        "tool_call_id": "call_1",
        "status": "success",
    }

    state = RunState.from_dict(result.to_state().to_dict())
    assert [step.type for step in state.steps[-2:]] == ["tool_call", "tool_result"]
    assert state.steps[-2].data["allowed"] is True


def test_denied_tool_policy_does_not_execute_tool():
    client = Client()
    provider = Mock()
    first_response = chat_response(None)
    first_response.choices[0].message = Message(
        role="assistant",
        tool_calls=[tool_call("lookup", '{"city": "Paris"}')],
    )
    provider.chat_completions_create.side_effect = [
        first_response,
        chat_response("I cannot use that tool."),
    ]
    client.providers["openai"] = provider
    calls = []

    def lookup(city: str) -> str:
        """Lookup a city."""
        calls.append(city)
        return f"{city} found"

    def policy(context):
        return ToolPolicyDecision(
            allowed=False,
            reason="lookup disabled",
            metadata={"policy": "deny_lookup"},
        )

    agent = Agent(name="assistant", model="openai:gpt-4o", tools=[lookup])

    result = Runner.run_sync(
        agent,
        "Find Paris",
        client=client,
        tool_policy=policy,
    )

    assert result.final_output == "I cannot use that tool."
    assert calls == []
    tool_message = provider.chat_completions_create.call_args_list[1].args[1][-1]
    assert tool_message["role"] == "tool"
    assert "Tool call denied by policy" in tool_message["content"]
    assert result.steps[-1].type == "tool_call"
    assert result.steps[-1].data == {
        "type": "tool_call",
        "tool_name": "lookup",
        "tool_call_id": "call_1",
        "allowed": False,
        "reason": "lookup disabled",
        "metadata": {"policy": "deny_lookup"},
    }
    assert "tool_result" not in [step.type for step in result.steps[-2:]]

    state = RunState.from_dict(result.to_state().to_dict())
    assert state.steps[-1].data["allowed"] is False


def test_class_based_tool_policy_works():
    client = Client()
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
        """Lookup a city."""
        calls.append(city)
        return f"{city} found"

    class AllowPolicy:
        def evaluate(self, context):
            return ToolPolicyDecision(allowed=True, reason="approved")

    agent = Agent(name="assistant", model="openai:gpt-4o", tools=[lookup])
    Runner.run_sync(agent, "Find Paris", client=client, tool_policy=AllowPolicy())

    assert calls == ["Paris"]

import json
from unittest.mock import Mock

import aisuite as ai
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
        "arguments": {"city": "Paris"},
        "allowed": True,
        "reason": None,
        "metadata": {},
    }
    assert result.steps[-1].data == {
        "type": "tool_result",
        "tool_name": "lookup",
        "tool_call_id": "call_1",
        "status": "success",
        "result_preview": '"Paris found"',
    }

    state = RunState.from_dict(result.to_state().to_dict())
    assert [step.type for step in state.steps[-2:]] == ["tool_call", "tool_result"]
    assert state.steps[-2].data["allowed"] is True


def test_tool_trace_events_are_emitted_in_execution_order():
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

    def lookup(city: str) -> str:
        """Lookup a city."""
        return f"{city} found"

    sink = ai.tracing.InMemoryTraceSink()
    agent = Agent(name="assistant", model="openai:gpt-4o", tools=[lookup])

    Runner.run_sync(agent, "Find Paris", client=client, trace_sinks=[sink])

    assert [event.event_type for event in sink.events] == [
        "run.started",
        "model.send",
        "model.response",
        "tool.allowed",
        "tool.started",
        "tool.completed",
        "model.send",
        "model.response",
        "run.completed",
    ]
    first_model_response = sink.events[2].data["response"]
    assert first_model_response["kind"] == "tool_calls"
    assert first_model_response["tool_calls"][0]["name"] == "lookup"
    tool_result = sink.events[5].data
    assert tool_result["tool_name"] == "lookup"
    assert tool_result["result_preview"] == '"Paris found"'
    final_model_response = sink.events[7].data["response"]
    assert final_model_response["kind"] == "text"
    assert final_model_response["text_preview"] == "done"


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
        "arguments": {"city": "Paris"},
        "allowed": False,
        "reason": "lookup disabled",
        "metadata": {"policy": "deny_lookup"},
    }
    assert "tool_result" not in [step.type for step in result.steps[-2:]]

    state = RunState.from_dict(result.to_state().to_dict())
    assert state.steps[-1].data["allowed"] is False


def test_tool_result_preview_remains_valid_json_when_truncated():
    client = Client()
    provider = Mock()
    first_response = chat_response(None)
    first_response.choices[0].message = Message(
        role="assistant",
        tool_calls=[tool_call("collect_output", "{}")],
    )
    provider.chat_completions_create.side_effect = [
        first_response,
        chat_response("done"),
    ]
    client.providers["openai"] = provider

    def collect_output() -> dict:
        """Collect output."""
        return {
            "exit_code": 0,
            "stdout": "x" * 5000,
            "stderr": "",
            "timed_out": False,
        }

    result = Runner.run_sync(
        Agent(name="assistant", model="openai:gpt-4o", tools=[collect_output]),
        "Collect output",
        client=client,
    )

    preview = json.loads(result.steps[-1].data["result_preview"])
    assert preview["exit_code"] == 0
    assert preview["stdout"].endswith("...")
    assert len(preview["stdout"]) < 1000


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

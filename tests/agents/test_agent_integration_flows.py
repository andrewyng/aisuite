from unittest.mock import Mock

import aisuite as ai
from aisuite.framework.message import ChatCompletionMessageToolCall, Function, Message
from tests.agents.helpers import chat_response


def tool_call(name, arguments, call_id):
    return ChatCompletionMessageToolCall(
        id=call_id,
        type="function",
        function=Function(name=name, arguments=arguments),
    )


def test_denied_tool_policy_flow_emits_denial_event():
    client = ai.Client()
    provider = Mock()
    first_response = chat_response(None)
    first_response.choices[0].message = Message(
        role="assistant",
        tool_calls=[tool_call("lookup_secret", '{"key": "token"}', "call_secret")],
    )
    provider.chat_completions_create.side_effect = [
        first_response,
        chat_response("I cannot access that."),
    ]
    client.providers["openai"] = provider
    sink = ai.tracing.InMemoryTraceSink()

    def lookup_secret(key: str) -> str:
        return f"secret: {key}"

    def deny_policy(context):
        return ai.ToolPolicyDecision(allowed=False, reason="blocked")

    result = ai.Runner.run_sync(
        ai.Agent(name="assistant", model="openai:gpt-4o", tools=[lookup_secret]),
        "Find the token",
        client=client,
        tool_policy=deny_policy,
        trace_sinks=[sink],
    )

    assert result.final_output == "I cannot access that."
    assert "tool.denied" in [event.event_type for event in sink.events]
    denied = next(event for event in sink.events if event.event_type == "tool.denied")
    assert denied.data["tool_name"] == "lookup_secret"
    assert denied.data["reason"] == "blocked"

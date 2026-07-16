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


def test_parent_agent_subagent_flow_is_reconstructable_from_jsonl_store(tmp_path):
    client = ai.Client()
    provider = Mock()
    first_response = chat_response(None)
    first_response.choices[0].message = Message(
        role="assistant",
        tool_calls=[
            tool_call(
                "research_topic",
                '{"input": "agent observability"}',
                "call_research",
            )
        ],
    )
    provider.chat_completions_create.side_effect = [
        first_response,
        chat_response("research notes"),
        chat_response("final recommendation"),
    ]
    client.providers["openai"] = provider
    trace_file = tmp_path / "events.jsonl"
    sink = ai.tracing.LocalTraceSink(trace_file)
    researcher = ai.Agent(name="researcher", model="openai:gpt-4o-mini")
    writer = ai.Agent(
        name="writer",
        model="openai:gpt-4o",
        tools=[ai.agent_tool(researcher, name="research_topic")],
    )

    parent = ai.Runner.run_sync(
        writer,
        "Write a recommendation",
        client=client,
        run_name="recommendation",
        group_id="workflow_1",
        tags=["integration"],
        metadata={"request_id": "req_1"},
        trace_sinks=[sink],
    )

    runs = ai.tracing.JsonlTraceStore(trace_file).list_runs()
    parent_run = next(run for run in runs if run["trace_id"] == parent.trace_id)
    child_run = next(run for run in runs if run["agent_name"] == "researcher")

    assert parent_run["final_output"] == "final recommendation"
    assert child_run["final_output"] == "research notes"
    assert child_run["parent_run_id"] == parent.trace_id
    assert {run["group_id"] for run in runs} == {"workflow_1"}
    assert [
        call.args[0] for call in provider.chat_completions_create.call_args_list
    ] == [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4o",
    ]


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
